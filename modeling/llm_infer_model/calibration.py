from __future__ import annotations

import json
import re
import statistics
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from .analytical import analyze_decode, state_bytes_by_layer
from .model import ModelConfig, config_from_dict
from .simulator import simulate_decode


H2D_PATTERN = re.compile(
    r"bandwidthTest-H2D-(?P<memory>Pinned|Paged),\s+"
    r"Bandwidth = (?P<bandwidth>[0-9.]+) MB/s,\s+"
    r"Time = (?P<time>[0-9.]+) s,\s+"
    r"Size = (?P<size>\d+) bytes"
)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: {command}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def parse_llama_bench_payload(
    payload: Any, *, prompt_tokens: int, generation_tokens: int
) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise RuntimeError("expected llama-bench JSON array")
    matches = [
        row
        for row in payload
        if int(row.get("n_prompt", -1)) == 0
        and int(row.get("n_gen", -1)) == generation_tokens
        and int(row.get("n_depth", 0)) == prompt_tokens
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one generation row at the requested cached depth"
        )
    result = matches[0]
    avg_tokens_per_second = float(result["avg_ts"])
    return {
        "result": result,
        "avg_token_seconds": 1.0 / avg_tokens_per_second,
        "avg_tokens_per_second": avg_tokens_per_second,
        "decode_samples_ns": list(result.get("samples_ns", [])),
        "decode_estimation_method": (
            "generation-only llama-bench row after restoring cached --n-depth state"
            if prompt_tokens > 0
            else "decode-only llama-bench row"
        ),
    }


def run_llama_bench(
    executable: str | Path,
    model: str | Path,
    *,
    generation_tokens: int = 128,
    prompt_tokens: int = 0,
    repetitions: int = 5,
    threads: int = 16,
    flash_attention: bool = True,
    window_size: int | None = None,
) -> dict[str, Any]:
    if generation_tokens <= 0 or prompt_tokens < 0 or repetitions <= 0 or threads <= 0:
        raise ValueError(
            "generation_tokens, repetitions, and threads must be positive; "
            "prompt_tokens must be non-negative"
        )
    if window_size is not None and window_size <= 0:
        raise ValueError("window_size must be positive")

    command = [
        str(Path(executable).resolve()),
        "-m",
        str(Path(model).resolve()),
        "-p",
        "0",
        "-n",
        str(generation_tokens),
        "-r",
        str(repetitions),
        "-t",
        str(threads),
        "-fa",
        "1" if flash_attention else "0",
        "-o",
        "json",
    ]
    if prompt_tokens > 0:
        command.extend(("-d", str(prompt_tokens)))
    if window_size is None:
        command.extend(("-ngl", "99"))
        benchmark_kind = "resident_decode"
    else:
        command.extend(("-pws", str(window_size)))
        benchmark_kind = "pipeline_decode"

    completed = _run(command)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"llama-bench did not emit JSON on stdout:\n{completed.stdout}"
        ) from exc
    parsed = parse_llama_bench_payload(
        payload,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
    )
    avg_tokens_per_second = float(parsed["avg_tokens_per_second"])
    return {
        "schema_version": 2,
        "benchmark": benchmark_kind,
        "command": command,
        "window_size": window_size,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "avg_token_seconds": parsed["avg_token_seconds"],
        "avg_tokens_per_second": avg_tokens_per_second,
        "decode_samples_ns": parsed["decode_samples_ns"],
        "decode_estimation_method": parsed["decode_estimation_method"],
        "result": parsed["result"],
        "stderr": completed.stderr,
    }


def run_h2d_bandwidth_test(
    executable: str | Path,
    *,
    memory: str = "pinned",
    start_mib: int = 32,
    end_mib: int = 160,
    increment_mib: int = 16,
    device: int = 0,
) -> dict[str, Any]:
    if memory not in {"pinned", "pageable"}:
        raise ValueError("memory must be pinned or pageable")
    if start_mib <= 0 or end_mib < start_mib or increment_mib <= 0:
        raise ValueError("invalid H2D size range")
    command = [
        str(Path(executable).resolve()),
        f"--device={device}",
        f"--memory={memory}",
        "--mode=range",
        f"--start={start_mib * 1024**2}",
        f"--end={end_mib * 1024**2}",
        f"--increment={increment_mib * 1024**2}",
        "--htod",
        "--csv",
    ]
    completed = _run(command)
    rows = [
        {
            "memory": match.group("memory").lower(),
            "size_bytes": int(match.group("size")),
            "time_seconds": float(match.group("time")),
            "bandwidth_MiB_per_s": float(match.group("bandwidth")),
            "bandwidth_bytes_per_s": float(match.group("bandwidth")) * 1024**2,
        }
        for match in H2D_PATTERN.finditer(completed.stdout)
    ]
    if not rows:
        raise RuntimeError(f"could not parse bandwidthTest output:\n{completed.stdout}")
    bandwidths = [float(row["bandwidth_bytes_per_s"]) for row in rows]
    return {
        "schema_version": 1,
        "benchmark": "cuda_bandwidth_test_h2d",
        "command": command,
        "memory": memory,
        "rows": rows,
        "mean_bandwidth_bytes_per_s": statistics.fmean(bandwidths),
        "median_bandwidth_bytes_per_s": statistics.median(bandwidths),
        "min_bandwidth_bytes_per_s": min(bandwidths),
        "max_bandwidth_bytes_per_s": max(bandwidths),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def evaluate_context_validation(
    config: ModelConfig,
    specification_path: str | Path,
) -> dict[str, Any]:
    """Compare contextual llama-bench rows with the calibrated simulator."""
    specification_file = Path(specification_path).resolve()
    specification = _read_json(specification_file)
    project_root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    for item in specification["measurements"]:
        result_path = Path(str(item["file"]))
        if not result_path.is_absolute():
            result_path = project_root / result_path
        result = _read_json(result_path)
        generation_tokens = int(result["generation_tokens"])
        measured_seconds = (
            float(result["result"]["avg_ns"]) / generation_tokens / 1e9
        )
        sample_seconds = [
            float(value) / generation_tokens / 1e9
            for value in result.get("decode_samples_ns", [])
        ]
        window_size = item.get("window_size")
        context_tokens = int(result["prompt_tokens"])
        row: dict[str, Any] = {
            "mode": str(item["mode"]),
            "measurement_role": str(item.get("role", "diagnostic")),
            "file": result_path.as_posix(),
            "window_size": int(window_size) if window_size is not None else None,
            "context_tokens": context_tokens,
            "generation_tokens": generation_tokens,
            "repetitions": len(sample_seconds),
            "measured_token_seconds": measured_seconds,
            "measured_tokens_per_second": 1.0 / measured_seconds,
            "measured_sample_min_seconds": min(sample_seconds),
            "measured_sample_max_seconds": max(sample_seconds),
        }
        if window_size is not None:
            point_config = config.with_overrides(
                window_size=int(window_size),
                context_tokens=context_tokens,
                state_placement="roundtrip",
            )
            analytical = analyze_decode(point_config)
            simulated = simulate_decode(point_config)
            row.update(
                {
                    "predicted_token_seconds": simulated.makespan_seconds,
                    "predicted_tokens_per_second": simulated.throughput_tokens_per_s,
                    "latency_relative_error": (
                        simulated.makespan_seconds / measured_seconds - 1.0
                    ),
                    "attention_kv_window_resident": (
                        analytical.attention_kv_window_resident
                    ),
                    "attention_kv_roundtrip_bytes": (
                        analytical.attention_kv_roundtrip_bytes_per_token
                    ),
                    "attention_kv_pcie_seconds": (
                        analytical.attention_kv_transfer_seconds
                    ),
                    "kv_storage_io_seconds": (
                        analytical.attention_kv_storage_io_seconds
                    ),
                }
            )
        else:
            row.update(
                {
                    "predicted_token_seconds": None,
                    "predicted_tokens_per_second": None,
                    "latency_relative_error": None,
                    "attention_kv_window_resident": True,
                    "attention_kv_roundtrip_bytes": 0,
                    "attention_kv_pcie_seconds": 0.0,
                    "kv_storage_io_seconds": 0.0,
                }
            )
        rows.append(row)

    baselines = {
        str(row["mode"]): float(row["measured_token_seconds"])
        for row in rows
        if int(row["context_tokens"]) == 0
    }
    predicted_baselines = {
        str(row["mode"]): float(row["predicted_token_seconds"])
        for row in rows
        if int(row["context_tokens"]) == 0
        and row["predicted_token_seconds"] is not None
    }
    for row in rows:
        mode = str(row["mode"])
        row["measured_increment_seconds_from_context_zero"] = (
            float(row["measured_token_seconds"]) - baselines[mode]
        )
        row["predicted_increment_seconds_from_context_zero"] = (
            float(row["predicted_token_seconds"]) - predicted_baselines[mode]
            if row["predicted_token_seconds"] is not None
            else None
        )

    return {
        "schema_version": 1,
        "config": config.name,
        "specification_file": specification_file.as_posix(),
        "measurement_method": (
            "llama-bench generation-only timer after restoring cached --n-depth state"
        ),
        "rows": rows,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fit_scheduler_overhead(measurements: dict[str, Any]) -> dict[str, Any]:
    """Measure the fixed pipeline-path delta from K=N and native runs.

    The per-layer normalization is retained as a diagnostic for old reports, but
    the unnormalized token delta is the quantity used by v0.4 configs. Nsight
    shows that this delta contains whole-token transfers and graph fragmentation,
    so treating it as a causal per-layer constant is misleading.
    """
    rows = []
    for model in measurements["models"]:
        pipeline_resident = model.get("pipeline_resident")
        if pipeline_resident is None:
            continue
        layer_count = int(model["layer_count"])
        native_seconds = 1.0 / float(model["resident"]["avg_tokens_per_second"])
        pipeline_seconds = 1.0 / float(pipeline_resident["avg_tokens_per_second"])
        token_overhead = pipeline_seconds - native_seconds
        layer_overhead = token_overhead / layer_count
        if layer_overhead < 0:
            raise ValueError("pipeline-resident run is faster than native resident run")
        rows.append(
            {
                "model_id": str(model["id"]),
                "layer_count": layer_count,
                "native_resident_seconds": native_seconds,
                "pipeline_resident_seconds": pipeline_seconds,
                "token_pipeline_overhead_s": token_overhead,
                "layer_scheduler_overhead_s": layer_overhead,
            }
        )
    if not rows:
        return {
            "mean_token_pipeline_overhead_s": 0.0,
            "layer_scheduler_overhead_s": 0.0,
            "rows": [],
        }
    token_overheads = [float(row["token_pipeline_overhead_s"]) for row in rows]
    overheads = [float(row["layer_scheduler_overhead_s"]) for row in rows]
    mean = statistics.fmean(overheads)
    return {
        "equation": "S_token(model) = T_pipeline,K=N - T_native_resident",
        "diagnostic_normalization": "S_layer = S_token / n_layers",
        "mean_token_pipeline_overhead_s": statistics.fmean(token_overheads),
        "layer_scheduler_overhead_s": mean,
        "min_layer_scheduler_overhead_s": min(overheads),
        "max_layer_scheduler_overhead_s": max(overheads),
        "relative_spread": (max(overheads) - min(overheads)) / mean if mean else 0.0,
        "rows": rows,
    }


def fit_pipeline_service(
    measurements: dict[str, Any],
    *,
    scheduler_overhead_s: float | None = None,
) -> dict[str, Any]:
    """Fit the streamed-weight lane from end-to-end pipeline measurements.

    The control lane is modeled separately and may overlap this lane. A non-zero
    scalar is retained for compatibility with older callers, but v0.4 fitting
    deliberately uses zero here rather than subtracting the full K=N delta.
    """
    if scheduler_overhead_s is None:
        scheduler_overhead_s = float(
            fit_scheduler_overhead(measurements)["layer_scheduler_overhead_s"]
        )
    rows: list[dict[str, float | str | int]] = []
    for model in measurements["models"]:
        target_seconds = 1.0 / float(model["pipeline"]["avg_tokens_per_second"])
        rows.append(
            {
                "model_id": str(model["id"]),
                "weight_bytes": int(model["layer_weight_bytes"]),
                "layer_count": int(model["layer_count"]),
                "target_seconds": target_seconds,
                "service_target_seconds": target_seconds - scheduler_overhead_s,
            }
        )

    sww = sum(float(row["weight_bytes"]) ** 2 for row in rows)
    swn = sum(
        float(row["weight_bytes"]) * float(row["layer_count"]) for row in rows
    )
    snn = sum(float(row["layer_count"]) ** 2 for row in rows)
    swy = sum(
        float(row["weight_bytes"]) * float(row["service_target_seconds"])
        for row in rows
    )
    sny = sum(
        float(row["layer_count"]) * float(row["service_target_seconds"])
        for row in rows
    )
    determinant = sww * snn - swn * swn
    if determinant == 0:
        raise ValueError("pipeline calibration matrix is singular")
    seconds_per_byte = (swy * snn - sny * swn) / determinant
    layer_latency_s = (sww * sny - swn * swy) / determinant
    if seconds_per_byte <= 0 or layer_latency_s < 0:
        raise ValueError(
            "pipeline measurements do not support a positive bandwidth/latency fit"
        )
    bandwidth = 1.0 / seconds_per_byte

    squared_relative_errors = []
    fitted_rows = []
    for row in rows:
        predicted_service = (
            float(row["weight_bytes"]) / bandwidth
            + float(row["layer_count"]) * layer_latency_s
        )
        predicted = predicted_service + scheduler_overhead_s
        target = float(row["target_seconds"])
        relative_error = predicted / target - 1.0
        squared_relative_errors.append(relative_error**2)
        fitted_rows.append(
            {
                **row,
                "predicted_service_seconds": predicted_service,
                "predicted_seconds": predicted,
                "relative_error": relative_error,
            }
        )
    return {
        "equation": "T_stream_lane ~= layer_weight_bytes / B_effective + layer_count * L_effective",
        "scheduler_overhead_used_s": scheduler_overhead_s,
        "effective_bandwidth_bytes_per_s": bandwidth,
        "effective_layer_latency_s": layer_latency_s,
        "relative_rmse": (sum(squared_relative_errors) / len(rows)) ** 0.5,
        "rows": fitted_rows,
    }


def fit_host_staging(measurements: dict[str, Any]) -> dict[str, Any]:
    """Fit host staging service exposed when K=2 has no prefetch distance."""
    rows: list[dict[str, Any]] = []
    for model in measurements["models"]:
        baseline_window = int(model["pipeline"]["window_size"])
        baseline_seconds = 1.0 / float(
            model["pipeline"]["avg_tokens_per_second"]
        )
        for point in model.get("window_validation", []):
            if int(point["window_size"]) != 2:
                continue
            if point.get("avg_tokens_per_second") is None:
                continue
            k2_seconds = 1.0 / float(point["avg_tokens_per_second"])
            exposed_seconds = k2_seconds - baseline_seconds
            if exposed_seconds <= 0:
                raise ValueError("K=2 must be slower than the calibrated baseline")
            weight_bytes = int(model["layer_weight_bytes"])
            rows.append(
                {
                    "model_id": str(model["id"]),
                    "baseline_window": baseline_window,
                    "k2_seconds": k2_seconds,
                    "baseline_seconds": baseline_seconds,
                    "exposed_host_staging_seconds": exposed_seconds,
                    "layer_weight_bytes": weight_bytes,
                    "host_staging_bandwidth_bytes_per_s": weight_bytes
                    / exposed_seconds,
                }
            )
    if not rows:
        return {
            "host_staging_bandwidth_bytes_per_s": None,
            "staging_overlap_window": 2,
            "rows": [],
        }
    bandwidth = statistics.fmean(
        float(row["host_staging_bandwidth_bytes_per_s"]) for row in rows
    )
    overlap_window = max(int(row["baseline_window"]) for row in rows)
    return {
        "equation": "T_K2 - T_Kbaseline = layer_weight_bytes / B_host_stage",
        "host_staging_bandwidth_bytes_per_s": bandwidth,
        "staging_overlap_window": overlap_window,
        "rows": rows,
    }


def fit_state_transfer(
    config: ModelConfig, nsys_decomposition: dict[str, Any]
) -> dict[str, Any]:
    """Fit effective state round-trip bandwidth from K=N CUDA copy service.

    At context zero, semantic state is the fixed recurrent state. Nsight's
    non-embedding copy service also contains tiny cache/control copies, so the
    resulting bandwidth is an effective model parameter rather than a link
    measurement.
    """
    direction_ms = {
        str(row["direction"]): float(row["gpu_milliseconds_per_token"])
        for row in nsys_decomposition["copy_summary"]["directions"]
    }
    embedding_ms = float(
        nsys_decomposition["embedding_fallback"][
            "d2h_gpu_milliseconds_per_token"
        ]
    )
    state_service_s = (
        direction_ms.get("Host-to-Device", 0.0)
        + direction_ms.get("Device-to-Host", 0.0)
        - embedding_ms
    ) / 1000.0
    state_by_layer = state_bytes_by_layer(config)
    recurrent_state_bytes = sum(
        size
        for layer, size in zip(config.layers, state_by_layer)
        if layer.kind == "ssm"
    )
    semantic_roundtrip_bytes = 2 * recurrent_state_bytes
    if state_service_s <= 0 or semantic_roundtrip_bytes <= 0:
        raise ValueError("state transfer fit requires recurrent bytes and positive service")
    return {
        "equation": "T_state,trace = 2 * recurrent_state_bytes / B_state_effective",
        "recurrent_state_bytes": recurrent_state_bytes,
        "semantic_roundtrip_bytes": semantic_roundtrip_bytes,
        "traced_non_embedding_copy_service_s": state_service_s,
        "effective_state_transfer_bandwidth_bytes_per_s": (
            semantic_roundtrip_bytes / state_service_s
        ),
    }


def fit_kv_storage_tiers(
    config: ModelConfig,
    specification: dict[str, Any],
    *,
    project_root: Path,
    state_transfer_bandwidth_bytes_per_s: float,
) -> dict[str, Any]:
    """Fit cached and uncached KV-file service from contextual decode runs.

    The latency at context zero is removed first, followed by the semantic KV
    PCIe round trip. The remaining latency is synchronous backing-file service.
    Low-context points fit a through-origin cached bandwidth. The final two
    high-context points identify an uncached slope and the cache knee.
    """

    def load_result(path_value: str) -> tuple[Path, dict[str, Any]]:
        path = Path(path_value)
        if not path.is_absolute():
            path = project_root / path
        return path, _read_json(path)

    baseline_path, baseline = load_result(str(specification["baseline"]))
    generation_tokens = int(baseline["generation_tokens"])
    baseline_seconds = float(baseline["result"]["avg_ns"]) / generation_tokens / 1e9
    kv_bytes_per_token = sum(layer.kind == "attention" for layer in config.layers) * (
        config.state.sequence_count
        * config.state.attention_kv_head_count
        * (config.state.attention_key_length + config.state.attention_value_length)
        * config.state.kv_element_bytes
        if config.state is not None
        else 0
    )
    if kv_bytes_per_token <= 0:
        raise ValueError("KV storage fit requires attention KV geometry")

    rows: list[dict[str, Any]] = []
    for path_value in specification["points"]:
        path, result = load_result(str(path_value))
        context_tokens = int(result["prompt_tokens"])
        tokens = int(result["generation_tokens"])
        measured_seconds = float(result["result"]["avg_ns"]) / tokens / 1e9
        one_way_kv_bytes = context_tokens * kv_bytes_per_token
        roundtrip_bytes = 2 * one_way_kv_bytes
        pcie_seconds = (
            roundtrip_bytes / state_transfer_bandwidth_bytes_per_s
        )
        storage_seconds = measured_seconds - baseline_seconds - pcie_seconds
        if storage_seconds <= 0:
            raise ValueError(
                f"{path.name}: contextual residual is not positive after PCIe service"
            )
        rows.append(
            {
                "file": path.as_posix(),
                "context_tokens": context_tokens,
                "measured_token_seconds": measured_seconds,
                "incremental_token_seconds": measured_seconds - baseline_seconds,
                "one_way_attention_kv_bytes": one_way_kv_bytes,
                "storage_roundtrip_bytes": roundtrip_bytes,
                "semantic_pcie_roundtrip_seconds": pcie_seconds,
                "inferred_storage_seconds": storage_seconds,
            }
        )
    rows.sort(key=lambda row: int(row["context_tokens"]))
    fast_max_context = int(specification["cached_fit_max_context_tokens"])
    fast_rows = [
        row for row in rows if int(row["context_tokens"]) <= fast_max_context
    ]
    slow_rows = [
        row for row in rows if int(row["context_tokens"]) > fast_max_context
    ]
    if len(fast_rows) < 2 or len(slow_rows) < 2:
        raise ValueError("KV storage tier fit requires at least two points per tier")

    numerator = sum(
        float(row["storage_roundtrip_bytes"])
        * float(row["inferred_storage_seconds"])
        for row in fast_rows
    )
    denominator = sum(
        float(row["storage_roundtrip_bytes"]) ** 2 for row in fast_rows
    )
    cached_seconds_per_byte = numerator / denominator
    cached_bandwidth = 1.0 / cached_seconds_per_byte

    high_a, high_b = slow_rows[-2:]
    byte_delta = float(high_b["storage_roundtrip_bytes"]) - float(
        high_a["storage_roundtrip_bytes"]
    )
    time_delta = float(high_b["inferred_storage_seconds"]) - float(
        high_a["inferred_storage_seconds"]
    )
    if byte_delta <= 0 or time_delta <= 0:
        raise ValueError("KV storage high-context points must be strictly increasing")
    uncached_bandwidth = byte_delta / time_delta
    slow_seconds_per_byte = 1.0 / uncached_bandwidth
    cache_roundtrip_bytes = (
        float(high_a["inferred_storage_seconds"])
        - float(high_a["storage_roundtrip_bytes"]) * slow_seconds_per_byte
    ) / (cached_seconds_per_byte - slow_seconds_per_byte)
    cache_bytes = max(0, round(cache_roundtrip_bytes / 2.0))

    def fitted_storage_seconds(one_way_bytes: int) -> float:
        cached_bytes = min(one_way_bytes, cache_bytes)
        uncached_bytes = max(0, one_way_bytes - cache_bytes)
        return 2.0 * (
            cached_bytes / cached_bandwidth
            + uncached_bytes / uncached_bandwidth
        )

    for row in rows:
        predicted_storage = fitted_storage_seconds(
            int(row["one_way_attention_kv_bytes"])
        )
        predicted_token = (
            baseline_seconds
            + float(row["semantic_pcie_roundtrip_seconds"])
            + predicted_storage
        )
        row["fitted_storage_seconds"] = predicted_storage
        row["predicted_token_seconds"] = predicted_token
        row["latency_relative_error"] = (
            predicted_token / float(row["measured_token_seconds"]) - 1.0
        )

    return {
        "equation": (
            "T_storage = 2 * [min(M_kv,C)/B_cached + "
            "max(M_kv-C,0)/B_uncached]"
        ),
        "baseline_file": baseline_path.as_posix(),
        "baseline_token_seconds": baseline_seconds,
        "cached_fit_max_context_tokens": fast_max_context,
        "kv_storage_cached_bandwidth_bytes_per_s": cached_bandwidth,
        "kv_storage_uncached_bandwidth_bytes_per_s": uncached_bandwidth,
        "kv_storage_cache_bytes": cache_bytes,
        "rows": rows,
    }


def calibrate_configs(
    measurements_path: str | Path,
    output_dir: str | Path,
) -> tuple[list[Path], dict[str, Any]]:
    measurement_file = Path(measurements_path).resolve()
    measurements = _read_json(measurement_file)
    project_root = Path(__file__).resolve().parents[1]
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = project_root / destination
    destination.mkdir(parents=True, exist_ok=True)

    scheduler_fit = fit_scheduler_overhead(measurements)
    scheduler_rows = {
        str(row["model_id"]): row for row in scheduler_fit["rows"]
    }
    fit = fit_pipeline_service(measurements, scheduler_overhead_s=0.0)
    staging_fit = fit_host_staging(measurements)
    bandwidth = float(fit["effective_bandwidth_bytes_per_s"])
    latency = float(fit["effective_layer_latency_s"])
    host_staging_bandwidth = staging_fit["host_staging_bandwidth_bytes_per_s"]
    staging_overlap_window = int(staging_fit["staging_overlap_window"])
    generated: list[Path] = []
    model_reports: list[dict[str, Any]] = []
    state_fit_rows: list[dict[str, Any]] = []
    storage_fit_rows: list[dict[str, Any]] = []

    for measured in measurements["models"]:
        config_path = Path(str(measured["config"]))
        if not config_path.is_absolute():
            config_path = project_root / config_path
        raw = _read_json(config_path)
        config = config_from_dict(raw)
        nsys_path = Path(str(measured["nsys_decomposition"]))
        if not nsys_path.is_absolute():
            nsys_path = project_root / nsys_path
        nsys_decomposition = _read_json(nsys_path)
        token_embedding_fallback = float(
            nsys_decomposition["embedding_fallback"][
                "d2h_gpu_milliseconds_per_token"
            ]
        ) / 1000.0
        state_fit = fit_state_transfer(config, nsys_decomposition)
        state_fit_rows.append({"model_id": str(measured["id"]), **state_fit})
        token_state_roundtrip = float(
            state_fit["traced_non_embedding_copy_service_s"]
        )
        state_transfer_bandwidth = float(
            state_fit["effective_state_transfer_bandwidth_bytes_per_s"]
        )
        storage_fit = None
        if measured.get("context_validation") is not None:
            storage_fit = fit_kv_storage_tiers(
                config,
                measured["context_validation"],
                project_root=project_root,
                state_transfer_bandwidth_bytes_per_s=state_transfer_bandwidth,
            )
            storage_fit_rows.append(
                {"model_id": str(measured["id"]), **storage_fit}
            )
        total_flops = sum(layer.flops for layer in config.layers)
        resident_tps = float(measured["resident"]["avg_tokens_per_second"])
        compute_seconds = 1.0 / resident_tps
        calibrated = deepcopy(raw)
        token_pipeline_overhead = float(
            scheduler_rows[str(measured["id"])]["token_pipeline_overhead_s"]
        )
        if token_embedding_fallback + token_state_roundtrip > token_pipeline_overhead:
            raise ValueError(
                f"{measured['id']}: traced embedding/state service exceeds total pipeline overhead"
            )
        calibrated["name"] = f"{config.name}-rtx4070-calibrated-v0.8"
        calibrated["notes"] = (
            f"{raw.get('notes', '')} Calibrated on RTX 4070 Laptop: resident decode "
            f"{resident_tps:.6f} token/s; streamed-weight lane uses "
            f"B={bandwidth / 1e9:.6f} GB/s and L={latency * 1000:.6f} ms/layer; "
            f"the measured fixed pipeline-path delta is "
            f"{token_pipeline_overhead * 1000:.6f} ms/token, including a directly "
            f"traced token_embd.weight D2H service of "
            f"{token_embedding_fallback * 1000:.6f} ms/token and state/cache copy "
            f"service of {token_state_roundtrip * 1000:.6f} ms/token. State traffic "
            f"uses B={state_transfer_bandwidth / 1e9:.6f} GB/s. Host staging uses "
            f"B={float(host_staging_bandwidth) / 1e9:.6f} GB/s with full overlap "
            f"assumed at K>={staging_overlap_window}."
            + (
                " Small-window attention KV backing-file I/O uses a fitted "
                f"cache knee of {int(storage_fit['kv_storage_cache_bytes']) / 1024**2:.2f} MiB "
                f"one-way, B_cached={float(storage_fit['kv_storage_cached_bandwidth_bytes_per_s']) / 1e9:.3f} GB/s, "
                f"and B_uncached={float(storage_fit['kv_storage_uncached_bandwidth_bytes_per_s']) / 1e9:.3f} GB/s."
                if storage_fit is not None
                else " KV backing-file service is not calibrated for this model."
            )
        ).strip()
        calibrated["hardware"]["gpu_effective_flops"] = total_flops / compute_seconds
        calibrated["hardware"]["h2d_bandwidth_bytes_per_s"] = bandwidth
        calibrated["hardware"]["h2d_latency_s"] = latency
        calibrated["hardware"][
            "host_staging_bandwidth_bytes_per_s"
        ] = host_staging_bandwidth
        calibrated["hardware"][
            "state_transfer_bandwidth_bytes_per_s"
        ] = state_transfer_bandwidth
        if storage_fit is not None:
            for key in (
                "kv_storage_cached_bandwidth_bytes_per_s",
                "kv_storage_uncached_bandwidth_bytes_per_s",
                "kv_storage_cache_bytes",
            ):
                calibrated["hardware"][key] = storage_fit[key]
        if measurements.get("environment", {}).get("vram_total_mib") is not None:
            calibrated["hardware"]["vram_capacity_bytes"] = int(
                float(measurements["environment"]["vram_total_mib"]) * 1024**2
            )
        calibrated["policy"]["window_size"] = int(measured["pipeline"]["window_size"])
        calibrated["policy"]["token_pipeline_overhead_s"] = token_pipeline_overhead
        calibrated["policy"]["token_embedding_fallback_s"] = token_embedding_fallback
        calibrated["policy"]["embedding_fallback_scale"] = 1.0
        calibrated["policy"]["staging_overlap_window"] = staging_overlap_window
        calibrated["policy"]["token_state_roundtrip_s"] = token_state_roundtrip
        calibrated["policy"]["state_placement"] = "roundtrip"
        calibrated["policy"]["layer_scheduler_overhead_s"] = 0.0
        calibrated["calibration"] = {
            "measurement_file": measurement_file.name,
            "resident_avg_tokens_per_second": resident_tps,
            "pipeline_avg_tokens_per_second": float(
                measured["pipeline"]["avg_tokens_per_second"]
            ),
            "pipeline_window_size": int(measured["pipeline"]["window_size"]),
            "effective_bandwidth_bytes_per_s": bandwidth,
            "effective_layer_latency_s": latency,
            "token_pipeline_overhead_s": token_pipeline_overhead,
            "nsys_decomposition_file": nsys_path.name,
            "token_embedding_fallback_s": token_embedding_fallback,
            "token_state_roundtrip_s": token_state_roundtrip,
            "state_transfer_bandwidth_bytes_per_s": state_transfer_bandwidth,
            "host_staging_bandwidth_bytes_per_s": host_staging_bandwidth,
            "staging_overlap_window": staging_overlap_window,
            "legacy_layer_scheduler_overhead_s": float(
                scheduler_rows[str(measured["id"])]["layer_scheduler_overhead_s"]
            ),
        }
        if storage_fit is not None:
            calibrated["calibration"]["kv_storage_fit"] = {
                key: storage_fit[key]
                for key in (
                    "equation",
                    "cached_fit_max_context_tokens",
                    "kv_storage_cached_bandwidth_bytes_per_s",
                    "kv_storage_uncached_bandwidth_bytes_per_s",
                    "kv_storage_cache_bytes",
                )
            }

        output = destination / f"qwen35_{measured['id']}_q4_k_m_rtx4070.json"
        output.write_text(
            json.dumps(calibrated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated.append(output)

        calibrated_config = config_from_dict(calibrated)
        analytical = analyze_decode(calibrated_config)
        simulation = simulate_decode(calibrated_config)
        no_embedding_config = calibrated_config.with_overrides(
            embedding_fallback_scale=0.0
        )
        no_embedding_analytical = analyze_decode(no_embedding_config)
        no_embedding_simulation = simulate_decode(no_embedding_config)
        target_seconds = 1.0 / float(measured["pipeline"]["avg_tokens_per_second"])
        validation_rows = []
        validation_points = list(measured.get("window_validation", []))
        if measured.get("pipeline_resident") is not None:
            validation_points.append(measured["pipeline_resident"])
        for point in validation_points:
            window = int(point["window_size"])
            predicted = simulate_decode(
                calibrated_config.with_overrides(window_size=window)
            )
            row = {
                "window_size": window,
                "status": point.get("status", "measured"),
                "simulated_tokens_per_second": predicted.throughput_tokens_per_s,
            }
            if point.get("avg_tokens_per_second") is not None:
                measured_tps = float(point["avg_tokens_per_second"])
                measured_seconds = 1.0 / measured_tps
                row.update(
                    {
                        "measured_tokens_per_second": measured_tps,
                        "latency_relative_error": predicted.makespan_seconds
                        / measured_seconds
                        - 1.0,
                    }
                )
            if point.get("note") is not None:
                row["note"] = point["note"]
            validation_rows.append(row)
        model_reports.append(
            {
                "model_id": measured["id"],
                "config": output.as_posix(),
                "resident_measured_tokens_per_second": resident_tps,
                "pipeline_measured_tokens_per_second": float(
                    measured["pipeline"]["avg_tokens_per_second"]
                ),
                "analytical_tokens_per_second": analytical.ideal_throughput_tokens_per_s,
                "simulated_tokens_per_second": simulation.throughput_tokens_per_s,
                "simulated_seconds": simulation.makespan_seconds,
                "target_seconds": target_seconds,
                "simulation_relative_error": simulation.makespan_seconds / target_seconds
                - 1.0,
                "no_embedding_counterfactual": {
                    "method": "subtract directly traced token_embd.weight D2H GPU service only",
                    "token_embedding_fallback_seconds_removed": token_embedding_fallback,
                    "analytical_tokens_per_second": no_embedding_analytical.ideal_throughput_tokens_per_s,
                    "simulated_tokens_per_second": no_embedding_simulation.throughput_tokens_per_s,
                    "simulated_speedup": no_embedding_simulation.throughput_tokens_per_s
                    / simulation.throughput_tokens_per_s,
                },
                "window_validation": validation_rows,
            }
        )

    report = {
        "schema_version": 6,
        "measurement_file": measurement_file.as_posix(),
        "scheduler_overhead_fit": scheduler_fit,
        "pipeline_service_fit": fit,
        "host_staging_fit": staging_fit,
        "state_transfer_fit": {
            "method": "per-model effective bandwidth from semantic recurrent bytes and Nsight non-embedding CUDA copy service",
            "rows": state_fit_rows,
        },
        "kv_storage_fit": {
            "method": "2B context-latency residual after semantic KV PCIe service; two-tier synchronous backing-file fit",
            "rows": storage_fit_rows,
        },
        "models": model_reports,
        "limitations": [
            "Resident llama-bench time is used as the total layer-compute proxy and absorbs output/KV/kernel overhead.",
            "Pipeline B and L are effective streamed-lane parameters fitted from K=4 end-to-end results, not raw PCIe link properties.",
            "The K=N minus native delta is modeled once per token. Nsight shows that it mixes embedding fallback, recurrent-state round trips, CUDA graph fragmentation, and synchronization; it is not a causal per-layer constant.",
            "The no-embedding counterfactual subtracts only the directly traced token_embd.weight D2H GPU duration. It does not assume that associated graph splits or synchronization gaps disappear, so it is conservative.",
            "The state-transfer component uses semantic recurrent-state bytes but all traced non-embedding CUDA copy service. Its per-model bandwidth absorbs small KV/control copies and copy-size effects; it is not raw PCIe bandwidth.",
            "Attention KV capacity and PCIe bytes grow linearly with context. Nsight validates the byte formula at 8K for 2B K=4 and shows no KV eviction at K=N.",
            "The two-tier KV backing-file fit is calibrated only on 2B K=4 at 4K/8K/16K/32K. Its cache knee and bandwidths are machine-, filesystem-, and run-state-specific and are not transferred to 4B/9B.",
            "The host-staging correction is fitted from the 2B K=2 minus K=4 latency delta. The linear K=3 interpolation and transfer to 4B/9B are model hypotheses, not validated measurements.",
            "The corrected 2B K=2 result is an in-sample fit, not independent validation. K=1 did not finish the short validation run and remains outside the model's valid range.",
            "Concurrent requests, prefill latency, and KV storage contention remain unmodeled.",
        ],
    }
    return generated, report
