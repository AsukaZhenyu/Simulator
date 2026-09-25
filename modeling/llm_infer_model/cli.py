from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analytical import analyze_decode
from .calibration import (
    calibrate_configs,
    evaluate_context_validation,
    run_h2d_bandwidth_test,
    run_llama_bench,
)
from .gguf_extract import (
    config_from_gguf_manifest,
    extract_gguf_manifest,
    manifest_layer_rows,
)
from .model import load_config
from .nsys_analysis import analyze_nsys_sqlite
from .simulator import simulate_decode


def _comma_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _comma_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _write_json(data: object, output: str | None) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _write_trace(path: str, events: tuple[object, ...]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "layer",
        "layer_name",
        "layer_kind",
        "resource",
        "start_s",
        "end_s",
        "bytes",
        "flops",
        "scheduler_seconds",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_dict())


def _read_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_analyze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if (
        args.window is not None
        or args.embedding_fallback_scale is not None
        or args.context_tokens is not None
        or args.state_placement is not None
    ):
        config = config.with_overrides(
            window_size=args.window,
            embedding_fallback_scale=args.embedding_fallback_scale,
            context_tokens=args.context_tokens,
            state_placement=args.state_placement,
        )
    _write_json(analyze_decode(config).to_dict(), args.output)


def command_simulate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if (
        args.window is not None
        or args.embedding_fallback_scale is not None
        or args.context_tokens is not None
        or args.state_placement is not None
    ):
        config = config.with_overrides(
            window_size=args.window,
            embedding_fallback_scale=args.embedding_fallback_scale,
            context_tokens=args.context_tokens,
            state_placement=args.state_placement,
        )
    result = simulate_decode(config)
    if args.trace:
        _write_trace(args.trace, result.events)
    _write_json(result.to_dict(include_events=args.include_events), args.output)


def command_sweep(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    windows = _comma_ints(args.windows)
    bandwidths = _comma_floats(args.bandwidth_gbps)
    rows: list[dict[str, object]] = []
    for bandwidth_gbps in bandwidths:
        for window in windows:
            config = base.with_overrides(
                window_size=window,
                h2d_bandwidth_bytes_per_s=bandwidth_gbps * 1e9,
                embedding_fallback_scale=args.embedding_fallback_scale,
                context_tokens=args.context_tokens,
                state_placement=args.state_placement,
            )
            analytical = analyze_decode(config)
            row: dict[str, object] = {
                "window_size": window,
                "bandwidth_GBps": bandwidth_gbps,
                "capacity_feasible": analytical.capacity_feasible,
                "analytical_lower_ms": round(
                    analytical.ideal_overlap_lower_bound_seconds * 1000, 6
                ),
                "analytical_upper_ms": round(
                    analytical.no_overlap_upper_bound_seconds * 1000, 6
                ),
                "bottleneck": analytical.bottleneck,
                "bytes_transferred_GB": analytical.bytes_transferred_per_token / 1e9,
            }
            if analytical.capacity_feasible:
                simulation = simulate_decode(config)
                row.update(
                    {
                        "simulated_ms": round(simulation.makespan_seconds * 1000, 6),
                        "throughput_tokens_per_s": round(
                            simulation.throughput_tokens_per_s, 6
                        ),
                        "gpu_utilization_pct": round(simulation.gpu_utilization * 100, 6),
                        "h2d_utilization_pct": round(simulation.h2d_utilization * 100, 6),
                        "peak_streamed_weight_GiB": round(
                            simulation.peak_streamed_weight_bytes / 1024**3, 6
                        ),
                        "error": "",
                    }
                )
            else:
                row.update(
                    {
                        "simulated_ms": "",
                        "throughput_tokens_per_s": "",
                        "gpu_utilization_pct": "",
                        "h2d_utilization_pct": "",
                        "peak_streamed_weight_GiB": "",
                        "error": "window exceeds available VRAM",
                    }
                )
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


def command_compare_embedding_fallback(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    rows: list[dict[str, object]] = []
    for window in _comma_ints(args.windows):
        baseline = base.with_overrides(
            window_size=window, embedding_fallback_scale=1.0
        )
        no_embedding = base.with_overrides(
            window_size=window, embedding_fallback_scale=0.0
        )
        baseline_analysis = analyze_decode(baseline)
        no_embedding_analysis = analyze_decode(no_embedding)
        row: dict[str, object] = {
            "window_size": window,
            "capacity_feasible": baseline_analysis.capacity_feasible,
            "weight_stream_ms": round(baseline_analysis.transfer_seconds * 1000, 6),
            "baseline_control_ms": round(baseline_analysis.compute_seconds * 1000, 6),
            "no_embedding_control_ms": round(
                no_embedding_analysis.compute_seconds * 1000, 6
            ),
            "embedding_d2h_ms_removed": round(
                baseline_analysis.token_embedding_fallback_seconds * 1000, 6
            ),
        }
        if baseline_analysis.capacity_feasible:
            baseline_simulation = simulate_decode(baseline)
            no_embedding_simulation = simulate_decode(no_embedding)
            row.update(
                {
                    "baseline_ms": round(
                        baseline_simulation.makespan_seconds * 1000, 6
                    ),
                    "baseline_tokens_per_second": round(
                        baseline_simulation.throughput_tokens_per_s, 6
                    ),
                    "no_embedding_ms": round(
                        no_embedding_simulation.makespan_seconds * 1000, 6
                    ),
                    "no_embedding_tokens_per_second": round(
                        no_embedding_simulation.throughput_tokens_per_s, 6
                    ),
                    "speedup": round(
                        no_embedding_simulation.throughput_tokens_per_s
                        / baseline_simulation.throughput_tokens_per_s,
                        6,
                    ),
                    "baseline_bottleneck": baseline_analysis.bottleneck,
                    "no_embedding_bottleneck": no_embedding_analysis.bottleneck,
                    "error": "",
                }
            )
        else:
            row.update(
                {
                    "baseline_ms": "",
                    "baseline_tokens_per_second": "",
                    "no_embedding_ms": "",
                    "no_embedding_tokens_per_second": "",
                    "speedup": "",
                    "baseline_bottleneck": "infeasible",
                    "no_embedding_bottleneck": "infeasible",
                    "error": "window exceeds available VRAM",
                }
            )
        rows.append(row)
    _write_rows(args.output, rows)
    print(f"wrote {len(rows)} counterfactual rows to {args.output}")


def command_compare_state_residency(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    if base.state is None:
        raise ValueError("compare-state-residency requires an explicit state model")
    rows: list[dict[str, object]] = []
    for context_tokens in _comma_ints(args.contexts):
        for window in _comma_ints(args.windows):
            roundtrip = base.with_overrides(
                window_size=window,
                context_tokens=context_tokens,
                state_placement="roundtrip",
            )
            resident = base.with_overrides(
                window_size=window,
                context_tokens=context_tokens,
                state_placement="resident",
            )
            roundtrip_analysis = analyze_decode(roundtrip)
            resident_analysis = analyze_decode(resident)
            row: dict[str, object] = {
                "context_tokens": context_tokens,
                "window_size": window,
                "runtime_state_MiB": round(
                    roundtrip_analysis.total_runtime_state_bytes / 1024**2, 6
                ),
                "recurrent_state_MiB": round(
                    roundtrip_analysis.total_recurrent_state_bytes / 1024**2, 6
                ),
                "attention_kv_MiB": round(
                    roundtrip_analysis.total_attention_kv_bytes / 1024**2, 6
                ),
                "attention_kv_window_resident": (
                    roundtrip_analysis.attention_kv_window_resident
                ),
                "recurrent_roundtrip_MiB_per_token": round(
                    roundtrip_analysis.recurrent_state_roundtrip_bytes_per_token
                    / 1024**2,
                    6,
                ),
                "attention_kv_roundtrip_MiB_per_token": round(
                    roundtrip_analysis.attention_kv_roundtrip_bytes_per_token
                    / 1024**2,
                    6,
                ),
                "roundtrip_bytes_MiB_per_token": round(
                    roundtrip_analysis.state_roundtrip_bytes_per_token / 1024**2,
                    6,
                ),
                "recurrent_state_pcie_ms": round(
                    roundtrip_analysis.recurrent_state_transfer_seconds * 1000,
                    6,
                ),
                "attention_kv_pcie_ms": round(
                    roundtrip_analysis.attention_kv_transfer_seconds * 1000,
                    6,
                ),
                "total_state_pcie_ms": round(
                    roundtrip_analysis.state_transfer_seconds * 1000, 6
                ),
                "kv_storage_io_ms": round(
                    roundtrip_analysis.attention_kv_storage_io_seconds * 1000,
                    6,
                ),
                "latency_break_even_context_tokens": (
                    roundtrip_analysis.state_latency_break_even_context_tokens
                ),
                "max_resident_context_tokens": (
                    resident_analysis.max_resident_state_context_tokens
                ),
                "roundtrip_required_MiB": round(
                    roundtrip_analysis.required_window_bytes / 1024**2, 6
                ),
                "resident_required_MiB": round(
                    resident_analysis.required_window_bytes / 1024**2, 6
                ),
                "additional_resident_MiB": round(
                    (
                        resident_analysis.required_window_bytes
                        - roundtrip_analysis.required_window_bytes
                    )
                    / 1024**2,
                    6,
                ),
                "roundtrip_feasible": roundtrip_analysis.capacity_feasible,
                "resident_feasible": resident_analysis.capacity_feasible,
            }
            roundtrip_simulation = (
                simulate_decode(roundtrip)
                if roundtrip_analysis.capacity_feasible
                else None
            )
            resident_simulation = (
                simulate_decode(resident)
                if resident_analysis.capacity_feasible
                else None
            )
            row.update(
                {
                    "roundtrip_tokens_per_second": (
                        round(roundtrip_simulation.throughput_tokens_per_s, 6)
                        if roundtrip_simulation is not None
                        else ""
                    ),
                    "resident_tokens_per_second": (
                        round(resident_simulation.throughput_tokens_per_s, 6)
                        if resident_simulation is not None
                        else ""
                    ),
                    "speedup": (
                        round(
                            resident_simulation.throughput_tokens_per_s
                            / roundtrip_simulation.throughput_tokens_per_s,
                            6,
                        )
                        if roundtrip_simulation is not None
                        and resident_simulation is not None
                        else ""
                    ),
                }
            )
            rows.append(row)
    _write_rows(args.output, rows)
    print(f"wrote {len(rows)} state-residency rows to {args.output}")


def command_extract_gguf(args: argparse.Namespace) -> None:
    manifest = extract_gguf_manifest(
        args.model,
        gguf_python_path=args.gguf_python_path,
    )
    _write_json(manifest, args.output)
    if args.layer_csv:
        _write_rows(args.layer_csv, manifest_layer_rows(manifest))
    print(
        f"extracted {manifest['layer_count']} layers and {manifest['tensor_count']} tensors "
        f"from {manifest['source_file']}"
    )


def command_config_from_gguf(args: argparse.Namespace) -> None:
    manifest = _read_json(args.manifest)
    template = _read_json(args.template)
    reference = _read_json(args.reference_manifest) if args.reference_manifest else None
    reference_seconds = (
        args.reference_compute_ms / 1000 if args.reference_compute_ms is not None else None
    )
    config = config_from_gguf_manifest(
        manifest,
        template,
        reference_manifest=reference,
        reference_compute_seconds=reference_seconds,
        keep_template_cpu=args.keep_template_cpu,
    )
    _write_json(config, args.output)
    print(f"wrote calibrated structure config to {args.output}")


def command_benchmark_llama(args: argparse.Namespace) -> None:
    result = run_llama_bench(
        args.executable,
        args.model,
        generation_tokens=args.generation_tokens,
        prompt_tokens=args.prompt_tokens,
        repetitions=args.repetitions,
        threads=args.threads,
        flash_attention=args.flash_attention,
        window_size=args.window,
    )
    _write_json(result, args.output)
    print(
        f"measured {result['benchmark']}: "
        f"{result['avg_tokens_per_second']:.6f} token/s"
    )


def command_benchmark_h2d(args: argparse.Namespace) -> None:
    result = run_h2d_bandwidth_test(
        args.executable,
        memory=args.memory,
        start_mib=args.start_mib,
        end_mib=args.end_mib,
        increment_mib=args.increment_mib,
        device=args.device,
    )
    _write_json(result, args.output)
    print(
        f"measured {args.memory} H2D median: "
        f"{result['median_bandwidth_bytes_per_s'] / 1e9:.6f} GB/s"
    )


def command_calibrate_configs(args: argparse.Namespace) -> None:
    generated, report = calibrate_configs(args.measurements, args.output_dir)
    _write_json(report, args.report)
    print(f"wrote {len(generated)} calibrated configs and report {args.report}")


def command_analyze_nsys(args: argparse.Namespace) -> None:
    result = analyze_nsys_sqlite(
        args.sqlite,
        generation_tokens=args.generation_tokens,
        layer_count=args.layer_count,
        attention_layer_count=args.attention_layers,
        ssm_layer_count=args.ssm_layers,
        embedding_bytes=args.embedding_bytes,
    )
    _write_json(result, args.output)
    fragmentation = result["graph_fragmentation"]
    fallback = result["embedding_fallback"]
    print(
        f"analyzed {fragmentation['graphs_per_token']} graphs/token; "
        f"embedding fallback once/token={fallback['matches_once_per_token']}"
    )


def command_validate_context(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    report = evaluate_context_validation(config, args.specification)
    _write_json(report, args.report)
    _write_rows(args.csv, report["rows"])
    print(f"validated {len(report['rows'])} contextual decode points")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-infer-model",
        description="Analytical and discrete-event models for layer-wise LLM inference.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="calculate analytical bounds")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--window", type=int)
    analyze.add_argument("--embedding-fallback-scale", type=float)
    analyze.add_argument("--context-tokens", type=int)
    analyze.add_argument("--state-placement", choices=("roundtrip", "resident"))
    analyze.add_argument("--output")
    analyze.set_defaults(func=command_analyze)

    simulate = subparsers.add_parser("simulate", help="run the discrete-event simulator")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--window", type=int)
    simulate.add_argument("--embedding-fallback-scale", type=float)
    simulate.add_argument("--context-tokens", type=int)
    simulate.add_argument("--state-placement", choices=("roundtrip", "resident"))
    simulate.add_argument("--output")
    simulate.add_argument("--trace", help="write an event trace CSV")
    simulate.add_argument("--include-events", action="store_true")
    simulate.set_defaults(func=command_simulate)

    sweep = subparsers.add_parser("sweep", help="sweep window size and H2D bandwidth")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--windows", default="1,2,4,8,16")
    sweep.add_argument("--bandwidth-gbps", default="3,6,12,24")
    sweep.add_argument("--embedding-fallback-scale", type=float, default=1.0)
    sweep.add_argument("--context-tokens", type=int)
    sweep.add_argument(
        "--state-placement", choices=("roundtrip", "resident"), default="roundtrip"
    )
    sweep.add_argument("--output", required=True)
    sweep.set_defaults(func=command_sweep)

    compare_embedding = subparsers.add_parser(
        "compare-embedding-fallback",
        help="compare the calibrated baseline with token embedding D2H removed",
    )
    compare_embedding.add_argument("--config", required=True)
    compare_embedding.add_argument("--windows", default="1,2,4,8,16,32")
    compare_embedding.add_argument("--output", required=True)
    compare_embedding.set_defaults(func=command_compare_embedding_fallback)

    compare_state = subparsers.add_parser(
        "compare-state-residency",
        help="compare per-token state round trips with cross-token VRAM residency",
    )
    compare_state.add_argument("--config", required=True)
    compare_state.add_argument("--contexts", default="0,512,2048,8192,32768")
    compare_state.add_argument("--windows", default="4,8,16,32")
    compare_state.add_argument("--output", required=True)
    compare_state.set_defaults(func=command_compare_state_residency)

    extract = subparsers.add_parser(
        "extract-gguf", help="extract exact per-layer tensor bytes from a GGUF file"
    )
    extract.add_argument("--model", required=True)
    extract.add_argument("--output", required=True, help="write full manifest JSON")
    extract.add_argument("--layer-csv", help="write a compact per-layer CSV")
    extract.add_argument(
        "--gguf-python-path",
        help="path containing the local gguf Python package",
    )
    extract.set_defaults(func=command_extract_gguf)

    config_from_gguf = subparsers.add_parser(
        "config-from-gguf",
        help="build a simulator config from a GGUF manifest and hardware template",
    )
    config_from_gguf.add_argument("--manifest", required=True)
    config_from_gguf.add_argument("--template", required=True)
    config_from_gguf.add_argument("--output", required=True)
    config_from_gguf.add_argument(
        "--reference-manifest",
        help="GGUF manifest used to calibrate effective GPU throughput",
    )
    config_from_gguf.add_argument(
        "--reference-compute-ms",
        type=float,
        help="measured/assumed total layer compute time of the reference manifest",
    )
    config_from_gguf.add_argument(
        "--keep-template-cpu",
        action="store_true",
        help="retain the template CPU throughput instead of disabling uncalibrated static comparison",
    )
    config_from_gguf.set_defaults(func=command_config_from_gguf)

    benchmark_llama = subparsers.add_parser(
        "benchmark-llama",
        help="run one reproducible resident or pipeline llama-bench calibration point",
    )
    benchmark_llama.add_argument("--executable", required=True)
    benchmark_llama.add_argument("--model", required=True)
    benchmark_llama.add_argument("--output", required=True)
    benchmark_llama.add_argument("--window", type=int, help="enable pipeline mode with K slots")
    benchmark_llama.add_argument("--generation-tokens", type=int, default=128)
    benchmark_llama.add_argument("--prompt-tokens", type=int, default=0)
    benchmark_llama.add_argument("--repetitions", type=int, default=5)
    benchmark_llama.add_argument("--threads", type=int, default=16)
    benchmark_llama.add_argument(
        "--flash-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    benchmark_llama.set_defaults(func=command_benchmark_llama)

    benchmark_h2d = subparsers.add_parser(
        "benchmark-h2d", help="run the CUDA Samples H2D bandwidth test"
    )
    benchmark_h2d.add_argument("--executable", required=True)
    benchmark_h2d.add_argument("--output", required=True)
    benchmark_h2d.add_argument("--memory", choices=("pinned", "pageable"), default="pinned")
    benchmark_h2d.add_argument("--start-mib", type=int, default=32)
    benchmark_h2d.add_argument("--end-mib", type=int, default=160)
    benchmark_h2d.add_argument("--increment-mib", type=int, default=16)
    benchmark_h2d.add_argument("--device", type=int, default=0)
    benchmark_h2d.set_defaults(func=command_benchmark_h2d)

    calibrate = subparsers.add_parser(
        "calibrate-configs",
        help="fit effective pipeline service and generate hardware-calibrated configs",
    )
    calibrate.add_argument("--measurements", required=True)
    calibrate.add_argument("--output-dir", required=True)
    calibrate.add_argument("--report", required=True)
    calibrate.set_defaults(func=command_calibrate_configs)

    analyze_nsys = subparsers.add_parser(
        "analyze-nsys",
        help="extract steady-state decode costs from an Nsight SQLite export",
    )
    analyze_nsys.add_argument("--sqlite", required=True)
    analyze_nsys.add_argument("--generation-tokens", required=True, type=int)
    analyze_nsys.add_argument("--layer-count", type=int)
    analyze_nsys.add_argument("--attention-layers", type=int)
    analyze_nsys.add_argument("--ssm-layers", type=int)
    analyze_nsys.add_argument("--embedding-bytes", type=int)
    analyze_nsys.add_argument("--output", required=True)
    analyze_nsys.set_defaults(func=command_analyze_nsys)

    validate_context = subparsers.add_parser(
        "validate-context",
        help="compare cached-depth contextual decode measurements with the simulator",
    )
    validate_context.add_argument("--config", required=True)
    validate_context.add_argument("--specification", required=True)
    validate_context.add_argument("--report", required=True)
    validate_context.add_argument("--csv", required=True)
    validate_context.set_defaults(func=command_validate_context)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
