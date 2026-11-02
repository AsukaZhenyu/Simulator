"""Read-only audit of retained Dense preconditions and CUDA copy implementation.

The interval CPU counter differences are retrospective diagnostics. They include
this process and must never be supplied as external future-resource predictions.
No timing parameter is fitted here and existing evidence is never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def mean(values):
    return statistics.mean(values) if values else None


def summarize(campaign):
    campaign = Path(campaign).resolve()
    rows = []
    for path in sorted((campaign / "jobs").glob("dense-primary-*/result.json")):
        native = read(path)
        trials = native["trials"]
        if len(trials) < 2:
            raise ValueError("at least two chronological blocks required")
        times = [r["wall_ns"] / r["repeats"] for r in trials]
        midpoint = len(times) // 2
        states = [r["preblock_state"] for r in trials]
        ram = [s["system_memory"]["available_bytes"] for s in states]
        gpu = [s["cuda_memory"]["free_bytes"] for s in states if s["cuda_memory"]]
        busy = []
        for earlier, later in zip(states, states[1:]):
            first, last = (s["cpu_cumulative_times_100ns"] for s in (earlier, later))
            total = (last["kernel_including_idle"] - first["kernel_including_idle"]
                     + last["user"] - first["user"])
            idle = last["idle"] - first["idle"]
            if total <= 0 or idle < 0 or idle > total:
                raise ValueError("invalid retrospective machine CPU counter interval")
            busy.append((total - idle) / total)
        early, late = mean(times[:midpoint]), mean(times[midpoint:])
        rows.append({"job": path.parent.name, "native": receipt(path),
                     "block_mean_ns": times, "early_mean_ns": early,
                     "late_mean_ns": late, "late_over_early_minus_one": late / early - 1,
                     "ram_available_min_bytes": min(ram),
                     "ram_available_max_bytes": max(ram),
                     "ram_available_last_minus_first_bytes": ram[-1] - ram[0],
                     "gpu_free_min_bytes": min(gpu) if gpu else None,
                     "gpu_free_max_bytes": max(gpu) if gpu else None,
                     "global_cpu_busy_counter_intervals": busy,
                     "global_cpu_busy_mean": mean(busy),
                     "gpu_clock_mhz": [s["gpu_clock_mhz"] for s in states],
                     "global_counters_are_external_only": False,
                     "states_observed_before_each_untimed_prime": True})
    if len(rows) != 24:
        raise ValueError("retained campaign must contain its complete 24 primary processes")

    cuda = ROOT.parent / "llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml/src/ggml-cuda/ggml-cuda.cu"
    backend = cuda.parent.parent / "ggml-backend.cpp"
    source = cuda.read_text(encoding="utf-8")
    for token in ("ggml_backend_cuda_buffer_set_tensor", "ggml_backend_cuda_buffer_get_tensor",
                  "cudaMemcpyHostToDevice, cudaStreamPerThread", "cudaStreamSynchronize(cudaStreamPerThread)"):
        if token not in source:
            raise ValueError("retained runtime-source semantics changed")
    return {"schema": "h2d-history-retained-prestate-audit-1", "rows": rows,
            "copy_source": {"cuda": receipt(cuda), "backend": receipt(backend),
                "observed_source_semantics": [
                    "host-source ggml_backend_tensor_copy dispatches tensor_set",
                    "CUDA buffer tensor_set uses cudaMemcpyAsync H2D on cudaStreamPerThread, then synchronizes that stream",
                    "host-destination dispatches tensor_get with corresponding completed D2H",
                    "GGML temporary malloc fallback is not the host-device branch",
                    "Executor backend synchronization follows the already-completed buffer copy"],
                "runtime_binary_equivalence_established_by_this_audit": False},
            "interpretation": [
                "memory capacity observations do not measure DRAM or PCIe bandwidth",
                "machine CPU intervals include own inference and prime and are not external supply",
                "constant free VRAM does not establish constant transfer service",
                "inclusive GGML copy response is not pure enqueue time or pure staging time",
                "physical device activity overlaps inclusive host API intervals and cannot be added again"],
            "new_measurement_run": False, "parameters_fitted": False,
            "causal_physical_cause_identified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=ROOT / "build/dac-accuracy-repair-20261026/campaign-v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.campaign)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"out": str(args.out.resolve()), "primary_processes": len(result["rows"])}))


if __name__ == "__main__":
    main()
