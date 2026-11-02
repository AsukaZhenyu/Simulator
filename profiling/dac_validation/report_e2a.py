"""Recompute quality failures and emit a paper-facing E2A report, without refit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_once_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError("existing derived artifact differs: " + str(path))
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def quality_breakdown(summary):
    programs = summary["programs"]
    processes = [dict(workload=p["workload"], strategy=p["strategy"], **r)
                 for p in programs for r in p["processes"]]
    pairs = [dict(workload=p["workload"], strategy=p["strategy"], **r)
             for p in programs for r in p["observer_pairs"]]
    failures = []
    for program in programs:
        causes = []
        if any(not r["mean_gate"] for r in program["processes"]):
            causes.append("process_mean_ape")
        if any(not r["cv_gate"] for r in program["processes"]):
            causes.append("process_cv")
        if program["trial_p90_ape"] > .2:
            causes.append("pooled_trial_p90_ape")
        if any(not r["passed"] for r in program["observer_pairs"]):
            causes.append("observer")
        failures.append(dict(workload=program["workload"], strategy=program["strategy"],
                             fully_qualified=program["fully_qualified"], failure_causes=causes))
    values = sorted(r["ape"] for r in processes)
    return dict(schema="dac-e2a-quality-breakdown-1", processes=len(processes),
        mean_ape_failed_processes=sum(not r["mean_gate"] for r in processes),
        cv_failed_processes=sum(not r["cv_gate"] for r in processes),
        p90_failed_programs=sum(p["trial_p90_ape"] > .2 for p in programs),
        observer_failed_pairs=sum(not r["passed"] for r in pairs),
        observer_failed_programs=sum(any(not r["passed"] for r in p["observer_pairs"]) for p in programs),
        mean_ape_failed_programs=sum(any(not r["mean_gate"] for r in p["processes"]) for p in programs),
        cv_failed_programs=sum(any(not r["cv_gate"] for r in p["processes"]) for p in programs),
        process_mean_ape_median=statistics.median(values),
        process_mean_ape_p90_nearest_rank=values[math.ceil(.9 * len(values)) - 1],
        process_mean_ape_max=max(values), worst_process=max(processes, key=lambda r:r["ape"]),
        worst_cv_process=max(processes, key=lambda r:r["cv"]),
        worst_observer_pair=max(pairs, key=lambda r:abs(r["relative_difference"])),
        program_failure_causes=failures, all_data_retained=True, refit=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    summary = read(out / "summary.json")
    quality = quality_breakdown(summary)
    write_once_json(out / "quality-breakdown.json", quality)
    counts = summary["counts"]
    text = ["# E2A: Independent component prices and complete Dense strategy selection", "",
        "This is a completed, frozen experiment with retained negative results. "
        "It is not a claim that all quality gates passed.", "",
        "## Main result", "",
        f"The frozen independent-action predictor selected the arithmetic-mean real-device winner in "
        f"{counts['regret_point_passes']}/{counts['declared_groups']} declared workload groups. "
        "Point selection regret and its paired-bootstrap upper estimate were zero in each group. "
        f"However, only {counts['qualified_programs']}/{counts['programs']} candidate programs and "
        f"{counts['qualified_groups']}/{counts['declared_groups']} complete groups satisfied every "
        "predeclared accuracy, variability and observer check. Selection agreement is therefore a "
        "descriptive result for this fixed backend, not a fully qualified general selection guarantee.", "",
        "| Workload | Real S0 / S1 / S2 / S3 (ms) | Predicted / real winner | Qualified candidates | Restricted CPU/GPU latency gap |", 
        "|---|---|---|---|---|"]
    for group in summary["workloads"]:
        rows = [p for p in summary["programs"] if p["workload"] == group["workload"]]
        means = " / ".join(f"{n/1e6:.4f}" for n in group["real_mean_ns"])
        text.append(f"| {group['workload']} | {means} | S{group['predicted_choice']} / S{group['real_choice']} | "
                    f"{sum(p['fully_qualified'] for p in rows)}/4 | {100*group['cpu_gpu_only_gap']:.2f}% |")
    text += ["", "The restricted comparator chooses the faster of S0 and S1 using real means. "
        "Its gap is `(restricted latency / best of four latency) - 1`, rather than percentage speedup. "
        "Positive gaps occurred in three groups; their one-sided simultaneous paired-bootstrap lower "
        "bounds were 99.20%, 39.32% and 99.56%. This only establishes the limitation of the two-policy "
        "restricted family in this experiment. S2 is already a known kind of heterogeneous placement; "
        "the study does not establish novelty over every published offloading system.", "",
        "## Accuracy and qualification", "",
        f"Across all 192 fresh heldout process means, median APE was {100*quality['process_mean_ape_median']:.4f}%, "
        f"nearest-rank P90 APE {100*quality['process_mean_ape_p90_nearest_rank']:.4f}%, "
        f"and maximum APE {100*quality['process_mean_ape_max']:.4f}%. "
        "The maximum error of the 24 candidate aggregate arithmetic means was 5.9800%. "
        "No observations were dropped and no target result was used to refit prices.", "",
        f"Mean APE exceeded 10% in {quality['mean_ape_failed_processes']}/192 processes; "
        f"within-process trial CV exceeded 10% in {quality['cv_failed_processes']}/192 processes. "
        f"No candidate failed the pooled trial-average P90 APE limit of 20%. "
        f"The stamped-versus-bare observer check exceeded 5% in {quality['observer_failed_pairs']}/96 pairs, "
        f"affecting {quality['observer_failed_programs']}/24 programs. "
        "These causes overlap. Observer consistency is the principal reason for zero completely "
        "qualified groups; low aggregate mean error does not override the failed gates.", "",
        "Full-stamped actions are post-freeze diagnostics only. They show differences between "
        "inclusive isolated action responses and responses in mixed execution. They do not isolate "
        "OS ready-queue delay, driver waiting, GPU clocks or cache state. In particular, diagnostic "
        "attribution for an observer-failing program is not an unbiased physical-stage ground truth.", "",
        "## Collection and prediction contract", "",
        "Six workload groups use H256/I512 or H512/I1024, four or eight Q heads, two KV heads, "
        "head width 64, batch one, and P16/history0, P64/history0 or D1/history128. "
        "All use F32, one CPU thread, one CUDA GPU, disabled CUDA graphs and a serial action executor. "
        "S0 is all-CPU; S1 all-GPU; S2 GPU projection/FFN with CPU attention; "
        "S3 has materialized half-width CPU/GPU FFN partitions with serial execution. "
        "It does not measure concurrent tensor parallelism. The block includes the declared "
        "RMSNorm, RoPE, Q/K normalization, sigmoid attention gate, SwiGLU and residual operations. "
        "It is a synthetic complete Dense block, not a named production model.", "",
        "All four candidates in each group use identical initial input, positions, weights and "
        "old KV bytes. Partition leaves reconstruct the same full parent weights exactly. "
        "Required final hidden and retained/new KV are in DRAM. Independent NumPy semantics "
        "and native materialized ports were checked. Profile and heldout seeds differ.", "",
        "The campaign contains 471 calibration processes (137 distinct semantic actions times three, "
        "plus ten sizes times two transfer routes times three), 192 heldout processes and 192 observer "
        "processes: 855 normal exits with numerical/contracts checked. Each independent primitive "
        "received at least 500 ms and 20 iterations of active warmup, then eight trials of eight "
        "fixed-input replays. Each heldout candidate has eight fresh processes, each with 12 trials "
        "of eight replays; candidate positions are balanced. Observer checks use four AB/BA pairs "
        "per candidate, eight trials of eight replays per process. Reported latency is a replay "
        "average, not autoregressive token latency or a serving tail percentile.", "",
        "Semantic keys contain operation/configuration, full operand/output geometry and layout, "
        "dtype, backend, threads, native timing boundary and binary/runtime/hardware context. "
        "They exclude program or strategy identity, tensor names, action ordinal, seed and values. "
        "Only exact coordinates are accepted. This tests independent composition at measured "
        "shapes; it does not test unseen-shape extrapolation. Untimed whole-program initialization "
        "prepares valid probe inputs, but complete-sequence timings and full-stamped costs are "
        "excluded from calibration. Prices and predictions were frozen before heldout began.", "",
        "The prediction is the serial sum of inclusive native COMPUTE, COPY, ADVANCE and EVICT "
        "responses. Dispatch and completion synchronization are included once. No extra submission "
        "cost is added. An additional history-dependent transition correction was not independently "
        "identified and is omitted, rather than asserted to be a measured zero. Startup, graph "
        "construction, allocation, input reset, numerical checking and serialization are outside "
        "the timing interval. Actual declared compute, transfer and completion work remain inside.", "",
        "IR compilation uses build_case -> ModelPlan.compile -> export_plan. The initial 1 ns "
        "prices, zero workspaces and 1 GiB pools are explicit compile placeholders for a fixed graph "
        "and order, never the measured prediction. The real CUDA private workspace remains unknown. "
        "The native executor preallocates backing and makes EVICT logical. This experiment "
        "therefore does not qualify actual allocator/capacity behavior, asynchronous prefetch, "
        "real eviction, MoE, multi-round generation or unified-memory coherence.", "",
        "## Transfer and KV diagnostic limits", "",
        "The separate ten-size transfer regression `time = fixed + bytes / bandwidth` failed the "
        "predeclared maximum relative residual limit of 10% on both routes: 28.7251% for VRAM-to-DRAM "
        "and 14.9414% for DRAM-to-VRAM. The fixed/payload decomposition is not qualified, although "
        "the exact measured COPY coordinates remain available. Changing only a link bandwidth "
        "must not silently scale the whole inclusive COPY price.", "",
        "D1/history128 S1 and S3 each perform 128 old-KV DRAM-to-VRAM copies (131072 bytes) "
        "and one new-KV VRAM-to-DRAM copy (1024 bytes). Their total independent KV COPY prediction "
        "is 2.821267708 ms. This is 20.9979%/20.7206% of H256 S1/S3 and 18.6200%/18.4650% of "
        "H512 S1/S3. S0 and S2 have no KV COPY. Token-block materialization and copy granularity "
        "are fixed backend contracts; copy coalescing was not evaluated. A finite priority-rule "
        "audit does not prove that such untested lower-level choices cannot alter strategy rankings.", "",
        "## Reproducibility and artifacts", "",
        "Native build: MSVC 19.44.35217.0/v143, Release/MaxSpeed, C++17, CUDA SDK 12.6.0, "
        "NVCC 12.6.20, CUDA runtime 12.6.37, compute architecture 89. The source asset is labelled "
        "llama.cpp-b8705 but has no verified exact git revision (build-info says unknown). "
        "File hashes bind the supplied sources and binary. The private F32 router runtime reused "
        "Release objects and rebuilt one patched translation unit; it is not a clean GGML rebuild. "
        "Loaded runtime module paths were not independently observed. Exact receipts are in "
        "native-reproducibility.json.", "",
        "Authoritative outputs: summary.json, process_metrics.csv (192 rows), trial_metrics.csv "
        "(2304 rows), selection_metrics.csv (six rows), quality-breakdown.json, profile-frozen.json "
        "and predictions.json. Every native request, result, stdout, stderr and execution receipt "
        "is retained and indexed by artifact-manifest.json. Post-freeze diagnostics are in "
        "component_diagnostics.csv, postfreeze-diagnostics.json and kv-copy-ledger.json. "
        "The visually checked figure is figures/real_strategy_selection_v2.png and .svg.", "",
        "The collector/model were frozen before heldout and were not revised after observing "
        "results. Report generation and figures are additive postprocessing. Seven focused "
        "cost-contract/balanced-order/numerical-boundary regressions passed. The explicit-build "
        "portable preparation adapter is supplied but has only been syntax-checked; a fresh "
        "cross-machine campaign has not been claimed.", ""]
    report = out / "E2A_REPORT.md"
    content = "\n".join(text)
    if report.exists() and report.read_text(encoding="utf-8") != content:
        raise ValueError("existing report differs")
    if not report.exists():
        with report.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    paths = ["summary.json", "profile-frozen.json", "predictions.json", "artifact-manifest.json",
             "quality-breakdown.json", "E2A_REPORT.md", "process_metrics.csv", "trial_metrics.csv",
             "selection_metrics.csv", "postfreeze-diagnostics.json", "component_diagnostics.csv",
             "kv-copy-ledger.json", "kv_copy_metrics.csv", "diagnostic-manifest.json",
             "native-reproducibility.json", "figures/real_strategy_selection_v2.png",
             "figures/real_strategy_selection_v2.svg", "figures/real_strategy_selection_v2-manifest.json"]
    delivery = dict(schema="dac-e2a-delivery-1", source=receipt(__file__),
                    artifacts=[receipt(out / p) for p in paths], native_timing_active=False,
                    target_results_refitted=False, complete_group_quality_qualified=False)
    write_once_json(out / "delivery-manifest.json", delivery)
    print(json.dumps(dict(report=str(report), counts=counts, quality=quality), indent=2))


if __name__ == "__main__":
    main()
