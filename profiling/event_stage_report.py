"""Render the already-frozen event-stage experiment; never fit parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact(value):
    if isinstance(value, dict):
        return value["numerator"] / value["denominator"]
    return value


def pct(value):
    return "unknown" if value is None else f"{100 * value:.2f}%"


def table(metrics):
    lines = ["| Declared candidate | Known / all calls | Median APE | P90 APE | Worst APE | Full point gate |",
             "|---|---:|---:|---:|---:|---|"]
    for name, row in metrics.items():
        lines.append(f"| {name} | {row['known_calls']}/{row['expected_calls']} | "
            f"{pct(row['known_subset_median_ape'])} | {pct(row['known_subset_p90_ape'])} | "
            f"{pct(row['known_subset_worst_ape'])} | {'pass' if row['point_accuracy_gate'] else 'fail'} |")
    return "\n".join(lines)


def plot(evaluation, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in evaluation["declared_candidates"] if n in {
        "profile_scalar_median", "cycle_after_host_no_calendar", "cycle_after_host_before_only", "cycle_after_host_causal_history"}]
    # Preserve declared names without requiring a particular candidate spelling.
    if not names:
        names = evaluation["declared_candidates"][:4]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in names:
        rows = [r for r in evaluation["calls"] if r["predictions"][name]["known"]]
        errors = sorted(abs(exact(r["predictions"][name]["predicted_ns"])-r["observed_wall_ns"])
                        / r["observed_wall_ns"] for r in rows)
        if errors:
            axes[0].plot([100*x for x in errors], [(i+1)/len(errors) for i in range(len(errors))], label=f"{name} ({len(errors)})")
    axes[0].axvline(20, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Absolute percentage error (%)", ylabel="CDF of known predictions", title="Fresh heldout; unknown coverage reported separately")
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=7)
    states = sorted(evaluation["by_condition"])
    for index, state in enumerate(states):
        values = [r["observed_wall_ns"]/1e6 for r in evaluation["calls"] if r["condition"]==state]
        axes[1].scatter([index + (i % 13 - 6)*.012 for i in range(len(values))], values, s=6, alpha=.35)
        axes[1].plot([index-.2, index+.2], [statistics.median(values)]*2, color="black")
    axes[1].set(xticks=range(len(states)), xticklabels=states, ylabel="Observed whole-call time (ms)", title="Every heldout call retained")
    axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        path=destination.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180)
    plt.close(fig)


def render(root, evaluation_path, audit_path, report_path, figure_path):
    root=Path(root).resolve()
    evaluation, audit=read(evaluation_path), read(audit_path)
    frozen=read(evaluation["frozen_file"])
    observer=audit["observer"]
    counts=audit["counts"]
    heldout_jobs=[r for r in audit["jobs"] if r["job"]["split"]=="heldout" and r["status"]=="included"]
    stable=sum(r["wall_ns"]["cv"] is not None and r["wall_ns"]["cv"]<=.1 for r in heldout_jobs)
    lines=["# Event-stage correction and fresh validation — 2026-10-22", "",
        "**This round fixes measurement and modeling mechanisms. It does not establish an accurate, continuous available-resource model.**",
        "The full qualification remains false: point accuracy, observer noninterference, source integrity and coverage are separate requirements. "
        "The cycle observer's 5% gate is reported independently and cannot be repaired by adjusting predicted means.", "",
        "## What changed", "",
        "1. Profiling actors block on parent-owned Windows GO/STOP events after READY. The old periodic file polling is retained only as a declared matched control. "
        "The executed layer actions and synchronization boundaries are unchanged.",
        "2. Modeling conserves raw CPU cycle work using `cpu_cycle/s`. It does not convert cycles with an assumed GHz. "
        "PROFILE and independent BEFORE capacity estimates use the same complete reference-iteration wall denominator.",
        "3. A submission graph separates enqueue acceptance, device-stream execution and completion waits. "
        "An already-enqueued device command can continue while the host has no CPU service. Inclusive host response cannot also be charged as device work.",
        "4. A coarse cycle-response graph tests explicit `after_host` and `overlap_host` assumptions in the common evaluator. "
        "Its fitted completion response is not measured GPU-active service or a reconstructed CUDA submission DAG.",
        "5. A fixed-parameter phase observer distinguishes BEFORE-only forecasting from updates using completed PAST target calls. "
        "Each prediction is sealed before that call's outcome is extracted; current/future target latency never supplies its own prediction.", "",
        "## Protocol and evidence", "",
        f"- PROFILE: 12 new processes × 96 calls; fresh bare-target heldout: 12 processes × 96 calls ({evaluation['expected_calls']} calls). "
        "Three CPU/GPU contention diagnostics use CUPTI; nine idle controls compare polling, events and cycle instrumentation.",
        "- One H=128, q=16, history=0 F32 dense layer, fixed single-CPU-thread placement and the same RTX 4070 Laptop GPU. "
        "This is a small controlled mixed CPU/GPU workload, not production-scale LLM validation.",
        "- PROFILE target cycle counters are enabled; heldout target counters are disabled. Heldout reference counters are collected but never read as prediction inputs. "
        "Legal capacity inputs are frozen PROFILE work and independent BEFORE reference wall/QPC observations.",
        "- Every heldout call, including long pauses, contributes to its applicable metric. Unknown calendar phase remains unknown. "
        "Conditional known-subset errors cannot establish full coverage.",
        "- Invocation entry timestamps are observed legal inputs. These are conditional per-call predictions, not an autonomous forecast of an entire token stream's future start times.",
        f"- Frozen parameter file SHA256: `{evaluation['frozen_file_sha256']}`. No heldout fitting, DNN, state-label lookup or whole-layer offset is used.",
        f"- Audit: {counts['included']}/{counts['planned']} declared jobs included; {counts['excluded']} excluded; "
        f"{counts['pending']} pending; {len(audit['source_errors'])} source errors. Integrity audit pass: **{audit['passed']}**.", "",
        f"- Retained infrastructure failures: **{len(audit.get('retained_failed_attempts', []))}**. "
        "The Windows completion-file read failure is excluded as a whole unit and replaced under an append-only same-configuration index. "
        "Original raw files and hashes remain intact; target latency never selects a retry.", "",
        f"Heldout process stability (whole-call CV ≤10%): **{stable}/{len(heldout_jobs)}**. This describes measured variation, "
        "not whether a time-varying model correctly predicts individual slow calls.", "",
        "## Observer controls", "",
        "| Repeat | Poll mean (ms) | Event mean (ms) | Event + cycles mean (ms) | Event versus poll | Cycles versus event |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in observer["controls"]:
        changes=row["relative_changes"]
        lines.append(f"| {row['repeat']} | {row['poll']['mean']/1e6:.4f} | {row['event']['mean']/1e6:.4f} | "
            f"{row['event_cycles']['mean']/1e6:.4f} | {pct(changes['event_vs_poll']['mean'])} | {pct(changes['cycles_vs_event']['mean'])} |")
    lines += ["", f"Cycle-observer ≤5% gate: **{'pass' if observer['cycle_observer_overhead_pass'] else 'fail'}**. "
        "These are separate process controls in fixed declared order. Their differences are observed protocol effects, not a randomized causal attribution. "
        "No observer overhead is subtracted from heldout measurements or predictions.", "", "## Fresh heldout results", "", table(evaluation["metrics"]), "",
        "APE = |predicted − observed| / observed. Percentiles pool all applicable calls, rather than averaging process percentiles. "
        "The point gate requires complete coverage, median APE ≤20% and P90 APE ≤20%. Hardware qualification additionally requires the observer and validity gates.", ""]
    for state, values in evaluation["by_condition"].items():
        lines += [f"### Offered-load condition: {state}", "", table(values), ""]
    hypothesis=frozen.get("CPU_blackout_hypothesis") or {}
    lines += ["## What the fitted quantities can and cannot establish", "",
        f"The PROFILE-only host-pause hypothesis has period `{hypothesis.get('period_candidate_ns')}` ns and width "
        f"`{hypothesis.get('pause_width_candidate_ns')}` ns. Phase is supplied separately, never fitted from a heldout call's own outcome. "
        "A periodic host-response pattern does not prove an OS scheduling mechanism or independently measure neutral CPU availability.", "",
        "The GPU reference feature includes host launch/wait response. CPU work and completion residuals are therefore scoped empirical response estimates. "
        "Nonnegative least squares being solvable does not establish causal identification, a neutral GPU capacity or independent physical components. "
        "Negative unprojected residuals remain in the frozen evidence; the constrained fit is explicitly underidentified.", "",
        "The completion residual is calibrated under an additive `after_host` assumption. Reusing it under `overlap_host` tests structural sensitivity; "
        "it is not a separately calibrated physical execution model. Fixed head/tail bounds and PROFILE counter-query work can retain instrumentation bias.", "",
        "The BEFORE references are stale workload-response probes. Causal-history updates use past target outcomes and must be reported separately from open-loop prediction. "
        "Calendar center predictions and assumed uncertainty growth are diagnostic forecasts, not physically proven future resource trajectories.", "",
        "## Startup, useful work and strategy latency", "",
        "Process/Executor construction and 20 warmups occur before READY and outside the per-invocation timer. "
        "The isolated benchmark reset is also outside that timer. Recurrent submissions, launches, copies and synchronizations remain inside. "
        "We optimize warm strategy latency while retaining activity diagnostics; a long API wall is not automatically a kernel-launch cost. "
        "See [Timing boundaries](TIMING_METRICS.md).", "",
        "| Traced diagnostic | Whole-call mean (ms) | GPU kernel envelope (ms) | GPU memcpy envelope (ms) | API union (ms) |",
        "|---|---:|---:|---:|---:|"]
    for row in audit['jobs']:
        diagnostic=row.get('diagnostic')
        if diagnostic:
            lines.append(f"| {row['job']['condition']} | {diagnostic['wall_ns']['mean']/1e6:.4f} | "
                f"{diagnostic['GPU_kernel_envelope_ns']['mean']/1e6:.4f} | {diagnostic['GPU_memcpy_envelope_ns']['mean']/1e6:.4f} | "
                f"{diagnostic['API_union_ns']['mean']/1e6:.4f} |")
    lines += ["", "These columns overlap and must not be added. GPU intervals are activity envelopes, not guaranteed useful arithmetic service. "
        "Pure CPU compute time remains unknown. CUPTI-instrumented diagnostics are separate from the bare-target accuracy cohort.", "",
        "## Remaining accuracy work", "",
        "- Lower or explicitly isolate per-action counter overhead before treating PROFILE parameters as noninterfering measurements.",
        "- Identify enqueue/host/wait components with independently timed backend microprograms; do not infer GPU active demand by subtracting host wall time.",
        "- Validate availability traces and recovery behavior independently of the target outcome. Separate replay with a supplied resource trace from online prediction of an unknown future trace.",
        "- After the small layer passes both median and tail gates, broaden to different shapes, token phases, partitions and hardware. "
        "This round cannot justify generalizing to production Dense/MoE strategies.", "",
        "## Reproduce", "", "```powershell",
        "python -m profiling.event_stage_evaluate evaluate --campaign profiling/available_service/results/event-v1-h128-t1 `",
        "  --frozen profiling/available_service/results/event-v1-h128-t1/profile-frozen.json `",
        "  --barrier profiling/available_service/results/event-v1-h128-t1/event-frozen-barrier.json --output build/event-reevaluation.json",
        "python -m profiling.event_stage_audit profiling/available_service/results/event-v1-h128-t1 --output build/event-reaudit.json",
        "```", "", "Existing outputs are write-once. Use new output paths for verification; do not overwrite frozen evidence.", ""]
    report_path=Path(report_path)
    with report_path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write("\n".join(lines))
    plot(evaluation,Path(figure_path))
    receipts={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in (evaluation_path,audit_path,report_path,Path(figure_path).with_suffix('.png'),Path(figure_path).with_suffix('.svg'))}
    return dict(report=str(report_path.resolve()),receipts=receipts,hardware_accuracy_qualified=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","evaluation","audit","report","figure"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(render(a.campaign,a.evaluation,a.audit,a.report,a.figure),indent=2))
