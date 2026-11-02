"""Assemble retained accuracy evidence without executing hardware or fitting.

Pure Python standard library. Source protocol verifiers remain authoritative;
this reader preserves their results and makes failed/missing coverage explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

try:
    from .highlevel_rankings import rank_highlevel
except ImportError:
    import importlib.util
    _ranking_spec=importlib.util.spec_from_file_location("accuracy_highlevel_rankings",Path(__file__).with_name("highlevel_rankings.py"))
    _ranking_module=importlib.util.module_from_spec(_ranking_spec)
    _ranking_spec.loader.exec_module(_ranking_module)
    rank_highlevel=_ranking_module.rank_highlevel

SCHEMA = "comprehensive-accuracy-evidence-report-0.1"
QUALITY = dict(max_within_process_cv=.15, max_between_process_cv=.20,
               max_heldout_relative_error=.20, min_processes_per_split=3)
CHAIN_STRATEGIES = {
    "ffn": ("cpu/resident", "gpu/resident", "hybrid/resident", "hybrid/preload",
            "cpu/joint_resident", "gpu/joint_resident", "hybrid/joint_resident"),
    "attention": ("cpu/resident", "gpu/resident", "hybrid/resident",
                  "cpu/joint_resident", "gpu/joint_resident"),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def maximum(values):
    values = [x for x in values if finite(x)]
    return max(values) if values else None


def error_summary(rows, key="absolute_relative_error"):
    values = sorted(row[key] for row in rows if finite(row.get(key)))
    return dict(unit="one declared case/path pooled-median relative error, unless stated otherwise",
                declared=len(rows), known=len(values), unknown=len(rows)-len(values),
                median=statistics.median(values) if values else None,
                max=values[-1] if values else None,
                within_20_percent=sum(x <= .20 for x in values))


def gate_counts(rows):
    names = sorted({name for row in rows for name in row.get("gates", {})})
    return {name: dict(passed=sum(row.get("gates", {}).get(name) is True for row in rows),
                       failed=sum(row.get("gates", {}).get(name) is False for row in rows),
                       unknown=sum(type(row.get("gates", {}).get(name)) is not bool for row in rows))
            for name in names}


def accepted(rows, field="accepted"):
    return bool(rows) and all(row.get(field) is True for row in rows)


class Inputs:
    def __init__(self):
        self.errors = []
        self.cache = {}

    def read(self, path, *, required=True, fallback=None):
        path = Path(path).resolve()
        if path in self.cache:
            return self.cache[path]
        if not path.is_file():
            if required:
                self.errors.append(dict(path=str(path), error="required evidence file is missing"))
            return {} if fallback is None else fallback
        try:
            value = strict_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.errors.append(dict(path=str(path), error=str(exc)))
            return {} if fallback is None else fallback
        self.cache[path] = value
        return value


def snapshot(paths):
    files, missing = {}, []
    for supplied in paths:
        root = Path(supplied).resolve()
        if root.is_file():
            files[str(root)] = sha(root)
        elif root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    files[str(path.resolve())] = sha(path)
        else:
            missing.append(str(root))
    return dict(files=files, missing_paths=sorted(missing))


def chain_summary(root, source):
    root = Path(root)
    suite = source.read(root / "suite.json")
    plan = source.read(root / "plan.json", fallback=[])
    rows = source.read(root / "measurements.json", fallback=[])
    report = source.read(root / "analysis/report.json")
    trace = source.read(root / "trace-report.json")
    shape = source.read(root / "shape-report.json")
    diagnostics = source.read(root / "diagnostic-summary.json")
    manifest = source.read(root / "manifest.json")
    baseline = {(r["case_id"], r["strategy"]): r for r in report.get("results", [])}
    traces = {(r["case_id"], r["strategy"]): r for r in trace.get("results", [])}
    shapes = {(r["case_id"], r["strategy"]): r for r in shape.get("results", [])}
    ids = Counter(r.get("id") for r in rows)
    expected = {r.get("id") for r in plan}
    actual = set(ids)
    failures = [dict(id=r.get("id"), case_id=r.get("case", {}).get("id"), split=r.get("split"),
                     error=r.get("error"), raw_path=r.get("raw_path"))
                for r in rows if r.get("status") != "measured"]
    source_targets = shape.get("profile_target_sources", [])
    target_between = shape.get("profile_target_between_process", [])
    paths, cases = [], []
    for case in suite.get("cases", []):
        cid = case["id"]
        selected = [r for r in rows if r.get("case", {}).get("id") == cid]
        by_split = {}
        for split in ("profile", "heldout"):
            found = [r for r in selected if r.get("split") == split]
            declared = [r for r in plan if r.get("case", {}).get("id") == cid and r.get("split") == split]
            by_split[split] = dict(declared=len(declared), retained=len(found),
                measured=sum(r.get("status") == "measured" for r in found),
                failed=sum(r.get("status") != "measured" for r in found),
                missing_ids=sorted({r["id"] for r in declared}-{r["id"] for r in found}),
                processes=[dict(id=r.get("id"), seed=r.get("seed"), repeat=r.get("repeat"),
                    status=r.get("status"), child_pid=r.get("child_pid"), raw_sha256=r.get("raw_sha256"),
                    metric_sample_counts={k:len(v) for k,v in r.get("metrics", {}).items()},
                    diagnostic_invocation_counts={k:len(v) for k,v in r.get("chain_traces", {}).items()}) for r in found])
        case_paths = []
        strategies = sorted(CHAIN_STRATEGIES.get(case.get("family"), ()))
        for strategy in strategies:
            b = baseline.get((cid, strategy), {})
            t, s = traces.get((cid, strategy), {}), shapes.get((cid, strategy), {})
            evidence, truth = b.get("source_evidence", []), b.get("heldout_evidence", [])
            profile_processes=len({x.get("row_id") for x in evidence})
            heldout_processes=len({x.get("row_id") for x in truth})
            item = dict(case_id=cid, family=case.get("family"), strategy=strategy,
                independent=dict(predicted_ns=b.get("predicted_ns"), observed_ns=b.get("observed_ns"),
                    absolute_relative_error=b.get("absolute_relative_error"), source_calibrated=b.get("calibrated"),
                    accepted=b.get("calibrated") is True and profile_processes>=3 and heldout_processes>=3,
                    gates={**b.get("gates", {}),"assembler_minimum_evidence_processes":profile_processes>=3 and heldout_processes>=3},
                    profile_component_processes=profile_processes, heldout_processes=heldout_processes,
                    profile_within_process_cv_max=maximum(x.get("cv") for x in evidence),
                    heldout_within_process_cv_max=maximum(x.get("cv") for x in truth),
                    between_process_cv=b.get("between_process_cv"),
                    unknown_reason=b.get("unknown_reason")),
                inchain=dict(predicted_ns=t.get("predicted_ns"), observed_ns=t.get("observed_uninstrumented_ns"),
                    absolute_relative_error=t.get("absolute_relative_error"), accepted=t.get("calibrated") is True,
                    gates=t.get("gates", {}), profile_cost_count=len(t.get("profile_cost_cv", {})),
                    profile_cost_cv=t.get("profile_cost_cv", {}),
                    heldout_within_process_cv=t.get("heldout_within_process_cv"),
                    heldout_between_process_medians_cv=t.get("heldout_between_process_medians_cv"),
                    status="available" if t else "unknown"))
            for mode, key in (("shared", "full"), ("leave_one_case_out", "leave_one_case_out")):
                prediction = s.get(key, {})
                target_sources = [x for x in source_targets if mode == "shared" or x.get("case_id") != cid]
                between_sources = [x for x in target_between if mode == "shared" or x.get("case_id") != cid]
                base_gates = {k:b.get("gates",{}).get(k) is True for k in (
                    "numeric_and_process_coverage","model_frozen_before_heldout","within_process_cv","between_process_cv")}
                base_gates["assembler_minimum_evidence_processes"]=profile_processes>=3 and heldout_processes>=3
                err = prediction.get("absolute_relative_error")
                gates = {**base_gates,
                    "profile_joint_target_within_process_cv": bool(target_sources) and all(finite(x.get("within_process_cv")) and x["within_process_cv"] <= .15 for x in target_sources),
                    "profile_joint_target_between_process_cv": bool(between_sources) and all(finite(x.get("between_process_cv")) and x["between_process_cv"] <= .20 and x.get("processes",0) >= 3 for x in between_sources),
                    "shared_model_frozen_before_heldout": shape.get("frozen_before_heldout") is True,
                    "heldout_relative_error": finite(err) and err <= .20,
                    "prediction_available": finite(prediction.get("predicted_ns"))}
                item[mode] = dict(predicted_ns=prediction.get("predicted_ns"), observed_ns=s.get("observed_ns"),
                    absolute_relative_error=err, correction_contributions_us=prediction.get("correction_contributions_us"),
                    unknown_reason=prediction.get("unknown_reason", "strategy outside this model's GPU/hybrid primary scope" if not s else None),
                    gates=gates, accepted=bool(s) and all(v is True for v in gates.values()),
                    gate_author="report assembler, original fixed thresholds including profile calibration-target CVs",
                    profile_target_process_path_observations=len(target_sources),
                    excluded_case_id=cid if mode != "shared" else None)
            case_paths.append(item)
            paths.append(item)
        cases.append(dict(case=case, processes=by_split, path_count=len(case_paths)))
    mode_summaries = {}
    for mode in ("independent", "inchain", "shared", "leave_one_case_out"):
        known = [p[mode] for p in paths if mode in ("independent", "inchain") or (p["case_id"],p["strategy"]) in shapes]
        mode_summaries[mode] = dict(path_count=len(known), accepted=sum(x["accepted"] for x in known),
            errors=error_summary(known), gates=gate_counts(known))
    family_counts = {}
    for family in sorted({c.get("family") for c in suite.get("cases", [])}):
        cids = {c["id"] for c in suite["cases"] if c["family"] == family}
        family_counts[family] = dict(cases=len(cids), declared_paths=sum(p["case_id"] in cids for p in paths),
            retained_processes=sum(r.get("case", {}).get("id") in cids for r in rows),
            measured_processes=sum(r.get("case", {}).get("id") in cids and r.get("status") == "measured" for r in rows),
            failures=sum(x.get("case_id") in cids for x in failures))
    return dict(directory=str(root.resolve()), scope="F32 compact FFN and post-RoPE attention, exact native paths",
        expected_cases=len(suite.get("cases", [])), retained_processes=len(rows), declared_processes=len(plan),
        families=family_counts, cases=cases, paths=paths, model_summaries=mode_summaries,
        failures=failures, missing_process_ids=sorted(expected-actual), unplanned_process_ids=sorted(actual-expected),
        duplicate_process_ids={str(k):v for k,v in ids.items() if v>1},
        process_coverage_complete=bool(plan) and expected==actual and not failures and all(v==1 for v in ids.values()),
        baseline_all_latency_gates_passed=report.get("all_declared_strategies_calibrated") is True,
        inchain_all_latency_gates_passed=trace.get("all_declared_strategies_calibrated") is True,
        assets_unchanged_during_campaign=manifest.get("assets_unchanged"),
        frozen_model_hashes={k:manifest.get(k) for k in ("model_frozen_before_heldout_sha256", "trace_model_frozen_before_heldout_sha256", "shape_models_frozen_before_heldout_sha256")},
        near_crossing_selection=report.get("near_crossing_selection", {}), rankings=report.get("rankings", []),
        diagnostic_summary=diagnostics, quality=suite.get("quality"), unknown_or_uncovered=report.get("unknown_or_uncovered", []))


def highlevel_summary(root, source):
    root = Path(root)
    campaign = source.read(root / "campaign.json")
    phase = source.read(root / "profile_phase_status.json")
    barrier = source.read(root / "all_profiles_frozen.json", required=False)
    report = source.read(root / "report.json")
    results, groups = report.get("results", []), report.get("case_groups", [])
    cases, process_records = [], []
    for item in campaign.get("cases", []):
        cid = item["id"]
        ledger = source.read(root / cid / "frozen_profile.json", required=False)
        process_stats = ledger.get("independent_process_statistics", [])
        local = [x for x in results if x.get("case_id") == cid]
        group = next((x for x in groups if x.get("case_id") == cid), {})
        records = {}
        for split, prefix in (("profile", "profile-process-"), ("heldout", "heldout-process-")):
            found = []
            for path in sorted((root / cid).glob(prefix + "*/execution.json")):
                raw = source.read(path)
                native = source.read(path.parent / "native.json", required=False)
                entry = dict(case_id=cid,split=split,path=str(path.resolve()),returncode=raw.get("returncode"),
                    timed_out=raw.get("timed_out",False),launch_failed=raw.get("launch_failed",False),
                    exit_success=raw.get("returncode")==0 and not raw.get("timed_out"),error=raw.get("stderr") if raw.get("returncode")!=0 else None,
                    native_mode=native.get("mode"),native_samples=native.get("samples"),
                    complete_path_sample_count=len(native.get("total_samples_ns",[])),
                    component_action_sample_counts={str(x.get("action_index")):len(x.get("samples_ns",[])) for x in native.get("components",[])})
                found.append(entry);process_records.append(entry)
            records[split] = dict(retained_raw_processes=len(found),successful_exits=sum(x["exit_success"] for x in found),
                                  failed_exits=sum(not x["exit_success"] for x in found))
        within = [x.get("cv") for p in process_stats for x in p.get("components", [])]
        between = [x.get("cv") for x in ledger.get("between_process_component_statistics", [])]
        cases.append(dict(case_id=cid,shape_and_strategy=item.get("kwargs"), processes=records,
            recorded_profile_processes=len(process_stats), recorded_heldout_processes=len(local),
            measured_component_actions=len(ledger.get("components", [])),
            profile_within_process_cv_max=maximum(within), profile_between_process_median_cv_max=maximum(between),
            source_quality_passed=ledger.get("source_quality_passed"),
            heldout_within_process_cv_max=maximum(x.get("heldout_within_process_cv") for x in local),
            group=group, absolute_relative_error=group.get("pooled_median_absolute_relative_error"),
            accepted=group.get("accepted") is True,
            actual_modelplan_policy_rerun=bool(local) and all(x.get("actual_modelplan_prediction",{}).get("actual_ModelPlan_policy_rerun") is True for x in local),
            native_complete_workspace_or_memory_claimed=ledger.get("native_complete_workspace_or_memory_claimed"),
            workspace_scope=ledger.get("workspace_scope")))
    failures = [dict(phase="profile",**x) for x in phase.get("failures", [])] + [dict(phase="heldout",**x) for x in report.get("failures", [])]
    for path in root.glob("*_phase_failure.json"):
        failures.append(dict(path=str(path.resolve()),**source.read(path)))
    expected_profile = len(cases)*barrier["profile_processes_per_case"] if "profile_processes_per_case" in barrier else None
    expected_heldout = len(cases)*report["independent_heldout_processes_per_case"] if "independent_heldout_processes_per_case" in report else None
    return dict(directory=str(root.resolve()),scope="actual public ModelPlan.compile/run exported graph, operands and accepted serial actions",
        cases=cases, case_count=len(cases),families={"dense":dict(cases=len(cases),declared_paths=len(cases),retained_raw_processes=len(process_records))},
        profile_raw_processes=sum(x["split"]=="profile" for x in process_records),
        heldout_raw_processes=sum(x["split"]=="heldout" for x in process_records),declared_profile_processes=expected_profile,
        declared_heldout_processes=expected_heldout, process_records=process_records,failures=failures,
        errors=error_summary(cases), accepted_cases=sum(x["accepted"] for x in cases),
        all_cases_accepted=report.get("all_cases_accepted") is True and not failures,
        all_profiles_frozen_before_any_heldout=report.get("all_profiles_frozen_before_any_heldout"),
        source_quality_passed=phase.get("source_quality_passed"), fixed_quality_gates=report.get("fixed_quality_gates"),
        profile_barrier_sha256=report.get("barrier_sha256"),
        common_boundary_strategy_comparisons=rank_highlevel(root,campaign,report),
        limitations=["One complete Dense invocation with optional AttentionStages/FFNPartition and explicit serial policy.",
                     "Preallocated independent backing with logical evictions; no native memory-capacity equivalence.",
                     "Profile placeholders never become measured predictions; measured policy action sequence must remain exact.",
                     "Assembler does not invoke standalone protocol verifiers. Saved acceptance requires a separately successful verifier receipt."])


def workload_summary(root, source, expected_backend):
    root=Path(root)
    suite=source.read(root/"suite.json")
    manifest=source.read(root/"manifest.json")
    report=source.read(root/"report.json")
    model=source.read(root/"frozen-model.json")
    records, memory=[],[]
    for path in sorted(root.glob("*.raw.json")):
        raw=source.read(path)
        case=raw.get("case",{})
        entry=dict(path=str(path.resolve()),case_id=case.get("id"),split=raw.get("split"),seed=case.get("seed"),
            backend=case.get("backend"),returncode=raw.get("returncode"),timed_out=raw.get("timed_out",False),
            measured=raw.get("returncode")==0 and not raw.get("validation_error") and not raw.get("timed_out"),
            validation_error=raw.get("validation_error"),frozen_model_sha256_at_start=raw.get("frozen_model_sha256_at_start"))
        records.append(entry)
        if entry["measured"]:
            try:native=strict_json(raw["stdout"])
            except (KeyError,ValueError) as exc:
                entry.update(measured=False,validation_error=str(exc));continue
            mem=native.get("memory",{})
            memory.append(dict(case_id=case.get("id"),split=raw.get("split"),seed=case.get("seed"),
                peak_tracked_reserved_bytes=mem.get("peak_tracked_reserved_bytes"),final_tracked_reserved_bytes=mem.get("final_tracked_reserved_bytes"),
                allocation_event_count=sum(x.get("event")=="allocate" for x in mem.get("events",[])),
                release_event_count=sum(x.get("event")=="release" for x in mem.get("events",[])),
                tracked_budget_pass=mem.get("tracked_budget_pass"),physical_capacity_verified=mem.get("physical_capacity_verified"),
                cuda_private_workspace_bytes=mem.get("cuda_private_workspace_bytes"),process_peak_bytes=mem.get("process_peak_bytes"),
                cuda_meminfo_scope=mem.get("cuda_meminfo_scope"),scope=mem.get("scope")))
    cases=[]
    for case in suite.get("cases",[]):
        cid=case["id"]
        found=[r for r in records if r["case_id"]==cid]
        expected={(split,seed) for split in ("profile","heldout") for seed in suite.get(split+"_seeds",[])}
        identities=Counter((r["split"],r["seed"]) for r in found)
        group=next((x for x in report.get("cases",[]) if x.get("case_id")==cid),{})
        local=[x for x in report.get("rows",[]) if x.get("case_id")==cid]
        cases.append(dict(case_id=cid,layer_kinds=[x.get("kind") for x in case.get("layers",[])],
            layers=len(case.get("layers",[])),rounds=len(case.get("rounds",[])),hidden=case.get("hidden"),intermediate=case.get("intermediate"),
            tokens_by_round=[len(x.get("hidden_input",[])) for x in case.get("rounds",[])],
            process_counts={split:dict(retained=sum(r["split"]==split for r in found),measured=sum(r["split"]==split and r["measured"] for r in found)) for split in ("profile","heldout")},
            missing_processes=[dict(split=s,seed=seed) for s,seed in sorted(expected-set(identities))],
            duplicate_processes=[dict(split=s,seed=seed,count=n) for (s,seed),n in identities.items() if n>1],
            report_group=group,process_results=local,accepted=group.get("accepted") is True,
            per_process_error_summary={**error_summary(local),"unit":"one heldout native process median, not pooled case median"}))
    failed=[x for x in records if not x["measured"]]
    sources=[dict(key=k,kind="stage",**v) for k,v in model.get("stages",{}).items()]+[dict(key=k,kind="effective_host_gap",**v) for k,v in model.get("host_gaps",{}).items()]
    complete=bool(cases) and not failed and all(not x["missing_processes"] and not x["duplicate_processes"] for x in cases)
    families={}
    for case in cases:
        family="+".join(sorted(set(case["layer_kinds"])))
        row=families.setdefault(family,dict(cases=0,declared_paths=0,retained_processes=0,measured_processes=0))
        row["cases"]+=1;row["declared_paths"]+=1
        row["retained_processes"]+=sum(x["retained"] for x in case["process_counts"].values())
        row["measured_processes"]+=sum(x["measured"] for x in case["process_counts"].values())
    return dict(directory=str(root.resolve()),backend=expected_backend,cases=cases,case_count=len(cases),
        families=families,
        declared_processes=len(cases)*(len(suite.get("profile_seeds",[]))+len(suite.get("heldout_seeds",[]))),
        retained_processes=len(records),manifest_records=len(manifest.get("records",[])),process_records=records,
        failures=failed,process_coverage_complete=complete,accepted=report.get("accepted") is True and complete,
        gate_counts=gate_counts(report.get("rows",[])),profile_cost_sources=sources,
        profile_within_process_cv_max=maximum(x.get("within_process_cv_max") for x in sources),
        profile_between_process_cv_max=maximum(x.get("between_process_cv") for x in sources),
        errors={**error_summary(report.get("rows",[])),"unit":"one heldout process median error"},
        memory_observations=memory,peak_tracked_reserved_bytes=maximum(x.get("peak_tracked_reserved_bytes") for x in memory),
        physical_capacity_verified=False,cuda_private_workspace_known=False,
        host_gap_is_effective=True,suitable_for_uninstrumented_runtime_prediction=False,
        scope="standalone traced multi-layer native evaluator with persistent KV, real layer outputs and selected-route expert lifecycle",
        compiler_bridge_distinction="Public facade audits shape/state/route/output contracts; exact native physical graph/cost equivalence is not claimed.",
        quality=report.get("quality"),frozen_model_sha256=manifest.get("frozen_model_sha256"))


def capacity_summary(root,source):
    root=Path(root)
    report=source.read(root/"capacity-report.json")
    raw=source.read(root/"baseline.raw.json",required=False)
    rows=report.get("rows",[])
    process_records=[]
    for path in sorted(root.glob("*.raw.json")):
        record=source.read(path)
        process_records.append(dict(path=str(path.resolve()),returncode=record.get("returncode"),
                                    timed_out=record.get("timed_out",False),
                                    expected_rejection=path.name=="one_byte_below.raw.json"))
    labels={x.get("case") for x in rows}
    raw_coverage={Path(x["path"]).name for x in process_records}=={"baseline.raw.json","one_byte_below.raw.json","exact_peak.raw.json","one_byte_above.raw.json"}
    peak=report.get("tracked_peak_bytes")
    budgets={"one_byte_below":peak-1,"exact_peak":peak,"one_byte_above":peak+1} if finite(peak) else {}
    return dict(directory=str(root.resolve()),backend=raw.get("case",{}).get("backend"),case_id=raw.get("case",{}).get("id"),
        rows=rows,tracked_peak_bytes=peak,retained_raw_processes=len(process_records),process_records=process_records,
        passed=report.get("passed") is True and raw_coverage and len(rows)==3 and labels==set(budgets)
        and all(x.get("expected_tracked_acceptance")==x.get("actual_tracked_acceptance") and x.get("budget_bytes")==budgets[x["case"]] for x in rows),
        allocation_release_checked=report.get("allocation_release_checked"),expert_reload_observed=report.get("expert_reload_observed"),
        physical_capacity_verified=False,process_peak_bytes=None,cuda_private_workspace_bytes=None,
        scope="below/exact/above explicit tracked allocator budget only; device physical capacity remains unknown")


def overlap_summary(root,source):
    root=Path(root)
    suite=source.read(root/"suite.json")
    plan=source.read(root/"plan.json",fallback=[])
    rows=source.read(root/"measurements.json",fallback=[])
    report=source.read(root/"report.json")
    manifest=source.read(root/"manifest.json")
    predictions={(x["case_id"],x["strategy"]):x for x in report.get("results",[])}
    strategies=("serial/one_copy_compute","overlap/one_copy_compute","serial/two_copy_compute","overlap/two_copy_compute")
    cases,paths,diagnostics=[],[],[]
    for case in suite.get("cases",[]):
        cid=case["id"];selected=[r for r in rows if r.get("case",{}).get("id")==cid]
        process_counts={split:dict(declared=sum(j.get("case",{}).get("id")==cid and j.get("split")==split for j in plan),
            retained=sum(r.get("split")==split for r in selected),measured=sum(r.get("split")==split and r.get("status")=="measured"for r in selected))for split in("profile","heldout")}
        cases.append(dict(case=case,processes=process_counts))
        for strategy in strategies:
            result=predictions.get((cid,strategy),{})
            sources=result.get("source_evidence",[])
            paths.append(dict(case_id=cid,strategy=strategy,predicted_ns=result.get("predicted_ns"),observed_ns=result.get("observed_ns"),
                absolute_relative_error=result.get("absolute_relative_error"),accepted=result.get("accepted")is True,gates=result.get("gates",{}),
                profile_within_process_cv_max=maximum(x.get("cv")for x in sources),source_between_process_cv=result.get("source_between_process_cv"),
                heldout_processes=result.get("heldout_processes"),heldout_within_process_cv=result.get("heldout_within_process_cv"),
                heldout_between_process_cv=result.get("heldout_between_process_cv"),
                independent_links_ideal_ns=result.get("independent_links_ideal_ns")if strategy=="overlap/two_copy_compute"else None,
                independent_links_ideal_relative_error=result.get("independent_links_ideal_relative_error")if strategy=="overlap/two_copy_compute"else None,
                unknown_reason=result.get("unknown_reason", "missing declared path"if not result else None)))
            for split in("profile","heldout"):
                observed=[r for r in selected if r.get("split")==split and r.get("status")=="measured"]
                traces=[t for r in observed for t in r.get("diagnostic_host_and_copy_events",{}).get(strategy,[])]
                totals=[v for r in observed for v in r.get("metrics",{}).get(strategy,[])]
                def median(values):return statistics.median(values)if values else None
                diagnostics.append(dict(case_id=cid,strategy=strategy,split=split,processes=len(observed),diagnostic_invocations=len(traces),
                    uninstrumented_total_median_ns=median(totals),instrumented_total_median_ns=median([t["total_ns"]for t in traces]),
                    host_compute_interval_median_ns=median([t["compute_end_ns"]-t["compute_begin_ns"]for t in traces]),
                    host_join_interval_median_ns=median([t["join_end_ns"]-t["join_begin_ns"]for t in traces]),
                    cuda_copy_a_event_elapsed_median_ns=median([t["copy_a_event_elapsed_ns"]for t in traces]),
                    cuda_copy_b_event_elapsed_median_ns=median([t["copy_b_event_elapsed_ns"]for t in traces if t.get("copy_b_event_elapsed_ns",-1)>=0]),
                    instrumentation_overhead_medians_ns={name:median([v for r in observed for v in r.get("instrumentation_overhead_ns",{}).get(name,[])])for name in("clock_pair","empty_cuda_event_pair_completed")}))
    planned={j["id"]for j in plan};identities=Counter(r.get("id")for r in rows)
    failures=[dict(id=r.get("id"),error=r.get("error"),raw_path=r.get("raw_path"))for r in rows if r.get("status")!="measured"]
    complete=bool(plan)and set(identities)==planned and all(n==1 for n in identities.values())and not failures
    return dict(directory=str(root.resolve()),scope="real asynchronous CE H2D to distinct next-payload storage while resident-current GGML FFN executes",
        family="compact_ffn_prefetch",cases=cases,case_count=len(cases),declared_paths=4*len(cases),retained_processes=len(rows),declared_processes=len(plan),
        process_coverage_complete=complete,missing_process_ids=sorted(planned-set(identities)),duplicate_process_ids={k:n for k,n in identities.items()if n>1},
        failures=failures,paths=paths,errors=error_summary(paths),gates=gate_counts(paths),diagnostics=diagnostics,
        all_paths_accepted=complete and bool(paths)and all(x["accepted"]for x in paths),
        common_fixed_tensor_compiler_runtime_exercised=report.get("common_fixed_tensor_compiler_runtime_exercised"),
        highlevel_ModelPlan_coverage=False,kernel_overlap_proved=False,copy_events_prove_kernel_overlap=False,physical_capacity_verified=False,
        instrumentation_overhead_subtracted=False,assets_unchanged_during_campaign=manifest.get("assets_unchanged"),
        frozen_model_sha256=manifest.get("model_frozen_before_heldout_sha256"),
        limitations=["Tests fixed sum/max laws with a shared PCIe service resource; it does not identify a general contention law.",
            "The second native stream is submitted before compute; the one-link tensor runtime queues its service after the first flow without pricing host submission overhead.",
            "Host spans and CUDA copy-event elapsed times are separate diagnostics; a CUDA kernel overlap proof still requires profiler evidence.",
            "Prefetch payload completion is checked, but a complete next decoder step or production prefetch speedup is not claimed."])


def percent(value):
    return "unknown" if not finite(value) else f"{100*value:.2f}%"


def cell(value):
    return str(value).replace("|","\\|").replace("\n"," ")


def markdown(report):
    chain,high=report["chain"],report["highlevel"]
    lines=["# Accuracy evidence report", "", "This report preserves statistical failures and missing coverage. A small prediction error alone is not an acceptance result.", "",
        f"Input files unchanged during assembly: **{report['input_assets_unchanged']}**. Evidence read errors: **{len(report['read_errors'])}**. Full accuracy-gap closure claimed: **False**.", "",
        "The fixed latency gates are within-process CV ≤ 15%, between-process CV of process medians ≤ 20%, pooled-median relative error ≤ 20%, at least three independent processes per split, and complete numerical/process coverage. Workload report errors are separately labeled when evaluated per heldout process.", "",
        "## Scope and coverage", "", "| Evidence | Cases | Retained / declared processes | Accepted scope units |", "|---|---:|---:|---:|",
        f"| Native compact FFN / post-RoPE attention | {chain['expected_cases']} | {chain['retained_processes']} / {chain['declared_processes']} | see model table |",
        f"| Actual compiled high-level Dense strategies | {high['case_count']} | {high['profile_raw_processes']+high['heldout_raw_processes']} / {high['declared_profile_processes']+high['declared_heldout_processes'] if finite(high['declared_profile_processes']) and finite(high['declared_heldout_processes']) else 'unknown'} | {high['accepted_cases']} cases |"]
    for work in report["workloads"]:
        lines.append(f"| Standalone workload {work['backend']} | {work['case_count']} | {work['retained_processes']} / {work['declared_processes']} | {sum(x['accepted'] for x in work['cases'])} cases |")
    lines += ["", "The high-level bridge exports the actual public compiler graph, operands and accepted serial action trace, then reruns its policy with frozen measured prices. The standalone workload evaluator executes multiple native layers/rounds with real activation propagation, persistent KV and traced expert loading. Its facade audit checks public shape/state/route semantics; it does not establish exact compiled physical topology or reusable macro-cost equivalence.", "",
        "## Native chain predictions", "", "| Model | Paths | Error ≤20% | Fully gated | Median error | Maximum error |", "|---|---:|---:|---:|---:|---:|"]
    for name,row in chain["model_summaries"].items():
        e=row["errors"];lines.append(f"| {name} | {row['path_count']} | {e['within_20_percent']} / {e['known']} | {row['accepted']} | {percent(e['median'])} | {percent(e['max'])} |")
    lines += ["", "Shared and leave-one-case-out models cover the original primary GPU/hybrid paths only. Their coefficients use profile complete-path targets. Each excluded-case fold retains that case's independent component prices; this is coefficient transfer, not prediction without profiling. Derived shared-model gates include the stability of their profile complete-path calibration sources.", "",
        "| Family | Cases | Paths | Measured processes | Failures |", "|---|---:|---:|---:|---:|"]
    for family,row in chain["families"].items():lines.append(f"| {family} | {row['cases']} | {row['declared_paths']} | {row['measured_processes']} | {row['failures']} |")
    lines += ["", "| Case | Path | Independent error / pass | In-chain error / pass | Shared error / pass | Excluded-case error / pass |", "|---|---|---|---|---|---|"]
    for row in chain["paths"]:
        details=[f"{percent(row[k].get('absolute_relative_error'))} / {row[k]['accepted']}" for k in ("independent","inchain","shared","leave_one_case_out")]
        lines.append("| "+" | ".join([cell(row["case_id"]),cell(row["strategy"]),*details])+" |")
    lines += ["", "### Source and heldout gates", "", "| Model | Gate | Pass | Fail | Unknown |", "|---|---|---:|---:|---:|"]
    for mode,data in chain["model_summaries"].items():
        for gate,counts in data["gates"].items():lines.append(f"| {mode} | {gate} | {counts['passed']} | {counts['failed']} | {counts['unknown']} |")
    lines += ["", "In-chain stage/gap prices include timestamp instrumentation. Diagnostics are separate invocations from the uninstrumented totals. Clock-pair, no-op stage/trace and empty-worker costs are measured separately and never subtracted. Host branch skew, join wait and bracket overlap cannot identify physical context-switch latency or prove a CUDA kernel timeline. Tiny variable gaps remain subject to the same source CV gates.", "",
        "### Strategy comparisons near crossings", ""]
    crossing=chain.get("near_crossing_selection",{})
    lines.append(f"Pairs are selected using profile predictions only, with ratio ≤ {crossing.get('threshold_profile_prediction_ratio','unknown')}. Retained selected pairs: {len(crossing.get('pairs',[]))}. Overlapping observed intervals are inconclusive.")
    lines += ["", "| Case | Pair | Predicted ratio | Correct median order | Disjoint observed intervals | Both latency gates pass |", "|---|---|---:|---|---|---|"]
    for p in crossing.get("pairs",[]):lines.append(f"| {cell(p.get('case_id'))} | {cell(p.get('left'))} / {cell(p.get('right'))} | {p.get('profile_prediction_ratio',0):.3f} | {p.get('median_order_correct')} | {p.get('observed_95pct_intervals_disjoint')} | {p.get('latency_quality_accepted')} |")
    lines += ["", "## Actual high-level compiler bridge", "", "| Case | Profile / heldout recorded processes | Pooled error | Profile source CV gate | Heldout within / between CV gates | Saved acceptance |", "|---|---:|---:|---|---|---|"]
    for row in high["cases"]:
        g=row["group"];lines.append(f"| {cell(row['case_id'])} | {row['recorded_profile_processes']} / {row['recorded_heldout_processes']} | {percent(row['absolute_relative_error'])} | {row['source_quality_passed']} | {g.get('heldout_within_process_cv_passed')} / {g.get('heldout_between_process_median_cv_passed')} | {row['accepted']} |")
    lines += ["", "This bridge supports one complete F32 Dense invocation, optional AttentionStages/FFNPartition and explicit serial launches. Native tensor backing is preallocated, eviction is logical liveness, and complete native memory accuracy is not claimed. The assembler records saved verdicts and does not invoke standalone protocol verifiers; acceptance requires a separately successful verifier receipt.", "",
              "### High-level strategy ranking at common completion boundaries", "",
              "These comparisons require matching configuration, initial/final locations, logical weight contents, hidden/position/KV inputs and numerical seed/process cohorts. Immutable FFN partitions are reconstructed into the same logical matrices before comparison. Every declared strategy and every pair is retained. Intervals resample independent process medians with a fixed seed; overlapping intervals are inconclusive. Ordering and regret remain descriptive when latency quality gates fail.", "",
              "| Geometry | Strategies | Common-boundary comparison | Predicted / observed best | Regret | Winner intervals separated | All latency quality gates |", "|---|---:|---|---|---:|---|---|"]
    comparisons=high["common_boundary_strategy_comparisons"]
    for ranking in comparisons["rankings"]:
        shape=ranking["declared_geometry"];label=f"H{shape['hidden']}/I{shape['intermediate']} q{shape['tokens']} history{shape['history']}"
        if not ranking["known"]:
            lines.append(f"| {label} | {len(ranking['declared_strategies'])} | unknown/incomparable: {cell('; '.join(ranking['reasons']))} | unknown | unknown | unknown | False |")
        else:
            lines.append(f"| {label} | {len(ranking['declared_strategies'])} | {cell(ranking['inference'])} | {cell(ranking['predicted_best'])} / {cell(ranking['observed_best'])} | {percent(ranking['selection_regret_relative'])} | {ranking['observed_winner_interval_separated']} | {ranking['latency_quality_accepted']} |")
    lines += ["", "All declared pair orders, profile prediction ratios, process-median intervals, overlap flags and regret in nanoseconds are preserved in report.json. Unknown or failed paths prevent selecting a winner from a surviving subset.", "", "## Multi-layer workload and memory", ""]
    for work in report["workloads"]:
        lines += [f"### {work['backend']}", "", f"Complete process coverage: **{work['process_coverage_complete']}**. All reported workload gates pass: **{work['accepted']}**. Peak tracked reserved bytes: **{work['peak_tracked_reserved_bytes']}**.", "",
            "| Case | Layers / rounds | Profile / heldout processes | Max per-process error | Accepted |", "|---|---:|---:|---:|---|"]
        for row in work["cases"]:
            counts=row["process_counts"];lines.append(f"| {cell(row['case_id'])} | {row['layers']} / {row['rounds']} | {counts['profile']['measured']} / {counts['heldout']['measured']} | {percent(row['per_process_error_summary']['max'])} | {row['accepted']} |")
    lines += ["", "The effective workload host gap includes graph construction, allocation, materialization, copies and weight lifecycle. Scalar reference generation, validation and output serialization are excluded from measured inference intervals. The workload model is not an uninstrumented production runtime predictor.", "",
        "Tracked memory covers declared backend buffers, graph metadata and explicit CPU work. Host/control/reference storage and private CUDA scratch are outside that accounting. Device-wide CUDA free-memory snapshots are diagnostics, not process peaks. Physical capacity and complete VRAM peak remain unverified.", "",
        "| Capacity evidence | Tracked peak bytes | Below / exact / above checks | Release checked | Expert reload observed |", "|---|---:|---|---|---|"]
    for row in report["capacity"]:lines.append(f"| {cell(row['directory'])} | {row['tracked_peak_bytes']} | {row['passed']} | {row['allocation_release_checked']} | {row['expert_reload_observed']} |")
    if not report["capacity"]:lines.append("| Not supplied | unknown | unknown | unknown | unknown |")
    lines += ["", "## Failures and evidence limits", "", f"Chain failed rows: {len(chain['failures'])}; missing rows: {len(chain['missing_process_ids'])}; duplicated IDs: {len(chain['duplicate_process_ids'])}. High-level failures: {len(high['failures'])}."]
    for work in report["workloads"]:lines.append(f"Workload {work['backend']} failed raw processes: {len(work['failures'])}; coverage complete: {work['process_coverage_complete']}.")
    for error in report["read_errors"]:lines.append(f"- Evidence read error: `{cell(error['path'])}` — {cell(error['error'])}")
    lines += ["", "No historical measurements or source artifacts were replaced. The assembler does not fit models, run hardware, remove samples, or rerun protocol verifiers. All source verdicts and failed gates remain visible in report.json, together with per-case/process counts and complete before/after hashes.", "",
        "## Input provenance", "", f"Files hashed before assembly: {len(report['input_hashes_before']['files'])}. Files hashed after assembly: {len(report['input_hashes_after']['files'])}.", "", "| Input | Directory/file |", "|---|---|"]
    for name,path in report["input_locations"].items():lines.append(f"| {cell(name)} | {cell(path)} |")
    if report.get("regressions") is not None:
        lines += ["", "## Regression evidence", "", "The supplied regression document is preserved verbatim as structured data in report.json; these are software checks, not hardware accuracy measurements.", "", "```json", json.dumps(report["regressions"],indent=2,ensure_ascii=False), "```"]
    if report.get("overlap")is not None:
        overlap=report["overlap"]
        lines += ["", "## Real CE prefetch / compute overlap and shared link", "",
            f"Cases: **{overlap['case_count']}**. Native processes: **{overlap['retained_processes']} / {overlap['declared_processes']}**. Complete process coverage: **{overlap['process_coverage_complete']}**. Fully accepted paths: **{sum(p['accepted']for p in overlap['paths'])} / {overlap['declared_paths']}**.", "",
            "Current F32 FFN weights stay resident while real asynchronous H2D copies target distinct next-payload allocations. Numerical output, exact copied bytes and current input immutability are checked outside timers. The fixed tensor IR/compiler/runtime evaluates serial A+compute, parallel max(A,compute), serial A+B+compute and shared-link max(A+B,compute). The two-flow independent-link ideal max(A,B,compute) is retained only as an alternative diagnostic.", "",
            "| Case | Path | Shared-resource prediction error | Independent-link ideal error (two-flow overlap only) | Profile max CV | Heldout max CV | Accepted |", "|---|---|---:|---:|---:|---:|---|"]
        for p in overlap["paths"]:lines.append(f"| {cell(p['case_id'])} | {cell(p['strategy'])} | {percent(p['absolute_relative_error'])} | {percent(p['independent_links_ideal_relative_error'])} | {percent(p['profile_within_process_cv_max'])} | {percent(maximum(p.get('heldout_within_process_cv')or[]))} | {p['accepted']} |")
        lines += ["", "This adds actual copy/compute concurrency attempts and shared-link timing evidence. It is fixed tensor-runtime coverage, not public high-level compiler coverage. Native host submission and the abstract one-link queue differ explicitly. Separate host intervals and CUDA copy events do not prove kernel overlap; instrumentation is never subtracted, and private CUDA workspace/capacity remains unknown."]
    return "\n".join(lines)+"\n"


def assemble(*,chain,highlevel,workload_cpu,workload_cuda,capacity=(),regressions=None,overlap=None,output):
    output=Path(output).resolve()
    locations=dict(chain=str(Path(chain).resolve()),highlevel=str(Path(highlevel).resolve()),
                   workload_cpu=str(Path(workload_cpu).resolve()),workload_cuda=str(Path(workload_cuda).resolve()))
    for index,path in enumerate(capacity):locations[f"capacity_{index}"]=str(Path(path).resolve())
    if regressions is not None:locations["regressions"]=str(Path(regressions).resolve())
    if overlap is not None:locations["overlap"]=str(Path(overlap).resolve())
    if output.exists():raise ValueError("output must be a new directory; retained reports are never replaced")
    for path in locations.values():
        root=Path(path)
        if root.is_dir() and output.is_relative_to(root):raise ValueError("output must lie outside all source input directories")
    inputs=[*locations.values(),Path(__file__).resolve(),Path(__file__).with_name("highlevel_rankings.py").resolve()]
    before=snapshot(inputs)
    reader=Inputs()
    report=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),quality=QUALITY,
        all_accuracy_gaps_closed=False,hardware_executed_by_assembler=False,protocol_verifiers_executed_by_assembler=False,
        chain=chain_summary(chain,reader),highlevel=highlevel_summary(highlevel,reader),
        workloads=[workload_summary(workload_cpu,reader,"CPU"),workload_summary(workload_cuda,reader,"CUDA")],
        capacity=[capacity_summary(p,reader) for p in capacity],
        overlap=overlap_summary(overlap,reader)if overlap is not None else None,
        regressions=reader.read(regressions) if regressions is not None else None,
        input_locations=locations,input_hashes_before=before)
    report["read_errors"]=reader.errors
    after=snapshot(inputs)
    report.update(input_hashes_after=after,input_assets_unchanged=before==after,
        evidence_assembly_complete=not reader.errors and not before["missing_paths"] and before==after)
    report["all_reported_latency_scopes_accepted"]=(report["evidence_assembly_complete"]
        and report["chain"]["process_coverage_complete"] and report["chain"]["baseline_all_latency_gates_passed"]
        and report["chain"]["inchain_all_latency_gates_passed"] and report["highlevel"]["all_cases_accepted"]
        and all(x["path_count"]>0 and x["accepted"]==x["path_count"] for x in report["chain"]["model_summaries"].values())
        and all(x["accepted"] for x in report["workloads"]))
    if report["overlap"]is not None:report["all_reported_latency_scopes_accepted"]=report["all_reported_latency_scopes_accepted"]and report["overlap"]["all_paths_accepted"]
    output.mkdir(parents=True)
    (output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    (output/"REPORT.md").write_text(markdown(report),encoding="utf-8",newline="\n")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("chain","highlevel","workload-cpu","workload-cuda","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--capacity",type=Path,action="append",default=[])
    parser.add_argument("--regressions",type=Path)
    parser.add_argument("--overlap",type=Path)
    args=parser.parse_args(argv)
    report=assemble(**vars(args))
    print(json.dumps(dict(output=str(args.output.resolve()),input_assets_unchanged=report["input_assets_unchanged"],
        evidence_assembly_complete=report["evidence_assembly_complete"],all_accuracy_gaps_closed=False,
        all_reported_latency_scopes_accepted=report["all_reported_latency_scopes_accepted"])))
    return 0 if report["evidence_assembly_complete"] else 2


if __name__=="__main__":raise SystemExit(main())
