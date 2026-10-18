"""Predeclared shared-coefficient and whole-geometry holdout experiments.

The existing three-feature implementation is reused unchanged. New chain
measurements are single completed invocations: the explicit normalization view
uses batch_repetitions=1, with raw ns unchanged. This is a new calibration, not a
claim that older batched coefficients have the same protocol applicability.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("accuracy_shared_shape_model", ROOT / "composition_calibration/model.py")
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
SCHEMA = "predeclared-shared-shape-models-0.1"


def freeze(profile_records):
    rows = list(profile_records)
    if not rows or any(r.get("split") != "profile" for r in rows):
        raise ValueError("profile-only shape calibration required")
    expected_cases = {}
    adapted = []
    for row in rows:
        case = row["case"]
        cid = case["id"]
        if cid in expected_cases and expected_cases[cid] != case:
            raise ValueError("conflicting exact geometry")
        expected_cases[cid] = deepcopy(case)
        if row.get("status") != "measured":
            continue
        item = {k: deepcopy(row[k]) for k in ("id", "split", "status", "case", "raw_sha256", "metrics")}
        for key in ("seed", "repeat", "raw_path"):
            if key in row:item[key]=row[key]
        item["batch_repetitions"] = 1
        adapted.append(item)
    full = shared.fit(adapted) if adapted else None
    folds = []
    for cid in sorted(expected_cases):
        train = [r for r in adapted if r["case"]["id"] != cid]
        folds.append(dict(excluded_case_id=cid, training_row_ids=[r["id"] for r in train],
            model=shared.fit(train) if train else None,
            excluded_shape_joint_targets_used_for_coefficient_fit=False,
            excluded_shape_independent_component_prices_retained=True))
    return dict(schema=SCHEMA,full=full,folds=folds,case_geometry=expected_cases,
        failed_profile_rows=[r["id"] for r in rows if r.get("status") != "measured"],
        profile_rows_used=[r["id"] for r in adapted],heldout_used_for_fitting=False,
        profile_joint_totals_used_for_calibration=True,
        normalization_view="new unbatched completed-invocation ns as batch_repetitions=1; raw sample integers unchanged",
        implementation_source=str(Path(shared.__file__).resolve()),
        evaluation_scope="same native protocol; exact independent component profiles at every evaluated geometry; shared coefficients tested on entirely excluded geometry")


def predict(bundle,case_id,strategy,*,leave_case_out=False):
    if bundle.get("schema")!=SCHEMA or bundle.get("heldout_used_for_fitting") is not False:
        raise ValueError("unknown or contaminated shape model bundle")
    full=bundle.get("full")
    if full is None:raise ValueError("no successful profile process")
    case=bundle["case_geometry"][case_id]
    prices=full["profile_component_prices_us"].get(case_id)
    if prices is None:raise ValueError("missing exact shape independent component prices")
    if leave_case_out:
        found=[fold for fold in bundle["folds"] if fold["excluded_case_id"]==case_id]
        if len(found)!=1 or found[0]["model"] is None:raise ValueError("no valid grouped training fold")
        model=found[0]["model"]
    else:model=full
    return shared.predict(case,strategy,prices,model)


def evaluate(bundle,rows,*,frozen_before_heldout):
    """Report fresh heldout errors, grouped transfer and target-source stability.

    Original independent component CV/coverage gates remain in the campaign's
    main report. This additional diagnostic does not override those gates.
    """
    profile=[r for r in rows if r["split"]=="profile" and r.get("status")=="measured"]
    heldout=[r for r in rows if r["split"]=="heldout" and r.get("status")=="measured"]
    sources=[]
    for row in profile:
        for strategy in shared.primary_strategies(row["case"]):
            values=row["metrics"][strategy]
            mean=statistics.mean(values)
            sources.append(dict(row_id=row["id"],case_id=row["case"]["id"],strategy=strategy,
                median_ns=statistics.median(values),within_process_cv=statistics.pstdev(values)/mean))
    source_groups={}
    for row in sources:source_groups.setdefault((row["case_id"],row["strategy"]),[]).append(row["median_ns"])
    between=[dict(case_id=key[0],strategy=key[1],processes=len(v),
        between_process_cv=statistics.pstdev(v)/statistics.mean(v) if len(v)>=2 else None)
        for key,v in sorted(source_groups.items())]
    output=[]
    for cid,case in sorted(bundle["case_geometry"].items()):
        selected=[r for r in heldout if r["case"]["id"]==cid]
        for strategy in shared.primary_strategies(case):
            values=[x for row in selected for x in row["metrics"].get(strategy,[])]
            actual=statistics.median(values) if values else None
            record=dict(case_id=cid,strategy=strategy,observed_ns=actual,heldout_processes=len(selected))
            for name,fold in (("full",False),("leave_one_case_out",True)):
                try:
                    result=predict(bundle,cid,strategy,leave_case_out=fold)
                    record[name]=dict(predicted_ns=result["predicted_ns"],baseline_ns=result["baseline_ns"],
                        absolute_relative_error=abs(result["predicted_ns"]-actual)/actual if actual is not None else None,
                        baseline_relative_error=abs(result["baseline_ns"]-actual)/actual if actual is not None else None,
                        correction_contributions_us=result["correction_contributions_us"])
                except (ValueError,KeyError,TypeError) as exc:record[name]=dict(unknown_reason=str(exc))
            output.append(record)
    return dict(schema=SCHEMA,frozen_before_heldout=frozen_before_heldout,
        heldout_used_for_fitting=False,profile_joint_totals_used_for_calibration=True,
        results=output,profile_target_sources=sources,profile_target_between_process=between,
        original_quality_gates_preserved_in_main_campaign_report=True,
        physical_capacity_validated=False,
        note="Grouped prediction excludes all complete-chain targets for one geometry, but retains its independent component measurements. This is coefficient transfer, not prediction without profiling.")
