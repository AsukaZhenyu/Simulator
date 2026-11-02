"""Recompute existing matched evidence as independent-component predictions."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MATCHED = HERE.parent / "matched_mechanisms"
sys.path[:0] = [str(HERE), str(MATCHED)]
import analysis as a
import experiment as e
from verify import verify


def analyze(experiment, output):
    experiment = Path(experiment).resolve()
    verification = verify(experiment)
    suite = e.document(experiment / "suite.json")
    original = e.document(experiment / "measurements.json")
    manifest = e.document(experiment / "manifest.json")
    original_model = e.document(experiment / "prediction-model.json")
    recipes, comparisons = {}, {}
    for case in suite["cases"]:
        cid = case["id"]
        devices = ("cpu", "gpu") if case["backend"] == "hybrid" else (case["backend"],)
        row_metrics = {device + "/serial" for device in devices}
        if case["backend"] == "hybrid":
            row_metrics |= {"hybrid/gpu_weights", "hybrid/cpu_rows", "hybrid/split_cpu_to_gpu"}
        recipes[cid] = {}
        comparisons[cid] = {}
        for metric in sorted(row_metrics):
            parts = e.components(metric, case["family"])
            if not parts:
                continue
            recipes[cid][metric] = {"sum": parts}
            # These old paths have distinct initial input/weight and final
            # output preparation contracts. Only their own execution boundaries
            # are calibrated; no cross-path fairness is established here.
            comparisons[cid][metric] = "distinct_initial_and_final_contract:" + metric
    recipes = {cid: value for cid, value in recipes.items() if value}
    rows = copy.deepcopy(original)
    for row in rows:
        row["raw_path"] = str(experiment / "raw" / (row["id"] + ".json"))
    model = a.freeze_model(rows, recipes)
    # Prove every component came from the original before-heldout artifact.
    original_lookup = {(entry["case_id"], entry["metric"]): entry for entry in original_model["entries"]}
    for case in model["cases"]:
        for metric, price in case["prices"].items():
            entry = original_lookup[case["case_id"], metric]
            if price["median_ns"] != entry["median_ns"] or {s["row_id"] for s in price["sources"]} != set(entry["profile_rows"]):
                raise ValueError("component differs from the original frozen profile model")
    frozen = a.sha(experiment / "prediction-model.json") == manifest["model_frozen_before_heldout_sha256"]
    report = a.evaluate(rows, model, suite["quality"], suite["bootstrap"],
                        scope="旧 matched native 的真实串行/混合 FFN 链，以及 Attention core+output 串行链；不等同于新增分片变换的校准。",
                        frozen_before_heldout=frozen, comparison_groups=comparisons)
    report["source_experiment"] = str(experiment)
    report["source_verification"] = verification
    report["source_frozen_model_sha256"] = manifest["model_frozen_before_heldout_sha256"]
    report["ranking_boundary_note"] = "Old paths have distinct initial tensor/weight residency and output/preparation contracts; cross-path strategy ranking is disabled. Only the new transform probe has matched resident input/output boundaries."
    report["unknown_or_uncovered"].extend([
        "旧 Attention probe 只测 attention_core 和 attention_output，未拆 QK/softmax/PV",
        "旧 hybrid split 是 gate/up/activation 在 CPU、down 在 GPU，未执行 FFN 中间维分片并行与求和",
        "旧路径的初始权重/输入驻留与输出交付合同不同；禁止据此作跨路径公平策略优劣排序",
        "传输 microbench 是独立诊断；没有据此宣称任意预载与计算重叠已校准"])
    a.write_outputs(output, report, model, rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.experiment, args.output)
        print(e.canonical({"output": str(args.output), "strategies": len(report["results"]),
                           "accepted": sum(r["calibrated"] for r in report["results"])}))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print("matched component analysis: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
