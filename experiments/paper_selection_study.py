"""Offline, frozen-data selection study. No profiling, fitting or runtime imports.

Prepare freezes an explicit input inventory; run fails on missing/changed evidence.
Historical failed quality gates are retained, not replaced by descriptive rankings.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAIN = Path("profiling/chain_calibration/results/campaign-v1")
OVERLAP = Path("profiling/overlap_accuracy/results/formal-v1")
STEADY = Path("profiling/steady_costs/results/steady-v1-f32")
HIGH = Path("profiling/highlevel_accuracy/runs/representative-v3")
QUALITY = {"max_within_process_cv": .15, "max_between_process_cv": .20,
           "max_heldout_relative_error": .20}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8", newline="\n")


def percentile(values, q):
    """Nearest-rank quantile; never interprets repeated calls as independent trials."""
    require(bool(values), "empty statistic")
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def metrics(values):
    return {"n": len(values), "median": st.median(values),
            "p90": percentile(values, .9), "worst": max(values)} if values else None


def cv(values):
    return st.pstdev(values) / st.mean(values)


def close(a, b):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), f"value mismatch: {a} vs {b}")


def ratio(value):
    return value["numerator"] / value["denominator"]


def expression_cost(expr, prices):
    if isinstance(expr, str):
        require(expr in prices, f"missing frozen price {expr}")
        return prices[expr]["median_ns"]
    require(isinstance(expr, dict) and len(expr) == 1, "invalid cost expression")
    op, terms = next(iter(expr.items()))
    require(op in ("sum", "max") and bool(terms), "unsupported cost expression")
    numbers = [expression_cost(term, prices) for term in terms]
    return sum(numbers) if op == "sum" else max(numbers)


def expression_leaves(expr):
    if isinstance(expr, str):
        return {expr}
    return set().union(*(expression_leaves(x) for xs in expr.values() for x in xs))


def check_statistical_gates(row, source_vectors, heldout_vectors):
    within = all(cv(v) <= .15 for v in source_vectors + heldout_vectors)
    # source_vectors are grouped as three process vectors per component.
    between = all(cv([st.median(v) for v in source_vectors[i:i+3]]) <= .20
                  for i in range(0, len(source_vectors), 3))
    between = between and cv([st.median(v) for v in heldout_vectors]) <= .20
    error = abs(row['predicted_ns'] / row['observed_ns'] - 1) <= .20
    for key, value in [('within_process_cv', within), ('between_process_cv', between),
                       ('heldout_relative_error', error)]:
        require(row['gates'][key] == value, 'historical statistical gate mismatch: ' + key)


def inventory(root):
    paths = [CHAIN / n for n in ("analysis/report.json", "prediction-model.json", "plan.json", "suite.json")]
    paths += [OVERLAP / n for n in ("report.json", "frozen-model.json", "plan.json", "suite.json")]
    paths += [STEADY / n for n in ("evaluation-v2.json", "profile-frozen.json", "manifest.json",
                                    "frozen-barrier.json", "heldout-executions.json", "profile-executions.json")]
    paths += [HIGH / "report.json", Path("docs/data/steady-costs-summary-20261022.json"),
              Path("docs/data/unified-service-response-20261022.json")]
    for directory in (CHAIN, OVERLAP):
        plan = read(root / directory / "plan.json")
        paths += [directory / "raw" / (job["id"] + ".json") for job in plan]
    # Bind all steady native evidence referenced by the immutable execution ledgers.
    for name in ("heldout-executions.json", "profile-executions.json"):
        for execution in read(root / STEADY / name):
            for receipt in (execution["native"], execution["program"]["program"],
                            execution["program"]["oracle"], execution["request"]):
                paths.append(Path(receipt["path"]).resolve().relative_to(root.resolve()))
    return sorted(set(paths), key=str)


def prepare(root, output):
    require(not output.exists(), "output exists: use a fresh experiment directory")
    inputs = [{"path": p.as_posix(), "sha256": sha(root / p)} for p in inventory(root)]
    protocol = {
        "schema": "paper-selection-study-protocol-1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_kind": "retrospective postprocessing of already inspected frozen evidence; not prospective preregistration",
        "script_sha256": sha(__file__), "inputs": inputs,
        "expected": {"chain_cases": 23, "chain_results": 143, "chain_resident_groups": 23,
                     "overlap_cases": 18, "overlap_results": 72, "overlap_groups": 36,
                     "steady_cases": 11, "steady_layers": 3, "highlevel_cases": 14},
        "prediction": "historical independent-component baseline only; no best-of-heldout predictor selection",
        "selection": "argmin frozen prediction; ties break lexicographically; all planned candidates retained",
        "regret": "(selected pooled-heldout-median / minimum pooled-heldout-median)-1",
        "uncertainty": "exact empirical paired process-cluster bootstrap: all 3^3 draws; whole call vectors retained; 95% nearest-rank interval",
        "near_tie": "selected-minus-observed-winner cluster-bootstrap interval contains zero; descriptive inconclusive, not equality",
        "qualification": "all candidates must retain every original quality gate; no survivor-only reranking",
        "matching": "within the same native process, same case/seed/F32/input/output boundary; exclude preload from resident comparisons; one-copy and two-copy pairs separate",
        "new_native_measurements": False, "refitting": False, "quality_thresholds_changed": False,
        "latest_whole_layer_selection": "not_evaluable: only one strategy for each distinct shape",
    }
    write(output / "protocol.json", protocol)
    (output / 'study-source.py').write_bytes(Path(__file__).read_bytes())
    return protocol


def verify_inventory(root, protocol):
    require(sha(__file__) == protocol["script_sha256"], "study source changed since preparation")
    for item in protocol["inputs"]:
        require(sha(root / item["path"]) == item["sha256"], f"changed evidence: {item['path']}")


def load_campaign(root, directory, model_name, expected_cases, expected_processes):
    plan = read(root / directory / "plan.json")
    require(len(plan) == expected_processes and len({p['id'] for p in plan}) == len(plan), "plan missing or duplicate jobs")
    require(len({p['case']['id'] for p in plan}) == expected_cases, "case denominator changed")
    frozen_hash = sha(root / directory / model_name)
    data = defaultdict(lambda: defaultdict(list))
    identities = set()
    for job in plan:
        rawpath = root / directory / "raw" / (job["id"] + ".json")
        raw = read(rawpath)
        require(raw["returncode"] == 0, f"failed native job {job['id']}")
        native = json.loads(raw["stdout"])
        require(native["all_checks_passed"] is True and all(c["passed"] for c in native["checks"].values()), "native numerical check failed")
        require(native["seed"] == job["seed"], "seed mismatch")
        for key in ("hidden", "intermediate", "tokens", "history", "heads", "kv_heads", "head_dim", "cpu_intermediate", "payload_bytes"):
            if key in job["case"]:
                require(native[key] == job["case"][key], "geometry mismatch: " + key)
        precision = 'GGML_PREC_F32_explicit' if directory == CHAIN else 'GGML_PREC_F32'
        require(native['cuda_graphs'] == 'disabled' and native['matmul_precision'] == precision, "precision/runtime boundary mismatch")
        identity = (raw["child_pid"], raw["started_utc"])
        require(identity not in identities, "reused process launch identity")
        identities.add(identity)
        if job["split"] == "heldout":
            require(raw["frozen_model_sha256_at_start"] == frozen_hash, "heldout does not bind the frozen predictor")
        for values in native["wall_ns"].values():
            require(len(values) == native["samples"] and all(isinstance(n, int) and n > 0 for n in values), "lost or invalid observations")
        data[job['case']['id']][job['split']].append({"job": job, "raw": raw, "native": native, "sha256": sha(rawpath)})
    for case, splits in data.items():
        require(set(splits) == {'profile', 'heldout'}, "missing split")
        for split, records in splits.items():
            require(len(records) == 3 and {r['job']['repeat'] for r in records} == {0, 1, 2}, "missing independent process")
            records.sort(key=lambda r: r['job']['repeat'])
        require({r['job']['seed'] for r in splits['profile']}.isdisjoint({r['job']['seed'] for r in splits['heldout']}), "heldout seed leakage")
        require(max(r['raw']['completed_utc'] for r in splits['profile']) < min(r['raw']['started_utc'] for r in splits['heldout']), "profile/heldout overlap")
    return data


def selection(group_id, rows, processes):
    require(len(rows) >= 2, "selection needs two complete candidates")
    names = sorted(r['strategy'] for r in rows)
    require(len(set(names)) == len(names), "duplicate strategy")
    require(set(names) == set(processes), "missing planned candidate")
    counts = {len(v) for v in processes.values()}
    require(counts == {3}, "exactly three real process clusters required")
    require(all(len({len(v) for v in processes[n]}) == 1 for n in names), "unequal retained counts")
    predictions = {r['strategy']: r['predicted_ns'] for r in rows}
    observed = {n: st.median(list(itertools.chain.from_iterable(processes[n]))) for n in names}
    chosen = min(names, key=lambda n: (predictions[n], n))
    winner = min(names, key=lambda n: (observed[n], n))
    regret = observed[chosen] / observed[winner] - 1
    draws, differences = [], []
    alternative_differences = {n: [] for n in names if n != winner}
    for draw in itertools.product(range(3), repeat=3):
        totals = {n: st.median([x for index in draw for x in processes[n][index]]) for n in names}
        draws.append(totals[chosen] / min(totals.values()) - 1)
        differences.append(totals[chosen] - totals[winner])
        for alternative in alternative_differences:
            alternative_differences[alternative].append(totals[alternative] - totals[winner])
    interval = [percentile(differences, .025), percentile(differences, .975)]
    alternative_intervals = {n: [percentile(v, .025), percentile(v, .975)] for n,v in alternative_differences.items()}
    winner_separated = all(interval[0] > 0 for interval in alternative_intervals.values())
    all_gates = all(all(r['gates'].values()) for r in rows)
    return {"group_id": group_id, "strategies": names, "predicted_ns": predictions, "observed_median_ns": observed,
            "selected_strategy": chosen, "observed_winner": winner, "exact_winner_match": observed[chosen] == observed[winner],
            "selection_regret": regret, "selection_regret_ns": observed[chosen] - observed[winner],
            "cluster_bootstrap_regret_95pct": [percentile(draws, .025), percentile(draws, .975)],
            "selected_minus_winner_ns_95pct": interval,
            "selected_vs_winner_inconclusive": interval[0] <= 0 <= interval[1],
            "mismatched_selection_inconclusive": chosen != winner and interval[0] <= 0 <= interval[1],
            "alternatives_minus_observed_winner_ns_95pct": alternative_intervals,
            "observed_winner_separated_from_all_alternatives": winner_separated,
            "qualified_decisive_winner_match": all_gates and winner_separated and chosen == winner,
            "all_original_gates_passed": all_gates, "interpretation": "qualified bounded comparison" if all_gates else "descriptive only; historical quality gates fail",
            "process_clusters": 3, "cluster_draws": 27}


def row_metrics(rows):
    return {"results": len(rows), "pooled_median_ape": metrics([abs(r['predicted_ns'] / r['observed_ns'] - 1) for r in rows]),
            "original_all_gates_passed": sum(all(r['gates'].values()) for r in rows)}


def group_metrics(groups):
    return {"groups": len(groups), "winner_matches": sum(g['exact_winner_match'] for g in groups),
            "mismatched_selection_inconclusive": sum(g['mismatched_selection_inconclusive'] for g in groups),
            "observed_winner_separated_from_all_alternatives": sum(g['observed_winner_separated_from_all_alternatives'] for g in groups),
            "qualified_decisive_winner_matches": sum(g['qualified_decisive_winner_match'] for g in groups),
            "original_all_gates_passed": sum(g['all_original_gates_passed'] for g in groups),
            "selection_regret": metrics([g['selection_regret'] for g in groups])}


def chain_analysis(root):
    data = load_campaign(root, CHAIN, 'prediction-model.json', 23, 138)
    report = read(root / CHAIN / 'analysis/report.json')
    model = read(root / CHAIN / 'prediction-model.json')
    require(report['quality'] == QUALITY and len(report['results']) == 143, "historical chain scope/gates changed")
    require(model['trained_on'] == 'profile_only', "not a profile-only model")
    models = {m['case_id']: m for m in model['cases']}
    for case, m in models.items():
        for metric, price in m['prices'].items():
            vectors = [p['native']['wall_ns'][metric] for p in data[case]['profile']]
            close(math.ceil(st.median(list(itertools.chain.from_iterable(vectors)))), price['median_ns'])
    rows_by_case = defaultdict(list)
    for row in report['results']:
        case = row['case_id']; m = models[case]; strategy = row['strategy']
        require(strategy in m['strategies'], "unplanned chain strategy")
        close(expression_cost(m['strategies'][strategy], m['prices']), row['predicted_ns'])
        rawsets = [r['native']['wall_ns'][row['observed_metric']] for r in data[case]['heldout']]
        close(st.median(list(itertools.chain.from_iterable(rawsets))), row['observed_ns'])
        sources = [p['native']['wall_ns'][metric] for metric in sorted(expression_leaves(m['strategies'][strategy]))
                   for p in data[case]['profile']]
        check_statistical_gates(row, sources, rawsets)
        for evidence in row['heldout_evidence'] + row['source_evidence']:
            records = data[case][evidence['split']]
            r = next(r for r in records if r['job']['id'] == evidence['row_id'])
            require(r['sha256'] == evidence['raw_sha256'], "chain report raw hash mismatch")
            close(st.median(r['native']['wall_ns'][evidence['metric']]), evidence['median_ns'])
        if strategy != 'hybrid/preload':
            rows_by_case[case].append(row)
    groups = []
    for case, rows in sorted(rows_by_case.items()):
        expected = set(models[case]['strategies']) - {'hybrid/preload'}
        require({r['strategy'] for r in rows} == expected, "missing chain candidate")
        for record in data[case]['heldout']:
            boundary = record['native']['boundary']
            require(boundary['input_location'] == 'all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu' and boundary['output_location'] == 'dram', "chain placement boundary changed")
        processes = {r['strategy']: [p['native']['wall_ns'][r['observed_metric']] for p in data[case]['heldout']] for r in rows}
        groups.append(selection(case, rows, processes))
    return {"summary": row_metrics(report['results']), "selection": group_metrics(groups), "groups": groups,
            "boundary": "F32 generated inputs/weights resident in BOTH CPU/GPU memories, output DRAM; no cold transfer; case partition fixed",
            "excluded_from_selection": {"hybrid/preload": 14, "reason": "different weight placement boundary; errors retained in all 143 rows"},
            "not_current_model_accuracy": True}


def overlap_analysis(root):
    data = load_campaign(root, OVERLAP, 'frozen-model.json', 18, 108)
    report = read(root / OVERLAP / 'report.json')
    frozen = read(root / OVERLAP / 'frozen-model.json')
    require(report['quality'] == QUALITY and len(report['results']) == 72, "historical overlap scope/gates changed")
    prices = {m['case']['id']: m['prices'] for m in frozen['cost_model']['cases']}
    for case, entries in prices.items():
        for metric, price in entries.items():
            vectors = [p['native']['wall_ns'][metric] for p in data[case]['profile']]
            close(math.ceil(st.median(list(itertools.chain.from_iterable(vectors)))), price['median_ns'])
    programs = {(p['case_id'], p['strategy']): p for p in frozen['programs']}
    rows_by_case = defaultdict(list)
    for row in report['results']:
        case = row['case_id']; name = row['strategy']; p = prices[case]
        a, b, c = (p[key]['median_ns'] for key in ('copy/a', 'copy/b', 'compute/current_ffn'))
        cost = {'serial/one_copy_compute': a+c, 'overlap/one_copy_compute': max(a,c),
                'serial/two_copy_compute': a+b+c, 'overlap/two_copy_compute': max(a+b,c)}[name]
        close(cost, row['predicted_ns']); close(cost, programs[case, name]['predicted_ns'])
        values = [r['native']['wall_ns'][name] for r in data[case]['heldout']]
        close(st.median(list(itertools.chain.from_iterable(values))), row['observed_ns'])
        used = ['copy/a', 'compute/current_ffn'] + (['copy/b'] if 'two_copy' in name else [])
        check_statistical_gates(row, [p['native']['wall_ns'][metric] for metric in used for p in data[case]['profile']], values)
        for record in data[case]['heldout']:
            n = record['native']
            require(n['prefetch_payload_checks_passed'] and n['prefetch_destinations_disjoint_from_each_other_and_all_current_ports'], "prefetch boundary invalid")
        for source in row['source_evidence']:
            record = next(r for r in data[case]['profile'] if r['job']['id'] == source['id'])
            require(source['split'] == 'profile' and record['sha256'] == source['raw_sha256'], 'overlap source provenance mismatch')
        rows_by_case[case].append(row)
    groups=[]
    for case, rows in sorted(rows_by_case.items()):
        require(len(rows) == 4 and len({r['strategy'] for r in rows}) == 4, "missing overlap candidate")
        for suffix in ('one_copy_compute', 'two_copy_compute'):
            subset = [r for r in rows if r['strategy'].endswith('/'+suffix)]
            processes = {r['strategy']: [p['native']['wall_ns'][r['strategy']] for p in data[case]['heldout']] for r in subset}
            groups.append(selection(case+'/'+suffix, subset, processes))
    return {"summary": row_metrics(report['results']), "selection": group_metrics(groups), "groups": groups,
            "boundary": "same resident compact FFN + same one/two future payload copies; GPU output; all copies complete; one/two-copy work never mixed",
            "not_highlevel_modelplan_coverage": True, "not_current_model_accuracy": True}


def steady_analysis(root):
    report = read(root / STEADY / 'evaluation-v2.json')
    require(report['frozen_file_sha256'] == sha(root / STEADY / 'profile-frozen.json'), 'steady frozen hash mismatch')
    require(not report['heldout_prices_fitted'] and report['all_trials_retained'], 'steady data leakage/filtering')
    require(len(report['cases']) == 11, 'missing steady cases')
    executions = {e['job']['id']: e for e in read(root / STEADY / 'heldout-executions.json')}
    require(len(executions) == 33, 'missing steady process')
    rows=[]
    for case in report['cases']:
        require(len(case['processes']) == 3, 'missing steady launch')
        errors=[]; trialerrors=[]
        for process in case['processes']:
            execution = executions[process['job_id']]
            require(execution['numerical']['passed'] and execution['status'] == 'completed', 'failed steady execution')
            receipt=execution['native']; require(sha(receipt['path']) == receipt['sha256'], 'steady raw changed')
            require(execution['process_id'] == process['process_identity']['pid'] and execution['started_at_utc'] == process['process_identity']['launch_epoch_utc'], 'steady launch identity mismatch')
            values = [ratio(x) for x in process['observed_trial_average_ns']]
            native = read(receipt['path'])
            rawvalues = [t['wall_ns']/t['repeats'] for t in native['trials']]
            require(rawvalues == values, 'steady raw trials differ from evaluation')
            err=abs(case['prediction_ns']/st.mean(values)-1)
            close(err, process['aggregate_mean_ape']);errors.append(err)
            trialerrors += [abs(case['prediction_ns']/x-1) for x in values]
        rows.append({'case_id': case['case_id'], 'kind': case['kind'], 'prediction_ns': case['prediction_ns'],
                     'process_mean_apes': errors, 'process_mean_error': metrics(errors), 'trial_average_error': metrics(trialerrors),
                     'accuracy_gate': case['mean_and_trial_p90_gate'], 'all_gates': case['hardware_accuracy_qualified']})
    return {'cases': rows, 'process_mean_ape': metrics([e for r in rows for e in r['process_mean_apes']]),
            'accuracy_passed': sum(r['accuracy_gate'] for r in rows), 'all_gates_passed': sum(r['all_gates'] for r in rows),
            'complete_layers': [r for r in rows if r['kind']=='layer'],
            'selection_status': 'not_evaluable', 'matched_whole_layer_strategy_groups': 0,
            'reason': 'Three layer shapes each have only one strategy; CPU/GPU component cases are not a complete-layer comparison and use different case seeds.'}


def run(root, output):
    protocol=read(output/'protocol.json');verify_inventory(root,protocol)
    chain=chain_analysis(root);overlap=overlap_analysis(root);steady=steady_analysis(root)
    high=read(root/HIGH/'report.json')
    require(len(high['case_groups']) == 14 and len(high['results']) == 42, 'historical highlevel missing results')
    result={
        'schema':'paper-selection-study-result-1', 'completed_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(output/'protocol.json'), 'execution_completed':True,
        'current_whole_layer_strategy_selection_qualified':False,
        'conclusion':'Existing evidence supports bounded latency diagnostics; current whole-layer strategy selection is not yet validated.',
        'chain_oct18':chain, 'overlap_oct18':overlap, 'steady_oct22':steady,
        'historical_highlevel_oct18':{'cases':14,'processes':42,'all_cases_accepted':high['all_cases_accepted'],
            'full_acceptance_cases':sum(bool(r.get('accepted',False)) for r in high['case_groups']),
            'max_single_process_ape':high['max_absolute_relative_error'],
            'scope':'preserved report-only evidence; no selection inference from neighboring cuts without data/placement identity proof'},
        'limitations':['Only three process launches per historical group: empirical intervals are coarse and do not establish future stability.',
                       'Within-process path order was fixed; sequence drift and order effects cannot be separated.',
                       'Synthetic-input microbenchmarks do not qualify production model weights, MoE, multi-layer or dynamic resource response.',
                       'Historical predictor scope and executable versions remain distinct from current IR strategies.',
                       'No fresh timings, fitted parameters, gate changes or unstable-case removal.'],
        'next_measurement_contract':{'status':'prepared_only_not_run', 'workload':'one existing H512/q64/history0 F32 Dense shape',
            'candidates':['CPU-only','GPU-only','CPU-attention/GPU-FFN','CPU/GPU FFN split'],
            'matching':'identical full weights/input seed/precision/initial residency/final output and KV boundary across all four plans',
            'training':'freeze profile-only scoped action prices before new heldout launches',
            'heldout':'three independent processes per plan with balanced order; one disjoint numerical seed shared by all heldout plans',
            'gates':'retain the existing steady cost gates; report regret for all candidates even if gates fail',
            'caveat':'native executor support for every compiled action must be checked before launch; no claim this proposed matrix is already executable'},
    }
    verify_inventory(root,protocol)
    write(output/'results.json',result)
    write(output/'completion.json',{'execution_completed':True,'input_hashes_unchanged':True,
          'result_sha256':sha(output/'results.json'),'current_whole_layer_selection_qualified':False,
          'source_sha256':sha(__file__),'input_files':len(protocol['inputs'])})
    print(json.dumps({'output':str(output),'chain':chain['selection'],'overlap':overlap['selection'],
                      'steady':{k:v for k,v in steady.items() if k not in ('cases','complete_layers')}},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    (prepare if args.mode=='prepare' else run)(args.root.resolve(),args.output.resolve())


if __name__=='__main__':
    main()
