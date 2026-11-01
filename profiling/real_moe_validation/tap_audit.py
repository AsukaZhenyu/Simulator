"""Strict, read-only pairing of callback-free CPU-consumer MoE observations.

This checks observed computation and partition evidence, never performance
accuracy. Forced output IDs are input constraints, not numerical evidence.
Identical logits do not independently establish equality of unobserved routes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .postcollection import CONTRACT, expected_forwards

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_STAGE = re.compile(r"ffn_moe_(up|gate|down)-(\d+)\Z")
_FORWARD_KEYS = ("index", "phase", "position", "tokens", "output_rows")


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _tokens(value):
    return isinstance(value, list) and bool(value) and all(type(x) is int and x >= 0 for x in value)


def _configuration(request):
    """Mode and teacher-forced token IDs are the only allowed differences."""
    result = {k: v for k, v in request.items() if k not in ("mode", "requests")}
    result["requests"] = [
        {k: v for k, v in row.items() if k != "output_token_ids"}
        for row in request.get("requests", [])
    ]
    return result


def pair_audit(reference, capture):
    """Pure analysis of two native records; acquisition is checked by audit_files.

    Both runs must enable step-logit and graph metadata observations. Their
    elapsed times consequently are diagnostic, including the reference run.
    """
    problems = {k: [] for k in ("binding", "structural", "numerical", "partition")}
    unknown = {k: [] for k in problems}
    unsupported = []
    cases = []
    totals = Counter()

    def check(domain, condition, message):
        if not condition:
            problems[domain].append(message)

    def need(domain, condition, message):
        if not condition:
            unknown[domain].append(message)

    for label, data in (("reference", reference), ("capture", capture)):
        check("binding", data.get("schema") == "real-moe-tap-native-4", label + ": native schema differs")
        check("binding", data.get("scheduler_callback_enabled") is False, label + ": scheduler callback must be disabled")
        check("binding", data.get("trace_callback_enabled") is False, label + ": legacy trace callback must be disabled")
        check("binding", data.get("model_weights_loaded") is True, label + ": actual model execution missing")
        check("binding", data.get("observed_model_contract") == CONTRACT, label + ": OLMoE model contract differs")
        check("binding", data.get("threadpool_policy") == "explicit_persistent_owned", label + ": owned threadpool policy differs")
        check("binding", data.get("graph_or_allocator_modified") is False, label + ": graph/allocator modification is outside this tap scope")
    check("binding", reference.get("mode") in ("generate", "replay"), "reference must be generate/replay")
    check("binding", capture.get("mode") == "capture", "capture mode required")
    check("binding", reference.get("cpu_consumer_tap_enabled") is False, "reference cannot enable the consumer tap")
    check("binding", capture.get("cpu_consumer_tap_enabled") is True, "capture consumer tap missing")
    check("binding", capture.get("tap_hook_available") is True, "private consumer hook unavailable")
    for key in ("model_sha256_verified_by_parent", "source_sha256"):
        check("binding", _sha(reference.get(key)) and reference.get(key) == capture.get(key), key + " differs or is missing")
    ref_req, cap_req = reference.get("request", {}), capture.get("request", {})
    check("binding", isinstance(ref_req, dict) and isinstance(cap_req, dict), "request objects missing")
    if not isinstance(ref_req, dict):
        ref_req = {}
    if not isinstance(cap_req, dict):
        cap_req = {}
    for label, req in (("reference", ref_req), ("capture", cap_req)):
        check("binding", req.get("batch_size") == 256, label + ": only reviewed 256/256 microbatches are supported")
        check("binding", req.get("model_contract") == CONTRACT, label + ": request model contract differs")
        need("binding", req.get("observe_logits") is True and req.get("observe_graph") is True,
             label + ": full step-logit and graph observations required")
    try:
        check("binding", _configuration(ref_req) == _configuration(cap_req), "model/input/execution parameter configuration differs")
    except (TypeError, AttributeError):
        check("binding", False, "malformed request rows")
    check("binding", reference.get("actual_placement") is not None and reference.get("actual_placement") == capture.get("actual_placement"),
          "actual weight placement inventory differs or is missing")

    ref_rows, cap_rows = reference.get("results", []), capture.get("results", [])
    if not isinstance(ref_rows, list) or not isinstance(cap_rows, list):
        ref_rows, cap_rows = [], []
        check("structural", False, "native result arrays required")
    ref_ids = [r.get("id") for r in ref_rows if isinstance(r, dict)]
    cap_ids = [r.get("id") for r in cap_rows if isinstance(r, dict)]
    check("structural", bool(ref_rows) and len(ref_ids) == len(ref_rows) and len(set(ref_ids)) == len(ref_ids), "reference request inventory empty, duplicate or malformed")
    check("structural", ref_ids == cap_ids and len(cap_ids) == len(cap_rows), "paired request order/inventory differs")
    for label, req, ids in (("reference", ref_req, ref_ids), ("capture", cap_req, cap_ids)):
        requested = req.get("requests", [])
        check("structural", isinstance(requested, list) and all(isinstance(r, dict) for r in requested)
              and [r.get("id") for r in requested] == ids, label + ": request/result rows differ")

    for normal, observed in zip(ref_rows, cap_rows):
        if not isinstance(normal, dict) or not isinstance(observed, dict):
            continue
        identifier = normal.get("id")
        prefix = str(identifier) + ": "
        before = {k: len(v) for k, v in problems.items()}
        prompt, outputs = normal.get("prompt_token_ids"), normal.get("output_token_ids")
        if not _tokens(prompt) or not _tokens(outputs):
            check("structural", False, prefix + "valid actual prompt/output token streams required")
            continue
        check("binding", observed.get("prompt_token_ids") == prompt and observed.get("output_token_ids") == outputs,
              prefix + "fixed input/output streams differ")
        check("binding", observed.get("teacher_forced") is True, prefix + "capture must replay actual reference outputs")
        forwards = expected_forwards(len(prompt), len(outputs))
        for label, row in (("reference", normal), ("capture", observed)):
            actual = row.get("forward_inventory", [])
            need("structural", isinstance(actual, list) and bool(actual), prefix + label + ": forward inventory missing")
            if isinstance(actual, list):
                check("structural", [{k: f.get(k) for k in _FORWARD_KEYS} for f in actual if isinstance(f, dict)] == forwards,
                      prefix + label + ": forward/output-mask inventory differs")
            check("structural", row.get("forward_count") == len(forwards) and row.get("decode_forward_count") == len(outputs) - 1,
                  prefix + label + ": forward/decode counts differ")
            observer = row.get("observer", {})
            check("binding", isinstance(observer, dict), prefix + label + ": observer object missing")
            if isinstance(observer, dict):
                check("binding", observer.get("step_logit_copy") is True and observer.get("graph_metadata") is True,
                      prefix + label + ": observer flags differ from required evidence")
                check("binding", observer.get("cpu_consumer_tap") is (label == "capture"), prefix + label + ": tap observer flag differs")
            check("binding", row.get("timing_usable_as_bare_reference") is False, prefix + label + ": observer timing cannot be bare truth")
            check("binding", row.get("trace_timing_is_performance_truth") is False, prefix + label + ": observed timing promoted to performance truth")
        check("binding", normal.get("routes") == [], prefix + "reference must not collect hidden routes")

        sampled = [normal.get("sampled_greedy_token_ids"), observed.get("sampled_greedy_token_ids")]
        for label, values in zip(("reference", "capture"), sampled):
            need("numerical", _tokens(values) and len(values) == len(outputs), prefix + label + ": sampled greedy ledger missing/incomplete")
        totals["greedy_token_denominator"] += len(outputs)
        if all(_tokens(v) and len(v) == len(outputs) for v in sampled):
            matches = sum(a == b for a, b in zip(*sampled))
            totals["greedy_token_matches"] += matches
            check("numerical", matches == len(outputs), prefix + "sampled greedy tokens differ (forced IDs are not evidence)")
            check("numerical", all(x == y for x, y in zip(sampled[0], outputs)), prefix + "reference sampled tokens do not reproduce its output trajectory")

        logits = []
        for label, row in (("reference", normal), ("capture", observed)):
            ledger = row.get("logit_observations", [])
            mapping = {}
            need("numerical", isinstance(ledger, list) and len(ledger) == len(outputs), prefix + label + ": step-logit coverage missing/incomplete")
            if isinstance(ledger, list):
                for entry in ledger:
                    if not isinstance(entry, dict) or type(entry.get("output_index")) is not int:
                        check("numerical", False, prefix + label + ": malformed logit output index")
                        continue
                    index = entry["output_index"]
                    check("numerical", index not in mapping and 0 <= index < len(outputs), prefix + label + ": duplicate/extra logit output index")
                    mapping[index] = entry
                    check("numerical", _sha(entry.get("full_f32_logits_sha256")), prefix + label + ": full-logits hash invalid")
                    top = entry.get("top_logits")
                    valid_top = isinstance(top, list) and len(top) >= 2 and all(
                        isinstance(v, dict) and type(v.get("token")) is int and v["token"] >= 0
                        and isinstance(v.get("logit"), (int, float)) and not isinstance(v["logit"], bool)
                        and math.isfinite(v["logit"]) for v in top)
                    check("numerical", valid_top, prefix + label + ": valid observed top logits missing")
                    if valid_top:
                        check("numerical", len({v["token"] for v in top}) == len(top)
                              and all(a["logit"] >= b["logit"] for a, b in zip(top, top[1:])), prefix + label + ": top-logit ordering/IDs invalid")
                        margin = entry.get("top1_top2_margin")
                        check("numerical", isinstance(margin, (int, float)) and not isinstance(margin, bool)
                              and math.isfinite(margin) and math.isclose(margin, top[0]["logit"] - top[1]["logit"], rel_tol=1e-6, abs_tol=1e-7),
                              prefix + label + ": observed top1/top2 margin inconsistent")
                        check("numerical", entry.get("greedy_token") == top[0]["token"], prefix + label + ": top logit and greedy token differ")
                    if _tokens(sampled[0 if label == "reference" else 1]) and index < len(sampled[0 if label == "reference" else 1]):
                        check("numerical", entry.get("greedy_token") == sampled[0 if label == "reference" else 1][index], prefix + label + ": greedy/logit ledger differs")
            logits.append(mapping)
        totals["full_logits_hash_denominator"] += len(outputs)
        for index in range(len(outputs)):
            if index not in logits[0] or index not in logits[1]:
                continue
            a, b = logits[0][index], logits[1][index]
            matched = _sha(a.get("full_f32_logits_sha256")) and a["full_f32_logits_sha256"] == b.get("full_f32_logits_sha256")
            totals["full_logits_hash_matches"] += int(matched)
            check("numerical", matched, prefix + f"output {index}: full F32 logits differ")
            check("numerical", a.get("top_logits") == b.get("top_logits") and a.get("greedy_token") == b.get("greedy_token")
                  and a.get("top1_top2_margin") == b.get("top1_top2_margin"), prefix + f"output {index}: observed logit summary differs")

        graphs = []
        for label, row in (("reference", normal), ("capture", observed)):
            ledger = row.get("graph_observations", [])
            mapping = {}
            need("partition", isinstance(ledger, list) and len(ledger) == len(forwards), prefix + label + ": graph coverage missing/incomplete")
            if isinstance(ledger, list):
                for graph in ledger:
                    if not isinstance(graph, dict) or type(graph.get("forward_index")) is not int:
                        check("partition", False, prefix + label + ": graph forward index malformed")
                        continue
                    index = graph["forward_index"]
                    check("partition", index not in mapping and 0 <= index < len(forwards), prefix + label + ": duplicate/extra graph forward")
                    mapping[index] = graph
                    check("partition", _sha(graph.get("graph_partition_fingerprint_sha256")), prefix + label + ": partition fingerprint invalid")
                    for key in ("node_count", "scheduler_splits", "scheduler_copies"):
                        check("partition", type(graph.get(key)) is int and graph[key] >= (1 if key == "node_count" else 0), prefix + label + ": " + key + " invalid")
                    need("partition", isinstance(graph.get("expert_nodes"), list), prefix + label + ": actual expert-node inventory missing")
            graphs.append(mapping)
        required_stages = {}
        for forward in forwards:
            index = forward["index"]
            totals["graph_forward_denominator"] += 1
            if index not in graphs[0] or index not in graphs[1]:
                continue
            a, b = graphs[0][index], graphs[1][index]
            matched = a.get("graph_partition_fingerprint_sha256") == b.get("graph_partition_fingerprint_sha256")
            totals["graph_forward_fingerprint_matches"] += int(matched)
            check("partition", matched, prefix + f"forward {index}: graph partition fingerprint changed")
            for key in ("node_count", "scheduler_splits", "scheduler_copies", "expert_nodes"):
                check("partition", a.get(key) == b.get(key), prefix + f"forward {index}: {key} differs")
            stage_layers = {}
            for node in b.get("expert_nodes", []) if isinstance(b.get("expert_nodes"), list) else []:
                if not isinstance(node, dict):
                    check("partition", False, prefix + "malformed expert node")
                    continue
                name = node.get("name", "")
                match = _STAGE.fullmatch(name) if isinstance(name, str) else None
                if not match:
                    unsupported.append(prefix + f"forward {index}: unknown expert-stage name {name!r}")
                    continue
                stage, layer = match.group(1), int(match.group(2))
                check("structural", 0 <= layer < 16, prefix + "expert node layer outside actual model")
                stages = stage_layers.setdefault(layer, set())
                check("structural", stage not in stages, prefix + "duplicate expert graph stage")
                stages.add(stage)
                if node.get("backend") != "CPU":
                    unsupported.append(prefix + f"forward {index}/layer{layer}: expert backend {node.get('backend')!r} is not CPU")
                if any(not isinstance(node.get(k), str) or not node[k] or node[k].lower() == "unknown" for k in ("weight_buffer", "weight_type")):
                    unsupported.append(prefix + f"forward {index}/layer{layer}: expert weight buffer/type unknown")
            for layer in range(16):
                if layer < 15 or forward["output_rows"]:
                    check("structural", bool(stage_layers.get(layer)), prefix + f"forward {index}/layer{layer}: no actual CPU expert stage inventoried")
            required_stages[index] = stage_layers

        route_map = {}
        routes = observed.get("routes", [])
        need("structural", isinstance(routes, list) and bool(routes), prefix + "actual consumer routes missing")
        for route in routes if isinstance(routes, list) else []:
            if not isinstance(route, dict):
                check("structural", False, prefix + "malformed route object")
                continue
            index, layer = route.get("forward_index"), route.get("layer")
            if type(index) is not int or type(layer) is not int or not 0 <= index < len(forwards) or not 0 <= layer < 16:
                check("structural", False, prefix + "route forward/layer invalid")
                continue
            check("structural", (index, layer) not in route_map, prefix + "duplicate consumer forward/layer route")
            route_map[index, layer] = route
        for forward in forwards:
            index = forward["index"]
            for layer in range(16):
                positions = list(range(forward["position"], forward["position"] + forward["tokens"])) if layer < 15 else (
                    [forward["position"] + forward["tokens"] - 1] if forward["output_rows"] else [])
                route = route_map.get((index, layer))
                if not positions:
                    check("structural", route is None, prefix + f"forward {index}/layer15: fabricated pruned consumer route")
                    totals["pruned_final_layer_invocations"] += 1
                    continue
                totals["required_nonempty_forward_layers"] += 1
                if route is None:
                    check("structural", False, prefix + f"forward {index}/layer{layer}: missing required consumer route")
                    continue
                totals["observed_forward_layers"] += 1
                check("structural", route.get("phase") == forward["phase"] and route.get("absolute_positions") == positions,
                      prefix + f"forward {index}/layer{layer}: phase/absolute positions differ")
                check("structural", route.get("consumer_backend") == "CPU", prefix + "consumer route backend is not CPU")
                ids = route.get("expert_ids", [])
                check("structural", isinstance(ids, list) and len(ids) == len(positions) and all(
                    isinstance(v, list) and len(v) == 8 and all(type(x) is int and 0 <= x < 64 for x in v)
                    and len(set(v)) == 8 for v in ids), prefix + f"forward {index}/layer{layer}: required row/top8 expert IDs invalid")
                stages = route.get("actual_stages_observed")
                valid_stages = isinstance(stages, list) and all(s in ("up", "gate", "down") for s in stages) and len(set(stages)) == len(stages)
                check("structural", valid_stages and bool(stages), prefix + "observed expert-stage list invalid")
                actual = set(stages) if valid_stages else set()
                expected = required_stages.get(index, {}).get(layer)
                if expected is not None:
                    check("structural", actual == expected, prefix + f"forward {index}/layer{layer}: consumer stages do not match actual graph stages")
                check("structural", route.get("all_three_stages_observed") is (actual == {"up", "gate", "down"}), prefix + "all-three flag disagrees with actual stage inventory")
                check("structural", route.get("observed_stage_routes_equal") is True, prefix + "expert stages did not consume equal routes")
                if "stage_count" in route:
                    check("structural", route["stage_count"] == len(actual), prefix + "stage count differs")
        cases.append(dict(id=identifier,forward_count=len(forwards),required_layers=16,
                          problems_added={k: len(v) - before[k] for k, v in problems.items()}))

    statuses = {k: "failed" if problems[k] else "unknown" if unknown[k] else "passed" for k in problems}
    if unsupported and statuses["partition"] != "failed":
        statuses["partition"] = "unsupported"
    overall = "failed" if "failed" in statuses.values() else "unsupported" if unsupported else "unknown" if "unknown" in statuses.values() else "passed"
    return dict(schema="real-moe-tap-pair-audit-1",status=overall,pair_status=overall,paired=overall == "passed",
                **{k + "_status": v for k, v in statuses.items()},problems=problems,missing_evidence=unknown,
                unsupported_reasons=unsupported,cases=cases,counts=dict(totals),
                forced_output_ID_equality_is_numerical_evidence=False,full_F32_logits_hash_equality_required=True,
                hidden_bare_routes_proven_identical=False,
                hidden_route_limit="Matching per-step logits/greedy and partition metadata do not independently prove equality of unobserved bare expert routes.",
                private_CUDA_fusion_decisions_proven_identical=False,capture_timing_is_bare_truth=False,
                observed_reference_timing_is_bare_truth=False,whole_model_accuracy_certified=False,
                mechanism="read-only CPU consumer tap; scheduler callback must be disabled; observer overhead remains separate")


def normal_runtime_equivalence(reference, candidate):
    """Compare two real generate records from original/private CPU runtimes.

    The probe source and workload must match. DLL provenance is bound by the
    caller's separate execution receipts, because the CPU runtime intentionally
    differs. Neither record is rewritten into capture mode or used as a timing
    predictor; both contain the same diagnostic logit/graph observers.
    """
    failures = []
    counts = Counter()

    def check(condition, message):
        if not condition:
            failures.append(message)

    for label, data in (("reference", reference), ("candidate", candidate)):
        check(data.get("schema") == "real-moe-tap-native-4", label + ": native schema differs")
        check(data.get("mode") == "generate" and data.get("request", {}).get("mode") == "generate", label + ": actual generate mode required")
        check(data.get("cpu_consumer_tap_enabled") is False, label + ": consumer tap must be disabled")
        check(data.get("scheduler_callback_enabled") is False and data.get("trace_callback_enabled") is False, label + ": scheduler/trace callbacks must be disabled")
        check(data.get("model_weights_loaded") is True and data.get("observed_model_contract") == CONTRACT, label + ": actual OLMoE execution required")
        check(data.get("threadpool_policy") == "explicit_persistent_owned" and data.get("graph_or_allocator_modified") is False, label + ": execution policy differs")
        request = data.get("request", {})
        check(request.get("observe_logits") is True and request.get("observe_graph") is True, label + ": complete diagnostic observations required")
        check(request.get("batch_size") == 256 and request.get("model_contract") == CONTRACT, label + ": reviewed model/microbatch contract differs")
        check(all("output_token_ids" not in row for row in request.get("requests", [])), label + ": generate request must not force output IDs")
    for key in ("source_sha256", "model_sha256_verified_by_parent"):
        check(_sha(reference.get(key)) and reference.get(key) == candidate.get(key), key + " differs or is missing")
    check(_configuration(reference.get("request", {})) == _configuration(candidate.get("request", {})), "model/input/execution configuration differs")
    check(reference.get("actual_placement") is not None and reference.get("actual_placement") == candidate.get("actual_placement"), "actual weight placement differs or is missing")
    rows = [reference.get("results", []), candidate.get("results", [])]
    ids = [[row.get("id") for row in group] for group in rows]
    check(bool(ids[0]) and ids[0] == ids[1] and len(set(ids[0])) == len(ids[0]), "request inventory/order differs, is empty or duplicate")
    counts["requests_denominator"] = len(rows[0])
    for original, actual in zip(*rows):
        prefix = str(original.get("id")) + ": "
        start = len(failures)
        prompt, outputs = original.get("prompt_token_ids"), original.get("output_token_ids")
        if not _tokens(prompt) or not _tokens(outputs):
            check(False, prefix + "actual token streams missing")
            continue
        check(actual.get("prompt_token_ids") == prompt and actual.get("output_token_ids") == outputs, prefix + "generated token stream differs")
        forwards = expected_forwards(len(prompt), len(outputs))
        for label, row in (("reference", original), ("candidate", actual)):
            check(row.get("teacher_forced") is False and row.get("routes") == [], prefix + label + ": actual unforced generation without routes required")
            check(row.get("sampled_greedy_token_ids") == row.get("output_token_ids"), prefix + label + ": sampled/output token ledger differs")
            observer = row.get("observer", {})
            check(observer.get("cpu_consumer_tap") is False and observer.get("step_logit_copy") is True and observer.get("graph_metadata") is True,
                  prefix + label + ": diagnostic observer configuration differs")
            check(row.get("timing_usable_as_bare_reference") is False and row.get("trace_timing_is_performance_truth") is False,
                  prefix + label + ": observer timing cannot be normal performance truth")
            check([{k: f.get(k) for k in _FORWARD_KEYS} for f in row.get("forward_inventory", [])] == forwards
                  and row.get("forward_count") == len(forwards) and row.get("decode_forward_count") == len(outputs) - 1,
                  prefix + label + ": physical forward/output-mask inventory differs")
        counts["output_steps_denominator"] += len(outputs)
        sampled = [original.get("sampled_greedy_token_ids", []), actual.get("sampled_greedy_token_ids", [])]
        greedy_matches = sum(a == b for a, b in zip(*sampled))
        counts["greedy_token_matches"] += greedy_matches
        check(len(sampled[0]) == len(outputs) and len(sampled[1]) == len(outputs) and greedy_matches == len(outputs), prefix + "sampled greedy tokens differ/incomplete")
        ledgers = [row.get("logit_observations", []) for row in (original, actual)]
        for label, ledger in zip(("reference", "candidate"), ledgers):
            check(len(ledger) == len(outputs) and [x.get("output_index") for x in ledger] == list(range(len(outputs))), prefix + label + ": complete ordered step-logit ledger required")
        for index, (a, b) in enumerate(zip(*ledgers)):
            matched = _sha(a.get("full_f32_logits_sha256")) and a.get("full_f32_logits_sha256") == b.get("full_f32_logits_sha256")
            counts["full_logits_hash_matches"] += int(matched)
            check(matched, prefix + f"output {index}: full F32 logits differ/unknown")
            for key in ("greedy_token", "top_logits", "top1_top2_margin", "vocabulary_size"):
                check(a.get(key) == b.get(key), prefix + f"output {index}: {key} differs")
            check(a.get("greedy_token") == outputs[index], prefix + f"output {index}: logit/greedy ledger differs")
        graph_ledgers = [row.get("graph_observations", []) for row in (original, actual)]
        counts["graph_forwards_denominator"] += len(forwards)
        for label, ledger in zip(("reference", "candidate"), graph_ledgers):
            check(len(ledger) == len(forwards) and [g.get("forward_index") for g in ledger] == list(range(len(forwards))), prefix + label + ": complete ordered graph ledger required")
        for index, (a, b) in enumerate(zip(*graph_ledgers)):
            matched = _sha(a.get("graph_partition_fingerprint_sha256")) and a.get("graph_partition_fingerprint_sha256") == b.get("graph_partition_fingerprint_sha256")
            counts["graph_fingerprint_matches"] += int(matched)
            check(matched, prefix + f"forward {index}: graph partition fingerprint differs/unknown")
            for key in ("node_count", "scheduler_splits", "scheduler_copies", "expert_nodes"):
                check(a.get(key) is not None and a.get(key) == b.get(key), prefix + f"forward {index}: {key} differs/unknown")
        counts["requests_passed"] += int(start == len(failures))
    return dict(schema="real-moe-normal-runtime-equivalence-1",status="failed" if failures else "passed",
                equivalent=not failures,denominator=dict(requests=counts["requests_denominator"],output_steps=counts["output_steps_denominator"],
                                                        graph_forwards=counts["graph_forwards_denominator"]),counts=dict(counts),failures=failures,
                input_records_modified=False,modes_rewritten=False,consumer_tap_enabled=False,
                timing_used_to_judge_accuracy=False,observer_timings_are_diagnostic=True,
                hidden_bare_routes_proven_identical=False,whole_model_accuracy_certified=False,
                runtime_provenance_limit="Caller must separately bind both original/private CPU DLL receipts; this compares observed normal computation, not performance.")


def _receipt(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path),sha256=digest.hexdigest(),bytes=path.stat().st_size)


def audit_files(reference_path, capture_path, out=None):
    """Accept only completed native records bound by adjacent execution receipts."""
    evidence, records = [], []
    for path in (reference_path, capture_path):
        path = Path(path).resolve()
        execution_path = path.parent / "execution.json"
        if not execution_path.is_file():
            raise ValueError("completed acquisition receipt missing: " + str(execution_path))
        data = json.loads(path.read_text(encoding="utf-8"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        native = _receipt(path)
        binding = execution.get("native", {})
        if not (execution.get("status") == "completed" and execution.get("process_id") == data.get("process_id")
                and binding.get("sha256") == native["sha256"]
                and Path(binding.get("path", "")).resolve() == path):
            raise ValueError("completed acquisition binding differs: " + str(path))
        evidence.append(dict(native=native,execution=_receipt(execution_path)))
        records.append(data)
    report = pair_audit(*records)
    report["acquisition_evidence"] = evidence
    report["acquisition_integrity_status"] = "passed"
    report["audit_tool"] = _receipt(__file__)
    if out is not None:
        out = Path(out)
        if out.exists():
            raise FileExistsError("audit must not overwrite existing evidence: " + str(out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    report = audit_files(args.reference, args.capture, args.out)
    print(json.dumps({k: report[k] for k in ("status", "paired", "binding_status", "structural_status", "numerical_status", "partition_status")}))
    return 0 if report["paired"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
