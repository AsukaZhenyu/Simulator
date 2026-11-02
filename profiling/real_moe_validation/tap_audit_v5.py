"""Pure audits of v5 CPU/GPU route observations, never performance accuracy.

Source/runtime receipts remain the collector's responsibility. Full F32 logit
hashes and actual graph metadata are checked directly; forced output token IDs,
CPU weight placement and top-logit margins are never proxies for expert routes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from .postcollection import CONTRACT, expected_forwards

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_EXPERT = re.compile(r"ffn_moe_(up|gate|down)-(\d+)\Z")
_STAGES = ("up", "gate", "down")
_FORWARD = ("index", "phase", "position", "tokens", "output_rows")


def _sha(value):
    return isinstance(value, str) and bool(_SHA.fullmatch(value))


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _tokens(value):
    return isinstance(value, list) and bool(value) and all(_integer(x) for x in value)


def _canonical_request(request):
    # Result-side observation ledgers are not execution inputs. No unrelated
    # request fields (threads, placement, graph policy, buffers, etc.) are erased.
    result = {k: v for k, v in request.items() if k not in ("schema", "source_sha256", "mode", "requests")}
    result["requests"] = [{k: v for k, v in row.items() if k != "output_token_ids"}
                          for row in request.get("requests", [])]
    return result


class _Audit:
    def __init__(self):
        self.problems = {key: [] for key in ("binding", "structural", "numerical", "partition")}
        self.missing = {key: [] for key in self.problems}
        self.unsupported = []
        self.totals = Counter()
        self.cases = []

    def check(self, domain, condition, message):
        if not condition:
            self.problems[domain].append(message)

    def need(self, domain, condition, message):
        if not condition:
            self.missing[domain].append(message)

    def unsupported_scope(self, message):
        self.unsupported.append(message)

    def result(self, schema):
        states = {key: "failed" if self.problems[key] else "unknown" if self.missing[key] else "passed"
                  for key in self.problems}
        if self.unsupported and states["partition"] != "failed":
            states["partition"] = "unsupported"
        status = ("failed" if "failed" in states.values() else "unsupported" if self.unsupported
                  else "unknown" if "unknown" in states.values() else "passed")
        return dict(schema=schema,status=status,**{key + "_status": value for key, value in states.items()},
                    problems=self.problems,missing_evidence=self.missing,unsupported_reasons=self.unsupported,
                    totals=dict(self.totals),cases=self.cases,
                    forced_output_ID_equality_is_numerical_evidence=False,
                    margins_used_as_expert_route_proxy=False,timing_used_to_judge_accuracy=False,
                    timings_are_observer_inclusive_diagnostics=True,whole_model_accuracy_certified=False,
                    hidden_bare_routes_proven_identical=False,private_CUDA_fusion_decisions_proven_identical=False,
                    source_and_runtime_receipts_must_be_verified_by_collector=True,
                    source_hashes_different_across_versions_are_expected_not_implicitly_trusted=True)


def _indexed(audit, domain, entries, key, expected_count, prefix):
    result = {}
    if not isinstance(entries, list):
        audit.need(domain, False, prefix + " ledger missing")
        return result
    audit.need(domain, len(entries) == expected_count, prefix + " ledger coverage incomplete")
    for row in entries:
        if not isinstance(row, dict) or not _integer(row.get(key)):
            audit.check(domain, False, prefix + " invalid ledger index")
            continue
        index = row[key]
        audit.check(domain, index < expected_count and index not in result, prefix + " duplicate/out-of-range ledger index")
        result[index] = row
    return result


def _logits(audit, row, outputs, prefix):
    ledger = _indexed(audit, "numerical", row.get("logit_observations"), "output_index", len(outputs), prefix + "logits")
    sampled = row.get("sampled_greedy_token_ids")
    audit.need("numerical", _tokens(sampled) and len(sampled) == len(outputs), prefix + "sampled greedy inventory missing")
    for index, entry in ledger.items():
        audit.check("numerical", _sha(entry.get("full_f32_logits_sha256")), prefix + "full F32 logits SHA missing/invalid")
        top = entry.get("top_logits")
        valid = isinstance(top, list) and len(top) >= 2 and all(isinstance(x, dict)
            and _integer(x.get("token")) and isinstance(x.get("logit"), (int, float))
            and not isinstance(x["logit"], bool) and math.isfinite(x["logit"]) for x in top)
        audit.check("numerical", valid, prefix + "finite top-logit diagnostic missing")
        if valid:
            audit.check("numerical", len({x["token"] for x in top}) == len(top)
                and all(x["logit"] >= y["logit"] for x, y in zip(top, top[1:])), prefix + "top logits unsorted/duplicate")
            margin = entry.get("top1_top2_margin")
            audit.check("numerical", isinstance(margin, (int, float)) and not isinstance(margin, bool)
                and math.isfinite(margin) and math.isclose(margin, top[0]["logit"] - top[1]["logit"], rel_tol=1e-6, abs_tol=1e-7),
                prefix + "top1/top2 diagnostic margin inconsistent")
            audit.check("numerical", entry.get("greedy_token") == top[0]["token"], prefix + "greedy/top-logit ledger differs")
        if _tokens(sampled) and index < len(sampled):
            audit.check("numerical", entry.get("greedy_token") == sampled[index], prefix + "sampled/logit ledger differs")
    return sampled, ledger


def _common(reference, candidate, *, cross_version):
    a = _Audit()
    versions = (4, 5) if cross_version else (5, 5)
    for label, data, version in zip(("reference", "candidate"), (reference, candidate), versions):
        a.check("binding", isinstance(data, dict), label + ": result object required")
        if not isinstance(data, dict):
            return a, []
        a.check("binding", data.get("schema") == f"real-moe-tap-native-{version}", label + ": native version differs")
        a.check("binding", data.get("scheduler_callback_enabled") is False and data.get("trace_callback_enabled") is False,
                label + ": scheduler callback must be disabled")
        a.check("binding", data.get("model_weights_loaded") is True and data.get("observed_model_contract") == CONTRACT,
                label + ": actual OLMoE model contract missing")
        a.check("binding", data.get("threadpool_policy") == "explicit_persistent_owned" and data.get("graph_or_allocator_modified") is False,
                label + ": graph/allocator/threadpool contract differs")
        request = data.get("request", {})
        if not isinstance(request, dict):
            a.check("binding", False, label + ": request object missing")
            return a, []
        a.check("binding", request.get("schema") == f"real-moe-tap-request-{version}", label + ": request version differs")
        a.check("binding", request.get("mode") == data.get("mode"), label + ": request/native execution mode differs")
        a.check("binding", _sha(data.get("source_sha256")) and request.get("source_sha256") == data.get("source_sha256"), label + ": source hash invalid or not bound to request")
        a.check("binding", _sha(data.get("model_sha256_verified_by_parent"))
                and request.get("model_sha256") == data.get("model_sha256_verified_by_parent"), label + ": pinned model hash differs")
        a.check("binding", request.get("model_contract") == CONTRACT and request.get("batch_size") == 256, label + ": reviewed model/batch boundary differs")
        a.need("binding", request.get("observe_logits") is True and request.get("observe_graph") is True, label + ": step logit/graph evidence required")
        for key in ("threads", "context_size", "expected_process_mask"):
            a.check("binding", _integer(request.get(key), 1), label + ": execution parameter missing: " + key)
    rr, cr = reference["request"], candidate["request"]
    try:
        a.check("binding", _canonical_request(rr) == _canonical_request(cr), "model/input/execution parameters differ")
    except (TypeError, AttributeError):
        a.check("binding", False, "request rows malformed")
    a.check("binding", reference.get("model_sha256_verified_by_parent") == candidate.get("model_sha256_verified_by_parent"), "model contents differ")
    if not cross_version:
        a.check("binding", reference.get("source_sha256") == candidate.get("source_sha256"), "same-v5 pair source differs")
    a.check("binding", reference.get("actual_placement") is not None and reference.get("actual_placement") == candidate.get("actual_placement"), "actual weight placement differs/missing")
    for key in ("NVIDIA_TF32_OVERRIDE", "GGML_CUDA_DISABLE_GRAPHS"):
        left = reference.get("observed_runtime_flags", {})
        right = candidate.get("observed_runtime_flags", {})
        a.need("binding", key in left and key in right, "observed runtime flag missing: " + key)
        a.check("binding", left.get(key) == right.get(key), "observed runtime flag differs: " + key)
    rows = (reference.get("results"), candidate.get("results"))
    if not all(isinstance(group, list) and all(isinstance(x, dict) for x in group) for group in rows):
        a.check("structural", False, "result arrays malformed/missing")
        return a, []
    ids = [[x.get("id") for x in group] for group in rows]
    a.check("structural", bool(ids[0]) and all(isinstance(x, str) for x in ids[0])
            and len(set(ids[0])) == len(ids[0]) and ids[0] == ids[1], "request inventory/order empty, duplicate or changed")
    for label, request, actual_ids in zip(("reference", "candidate"), (rr, cr), ids):
        requested = request.get("requests", [])
        a.check("structural", isinstance(requested, list) and all(isinstance(x, dict) for x in requested)
                and [x.get("id") for x in requested] == actual_ids, label + ": request/result inventory differs")
    paired_rows = []
    for left, right in zip(*rows):
        prefix = str(left.get("id")) + ": "
        p, outputs = left.get("prompt_token_ids"), left.get("output_token_ids")
        a.totals["requests_denominator"] += 1
        if not _tokens(p) or not _tokens(outputs):
            a.check("structural", False, prefix + "actual token streams missing")
            continue
        a.check("binding", right.get("prompt_token_ids") == p and right.get("output_token_ids") == outputs, prefix + "fixed input/generated output streams differ")
        for label, data, row in (("reference", reference, left), ("candidate", candidate, right)):
            requested = [x for x in data["request"].get("requests", []) if isinstance(x, dict) and x.get("id") == row.get("id")]
            a.check("binding", len(requested) == 1, prefix + label + ": unique source request missing")
            if len(requested) == 1:
                source = requested[0]
                a.check("binding", source.get("prompt_token_ids") == row.get("prompt_token_ids"), prefix + label + ": actual prompt differs from requested inputs")
                if "output_token_ids" in source:
                    a.check("binding", source["output_token_ids"] == row.get("output_token_ids"), prefix + label + ": forced input stream differs from request")
        forwards = expected_forwards(len(p), len(outputs))
        graphs, numbers = [], []
        for label, row in (("reference", left), ("candidate", right)):
            f = row.get("forward_inventory", [])
            a.check("structural", isinstance(f, list) and all(isinstance(x, dict) for x in f)
                and [{k: x.get(k) for k in _FORWARD} for x in f] == forwards
                and row.get("forward_count") == len(forwards) and row.get("decode_forward_count") == len(outputs) - 1,
                prefix + label + ": physical forward/output-mask inventory differs")
            observer = row.get("observer", {})
            a.check("binding", observer.get("step_logit_copy") is True and observer.get("graph_metadata") is True,
                    prefix + label + ": diagnostic observers required")
            a.check("binding", row.get("timing_usable_as_bare_reference") is False and row.get("trace_timing_is_performance_truth") is False,
                    prefix + label + ": observer timing must not be bare truth")
            numbers.append(_logits(a, row, outputs, prefix + label + ": "))
            graphs.append(_indexed(a, "partition", row.get("graph_observations"), "forward_index", len(forwards), prefix + label + ": graph"))
        a.totals["output_steps_denominator"] += len(outputs)
        sampled = [x[0] for x in numbers]
        if all(_tokens(x) and len(x) == len(outputs) for x in sampled):
            matches = sum(x == y for x, y in zip(*sampled))
            a.totals["greedy_token_matches"] += matches
            a.check("numerical", matches == len(outputs), prefix + "sampled greedy trajectory differs")
            a.check("numerical", sampled[0] == outputs, prefix + "reference does not reproduce its own greedy trajectory")
        for index in range(len(outputs)):
            one, two = numbers[0][1].get(index), numbers[1][1].get(index)
            if one is None or two is None:
                continue
            matched = _sha(one.get("full_f32_logits_sha256")) and one.get("full_f32_logits_sha256") == two.get("full_f32_logits_sha256")
            a.totals["full_logits_hash_matches"] += int(matched)
            a.check("numerical", matched, prefix + f"output {index}: full F32 logits differ")
            for key in ("greedy_token", "top_logits", "top1_top2_margin", "vocabulary_size"):
                a.check("numerical", one.get(key) == two.get(key), prefix + f"output {index}: {key} differs")
        a.totals["graph_forwards_denominator"] += len(forwards)
        for forward in forwards:
            index = forward["index"]
            one, two = graphs[0].get(index), graphs[1].get(index)
            if one is None or two is None:
                continue
            matched = _sha(one.get("graph_partition_fingerprint_sha256")) and one.get("graph_partition_fingerprint_sha256") == two.get("graph_partition_fingerprint_sha256")
            a.totals["graph_fingerprint_matches"] += int(matched)
            a.check("partition", matched, prefix + f"forward {index}: actual graph partition fingerprint differs/unknown")
            for key in ("node_count", "scheduler_splits", "scheduler_copies", "expert_nodes"):
                a.check("partition", one.get(key) is not None and one.get(key) == two.get(key), prefix + f"forward {index}: {key} differs/missing")
            for key in ("node_count", "scheduler_splits", "scheduler_copies"):
                a.check("partition", _integer(two.get(key), 1 if key == "node_count" else 0), prefix + "invalid actual graph count: " + key)
        a.cases.append(dict(id=left.get("id"),outputs=len(outputs),forwards=len(forwards)))
        paired_rows.append((left, right, forwards, graphs[0], graphs[1]))
    return a, paired_rows


def _normal_record(a, record, label):
    a.check("binding", record.get("mode") == "generate" and record.get("request", {}).get("mode") == "generate", label + ": real generate mode required")
    a.check("binding", record.get("cpu_consumer_tap_enabled") is False, label + ": CPU consumer tap must be disabled")
    for row in record.get("results", []):
        a.check("binding", row.get("teacher_forced") is False and row.get("routes") == [], label + ": unforced normal generation without routes required")
        a.check("binding", row.get("observer", {}).get("cpu_consumer_tap") is False, label + ": normal CPU observer unexpectedly enabled")
        if record.get("schema") == "real-moe-tap-native-5":
            a.check("binding", row.get("gpu_observer_copy_count") == 0 and row.get("gpu_copy_inventory") == [] and row.get("gpu_routes") == [], label + ": normal run performed GPU route copies")


def normal_equivalence(original_result, private_result):
    """Compare genuine schema4→schema5 generate runs without relabeling either."""
    a, _ = _common(original_result, private_result, cross_version=True)
    _normal_record(a, original_result, "original")
    _normal_record(a, private_result, "private")
    result = a.result("real-moe-tap-v5-normal-audit-1")
    result.update(equivalent=result["status"] == "passed",input_records_modified=False,modes_rewritten=False,
                  original_CUDA_graph_decisions_observed=all("cuda_graph_decisions" in r for r in original_result.get("results", [])),
                  actual_CUDA_graph_decision_equivalence="not established by cross-version partition fingerprints",
                  runtime_change="explicit private GPU observer build; actual source/DLL hashes must be bound by collector")
    return result


def _positions(forward, layer):
    if layer < 15:
        return list(range(forward["position"], forward["position"] + forward["tokens"]))
    return [forward["position"] + forward["tokens"] - 1] if forward["output_rows"] else []


def _expert_inventory(a, graphs, forwards, prefix):
    inventory = {}
    for forward in forwards:
        index = forward["index"]
        graph = graphs.get(index)
        if graph is None:
            continue
        nodes = graph.get("expert_nodes")
        if not isinstance(nodes, list):
            a.need("partition", False, prefix + "actual expert-node inventory missing")
            continue
        stage_map = {}
        for node in nodes:
            match = _EXPERT.fullmatch(node.get("name", "")) if isinstance(node, dict) and isinstance(node.get("name"), str) else None
            if not match:
                a.unsupported_scope(prefix + f"forward {index}: unknown actual expert stage")
                continue
            stage, layer = match.group(1), int(match.group(2))
            a.check("structural", layer < 16 and (layer, stage) not in stage_map, prefix + "duplicate/out-of-range graph expert stage")
            stage_map[layer, stage] = node.get("backend")
            if node.get("backend") not in ("CPU", "CUDA0"):
                a.unsupported_scope(prefix + f"forward {index}/layer{layer}/{stage}: unknown consumer backend {node.get('backend')!r}")
            if any(not isinstance(node.get(k), str) or not node[k] or node[k].lower() == "unknown" for k in ("weight_type", "weight_buffer")):
                a.unsupported_scope(prefix + f"forward {index}/layer{layer}/{stage}: actual weight binding unknown")
        for layer in range(16):
            backends = {backend for (which, _), backend in stage_map.items() if which == layer}
            if _positions(forward, layer):
                a.check("structural", bool(backends), prefix + f"forward {index}/layer{layer}: actual expert graph stage missing")
                if len(backends) > 1:
                    a.unsupported_scope(prefix + f"forward {index}/layer{layer}: cross-backend expert stages cannot be merged silently")
        inventory[index] = stage_map
    return inventory


def _graph_decisions(a, row, inventory, forwards, prefix):
    entries = row.get("cuda_graph_decisions")
    if not isinstance(entries, list):
        a.unsupported_scope(prefix + "actual CUDA graph decision ledger missing")
        return
    by_forward = defaultdict(list)
    for event in entries:
        if not isinstance(event, dict) or not _integer(event.get("forward_index")) or event["forward_index"] >= len(forwards):
            a.check("partition", False, prefix + "invalid CUDA graph event forward")
            continue
        index = event["forward_index"]
        by_forward[index].append(event)
        for key in ("partition_nodes", "expert_nodes"):
            a.check("partition", _integer(event.get(key), 1 if key == "partition_nodes" else 0), prefix + "invalid CUDA graph event count")
        flags = all(type(event.get(k)) is bool for k in ("USE_CUDA_GRAPH_compiled", "use_cuda_graph", "cuda_graph_update_required"))
        if not flags:
            a.unsupported_scope(prefix + f"forward {index}: actual CUDA graph decision unknown")
            continue
        expected = "direct" if not event["use_cuda_graph"] else "capture_update_launch" if event["cuda_graph_update_required"] else "replay_launch"
        a.check("partition", event.get("execution") == expected and event.get("device_completion_observed_here") is False,
                prefix + "CUDA graph event semantics inconsistent")
        a.check("partition", event["USE_CUDA_GRAPH_compiled"] or not event["use_cuda_graph"], prefix + "CUDA graph use reported without compile support")
        if event.get("expert_nodes", 0) and (event["use_cuda_graph"] or expected != "direct"):
            a.unsupported_scope(prefix + f"forward {index}: GPU expert capture/replay has no per-call host route coverage")
    for forward in forwards:
        index = forward["index"]
        gpu_count = sum(backend == "CUDA0" for backend in inventory.get(index, {}).values())
        if gpu_count:
            events = by_forward.get(index, [])
            counts = [e.get("expert_nodes") for e in events]
            if not events or not all(_integer(x) for x in counts) or sum(counts) != gpu_count:
                a.unsupported_scope(prefix + f"forward {index}: actual GPU expert partitions not completely observed")


def _copy_ledger(a, row, inventory, forwards, prefix):
    entries = row.get("gpu_copy_inventory")
    if not isinstance(entries, list):
        a.need("structural", False, prefix + "GPU copy inventory missing")
        return {}
    a.check("structural", row.get("gpu_observer_copy_count") == len(entries), prefix + "GPU copy count differs")
    capacity = row.get("gpu_pinned_capacity_bytes")
    a.check("structural", _integer(capacity) and capacity <= 80000000, prefix + "bounded pinned capacity missing/invalid")
    result, intervals = {}, []
    for entry in entries:
        if not isinstance(entry, dict) or not all(_integer(entry.get(k)) for k in ("forward_index", "layer", "stage")):
            a.check("structural", False, prefix + "GPU copy identity malformed")
            continue
        index, layer, stage = entry["forward_index"], entry["layer"], entry["stage"]
        if index >= len(forwards) or layer >= 16 or stage >= 3:
            a.check("structural", False, prefix + "GPU copy identity out of range")
            continue
        key = index, layer, _STAGES[stage]
        a.check("structural", key not in result, prefix + "duplicate GPU stage-copy receipt")
        result[key] = entry
        expected_rows = len(_positions(forwards[index], layer))
        a.check("structural", expected_rows > 0 and inventory.get(index, {}).get((layer, _STAGES[stage])) == "CUDA0",
                prefix + "copy for pruned/non-GPU/unknown expert stage")
        size = expected_rows * 8 * 4
        a.check("structural", entry.get("rows") == expected_rows and entry.get("slots") == 8 and entry.get("bytes") == size,
                prefix + "copy rows/top8/exact bytes differ")
        a.check("structural", entry.get("source_nb0") == 4 and _integer(entry.get("source_nb1"), 32), prefix + "GPU source ID stride/pitch invalid")
        a.check("structural", _integer(entry.get("actual_consumer_stream_opaque_id"), 1) and entry.get("same_stream_before_consumer") is True,
                prefix + "actual same-stream consumer identity missing")
        offset = entry.get("pinned_destination_offset_bytes")
        valid = _integer(offset) and offset % 4 == 0 and _integer(capacity) and offset + size <= capacity
        a.check("structural", valid, prefix + "pinned slot exceeds actual capacity or alignment")
        if valid:
            intervals.append((offset, offset + size))
    cursor = 0
    for begin, end in sorted(intervals):
        a.check("structural", begin == cursor, prefix + "pinned slots overlap, repeat or are noncontiguous")
        cursor = end
    a.totals["GPU_copy_receipts"] += len(entries)
    return result


def pair_audit(normal_result, capture_result):
    """Strict same-v5 normal/replay→capture pairing across actual CPU/GPU stages."""
    a, rows = _common(normal_result, capture_result, cross_version=False)
    a.check("binding", normal_result.get("mode") in ("generate", "replay"), "normal mode must be generate/replay")
    a.check("binding", capture_result.get("mode") == "capture", "capture mode required")
    a.check("binding", normal_result.get("cpu_consumer_tap_enabled") is False and capture_result.get("cpu_consumer_tap_enabled") is True,
            "CPU tap enablement differs from normal/capture roles")
    a.check("binding", capture_result.get("tap_hook_available") is True and capture_result.get("gpu_tap_hook_available") is True,
            "explicit private CPU/GPU hook ABI missing")
    for normal, capture, forwards, normal_graphs, capture_graphs in rows:
        prefix = str(normal.get("id")) + ": "
        a.check("binding", normal.get("routes") == [] and normal.get("gpu_routes") == [] and normal.get("gpu_copy_inventory") == []
                and normal.get("gpu_observer_copy_count") == 0, prefix + "normal run cannot collect consumer routes")
        a.check("binding", capture.get("teacher_forced") is True, prefix + "capture must replay actual normal output inputs")
        a.check("binding", normal.get("observer", {}).get("cpu_consumer_tap") is False
                and capture.get("observer", {}).get("cpu_consumer_tap") is True, prefix + "CPU tap observer roles differ")
        a.check("binding", normal.get("observer", {}).get("cpu_consumer_tap") is False
                and capture.get("observer", {}).get("cpu_consumer_tap") is True, prefix + "CPU observer roles differ")
        a.check("binding", capture.get("observer_extra_flush_ns") == 0
                and capture.get("observer_completion_boundary") == "existing llama_synchronize only", prefix + "new GPU completion boundary/flush unsupported")
        code = capture.get("gpu_capture_error_code")
        if not _integer(code):
            a.need("structural", False, prefix + "GPU capture error/status evidence missing")
        elif code:
            a.unsupported_scope(prefix + f"GPU capture rejected code {code}; missing routes cannot be repaired from cached or CPU weight data")
        a.check("binding", capture.get("gpu_capture_status") == ("unsupported" if code else "observed"), prefix + "GPU capture status/error disagree")
        inventory = _expert_inventory(a, capture_graphs, forwards, prefix)
        _graph_decisions(a, capture, inventory, forwards, prefix)
        # Same-v5 actual graph choices are also compared. Schema4 has no such
        # ledger, so the cross-version normal comparator never fabricates it.
        a.check("partition", normal.get("cuda_graph_decisions") == capture.get("cuda_graph_decisions"), prefix + "normal/capture actual CUDA graph decisions differ")
        copies = _copy_ledger(a, capture, inventory, forwards, prefix)
        route_map = defaultdict(list)
        routes = capture.get("routes")
        if not isinstance(routes, list):
            a.need("structural", False, prefix + "consumer route records missing")
            routes = []
        gpu_routes = []
        for route in routes:
            if not isinstance(route, dict) or not _integer(route.get("forward_index")) or not _integer(route.get("layer")):
                a.check("structural", False, prefix + "route forward/layer malformed")
                continue
            index, layer = route["forward_index"], route["layer"]
            if index >= len(forwards) or layer >= 16:
                a.check("structural", False, prefix + "route forward/layer out of range")
                continue
            route_map[index, layer].append(route)
            if route.get("consumer_backend") == "CUDA0":
                gpu_routes.append(route)
        a.check("structural", capture.get("gpu_routes") == gpu_routes, prefix + "merged GPU routes differ from explicit GPU capture ledger")
        for forward in forwards:
            index = forward["index"]
            for layer in range(16):
                positions = _positions(forward, layer)
                observed = route_map.get((index, layer), [])
                if not positions:
                    a.check("structural", not observed, prefix + "fabricated last-layer pruned route")
                    continue
                a.totals["required_nonempty_forward_layers"] += 1
                stages = {stage: backend for (which, stage), backend in inventory.get(index, {}).items() if which == layer}
                backends = set(stages.values())
                if len(backends) != 1 or not backends <= {"CPU", "CUDA0"}:
                    a.unsupported_scope(prefix + f"forward {index}/layer{layer}: actual consumer backend/stages unsupported")
                    continue
                backend = next(iter(backends))
                if not observed:
                    if backend == "CUDA0":
                        a.unsupported_scope(prefix + f"forward {index}/layer{layer}: no current GPU route observation (fusion/replay may bypass host consumer)")
                    else:
                        a.check("structural", False, prefix + f"forward {index}/layer{layer}: actual CPU consumer route missing")
                    continue
                a.check("structural", len(observed) == 1, prefix + "duplicate/cross-backend forward-layer route")
                for route in observed:
                    a.totals["observed_forward_layer_records"] += 1
                    a.check("structural", route.get("consumer_backend") == backend and route.get("phase") == forward["phase"]
                            and route.get("absolute_positions") == positions, prefix + "actual route backend/phase/positions differ")
                    ids = route.get("expert_ids")
                    a.check("structural", isinstance(ids, list) and len(ids) == len(positions) and all(isinstance(row, list)
                        and len(row) == 8 and all(_integer(v) and v < 64 for v in row) and len(set(row)) == 8 for row in ids), prefix + "route rows/top8/IDs invalid")
                    actual = route.get("actual_stages_observed")
                    valid = isinstance(actual, list) and bool(actual) and all(s in _STAGES for s in actual) and len(set(actual)) == len(actual)
                    a.check("structural", valid and set(actual) == set(stages), prefix + "observed stage inventory differs from actual graph, not assumed three-stage")
                    actual_set = set(actual) if valid else set()
                    a.check("structural", route.get("all_three_stages_observed") is (actual_set == set(_STAGES))
                            and route.get("observed_stage_routes_equal") is True, prefix + "native stage-ID equality/all-three evidence invalid")
                    if "stage_count" in route:
                        a.check("structural", route["stage_count"] == len(actual_set), prefix + "actual stage count differs")
                    if backend == "CUDA0":
                        a.check("structural", route.get("observation_transport") == "same_stream_async_D2H_before_consumer", prefix + "GPU observation transport unknown")
                        for stage in stages:
                            a.check("structural", (index, layer, stage) in copies, prefix + f"forward {index}/layer{layer}/{stage}: corresponding actual GPU copy receipt missing")
        # Extra copy receipts cannot legitimize a missing route or unknown stage.
        for index, layer, stage in copies:
            if not code and not route_map.get((index, layer)):
                a.check("structural", False, prefix + f"forward {index}/layer{layer}/{stage}: GPU copy has no exported current route")
    result = a.result("real-moe-tap-v5-pair-audit-1")
    result.update(paired=result["status"] == "passed",normal_timing_is_bare_truth=False,capture_timing_is_bare_truth=False,
                  GPU_stage_route_equality_scope="native compares actual per-stage copied IDs; per-stage full ID arrays are not separately retained",
                  pinned_slots_and_copy_metadata_verified=True if result["status"] == "passed" else False,
                  graph_replay_routes_filled_from_cache=False,expert_stages_inferred_from_CPU_weights=False)
    return result
