"""Strict, reusable inference-route adapters for M2 MoE.

No training-row/session inference, expert remapping, top-k truncation, or missing
route imputation is permitted. The llama adapter targets the explicit local
llama_eval_callback_v1 collector: every prompt token is marked for output and a
GENTOK marker precedes the forward pass of each sampled decode input token.
Timing printed by this instrumented callback is not a performance measurement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

SCHEMA_VERSION = "moe_inference_routes_v1"
COLLECTOR_CONTRACT = "llama_eval_callback_v1"


class RouteError(ValueError):
    """Missing, malformed, ambiguous, or unsupported route evidence."""


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RouteError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RouteError(f"{label} must be a nonempty string")
    return value


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _source(path: Path, raw: bytes) -> dict:
    return {"path": str(path), "sha256": _sha(raw), "size_bytes": len(raw)}


def dump(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def load(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RouteError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(RouteError(f"invalid JSON number {value}")))


def _metadata(stderr: str) -> dict:
    def count(pattern: str, label: str) -> int:
        values = re.findall(pattern, stderr, re.MULTILINE)
        if len(values) != 1:
            raise RouteError(f"stderr must declare exactly one {label}; got {len(values)}")
        return _int(int(values[0]), label, 1)
    layers = count(r"^print_info:\s+n_layer\s*=\s*(\d+)\s*$", "layer count")
    experts = count(r"^print_info:\s+n_expert\s*=\s*(\d+)\s*$", "expert count")
    top_k = count(r"^print_info:\s+n_expert_used\s*=\s*(\d+)\s*$", "top-k")
    prompt_tokens = count(r"^number of input tokens = (\d+)\s*$", "prompt token count")
    if top_k > experts:
        raise RouteError("top-k cannot exceed expert count")
    def optional(pattern):
        values = re.findall(pattern, stderr, re.MULTILINE)
        if len(values) > 1:
            raise RouteError("ambiguous repeated model metadata")
        return values[0].strip() if values else None
    return {"model_name": optional(r"^print_info:\s+general.name\s*=\s*(.+)$"),
            "architecture": optional(r"^print_info:\s+arch\s*=\s*(.+)$"),
            "layer_count": layers, "experts": experts, "top_k": top_k,
            "prompt_tokens": prompt_tokens, "model_revision": None,
            "model_file_sha256": None}


def parse_llama_trace(stdout: str, stderr: str, *, request_id: str,
                      collector_contract: str) -> dict:
    """Parse one independently executed request; validate every layer and token.

    The supplied collector contract is an explicit declaration, not a guess from
    filenames. Legacy trace captures have no recorded binary digest or exit code;
    these fields remain unknown. Full route coverage is checked independently.
    """
    _text(request_id, "request_id")
    if request_id == "__moe_batch__":
        raise RouteError("request_id __moe_batch__ is reserved by M2")
    if collector_contract != COLLECTOR_CONTRACT:
        raise RouteError(f"unsupported collector contract {collector_contract!r}")
    if not stdout.endswith("\n"):
        raise RouteError("stdout has an incomplete final line; no partial route is dropped")
    meta = _metadata(stderr)
    if not re.search(r"^llama_perf_context_print:\s+total time =", stderr, re.MULTILINE):
        raise RouteError("stderr lacks end-of-run summary; capture may still be running")
    if re.search(r"failed to (?:eval|init)|end of generation.*failed", stderr, re.IGNORECASE):
        raise RouteError("collector reported failed execution")
    layers, experts, top_k = meta["layer_count"], meta["experts"], meta["top_k"]
    prompt_tokens = meta["prompt_tokens"]
    records, generated_token_ids = [], []
    expected_layer = 0
    batch_tokens = None
    prompt_seen = 0
    generated_seen = 0
    pending_generated = None
    batch_phase = None
    batch_start = None
    route_lines = 0
    for line_no, line in enumerate(stdout.splitlines(), 1):
        if line.startswith("GENTOK"):
            match = re.fullmatch(r"GENTOK (\d+)", line)
            if match is None or expected_layer != 0 or pending_generated is not None:
                raise RouteError(f"line {line_no}: malformed or misplaced GENTOK")
            if prompt_seen != prompt_tokens:
                raise RouteError("generation starts before all prompt tokens were routed")
            pending_generated = int(match.group(1))
            continue
        if not line.startswith("MOETOPK"):
            continue
        match = re.fullmatch(r"MOETOPK ffn_moe_topk-(\d+) (\d+) (\d+)(.*)", line)
        if match is None:
            raise RouteError(f"line {line_no}: unsupported MOETOPK syntax")
        layer, k, n = (int(match.group(i)) for i in range(1, 4))
        if layer != expected_layer or layer >= layers:
            raise RouteError(f"line {line_no}: expected layer {expected_layer}, got {layer}; no missing layer is dropped")
        if k != top_k or n < 1:
            raise RouteError(f"line {line_no}: top-k mismatch or missing token routes")
        if expected_layer == 0:
            batch_tokens = n
            if pending_generated is None:
                if prompt_seen + n > prompt_tokens:
                    raise RouteError("unmarked post-prompt routes cannot be called decode")
                batch_phase, batch_start = "prefill", prompt_seen
            else:
                if n != 1:
                    raise RouteError("GENTOK decode forward must route exactly one token")
                batch_phase, batch_start = "decode", prompt_tokens + generated_seen
        elif n != batch_tokens:
            raise RouteError(f"line {line_no}: layer micro-batch token counts differ")
        body = match.group(4).strip()
        groups = body.split("|")
        if groups[0].strip() or len(groups) != n + 1:
            raise RouteError(f"line {line_no}: token separators/count mismatch")
        for offset, group in enumerate(groups[1:]):
            words = group.split()
            if len(words) != k or any(re.fullmatch(r"\d+", word) is None for word in words):
                raise RouteError(f"line {line_no}: expected exactly {k} integer expert IDs per token")
            ids = [int(word) for word in words]
            if len(set(ids)) != k or any(not 0 <= expert < experts for expert in ids):
                raise RouteError(f"line {line_no}: repeated or out-of-range expert ID")
            records.append({"request_id": request_id, "phase": batch_phase,
                            "absolute_position": batch_start + offset,
                            "layer_id": str(layer), "expert_ids": ids})
        route_lines += 1
        expected_layer += 1
        if expected_layer == layers:
            expected_layer = 0
            if batch_phase == "prefill":
                prompt_seen += batch_tokens
            else:
                generated_token_ids.append(pending_generated)
                pending_generated = None
                generated_seen += 1
    if expected_layer != 0 or pending_generated is not None:
        raise RouteError("capture ends inside a layer cycle or before a generated-token forward completes")
    if prompt_seen != prompt_tokens:
        raise RouteError(f"prompt route coverage is {prompt_seen}, expected {prompt_tokens}")
    if len(records) != layers * (prompt_tokens + generated_seen):
        raise RouteError("route coverage mismatch")
    result = {"schema_version": SCHEMA_VERSION,
              "origin": "observed_inference_callback",
              "collector_contract": collector_contract,
              "model": meta,
              "requests": [{"request_id": request_id, "position_start": 0,
                            "prefill_tokens": prompt_tokens,
                            "decode_tokens": generated_seen,
                            "generated_token_ids": generated_token_ids}],
              "records": records,
              "verification": {"complete_layer_token_coverage": True,
                               "route_lines": route_lines,
                               "scores_observed": False,
                               "expert_ids_remapped": False,
                               "top_k_truncated": False,
                               "timing_is_performance_calibration": False,
                               "collector_binary_sha256": None,
                               "process_exit_code": None,
                               "decode_position_semantics": "GENTOK is sampled from previous logits, then forwarded at prompt_tokens + decode_index"},
              "sources": {}}
    validate_bundle(result)
    return result


def convert_llama_files(stdout_path: Path, stderr_path: Path, *, request_id: str,
                        collector_contract: str) -> dict:
    stdout_raw, stderr_raw = stdout_path.read_bytes(), stderr_path.read_bytes()
    result = parse_llama_trace(stdout_raw.decode("utf-8"), stderr_raw.decode("utf-8"),
                              request_id=request_id, collector_contract=collector_contract)
    if stdout_path.read_bytes() != stdout_raw or stderr_path.read_bytes() != stderr_raw:
        raise RouteError("source trace changed during conversion")
    result["sources"] = {"stdout": _source(stdout_path, stdout_raw),
                         "stderr": _source(stderr_path, stderr_raw)}
    return result


def validate_bundle(bundle: Any) -> None:
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise RouteError("unsupported route bundle; training arrays cannot be inferred as inference sessions")
    if bundle.get("origin") not in ("observed_inference_callback", "observed_external_freetoken_trace"):
        raise RouteError("route origin must explicitly declare observed inference")
    meta = bundle.get("model", {})
    experts, top_k = _int(meta.get("experts"), "experts", 1), _int(meta.get("top_k"), "top_k", 1)
    layers = _int(meta.get("layer_count"), "layer_count", 1)
    if top_k > experts:
        raise RouteError("top_k exceeds experts")
    requests = bundle.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RouteError("requests must be explicitly declared")
    request_by_id = {}
    expected = set()
    for request in requests:
        if not isinstance(request, dict):
            raise RouteError("request must be an object")
        name = _text(request.get("request_id"), "request_id")
        if name == "__moe_batch__" or name in request_by_id:
            raise RouteError("reserved or duplicate request ID")
        start = _int(request.get("position_start"), "position_start")
        prefill = _int(request.get("prefill_tokens"), "prefill_tokens", 1)
        decode = _int(request.get("decode_tokens"), "decode_tokens")
        request_by_id[name] = request
        for position in range(start, start + prefill + decode):
            phase = "prefill" if position < start + prefill else "decode"
            expected.update((name, phase, str(layer), position) for layer in range(layers))
    records = bundle.get("records")
    if not isinstance(records, list):
        raise RouteError("records must be a list")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise RouteError("record must be an object")
        name = _text(record.get("request_id"), "record.request_id")
        phase = record.get("phase")
        layer = _text(record.get("layer_id"), "record.layer_id")
        position = _int(record.get("absolute_position"), "record.absolute_position")
        key = (name, phase, layer, position)
        if key not in expected or key in seen:
            raise RouteError(f"unknown, duplicate, or phase-mismatched token {key!r}")
        seen.add(key)
        ids = record.get("expert_ids")
        if not isinstance(ids, list) or len(ids) != top_k:
            raise RouteError("each record requires exactly top_k experts")
        if "scores" in record:
            import math
            values = record["scores"]
            if not isinstance(values, list) or len(values) != top_k or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in values):
                raise RouteError("observed scores must contain exactly top_k finite numbers")
        checked = [_int(expert, "expert_id") for expert in ids]
        if len(set(checked)) != top_k or any(expert >= experts for expert in checked):
            raise RouteError("duplicate or out-of-range experts")
    if seen != expected:
        raise RouteError(f"incomplete inference coverage: missing {len(expected - seen)} layer-token records")


def select_m2_routes(bundle: dict, *, request_id: str, source_layer: int,
                     m2_layer_id: str, phase: str, position_start: int,
                     tokens: int, information_mode: str) -> dict:
    """Explicitly select one contiguous layer/phase span; expert IDs stay exact.

    Source-layer -> M2 layer-name binding is declared and recorded. This is not
    an expert-ID remapping. A mid-request span still requires real M2 history.
    """
    validate_bundle(bundle)
    source_layer = _int(source_layer, "source_layer")
    if source_layer >= bundle["model"]["layer_count"]:
        raise RouteError("source_layer is outside source model")
    _text(m2_layer_id, "m2_layer_id")
    _text(request_id, "request_id")
    position_start, tokens = _int(position_start, "position_start"), _int(tokens, "tokens", 1)
    if phase not in ("prefill", "decode") or information_mode not in ("causal_replay", "offline_oracle"):
        raise RouteError("explicit supported phase and information_mode are required")
    if phase == "decode" and tokens != 1:
        raise RouteError("M2 decode invocation must select exactly one token")
    matched = [copy.deepcopy(record) for record in bundle["records"]
               if record["request_id"] == request_id and record["layer_id"] == str(source_layer)
               and record["phase"] == phase and position_start <= record["absolute_position"] < position_start + tokens]
    matched.sort(key=lambda record: record["absolute_position"])
    if [record["absolute_position"] for record in matched] != list(range(position_start, position_start + tokens)):
        raise RouteError("requested phase/span has no exact observed coverage; no values are imputed")
    trace = [{"request_id": request_id, "layer_id": m2_layer_id,
              "absolute_position": record["absolute_position"], "expert_ids": record["expert_ids"],
              **({"scores": record["scores"]} if "scores" in record else {})}
             for record in matched]
    source_hash = _sha(_json_bytes(bundle))
    return {"schema_version": "m2_moe_route_fragment_v1", "phase": phase,
            "request": {"request_id": request_id, "tokens": tokens, "position_start": position_start},
            "route_trace": trace,
            "route_source": f"observed_inference:bundle_sha256={source_hash}:source_layer={source_layer}:span={position_start}+{tokens}",
            "information_mode": information_mode,
            "provenance": {"source_bundle_sha256": source_hash,
                           "selected_records_sha256": _sha(_json_bytes(matched)),
                           "source_layer": source_layer, "m2_layer_id": m2_layer_id,
                           "model": copy.deepcopy(bundle["model"]),
                           "sources": copy.deepcopy(bundle.get("sources", {})),
                           "source_metadata": copy.deepcopy(bundle.get("source_metadata")),
                           "source_run_header": copy.deepcopy(bundle.get("source_run_header")),
                           "source_verification": copy.deepcopy(bundle.get("verification")),
                           "expert_ids_remapped": False, "top_k_truncated": False,
                           "initial_state_required": position_start > 0,
                           "performance_costs_provided": False}}


def invocation_slice(bundle: dict, *, request_id: str, source_layer: int,
                     m2_layer_id: str, phase: str, position_start: int,
                     tokens: int, information_mode: str) -> dict:
    selected = select_m2_routes(bundle, request_id=request_id, source_layer=source_layer,
                                m2_layer_id=m2_layer_id, phase=phase,
                                position_start=position_start, tokens=tokens,
                                information_mode=information_mode)
    return {"phase": selected["phase"], "requests": [selected["request"]],
            "route_trace": selected["route_trace"], "route_source": selected["route_source"],
            "information_mode": selected["information_mode"]}


def parse_freetoken_ranges(header: bytes, chunks: list[tuple[int, bytes]], *,
                           request_uid: int, request_id: str, source_metadata: dict) -> dict:
    """Read routing only from explicitly aligned byte ranges, without remote code.

    Main-stream token blocks are unchanged in FreeToken v1/v2/v3. Separate
    states/candidates/timing sections are deliberately not read or interpreted.
    The selected request must contain its complete prefill from position zero and
    a contiguous decode prefix from forward index zero. A range sample is never
    claimed to validate the whole publisher file.
    """
    import math
    import struct
    request_uid = _int(request_uid, "request_uid")
    _text(request_id, "request_id")
    if not isinstance(source_metadata, dict):
        raise RouteError("source_metadata must be an explicit object")
    if len(header) < 32:
        raise RouteError("truncated FreeToken fixed header")
    magic, version, endian, hbytes, sbytes, lbytes, layers, top_k, flags, jbytes = struct.unpack_from("<8sHHIHHHHII", header)
    if magic != b"FTEXTRC\0" or version not in (1, 2, 3) or endian != 0x0102:
        raise RouteError("unsupported FreeToken magic/version/endianness")
    if len(header) != hbytes or hbytes < 32 + jbytes or hbytes % 8:
        raise RouteError("header must contain exact fixed+JSON+padded bytes")
    if any(header[32 + jbytes:]):
        raise RouteError("nonzero header padding")
    try:
        raw_meta = json.loads(header[32:32 + jbytes].decode("utf-8"))
    except (ValueError, UnicodeError) as error:
        raise RouteError("invalid FreeToken run-header JSON") from error
    if not isinstance(raw_meta, dict) or not isinstance(raw_meta.get("model_geometry"), dict):
        raise RouteError("FreeToken header requires explicit model_geometry")
    geometry = raw_meta["model_geometry"]
    experts = _int(geometry.get("num_experts"), "model_geometry.num_experts", 1)
    if _int(geometry.get("num_moe_layers"), "num_moe_layers", 1) != layers or _int(geometry.get("top_k"), "top_k", 1) != top_k:
        raise RouteError("fixed and JSON header geometry disagree")
    if layers > 256 or top_k != 10 or top_k > experts or flags & ~0x3FFF or flags & (1 << 12):
        raise RouteError("unsupported FreeToken route layout/flags")
    expected_sbytes = 176 if flags & 128 else 32
    expected_lbytes = 72 + (28 if flags & 64 else 0) + (138 if flags & 128 else 0)
    if sbytes != expected_sbytes or lbytes != expected_lbytes:
        raise RouteError("unsupported FreeToken step/layer record width")
    block_bytes = sbytes + layers * lbytes
    selected = []
    selected_file_rows = []
    parsed_file_rows = set()
    range_meta = []
    for offset, payload in chunks:
        offset = _int(offset, "range offset")
        if offset < hbytes or (offset - hbytes) % block_bytes or not payload or len(payload) % block_bytes:
            raise RouteError("every range must begin and end on a main-stream token-block boundary")
        range_meta.append({"offset": offset, "size_bytes": len(payload), "sha256": _sha(payload)})
        for start in range(0, len(payload), block_bytes):
            file_index = (offset - hbytes + start) // block_bytes
            if file_index in parsed_file_rows:
                raise RouteError("overlapping ranges duplicate main-stream token rows")
            parsed_file_rows.add(file_index)
            uid, phase, rebuild, index, wall, total, used, reserved = struct.unpack_from("<IBB2xIQIII", payload, start)
            if phase not in (0, 1, 2) or rebuild not in (0, 1) or used > total:
                raise RouteError("invalid step phase/rebuild/slot counters")
            token_id = struct.unpack_from("<I", payload, start + 32)[0] if flags & 128 else None
            layer_rows = []
            for expected_layer in range(layers):
                base = start + sbytes + expected_layer * lbytes
                luid, lphase, layer, hit, lindex = struct.unpack_from("<IBBHI", payload, base)
                if (luid, lphase, layer, lindex) != (uid, phase, expected_layer, index):
                    raise RouteError("step/layer repeated fields or layer order disagree")
                ids = list(struct.unpack_from("<10H", payload, base + 12))
                if len(set(ids)) != top_k or any(expert >= experts for expert in ids):
                    raise RouteError("duplicate or out-of-range FreeToken expert IDs")
                scores = list(struct.unpack_from("<10e", payload, base + 32)) if flags & 1 else None
                if scores is not None and any(not math.isfinite(score) or score < 0 or score > 1 for score in scores):
                    raise RouteError("invalid observed f16 gate weight")
                if flags & 128:
                    top32 = list(struct.unpack_from("<32H", payload, base + 72))
                    if top32[:top_k] != ids:
                        raise RouteError("router top32 execution prefix disagrees with raw experts")
                layer_rows.append({"request_id": request_id, "phase": "decode" if phase == 1 else "prefill",
                                   "layer_id": str(layer), "expert_ids": ids,
                                   "source_request_uid": uid, "source_phase": phase,
                                   "source_tok_idx": index, "source_file_token_index": file_index,
                                   "input_token_id": token_id, **({"scores": scores} if scores is not None else {})})
            if uid == request_uid:
                selected.append((phase, index, layer_rows))
                selected_file_rows.append(file_index)
    if not selected:
        raise RouteError("selected request UID is absent from downloaded ranges")
    prefill_indices = sorted(index for phase, index, rows in selected if phase != 1)
    decode_indices = sorted(index for phase, index, rows in selected if phase == 1)
    if not prefill_indices or prefill_indices != list(range(len(prefill_indices))):
        raise RouteError("selected request requires complete observed prefill from absolute position zero")
    if decode_indices != list(range(len(decode_indices))):
        raise RouteError("selected decode forwards must be contiguous from source tok_idx zero")
    # The complete prefill length is established by observing the request's next
    # row as decode index zero. A selected prefill prefix alone cannot prove it.
    if not decode_indices:
        raise RouteError("a range sample needs decode index zero to establish the full prompt length")
    prefill_count = len(prefill_indices)
    records = []
    decode_tokens = []
    for phase, index, rows in sorted(selected, key=lambda item: (item[0] == 1, item[1])):
        for row in rows:
            row["absolute_position"] = prefill_count + index if phase == 1 else index
            records.append(row)
        if phase == 1:
            decode_tokens.append(rows[0]["input_token_id"])
    model = {"model_name": source_metadata.get("model"), "architecture": None,
             "layer_count": layers, "experts": experts, "top_k": top_k,
             "prompt_tokens": prefill_count, "model_revision": None,
             "model_file_sha256": None, "geometry": copy.deepcopy(geometry),
             "source_quantization": raw_meta.get("quant")}
    result = {"schema_version": SCHEMA_VERSION, "origin": "observed_external_freetoken_trace",
              "collector_contract": f"freetoken_main_stream_v{version}", "model": model,
              "requests": [{"request_id": request_id, "source_request_uid": request_uid,
                            "position_start": 0, "prefill_tokens": prefill_count,
                            "decode_tokens": len(decode_indices), "generated_token_ids": decode_tokens}],
              "records": records, "sources": {"header": {"size_bytes": len(header), "sha256": _sha(header)},
                                                "ranges": range_meta},
              "source_run_header": raw_meta,
              "source_metadata": copy.deepcopy(source_metadata),
              "verification": {"complete_selected_layer_token_coverage": True,
                               "full_publisher_file_downloaded": False,
                               "publisher_file_sha256_verified": False,
                               "parsed_token_blocks": len(parsed_file_rows),
                               "selected_file_token_indices": selected_file_rows,
                               "expert_ids_remapped": False, "top_k_truncated": False,
                               "scores_observed": bool(flags & 1),
                               "timing_is_performance_calibration": False,
                               "decode_tokens_are_executed_forwards": True,
                               "http_completion_token_count_equivalent": False,
                               "source_cache_masks_used_as_policy": False,
                               "whole_run_or_model_reproduced": False,
                               "decode_position_semantics": "observed complete prompt length + zero-based phase-1 forward index; includes any in-flight discarded forward"}}
    validate_bundle(result)
    return result


def convert_freetoken_files(header_path: Path, ranges: list[tuple[int, Path]], *,
                           request_uid: int, request_id: str, source_metadata: dict) -> dict:
    header = header_path.read_bytes()
    chunks = [(offset, path.read_bytes()) for offset, path in ranges]
    result = parse_freetoken_ranges(header, chunks, request_uid=request_uid,
                                    request_id=request_id, source_metadata=source_metadata)
    if header_path.read_bytes() != header or any(path.read_bytes() != raw for (offset, path), (_, raw) in zip(ranges, chunks)):
        raise RouteError("source ranges changed during conversion")
    result["sources"]["header"]["path"] = str(header_path)
    for entry, (_, path) in zip(result["sources"]["ranges"], ranges):
        entry["path"] = str(path)
    return result



def download_range(url: str, *, offset: int, size_bytes: int, output: Path,
                   metadata_output: Path, max_bytes: int = 10 * 1024 * 1024) -> dict:
    """Download a bounded public HTTPS byte range, rejecting ignored ranges.

    No credentials or account terms are accepted. The original immutable URL,
    exact Content-Range and local range digest are retained; the whole remote
    object's digest is explicitly unknown. Use a commit-pinned URL for studies.
    """
    import urllib.request
    import urllib.parse
    _text(url, "url")
    if urllib.parse.urlparse(url).scheme != "https":
        raise RouteError("public range download requires HTTPS")
    offset, size_bytes = _int(offset, "offset"), _int(size_bytes, "size_bytes", 1)
    max_bytes = _int(max_bytes, "max_bytes", 1)
    if size_bytes > max_bytes:
        raise RouteError("requested byte range exceeds explicit download budget")
    if output.exists() or metadata_output.exists():
        raise RouteError("download outputs already exist")
    end = offset + size_bytes - 1
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if response.status != 206 or match is None or (int(match[1]), int(match[2])) != (offset, end):
            raise RouteError("server did not honor the exact requested byte range; no full download attempted")
        total = int(match[3])
        raw = response.read(size_bytes + 1)
        if len(raw) != size_bytes:
            raise RouteError("returned range byte count disagrees with request")
        metadata = {"requested_url": url, "offset": offset, "size_bytes": size_bytes,
                    "remote_total_bytes": total, "content_range": content_range,
                    "etag": response.headers.get("ETag"), "sha256": _sha(raw),
                    "full_object_sha256_verified": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    dump(metadata_output, metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download-range", help="Bounded public HTTPS range download; requires network access")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--offset", type=int, required=True)
    fetch.add_argument("--size-bytes", type=int, required=True)
    fetch.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--metadata-output", type=Path, required=True)
    convert = commands.add_parser("convert-llama", help="Strictly convert one llama_eval_callback_v1 inference request")
    convert.add_argument("--stdout", type=Path, required=True)
    convert.add_argument("--stderr", type=Path, required=True)
    convert.add_argument("--request-id", required=True)
    convert.add_argument("--collector-contract", choices=(COLLECTOR_CONTRACT,), required=True)
    convert.add_argument("--output", type=Path, required=True)
    public = commands.add_parser("convert-freetoken", help="Convert explicit aligned local byte ranges from FreeToken")
    public.add_argument("--header", type=Path, required=True)
    public.add_argument("--range", dest="ranges", action="append", required=True, metavar="OFFSET=PATH")
    public.add_argument("--request-uid", type=int, required=True)
    public.add_argument("--request-id", required=True)
    public.add_argument("--source-metadata", type=Path, required=True)
    public.add_argument("--output", type=Path, required=True)
    select = commands.add_parser("select", help="Export an exact M2 single-layer invocation route fragment")
    select.add_argument("--bundle", type=Path, required=True)
    select.add_argument("--request-id", required=True)
    select.add_argument("--source-layer", type=int, required=True)
    select.add_argument("--m2-layer-id", required=True)
    select.add_argument("--phase", choices=("prefill", "decode"), required=True)
    select.add_argument("--position-start", type=int, required=True)
    select.add_argument("--tokens", type=int, required=True)
    select.add_argument("--information-mode", choices=("causal_replay", "offline_oracle"), required=True)
    select.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download-range":
        result = download_range(args.url, offset=args.offset, size_bytes=args.size_bytes,
                                output=args.output, metadata_output=args.metadata_output, max_bytes=args.max_bytes)
        print(json.dumps(result))
        return 0
    if args.output.exists():
        raise RouteError("output already exists; choose a new artifact path")
    if args.command == "convert-llama":
        result = convert_llama_files(args.stdout, args.stderr, request_id=args.request_id,
                                     collector_contract=args.collector_contract)
    elif args.command == "convert-freetoken":
        ranges = []
        for item in args.ranges:
            if "=" not in item:
                raise RouteError("range must be OFFSET=PATH")
            offset, path = item.split("=", 1)
            ranges.append((int(offset), Path(path)))
        result = convert_freetoken_files(args.header, ranges, request_uid=args.request_uid,
                                         request_id=args.request_id, source_metadata=load(args.source_metadata))
    else:
        result = select_m2_routes(load(args.bundle), request_id=args.request_id,
                                 source_layer=args.source_layer, m2_layer_id=args.m2_layer_id,
                                 phase=args.phase, position_start=args.position_start,
                                 tokens=args.tokens, information_mode=args.information_mode)
    dump(args.output, result)
    print(json.dumps({"output": str(args.output), "records": len(result.get("records", result.get("route_trace", []))),
                      "performance_costs_provided": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RouteError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"route conversion rejected: {error}")
