#!/usr/bin/env python3
"""Re-instantiate the 800 coordination topologies on a real model geometry.

The corpus in ``corpus_es.json.gz`` (unpack with ``corpus_es.py --unpack cases-es``)
carries a real schedule topology -- which action runs where, with which deps and which
copies -- but a toy substrate: ``lowering/PROTOCOL.json`` declares ``geometry {hidden 8,
intermediate 16, layers 1, F32}``, node durations are synthetic constants and copy sizes
are 4--1536 bytes at 1 ns per byte.

This script keeps the topology and replaces the substrate: every compute node becomes
the same action at the model's real geometry, every copy node carries the real byte
count of the tensor it names.  Predicted durations come from the frozen table later, in
the driver -- nothing here reads a price.

Every path is a CLI default that can be overridden, because the same corpus is
instantiated against price exports made on different machines::

    python -m profiling.ir_ggml.instantiate --model qwen3-0.6b --tokens 64
    python -m profiling.ir_ggml.instantiate --model all --tokens 64 --list
    python -m profiling.ir_ggml.instantiate --model all --cases cases-v4 --out cases-v4-out

``--cases`` takes a directory of ``<case>/nodes.json``; ``corpus_v4.py`` writes one, and
``corpus_es.py --unpack`` writes the other.  ``EXPORTS`` below names the frozen price
exports shipped in ``prices/``; ``--prices`` points at another root with the same layout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parents[1]                      # .../Simulator
PRICES = Path(os.environ.get("IR_PRICES", HERE / "prices"))
CASES = Path(os.environ.get("IR_CASES", HERE / "cases-es"))
MANIFEST = SIMULATOR / "profiling" / "multimodel" / "models.json"
EXPORTS = {name: PRICES / name for name in (
    "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b",
    "granite-3.0-3b-a800m", "qwen1.5-moe-a2.7b", "olmoe-1b-7b")}

DTYPE_BYTES = {"f32": (1, 4), "f16": (1, 2), "bf16": (1, 2), "q8_0": (32, 34),
               "q4_0": (32, 18), "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210)}

OP_OF_TAIL = {
    "request_arrival": "arrival",
    "proj": "proj", "attn": "attn",
    "attn.qk": "attn.qk", "attn.softmax": "attn.softmax",
    "attn.pv": "attn.pv", "attn.output": "attn.output",
    "ffn": "ffn", "router": "router", "combine": "combine",
    "expert.0": "expert",
}
PRICED_UNIT = {
    "proj": "projection",
    "attn": "attention", "attn.qk": "attention", "attn.softmax": "attention",
    "attn.pv": "attention", "attn.output": "attention",
    "ffn": "dense_ffn", "moe_ffn": "moe_ffn",
    "router": "moe_ffn", "expert": "moe_ffn", "combine": "moe_ffn",
}


# ---------------------------------------------------------------------------
# Copy prices: a lookup at (machine, direction, bytes), never a fitted line.
#
# `PREREG-gpu-prices.md` §"4b′ 预注册" freezes this form.  An affine
# `t = a + b*bytes` was measured first and refuted: bandwidth is not monotone over the
# corpus's own range (cpu>gpu reads 7.6 / 10.4 / 6.5 GB/s at 1 / 8 / 64 MiB), so the fit
# had a negative intercept and residuals up to 1648%.  A lookup at the sizes the corpus
# declares is the same discipline the compute tables already use -- they are a grid of
# measurements, not a function -- and it is stricter: it cannot interpolate.

COPY_PRICE_KEY = ("machine", "direction", "bytes")


def load_copy_prices(path: Path) -> dict:
    """`(machine, direction, bytes) -> ns`.  Only `stable` points become prices.

    An unstable point is *absent*, not averaged: a price built from two runs that disagree
    by more than the gate describes neither of them.  The corpus cases that then lose their
    price are named by `copy_unpriced_reason`, not silently charged zero.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != "transfer_lookup_v1":
        raise SystemExit(f"{path}: schema {document.get('schema')!r}, expected "
                         f"'transfer_lookup_v1' -- an affine `transfer_prices_v1` file "
                         f"cannot be used as a lookup")
    prices = {}
    for point in document["points"]:
        if not point.get("stable"):
            continue
        prices[("pcie", str(point["direction"]), int(point["bytes"]))] = float(
            point["median_ns"])
    if not prices:
        raise SystemExit(f"{path}: no stable point -- every copy would be unpriceable")
    return prices


def copy_price(prices: dict, machine: str, node: dict) -> tuple[float | None, str | None]:
    """`(ns, None)` or `(None, reason)` for one copy node's declared direction and bytes.

    Two spellings reach this function: a case-file node carries `params["dir"]`, while a
    node from `corpus_v4.render()` carries `src`/`dst`.  Reading both is not tolerance for
    sloppiness -- the two graphs describe the same copy, and refusing one of them would
    mean the search and the driver price different programs.
    """
    direction = node.get("dir")
    if direction is None and node.get("src") and node.get("dst"):
        direction = f"{node['src']}>{node['dst']}"
    if direction is None:
        return None, "copy_without_direction"
    size = int(node.get("bytes", 0))
    key = (machine, direction, size)
    if key in prices:
        return prices[key], None
    if not any(k[0] == machine and k[1] == direction for k in prices):
        return None, f"no_price_for_direction:{direction}"
    return None, f"no_price_at_bytes:{direction}/{size}B"


def annotate_copy_nodes(nodes: dict, spec: dict, tokens: int) -> dict:
    """Put the real `bytes` and `dir` on every copy node of a rendered topology.

    `corpus_v4.topology` writes `bytes: 0` because it has no model geometry; the case file
    it is emitted to carries the real count, computed here by `bytes_for`.  So the
    in-memory graph and the text the executor runs disagreed on the one attribute a copy
    price is a function of.  Nothing read the field until the copy price existed, which is
    why it went unnoticed, and a price table looked up against `bytes = 0` would have
    charged every copy the same number while looking perfectly normal.

    `spec` must be the merged model + strategy dict (`load_model()` updated with the case
    spec), because `bytes_for` needs the geometry and the dtypes.
    """
    for node in nodes.values():
        if node["kind"] != "copy":
            continue
        direction = f"{node['src']}>{node['dst']}"
        node["dir"] = direction
        node["bytes"] = int(bytes_for(node["role"], direction, spec, tokens))
    return nodes


def nbytes(nelem: int, dtype: str) -> int:
    block, size = DTYPE_BYTES[dtype]
    if nelem % block:
        raise ValueError(f"{nelem} elements is not a whole number of {dtype} blocks")
    return nelem // block * size


def load_model(model: str) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    meta = next(m for m in manifest["models"] if m["id"] == model)
    geo = meta["geometry"]
    document = json.loads((EXPORTS[model] / "costs.json").read_text(encoding="utf-8"))
    attention = [e for e in document["entries"] if e["kind"] == "attention" and e["phase"] == "decode"]
    signature = attention[0]["physical_signature"]
    config = signature["config"]
    # the MoE exports carry the expert dtypes under *_exps; the IR asks for one dtype
    # per expert matrix, and the block is measured with that same dtype
    dtypes = dict(config["weight_dtypes"])
    for role, exps in (("gate", "gate_exps"), ("up", "up_exps"), ("down", "down_exps")):
        if role not in dtypes and exps in dtypes:
            dtypes[role] = dtypes[exps]
    moe = geo["ffn"]
    return {
        "model": model, "layers": int(geo["block_count"]),
        "hidden": int(geo["hidden_size"]),
        "intermediate": int(moe.get("expert_intermediate_size") or geo["intermediate_size"]),
        "heads": int(geo["attention"]["heads"]), "kv_heads": int(geo["attention"]["kv_heads"]),
        "head_dim": int(geo["attention"]["head_dim"]),
        "rope_dims": int(geo["attention"].get("rope_dims") or geo["attention"]["head_dim"]),
        "rope_base": float(config.get("rope_base", 10000.0)),
        "norm_eps": float(geo["attention"]["norm_epsilon"]),
        "qk_norm": config.get("qk_norm", "none"),
        "attention_gate": config.get("attention_gate", "none"),
        "experts": int(moe.get("experts") or 0), "top_k": int(moe.get("top_k") or 0),
        "dtypes": dtypes,
        "kv_dtype": signature.get("kv_dtype") or config.get("kv_dtype") or "f16",
        "intermediate_dense": int(geo["intermediate_size"] or 0),
    }


def bytes_for(role: str, direction: str, spec: dict, tokens: int) -> int:
    """Real bytes of the tensor a copy node names, at this model's geometry."""
    h, i = spec["hidden"], spec["intermediate"]
    qd = spec["heads"] * spec["head_dim"]
    kd = spec["kv_heads"] * spec["head_dim"]
    d = spec["dtypes"]
    act = 4  # activations are f32 throughout, as the tables declare
    table = {
        "in.hidden": nbytes(tokens * h, "f32"),
        "in": nbytes(tokens * h, "f32"),
        "in.positions": nbytes(tokens, "f32"),
        "proj.q": nbytes(tokens * qd, "f32"),
        "proj.k": nbytes(tokens * kd, "f32"),
        "proj.v": nbytes(tokens * kd, "f32"),
        "attn": nbytes(tokens * h, "f32"),
        "attn.scores": nbytes(spec["heads"] * tokens * tokens, "f32"),
        "attn.probabilities": nbytes(spec["heads"] * tokens * tokens, "f32"),
        "attn.context": nbytes(tokens * qd, "f32"),
        "router": nbytes(tokens * max(spec["experts"], 1), "f32"),
        "combine": nbytes(tokens * h, "f32"),
        "ffn": nbytes(tokens * h, "f32"),
        "expert.0": nbytes(tokens * h, "f32"),
        "proj": nbytes(tokens * h, "f32"),
        "0": nbytes(tokens, "f32"),
        "W_input_norm": nbytes(h, "f32"),
        "W_post_attention_norm": nbytes(h, "f32"),
        "W_q": nbytes(qd * h, d["q"]),
        "W_k": nbytes(kd * h, d["k"]),
        "W_v": nbytes(kd * h, d["v"]),
        "W_output": nbytes(h * qd, d["output"]),
        "W_proj": nbytes(qd * h, d["q"]),
        "W_gate": nbytes(i * h, d["gate"]),
        "W_up": nbytes(i * h, d["up"]),
        "W_down": nbytes(h * i, d["down"]),
        "W_router": nbytes(max(spec["experts"], 1) * h, "f32"),
        "E0": (nbytes(i * h, d.get("gate_exps", d["gate"]))
               + nbytes(i * h, d.get("up_exps", d["up"]))
               + nbytes(h * i, d.get("down_exps", d["down"]))),
    }
    if role not in table:
        raise KeyError(f"copy role {role!r} ({direction}) has no real byte definition")
    return table[role]


def read_topology(case_dir: Path) -> dict:
    return json.loads((case_dir / "nodes.json").read_text(encoding="utf-8"))


OVERRIDE_KEYS = ("dtypes", "kv_dtype", "request_tokens",
                 "vram_blocks", "dram_blocks", "ssd_blocks", "reclaim", "release_order", "uma")


def block_weight_bytes(spec: dict) -> int:
    """Bytes of one block's resident weights at this model's geometry and precision.

    A capacity budget has to be declared in bytes before the model is known, so the corpus
    declares it in *blocks* and it is converted here.  A budget of `k` blocks is the exact
    size of k blocks' weights -- not a round number -- so "does this plan fit" is a
    statement about this model rather than about the rounding.
    """
    roles = ["W_input_norm", "W_q", "W_k", "W_v", "W_post_attention_norm"]
    roles += ["W_router", "E0"] if spec["experts"] > 0 else ["W_gate", "W_up", "W_down"]
    return sum(bytes_for(role, "", spec, 1) for role in roles)


def case_overrides(case_dir: Path) -> dict:
    """Per-case strategy attributes that change *bytes or shapes*, never the node graph.

    A corpus case is a graph plus a few declarations; `corpus_v4.write()` puts the
    declarations in `axes.json` and the graph in `nodes.json`.  Weight precision and the
    KV precision do not add or remove a node, so they live here and are read back at
    instantiation time.  The archived 800-case corpus has no `axes.json` and gets
    `{}` -- every one of those cases instantiates exactly as before.
    """
    path = case_dir / "axes.json"
    if not path.exists():
        return {}
    axes = json.loads(path.read_text(encoding="utf-8"))
    return {key: axes[key] for key in OVERRIDE_KEYS if key in axes}


def apply_overrides(spec: dict, overrides: dict) -> dict:
    """The model spec as this case executes it.

    A MoE export carries the expert matrices' dtypes under `*_exps`; those are real tensors
    too, so a weight-precision level has to move them with the role it names -- otherwise
    `E0` would be priced at the export's precision while `W_gate` is at the case's.
    """
    if not overrides:
        return spec
    case = dict(spec)
    weights = dict(spec["dtypes"])
    for role, dtype in overrides.get("dtypes", {}).items():
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"case dtype {role}={dtype} has no byte size in this build")
        if role not in weights and role not in ("q", "k", "v", "output", "gate", "up", "down"):
            raise ValueError(f"case dtype override names unknown weight role {role!r}")
        weights[role] = dtype
        if f"{role}_exps" in weights:
            weights[f"{role}_exps"] = dtype
    case["dtypes"] = weights
    if "kv_dtype" in overrides:
        if overrides["kv_dtype"] not in DTYPE_BYTES:
            raise ValueError(f"case kv_dtype {overrides['kv_dtype']} has no byte size")
        case["kv_dtype"] = overrides["kv_dtype"]
    return case


def case_layers(case_dir: Path) -> int:
    """How many blocks this case asks for.

    The depth family spans 1..28 blocks, so a case can ask for more layers than the model
    under test has.  That point simply does not exist for a 16-layer model; the caller
    skips it by name rather than letting the instantiator wrap or clamp silently.
    """
    seen = {int(number) for node_id in read_topology(case_dir)
            for number in LAYER_RE.findall(node_id)}
    return max(seen) + 1 if seen else 0


MOE_TAILS = {"router", "expert.0", "combine"}


def case_is_moe(directory: Path) -> bool:
    """Does this one topology use router/expert.0/combine instead of a single `ffn`?"""
    return any(node["kind"] == "compute" and node["id"].split(":")[-1] in MOE_TAILS
               for node in read_topology(directory).values())


def cases_are_moe(dirs) -> bool:
    """Does this case set use the MoE op decomposition?

    A dense case set is legal for *any* model -- `ffn` becomes `moe_ffn` on a MoE model,
    which is what the archived corpus does.  The converse is not: a MoE case set run against
    a dense model would execute `router`/`expert.0`/`combine` on a model that has none, so
    the mismatch is refused rather than silently priced.
    """
    for directory in dirs:
        for node in read_topology(directory).values():
            if node["kind"] == "compute" and node["id"].split(":")[-1] in MOE_TAILS:
                return True
    return False


DECODE_RE = re.compile(r"-(prefill|decode)(\d*)-layer\d+:")
LAYER_RE = re.compile(r"layer(\d+)")
DECODE_ONLY_RE = re.compile(r"-decode\d*-layer\d+:")


def has_decode(names) -> bool:
    return any(DECODE_ONLY_RE.search(n) for n in names)


def decode_histories(model: str) -> set[int]:
    """The `visible_history` grid the frozen export prices decodes at -- the authority."""
    document = json.loads((EXPORTS[model] / "costs.json").read_text(encoding="utf-8"))
    return {int(entry.get("visible_history", 0))
            for entry in document["entries"] if entry["phase"] == "decode"}


REQUEST_RE = re.compile(r"g:request(\d+)-")


def node_shape(node_id: str, tokens: int, history: int,
               decode_history: int | tuple[int, ...] | None = None,
               request_tokens: tuple[int, ...] | None = None) -> tuple[int, int]:
    """The (tokens, history) this node executes at.

    A case header declares one shape, and every family is single-shape except pd, which
    carries a prefill round followed by a decode round.  A prefill node runs the case's
    own shape.  A decode node emits **one** token against whatever context exists when it
    runs: by default that is the request's own prefill plus every decode round before it
    (the physical chain), and with ``decode_history`` pinned it is that declared visible
    history instead -- which is how a decode lands on a point the frozen table prices
    (``visible_history`` 0/128/512/1024).

    ``decode_history`` may be a single value (every round pinned to it) or one value per
    round, so a multi-round case can keep *every* round on the frozen grid instead of only
    the first.  If a case has more rounds than declared histories, later rounds reuse the
    last one; ``instantiate`` records that as ``decode_history_clamped true`` in the case
    header rather than letting it happen silently.

    The same function is used by the driver to price a node, so the executed shape and
    the priced shape cannot drift apart.
    """
    if request_tokens:
        # a heterogeneous batch: each request runs its own prefill length.  The node id
        # carries the request number, so the shape is a lookup rather than a new syntax.
        owner = REQUEST_RE.search(node_id)
        if owner:
            tokens = request_tokens[int(owner.group(1))]
    match = DECODE_RE.search(node_id)
    if not match or match.group(1) == "prefill":
        return tokens, history
    rounds_before = int(match.group(2) or 0)
    if decode_history is None:
        return 1, history + tokens + rounds_before
    pinned = ((decode_history,) if isinstance(decode_history, int)
              else tuple(decode_history))
    return 1, pinned[min(rounds_before, len(pinned) - 1)]


def find_copy_role(node: dict) -> str:
    parts = node["id"].split(":")
    return parts[-2]


def instantiate(case_dir: Path, spec: dict, tokens: int,
                cpu_mask: str = "2,4", history: int = 0,
                decode_history: int | None = None,
                request_tokens: tuple[int, ...] | None = None,
                overrides: dict | None = None) -> tuple[str, list[dict]]:
    topo = read_topology(case_dir)
    order = list(topo)                       # insertion order == the case's own order
    index_of = {name: i for i, name in enumerate(order)}
    lines, manifest = [], []
    for i, name in enumerate(order):
        node = topo[name]
        tail = node["id"].split(":")[-1]
        node_tokens, node_history = node_shape(name, tokens, history, decode_history,
                                               request_tokens)
        if node["kind"] == "copy":
            role = find_copy_role(node)
            direction = node["id"].split(":")[-1]
            op = "copy"
            params = {"bytes": bytes_for(role, direction, spec, node_tokens), "dir": direction}
            # the tensor's identity, not just its size: a capacity budget evicts *which*
            # tensor, and a plan that moves the same tensor twice must not be charged for
            # admitting it twice
            if node.get("tensor"):
                params["tensor"] = node["tensor"].replace(",", "_")
        else:
            if tail not in OP_OF_TAIL:
                if "trace_next" in node["id"] or "request_arrival" in node["id"]:
                    tail = "request_arrival"      # a trace marker with no work, like arrival
                else:
                    raise KeyError(f"compute op {tail!r} has no lowering")
            op = OP_OF_TAIL[tail]
            # the `ffn` action is whatever the model's FFN is: a dense block, or the
            # routed expert block with its router and combine folded in
            if op == "ffn" and spec["experts"] > 0:
                op = "moe_ffn"
            params = {}
        # a node whose shape is the case header's shape says nothing; only the pd family's
        # decode round carries its own `t=`/`h=`, so every single-shape case file stays
        # byte-identical to the ones the prefill tranche was measured from
        if (node_tokens, node_history) != (tokens, history):
            params["t"] = node_tokens
            params["h"] = node_history
        deps = [index_of[d] for d in node["deps"]]
        lines.append(f"node {i} {op} {node['machine']} "
                     f"{','.join(map(str, deps)) if deps else '-'} "
                     + ",".join(f"{k}={v}" for k, v in params.items()))
        manifest.append({"index": i, "id": name, "op": op, "machine": node["machine"],
                         "priced_unit": PRICED_UNIT.get(op), "bytes": params.get("bytes", 0),
                         "tokens": node_tokens, "history": node_history})
    header = [
        f"case {case_dir.name}",
        f"model {spec['model']}",
        f"threads {len(cpu_mask.split(','))}",
        f"cpu_mask {cpu_mask}",
        f"tokens {tokens}",
        f"history {history}",
    ]
    if request_tokens:
        # a heterogeneous batch: the header declares every request's shape, and each node
        # below carries `t=` because its own shape differs from this one
        header.append("request_tokens " + ",".join(map(str, request_tokens)))
    # Edge-tier declarations.  A budget is declared in *blocks* by the corpus and written
    # here in bytes, so the budget is a statement about this model's weights.
    edge = {k: overrides[k] for k in ("vram_blocks", "dram_blocks", "ssd_blocks", "reclaim",
                                      "release_order", "uma")
            if overrides and k in overrides}
    if edge:
        per_block = block_weight_bytes(spec)
        for role, key in (("vram", "vram_blocks"), ("dram", "dram_blocks"), ("ssd", "ssd_blocks")):
            if key in edge:
                header.append(f"{role}_bytes {edge[key] * per_block}")
        for key in ("reclaim", "release_order"):
            if key in edge:
                header.append(f"{key} {edge[key]}")
        if "uma" in edge:
            header.append(f"uma {edge['uma']}")
    if overrides and ("dtypes" in overrides or "kv_dtype" in overrides):
        # The frozen table holds one signature per export.  A case that re-quantizes its
        # weights or its KV cache executes tensors the table never measured, so the driver
        # must not charge it a frozen price -- this line is how it knows.
        named = [f"{role}={dtype}" for role, dtype in overrides.get("dtypes", {}).items()]
        if "kv_dtype" in overrides:
            named.append(f"kv={overrides['kv_dtype']}")
        header.append("price_signature_override " + ",".join(named))
    if has_decode(order):
        # self-describing: the case text says how its decode round is shaped
        pinned = ("chain" if decode_history is None
                  else ",".join(map(str, decode_history))
                  if isinstance(decode_history, tuple) else str(decode_history))
        header.append("decode_history " + pinned)
        rounds = [int(m.group(2) or 0) for m in map(DECODE_RE.search, order)
                  if m and m.group(1) != "prefill"]
        if decode_history is not None:
            count = (1 if isinstance(decode_history, int) else len(decode_history))
            if max(rounds) >= count:
                header.append("decode_history_clamped true")
    header += [
        f"gpu 1",
        f"experts {spec['experts']}",
        f"top_k {spec['top_k']}",
        f"geometry_hidden {spec['hidden']}",
        f"geometry_intermediate {spec['intermediate']}",
        f"geometry_heads {spec['heads']}",
        f"geometry_kv_heads {spec['kv_heads']}",
        f"geometry_head_dim {spec['head_dim']}",
        f"geometry_rope_dims {spec['rope_dims']}",
        f"geometry_rope_base {spec['rope_base']}",
        f"geometry_norm_eps {spec['norm_eps']}",
        f"geometry_qk_norm {spec['qk_norm']}",
        f"geometry_attention_gate {spec['attention_gate']}",
        f"dtype_q {spec['dtypes']['q']}",
        f"dtype_k {spec['dtypes']['k']}",
        f"dtype_v {spec['dtypes']['v']}",
        f"dtype_o {spec['dtypes']['output']}",
        f"dtype_gate {spec['dtypes'].get('gate', 'q4_k')}",
        f"dtype_up {spec['dtypes'].get('up', 'q4_k')}",
        f"dtype_down {spec['dtypes'].get('down', 'q6_k')}",
        f"dtype_kv {spec['kv_dtype']}",
    ]
    return "\n".join(header + lines) + "\n", manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen3-0.6b")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--history", type=int, default=0,
                        help="the case header's visible history; only the C arm's decode "
                             "cases need a non-zero value, where the header should describe "
                             "the nodes instead of differing from all of them")
    parser.add_argument("--out", type=Path, default=HERE / "cases")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--prices", type=Path, default=None,
                        help="root holding <model>/costs.json (default: the frozen exports "
                             "in prices/); overrides IR_PRICES")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict to case directories with these prefixes")
    parser.add_argument("--cpu-mask", default="2,4",
                        help="the tables' explicit_strict_cpu_mask; part of the instrument")
    parser.add_argument("--decode-history", default="chain",
                        help="visible history of the pd family's decode rounds: `chain` (the "
                             "request's own prefill length, the physical chain), a declared "
                             "history from the frozen decode grid, or one per round as "
                             "`h0,h1,...` (see decode_histories)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.prices:
        global EXPORTS
        EXPORTS = {name: args.prices / name for name in EXPORTS}
    missing = [name for name, root in EXPORTS.items() if not (root / "costs.json").exists()]
    if not args.list and missing:
        raise SystemExit(f"no costs.json for {missing} under {next(iter(EXPORTS.values())).parent}; "
                         f"pass --prices ROOT with <model>/costs.json")
    if not args.cases.exists() or not any((p / "nodes.json").exists()
                                          for p in args.cases.iterdir()):
        # The corpus ships as an 83 KiB pack; materialise it on first use rather than
        # asking a fresh clone to run an unpack step it can forget.
        pack = HERE / "corpus_es.json.gz"
        if args.cases == CASES and pack.exists():
            sys.path.insert(0, str(HERE))
            import corpus_es
            print(f"unpacking {pack.name} -> {args.cases}")
            corpus_es.unpack(pack, args.cases)
        else:
            raise SystemExit(f"{args.cases} holds no <case>/nodes.json; "
                             f"run `python -m profiling.ir_ggml.corpus_es --unpack {args.cases}` "
                             f"or point --cases at another corpus directory")

    if args.decode_history != "chain":
        pinned = tuple(int(x) for x in args.decode_history.split(","))
        args.decode_history = pinned[0] if len(pinned) == 1 else pinned

    if args.model == "all":
        models = sorted(EXPORTS)
    else:
        models = [m.strip() for m in args.model.split(",") if m.strip()]
        unknown = [m for m in models if m not in EXPORTS]
        if unknown:
            raise SystemExit(f"unknown model(s) {unknown}; known: {sorted(EXPORTS)}")
    dirs = sorted(p for p in args.cases.iterdir() if (p / "nodes.json").exists())
    if args.only:
        dirs = [d for d in dirs if any(d.name.startswith(p) for p in args.only)]
    if args.list:
        for d in dirs:
            print(d.name)
        print(f"{len(dirs)} cases", file=sys.stderr)
        return 0

    moe_of = {d.name: case_is_moe(d) for d in dirs}
    depth_of = {d.name: case_layers(d) for d in dirs}

    for model in models:
        spec = load_model(model)
        if spec["experts"] == 0 and all(moe_of.values()):
            # A set where *every* topology uses router/expert.0/combine is a MoE corpus
            # pointed at a dense model -- a mistake, not a mixture.  A mixed corpus is
            # legal: `run.py --families` keeps the half this model can execute and counts
            # the rest as not applicable, which is exactly what the archived 800 did.
            raise SystemExit(f"case set {args.cases} uses the MoE op decomposition "
                             f"(router/expert.0/combine) but {model} is dense; "
                             f"instantiate a dense case set against it instead")
        # A case that asks for 28 blocks does not exist on a 16-layer model.  Skipping it
        # is a property of the model, not a silent loss, so it is recorded by name.
        kept = [d for d in dirs if depth_of[d.name] <= spec["layers"]]
        skipped = sorted(d.name for d in dirs if depth_of[d.name] > spec["layers"])
        grid = decode_histories(model)
        if args.decode_history != "chain":
            pinned = ((args.decode_history,) if isinstance(args.decode_history, int)
                      else tuple(args.decode_history))
            outside = sorted(set(pinned) - grid)
            if outside:
                raise SystemExit(f"--decode-history {outside} is outside {model}'s "
                                 f"frozen decode grid {sorted(grid)}")
        out = args.out / model
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"model": model, "tokens": args.tokens, "cpu_mask": args.cpu_mask,
                    "decode_history": (args.decode_history if args.decode_history == "chain"
                                       else list(pinned) if len(pinned) > 1 else pinned[0]),
                    "skipped_deeper_than_model": skipped,
                    "cases": {}}
        for d in kept:
            overrides = case_overrides(d)
            case_spec = apply_overrides(spec, overrides)
            shapes = overrides.get("request_tokens")
            text, nodes = instantiate(d, case_spec, args.tokens, args.cpu_mask,
                                      history=args.history,
                                      decode_history=(None if args.decode_history == "chain"
                                                      else args.decode_history),
                                      request_tokens=tuple(shapes) if shapes else None,
                                      overrides=overrides)
            (out / f"{d.name}.txt").write_text(text, encoding="utf-8")
            manifest["cases"][d.name] = {"nodes": nodes, "request_tokens": shapes,
                                         "price_signature_override":
                                             {k: overrides[k] for k in ("dtypes", "kv_dtype")
                                              if k in overrides} or None}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        # `run.py` drives itself from this manifest, so a leftover `.txt` from an earlier
        # corpus is never executed -- but it *is* in the directory, and the next person to
        # `ls` it reads a case that no longer exists in the corpus (this happened: the
        # retired `ssd-g2-v2` survived three re-instantiations).  Remove what this run did
        # not write, by name, and say so.
        stale = sorted(p.name for p in out.glob("*.txt")
                       if p.stem not in manifest["cases"])
        for name in stale:
            (out / name).unlink()
        if stale:
            print(f"{model}: removed {len(stale)} stale case file(s) not in the corpus: "
                  + ", ".join(stale))
        note = (f"  (skipped {len(skipped)} deeper than its {spec['layers']} layers)"
                if skipped else "")
        print(f"{model}: {len(kept)} cases -> {out}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
