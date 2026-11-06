#!/usr/bin/env python3
"""Corpus v4: an edge-heterogeneous strategy corpus, generated from its own axes.

Why this exists
---------------
The 800-case corpus (``build/coordination-metrics-approved-20261030/exact/cases``) is a
**complete power set over per-semantic-op placement bits** -- ``itertools.product(('cpu',
'gpu'), repeat=len(groups))`` -- with everything else frozen: one layer, no prefetch, one
cache budget, toy geometry.  It answers "which of the *n* semantic ops runs on the GPU"
and nothing else.  For an edge device the interesting strategies are about the **block
stack** -- how many of the L transformer blocks are pinned on the accelerator, which ones
are streamed, what is warm -- and those are not expressible in a 1-layer, per-op-bit space.

This generator emits the same ``nodes.json`` grammar the executor already consumes, so
``instantiate.py`` -> ``run.py`` need no new machinery; only the layer-index regex in
``instantiate.py`` changes from ``layer0`` to ``layer<N>``.

Only what the executor can *do* is generated.  A knob that changes the simulator's own
prediction but not the executed graph is not a strategy in this corpus: it is listed in
``PREREG-corpus-v4.md`` §5 as ``simulator_only`` and is not emitted here.

Node grammar (verified against the archived 800):
    compute  g:request{r}-{prefill|decode<n>}-layer{l}:op:layer{l}:request{r}:{tail}
    weight   copy:w:layer{l}:{weight}:{src}>{dst}
    tensor   copy:g:<producer node id>:t:layer{l}:request{r}:{role}:{src}>{dst}
``instantiate.find_copy_role`` reads ``parts[-2]``, which is the weight name or the tensor
role in both copy forms.

Usage::

    python corpus_v4.py --list                # the axes and the case count, writes nothing
    python corpus_v4.py                       # writes cases-v4/<case>/nodes.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cases-v4"
OUT_MOE = HERE / "cases-v4-moe"     # `--model moe` defaults here; see main()

# The declared block stack.  D is a *declared* depth, not the model's block_count: the
# corpus is a scheduling topology and instantiate.py prices one node as one block.  D=4 is
# the smallest depth at which "first g of D on the accelerator" is a non-trivial statement,
# and it keeps the executed graph at 4x the 1-layer cost.  The real block_count is carried
# by composing's layer term -- which is exactly what the `depth` family measures.
D = 4

# tail -> (weight roles needed in its own address space, input roles, output roles)
DENSE_OPS = (
    ("proj", ("W_input_norm", "W_q", "W_k", "W_v"),
     ("in.hidden", "in.positions"), ("proj.q", "proj.k", "proj.v")),
    ("attn.qk", (), ("proj.q", "proj.k"), ("attn.scores",)),
    ("attn.softmax", (), ("attn.scores",), ("attn.probabilities",)),
    ("attn.pv", (), ("attn.probabilities", "proj.v"), ("attn.context",)),
    ("attn.output", (), ("attn.context",), ("attn",)),
    ("ffn", ("W_post_attention_norm", "W_gate", "W_up", "W_down"),
     ("attn",), ("ffn",)),
)
MOE_OPS = (
    ("proj", ("W_input_norm", "W_q", "W_k", "W_v"),
     ("in.hidden", "in.positions"), ("proj.q", "proj.k", "proj.v")),
    ("attn", (), ("proj.q",), ("attn",)),
    ("router", ("W_router",), ("attn",), ("router",)),
    ("expert.0", ("E0",), ("attn", "router"), ("expert.0",)),
    ("combine", (), ("expert.0", "attn", "router"), ("combine",)),
)

HOME = "cpu"        # weights live in DRAM; an accelerator must be given them

# The depth sweep.  `D = 4` is the reference every other family holds; these are the
# points that make depth a variable.  28 is the smallest published model's layer count,
# so the whole sweep is instantiable against every export.
DEPTHS = (1, 2, 4, 8, 16, 28)

DENSE_SUBOPS = ("proj", "attn.qk", "attn.softmax", "attn.pv", "attn.output", "ffn")
MOE_SUBOPS = ("proj", "attn", "router", "expert.0", "combine")

# Precision of the stored weight matrices, and the levels the *executor* can build a
# tensor at (`ir_runner.cpp` `Geometry::dtype_of`).  A level below f16 is what an edge
# deployment actually ships, and it moves both terms at once: the copy node's byte count
# is the quantized row size, and the matmul runs at that type.
QUANT_ROLES = ("q", "k", "v", "output", "gate", "up", "down")
QUANT_LEVELS = ("q4_0", "q4_k", "q5_k", "q6_k", "q8_0")
KV_LEVELS = ("q8_0", "q4_0")     # f16 is the export's own level and is the reference


def quant(level):
    """One precision for every weight matrix -- the `-q` of a quantized gguf."""
    return {role: level for role in QUANT_ROLES}


def phase_label(phase: str, rnd: int) -> str:
    return "prefill" if phase == "prefill" else f"decode{rnd}"


def topology(*, layers, place, moe=False, storage="demand", warm=(), reload_blocks=(),
             rounds=0, requests=1, prefetch_gap=0, phases=("prefill", "decode")):
    """Emit one case's node dict.

    ``place(phase, layer, tail) -> 'cpu'|'gpu'`` is the whole strategy.
    ``warm``      -- layers whose accelerator weights are pre-placed at t=0 (a weight cache).
    ``reload_blocks`` -- layers re-copied on *every* decode round (a stack that does not fit).
    ``storage``   -- ``'spare'`` pre-places every accelerator weight; ``'demand'`` loads each
                     weight at the op that first needs it.
    ``prefetch_gap`` -- start layer ``l``'s weight copy when block ``l - gap`` finishes instead
                     of when layer ``l``'s own inputs are ready (double buffering).  ``gap=0``
                     is ``demand`` and ``gap=inf`` is ``spare``, so this fills in between.
    """
    ops = MOE_OPS if moe else DENSE_OPS
    last_role = ops[-1][3][0]
    nodes: dict[str, dict] = {}

    def add(name, row):
        row["id"] = name
        nodes[name] = row

    def compute(name, machine, deps):
        add(name, {"kind": "compute", "machine": machine, "deps": list(deps),
                   "duration": 1000, "release": 0, "bytes": 0})

    def copy(name, src, dst, deps, tensor, role, machine="pcie"):
        add(name, {"kind": "copy", "machine": machine, "deps": list(deps),
                   "duration": 1, "release": 0, "bytes": 0,
                   "tensor": tensor, "src": src, "dst": dst, "role": role})

    for r in range(requests):
        arrival = f"g:request{r}:request_arrival"
        compute(arrival, "cpu", [])

        def run_phase(phase, rnd):
            label = phase_label(phase, rnd)
            produced: dict[str, tuple[str, str]] = {}     # role -> (node, machine)
            block_done: dict[int, str] = {}               # layer -> its last node
            for layer in range(layers):
                for tail, weights, inputs, outputs in ops:
                    machine = place(phase, layer, tail)
                    scope = f"g:request{r}-{label}-layer{layer}:op:layer{layer}"
                    name = (f"{scope}:__moe_batch__:expert.0" if tail == "expert.0"
                            else f"{scope}:request{r}:{tail}")
                    deps = []
                    for role in inputs:
                        src_id, src_machine = produced.get(role, (arrival, "cpu"))
                        if src_machine == machine:
                            deps.append(src_id)
                            continue
                        cname = (f"copy:g:{src_id}:t:layer{layer}:request{r}:"
                                 f"{role}:{src_machine}>{machine}")
                        if cname not in nodes:
                            copy(cname, src_machine, machine, [src_id], src_id, role)
                        deps.append(cname)
                    if tail == "attn.output":
                        deps.append(arrival)      # the residual add
                    deps = list(dict.fromkeys(deps))
                    if machine != HOME:
                        reload_here = phase != "prefill" and layer in reload_blocks
                        load_here = phase == "prefill" or reload_here
                        if load_here:
                            preplaced = storage == "spare" or layer in warm
                            # double buffering: start this block's weight copy when an earlier
                            # block finished, not when this block's own inputs are ready
                            ahead = layer - prefetch_gap
                            if preplaced:
                                anchor = []
                            elif prefetch_gap and ahead in block_done:
                                anchor = [block_done[ahead]]
                            else:
                                anchor = list(deps)
                            tag = "" if phase == "prefill" else f"decode{rnd}:"
                            for weight in weights:
                                if storage == "ssd":
                                    # the third tier: the weight lives in a file, so the
                                    # *write* leg happens once (the gguf is laid down /
                                    # page-cached out) and every load is a read through the
                                    # same file.  Hence the down leg carries no round tag --
                                    # charging a disk write per decode round would model a
                                    # runtime that rewrites its own weight file每 token.
                                    what = f"w:layer{layer}:{weight}"
                                    down = f"copy:w:layer{layer}:{weight}:cpu>ssd"
                                    if down not in nodes:
                                        copy(down, HOME, "ssd", anchor, what, weight, "ssd")
                                    up = f"copy:w:layer{layer}:{tag}{weight}:ssd>{machine}"
                                    if up not in nodes:
                                        copy(up, "ssd", machine, [down], what, weight, "ssd")
                                    deps.append(up)
                                    continue
                                wname = (f"copy:w:layer{layer}:{tag}{weight}:"
                                         f"{HOME}>{machine}")
                                if wname not in nodes:
                                    copy(wname, HOME, machine, anchor,
                                         f"w:layer{layer}:{weight}", weight)
                                deps.append(wname)
                    compute(name, machine, deps)
                    for role in outputs:
                        produced[role] = (name, machine)
                # block l+1's projection reads block l's residual; wire the chain
                produced["in.hidden"] = produced[last_role]
                block_done[layer] = produced[last_role][0]
                # terminal handoff back to DRAM
                if layer == layers - 1:
                    final, machine = produced[last_role]
                    if machine != HOME:
                        role = ops[-1][3][0]
                        cname = (f"copy:g:{final}:t:layer{layer}:request{r}:"
                                 f"{role}:{machine}>{HOME}")
                        if cname not in nodes:
                            copy(cname, machine, HOME, [final], final, role)

        # `phases` exists for the C arm's decode comparison (PREREG-real-arm.md §10):
        # llama.cpp's generation test times one token at a given depth with the weights
        # already resident, so the comparable IR graph is a decode round on its own -- no
        # prompt, hence no weight loads either (`load_here` is false outside prefill unless
        # the plan reloads).  A decode round with no prefill in front of it anchors its
        # inputs at `arrival` through the existing fallback, which is the honest reading of
        # "the residual is an input" rather than a rewrite of the dependency structure.
        # The default is both phases, so every existing case renders byte-identically
        # (`_corpus_v4_phase_check.py`).
        if "prefill" in phases:
            run_phase("prefill", 0)
        # decode rounds are numbered from 0, as the archived corpus does: `node_shape`
        # reads that number as the number of rounds already emitted, so round 0 is the
        # first decode and its chain history is exactly the prefill length.
        if "decode" in phases:
            for rnd in range(rounds):
                run_phase("decode", rnd)
    return nodes


# ---------------------------------------------------------------------------
# The axes.  Each family holds everything else at a declared reference and varies
# only its own knob, so a family is a statement that can be read off its name.

def residency(g, where, d=D):
    """`g` of `d` blocks on the accelerator; `where` picks which ones."""
    if where == "first":
        chosen = list(range(g))
    elif where == "last":
        chosen = list(range(d - g, d))
    else:                                   # interleaved: even blocks first, then odd
        chosen = [i for i in range(0, d, 2)] + [i for i in range(1, d, 2)]
        chosen = chosen[:g]
    gpu = set(chosen)
    return lambda phase, layer, tail: "gpu" if layer in gpu else "cpu"


def split_place(g, split, subops):
    """`g` accelerator blocks, with an intra-block op split (`all-*`, `attn-*`, `ffn-*`)."""
    heavies = {"ffn", "router", "expert.0", "combine"}

    def f(phase, layer, tail):
        if layer >= g:
            return "cpu"
        if split == "all-gpu":
            return "gpu"
        if split == "all-cpu":
            return "cpu"
        if split == "attn-gpu":
            return "cpu" if tail in heavies else "gpu"
        return "gpu" if tail in heavies else "cpu"
    return f


def op_bits(mask, subops, g=2):
    """One bit per sub-op, applied uniformly to the first `g` accelerator blocks.

    This is the archived corpus's own object -- a complete power set over per-op placement
    bits -- carried onto a block stack so it can coexist with partial offload.
    """
    on = {tail for i, tail in enumerate(subops) if mask >> i & 1}
    return lambda phase, layer, tail: "gpu" if layer < g and tail in on else "cpu"


def phase_place(prefill, decode):
    return lambda phase, layer, tail: prefill if phase == "prefill" else decode


def families(model_is_moe):
    """(family, [(name, kwargs)]) -- the corpus, for one model kind.

    Every family on a MoE model uses the MoE op decomposition (proj / attn / router /
    expert.0 / combine), because that is the graph the model actually has; the dense
    decomposition of the same block would be a different model.  `moe` is therefore a
    property of the corpus, not a per-family knob.
    """
    out = []
    moe = model_is_moe
    subops = MOE_SUBOPS if moe else DENSE_SUBOPS

    def fam(name, rows):
        out.append((name, [(case_name, dict(moe=moe, **kwargs))
                           for case_name, kwargs in rows]))

    # F1 -- the affine layer term, and the only family that treats depth as a *variable*.
    # Every other family holds d at `D = 4`, so before this sweep the corpus could not see
    # a depth dependence at all -- while the price campaigns already know the probe/engine
    # ratio and the slow band are both functions of the depth window (G8, G11).  Two arms
    # per depth so the slope is measured against a placement that actually does work:
    # all-CPU (no copies; the prediction is boundary + d * block) and half-resident.
    # `storage` has no meaning with no accelerator block (demand and spare render
    # identically), so the all-CPU arm carries no readiness level -- which is itself the
    # control that readiness is only live once a block is offloaded.  At d = 1 the
    # half-resident arm has g = 0 and coincides with all-CPU; `all_cases` drops it.  At
    # d = `D` it is `offload-d{D}-g{D//2}-first-demand`, so that point stays filed under
    # the offload family rather than being silently renamed by the dedup.
    fam("depth", [(f"depth-d{d}-allcpu", dict(layers=d, place=residency(0, "first", d),
                                              storage="demand"))
                  for d in DEPTHS]
                 + [(f"depth-d{d}-halfgpu", dict(layers=d, place=residency(d // 2, "first", d),
                                                 storage="demand"))
                    for d in DEPTHS if d != D])

    # F2 -- partial offload: the `-ngl` sweep, crossed with weight readiness.  `g = 0` is
    # *not* generated here: with no accelerator block the three `where` patterns coincide,
    # and `depth-d4-allcpu-*` already is that reference.  `g = D` has no `where` either.
    fam("offload", [(f"offload-d{D}-g{g}-{w}-{s}",
                     dict(layers=D, place=residency(g, w), storage=s))
                    for g in range(1, D + 1)
                    for w in (("first",) if g == D else ("first", "last", "interleaved"))
                    for s in ("demand", "spare")])

    # F3 -- the intact power set over per-sub-op placement bits (64 dense / 32 MoE),
    # applied to the first two blocks, so "which op" and "which block" coexist.
    width = len(subops)
    fam("opsplit", [(f"opsplit-g2-b{mask:0{width}b}-demand",
                     dict(layers=D, place=op_bits(mask, subops), storage="demand"))
                    for mask in range(1 << width)]
                 # the four named splits with the other readiness, so readiness is crossed
                 # at the op-axis interior rather than only at its corners
                 + [(f"opsplit-g2-{p}-spare",
                     dict(layers=D, place=split_place(2, p, subops), storage="spare"))
                    for p in ("all-gpu", "attn-gpu", "ffn-gpu", "all-cpu")])

    # F4 -- a weight cache of size k, realised as k warm blocks out of g.  `k = 0` is the
    # offload family's own reference and is not repeated here.
    fam("warm", [(f"warm-g{g}-k{k}", dict(layers=D, place=residency(g, "first"),
                                          storage="demand", warm=tuple(range(k))))
                 for g in (2, 4) for k in range(1, g + 1)])

    # F5 -- the stack does not fit: `st` streamed blocks re-load every decode round.
    fam("stream", [(f"stream-g{g}-s{st}-r{rounds}",
                    dict(layers=D, place=residency(g, "first"), storage="demand",
                         rounds=rounds, reload_blocks=tuple(range(g - st, g))))
                   for g in (2, 4) for st in range(0, min(g, 2) + 1) for rounds in (1, 2, 4)])

    # F6 -- prefill/decode phase split: attention on one device, decode on the other.
    fam("pd", [(f"pd-pf{pf}-dc{dc}", dict(layers=D, rounds=1, storage="demand",
                                          place=phase_place(pf, dc)))
               for pf in ("cpu", "gpu") for dc in ("cpu", "gpu")])

    # F7 -- concurrency: N requests sharing one device pair.  N=1,g=0 is the depth
    # family's own reference and is not repeated here.  N goes to 8 because that is where
    # an edge device stops being "one user": beyond it the 4 GB accelerator cannot hold
    # one request's KV cache, let alone eight.
    fam("concurrency", [(f"concurrency-n{n}-g{g}", dict(layers=D, place=residency(g, "first"),
                                                        requests=n, storage="demand"))
                        for n in (1, 2, 4, 8) for g in (0, 2, 4)
                        if (n, g) != (1, 0)])

    # F8 -- double buffering: begin block l's weight copy when block l-gap finished.
    # `gap = 0` is the offload family's `demand` corner and `gap >= g` collapses to its
    # `spare` corner, so only the interior is new; hence `1 <= gap < g`.
    fam("prefetch", [(f"prefetch-g{g}-gap{gap}",
                      dict(layers=D, place=residency(g, "first"), storage="demand",
                           prefetch_gap=gap))
                     for g in (2, 4) for gap in range(1, g)])

    # F9 -- precision, as a strategy variable rather than a property of the download.
    # Weight precision is declared per case and lands in the case's `dtype_*` lines, so
    # the executor builds its weight tensors at that type and every weight copy carries
    # that type's row size.  Two arms per level: all-CPU (no copies -- the compute-only
    # effect) and g=2 (copies and compute together), because a precision that only changes
    # bytes is invisible on the all-CPU arm and a precision that only changes compute is
    # invisible on the offload arm.  The reference for both is the same graph at the
    # export's own dtypes, which is `depth-d4-allcpu` and `offload-d4-g2-first-demand`;
    # they are not regenerated here.
    #
    # The frozen table holds exactly one signature per export, so these cases are
    # *executed but not priced*: `instantiate.py` writes `dtype_override` into the case
    # header and `run.py` refuses to charge a frozen price to a signature it was not
    # measured under.  What they measure is the execution side.
    fam("quant", [(f"quant-allcpu-{level}", dict(layers=D, place=residency(0, "first"),
                                                 storage="demand", dtypes=quant(level)))
                  for level in QUANT_LEVELS]
                 + [(f"quant-offload-g2-{level}",
                     dict(layers=D, place=residency(2, "first"), storage="demand",
                          dtypes=quant(level)))
                    for level in QUANT_LEVELS]
                 # the KV cache is a second, independent precision (`-ctk`/`-ctv`): it
                 # never crosses PCIe, so it can only move compute, and it is the one that
                 # decides whether N concurrent requests fit at all
                 + [(f"quant-offload-g2-kv{kv}", dict(layers=D, place=residency(2, "first"),
                                                      storage="demand", kv_dtype=kv))
                    for kv in KV_LEVELS]
                 # crossed, so the two precisions are shown not to be one knob
                 + [("quant-offload-g2-q4_0-kvq8_0",
                     dict(layers=D, place=residency(2, "first"), storage="demand",
                          dtypes=quant("q4_0"), kv_dtype="q8_0"))])

    # F10 -- request heterogeneity: one device pair serving *different* shapes at once.
    # Every other family gives all N requests the same shape, which is the one assumption
    # an edge server never gets to make.  The shapes are the frozen table's own prefill
    # points (16 / 64 / 256 tokens) -- 512 is a *decode history* point there, not a prefill
    # one, and using it would silently push a third of this family's actions off the price
    # grid (`_corpus_v4_audit.py` reports that as a `shape` gap, not as an execution error).
    #
    # What this family can and cannot show: the executor gives each machine one FIFO
    # queue, so the *makespan* of a heterogeneous batch is queue-bound and interleaving
    # order does not move it.  The measurable object is the per-request chain -- whether
    # a long request's actions read the same price when a short request is in flight --
    # which is what `run.py` reports per action, not per case.
    #
    # Arrival staggering is deliberately *not* emitted: an arrival node carries no
    # duration in this executor, and anchoring it on an earlier node would only reorder
    # work that the FIFO queue already orders.  A vacuous axis is a lie about coverage,
    # so it is recorded as inexpressible in `PREREG-corpus-v4.md` §5 instead.
    fam("mixed", [(f"mixed-{tag}-g{g}", dict(layers=D, place=residency(g, "first"),
                                             storage="demand", requests=len(tokens),
                                             request_tokens=tokens))
                  for tag, tokens in (("longshort", (16, 256)),
                                      ("ladder", (16, 64, 256)))
                  for g in (0, 2)])

    # F11 -- capacity: the accelerator's budget is smaller than the resident set, so
    # admitting one block's weights requires another block to leave.  The three release
    # orders are the simulator's own `release_order` declaration; `decline` is the arm
    # where the plan simply does not fit and the execution must fail rather than quietly
    # take less work.  Budgets are declared in *blocks* -- instantiate.py turns them into
    # this model's bytes -- so "does it fit" is a statement about the model, not a round
    # number that happens to work for one export.
    fam("capacity", [(f"capacity-g4-v{v}-{order}",
                      dict(layers=D, place=residency(4, "first"), storage="demand",
                           vram_blocks=v, reclaim="release_clean_weights", release_order=order))
                     for v in (1, 2, 3)
                     for order in ("lru", "largest_first", "history_frequency")]
                    + [("capacity-g4-v1-decline",
                        dict(layers=D, place=residency(4, "first"), storage="demand",
                             vram_blocks=1, reclaim="decline"))])

    # F12 -- the third tier: the weights do not fit in DRAM, so every load is written to a
    # file-backed tier once and read back through it.  `vram_blocks` is the accelerator's
    # budget, so the plan is "how much of the stack is warm where" against two budgets at
    # once -- which is the actual edge question and the one no two-tier corpus can ask.
    #
    # `v < g` only: at `v = g` the resident set fits exactly, `acquire()` never runs, and
    # the case would be filed as an eviction experiment that evicted nothing.  The arm
    # that isolates the *read path* from the *eviction* is the `unbounded` one, which
    # declares no accelerator budget at all -- and it is emitted for both `g` so the
    # comparison `v < g` versus `no budget` is available at each placement.
    fam("ssd", [(f"ssd-g{g}-v{v}", dict(layers=D, place=residency(g, "first"), storage="ssd",
                                        ssd_blocks=4, vram_blocks=v,
                                        reclaim="release_clean_weights", release_order="lru"))
                for g in (2, 4) for v in range(1, g)]
               + [(f"ssd-g{g}-unbounded", dict(layers=D, place=residency(g, "first"),
                                               storage="ssd", ssd_blocks=4))
                  for g in (2, 4)]
               # the arm the family exists for: a decode round that has to read the
               # streamed block back through the file tier *again*.  Without rounds the
               # family would only measure one pass through the file (write once, read
               # once), which is the cheap case; the per-token re-read is the edge cost.
               + [(f"ssd-g{g}-stream-r{rounds}",
                   dict(layers=D, place=residency(g, "first"), storage="ssd", ssd_blocks=4,
                        vram_blocks=1, reclaim="release_clean_weights", release_order="lru",
                        rounds=rounds, reload_blocks=(g - 1,)))
                  for g in (2, 4) for rounds in (2,)])

    # F13 -- UMA: an integrated device shares the host's address space, so a cpu<->gpu copy
    # is an alias and moves nothing.  Same graph as the offload family on purpose -- the
    # pair (offload-d4-g2-first-demand, uma-g2) is the same plan on two hardware contracts,
    # which is exactly the comparison the accelerator-versus-iGPU question needs.
    fam("uma", [(f"uma-g{g}", dict(layers=D, place=residency(g, "first"), storage="demand",
                                   uma=1)) for g in (2, 4)])

    return out


def fingerprint(spec) -> bytes:
    """A case's identity: its graph *and* the attributes that change bytes but not nodes.

    Weight precision and KV precision add and remove no node -- they are the same schedule
    over differently sized tensors -- so a graph digest alone would collapse the quant
    family into its own reference.  Per-request shapes are the same case: the graph is
    identical and the shapes are carried in the case header.
    """
    tiers = {k: spec.get(k) for k in ("vram_blocks", "dram_blocks", "ssd_blocks", "reclaim",
                                      "release_order", "uma")}
    return json.dumps([render(spec), spec.get("dtypes"), spec.get("kv_dtype"),
                       spec.get("request_tokens"), tiers], sort_keys=True).encode()


def all_cases(model_is_moe):
    """Every case, with cells whose rendered graph repeats an earlier one dropped.

    A level that cannot change the graph is not a cover of that axis: at D=4 the
    `interleaved` order coincides with `first` at g=1 and g=3, and readiness is vacuous
    with no accelerator block.  Dropping them at generation time keeps the case count
    equal to the number of distinct plans, which is the number that is reported.
    """
    cases: dict[str, dict] = {}
    digests: dict[bytes, str] = {}
    dropped: list[tuple[str, str]] = []
    for family, rows in families(model_is_moe):
        for name, kwargs in rows:
            if name in cases:
                raise ValueError(f"duplicate case name {name!r} across families")
            spec = dict(family=family, **kwargs)
            if "request_tokens" in spec and len(spec["request_tokens"]) != spec["requests"]:
                raise ValueError(f"{name}: {len(spec['request_tokens'])} shapes declared "
                                 f"for {spec['requests']} requests")
            digest = fingerprint(spec)
            if digest in digests:
                dropped.append((name, digests[digest]))
                continue
            digests[digest] = name
            cases[name] = spec
    return cases, dropped


def render(spec):
    return topology(layers=spec["layers"], place=spec["place"],
                    moe=spec.get("moe", False), storage=spec.get("storage", "demand"),
                    warm=spec.get("warm", ()), reload_blocks=spec.get("reload_blocks", ()),
                    rounds=spec.get("rounds", 0), requests=spec.get("requests", 1),
                    prefetch_gap=spec.get("prefetch_gap", 0),
                    phases=tuple(spec.get("phases", ("prefill", "decode"))))


def write(cases, out: Path = OUT) -> None:
    if out.exists():
        shutil.rmtree(out)
    for name, spec in cases.items():
        directory = out / name
        directory.mkdir(parents=True)
        (directory / "nodes.json").write_text(
            json.dumps(render(spec), indent=1), encoding="utf-8")
        axes = {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in spec.items() if k != "place"}
        (directory / "axes.json").write_text(json.dumps(axes, indent=1), encoding="utf-8")


def check_sound(cases) -> None:
    """Every case must be topologically ordered with no dangling or forward deps, and no
    two cases may render to the same graph.

    `instantiate.py` turns insertion order into node indices and `run.py` checks the
    executor's node list against the manifest by index, so a forward dep would silently
    describe a schedule the executor cannot build.  Byte-identical cases are the defect the
    archived corpus carried 38 of (demand/spare pairs that coincide); refusing them here
    means a `where` or `w` level that cannot change the graph is removed at design time
    instead of inflating the case count.
    """
    seen: dict[bytes, str] = {}
    for name, spec in cases.items():
        nodes = render(spec)
        digest = fingerprint(spec)
        if digest in seen:
            raise ValueError(f"{name} is the same plan as {seen[digest]}")
        seen[digest] = name
        index = {key: i for i, key in enumerate(nodes)}
        for i, (key, row) in enumerate(nodes.items()):
            for dep in row["deps"]:
                if dep not in index:
                    raise ValueError(f"{name}: {key} depends on unknown {dep}")
                if index[dep] >= i:
                    raise ValueError(f"{name}: {key} depends on later {dep}")
        if not nodes:
            raise ValueError(f"{name}: empty topology")


_LAYER_RE = re.compile(r"layer(\d+)")


def check_budgets(cases) -> None:
    """A declared budget must bind, or the case is not the experiment it is named for.

    `capacity` and `ssd` declare the accelerator's budget in *blocks*.  If that budget is
    at least the number of blocks the plan puts on the accelerator, `acquire()` is never
    consulted, nothing is ever evicted, and the case still runs to completion and still
    reports a `tiers` block -- a successful-looking eviction experiment that evicted
    nothing.  The rendered graph is identical either way, so `check_sound` cannot see it;
    the check has to be made against the plan's own placement.

    The reverse direction needs no check here: `decline` is required to fail, and `run.py`
    turns stderr's "capacity exceeded" into an `infeasible_as_declared` row.  What is
    checked is only that the budget is tight enough for that failure to be possible.
    """
    for name, spec in cases.items():
        blocks = spec.get("vram_blocks")
        if blocks is None:
            continue
        resident = set()
        for key, row in render(spec).items():
            if row["kind"] != "compute" or row["machine"] != "gpu":
                continue
            match = _LAYER_RE.search(key)
            if match:
                resident.add(int(match.group(1)))
        if blocks >= len(resident):
            raise ValueError(
                f"{name}: vram_blocks={blocks} but the plan puts {len(resident)} blocks on "
                f"the accelerator -- the budget does not bind and the case would record an "
                f"eviction run that evicted nothing")


def audit(cases) -> dict:
    by_family: dict[str, list] = {}
    for spec in cases.values():
        by_family.setdefault(spec["family"], []).append(spec)
    print(f"{'family':13s} {'cases':>6s} {'nodes/case':>12s} {'gpu':>7s} {'copies':>8s}")
    total = {"cases": 0, "nodes": 0, "gpu": 0, "copies": 0}
    for family in sorted(by_family):
        sizes = gpu = copies = 0
        lo, hi = 10 ** 9, 0
        for spec in by_family[family]:
            nodes = render(spec)
            sizes += len(nodes)
            lo, hi = min(lo, len(nodes)), max(hi, len(nodes))
            for row in nodes.values():
                gpu += row["kind"] == "compute" and row["machine"] == "gpu"
                copies += row["kind"] == "copy"
        print(f"{family:13s} {len(by_family[family]):6d} {lo:5d}..{hi:<6d} "
              f"{gpu / sizes:6.1%} {copies / sizes:7.1%}")
        total["cases"] += len(by_family[family])
        total["nodes"] += sizes
        total["gpu"] += gpu
        total["copies"] += copies
    print(f"{'TOTAL':13s} {total['cases']:6d} {total['nodes']:12d} "
          f"{total['gpu'] / total['nodes']:6.1%} {total['copies'] / total['nodes']:7.1%}")
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="dense", choices=("dense", "moe"))
    parser.add_argument("--list", action="store_true", help="print the axes, write nothing")
    parser.add_argument("--out", type=Path, default=None,
                        help="case directory; defaults to cases-v4 for --model dense and "
                             "cases-v4-moe for --model moe (writing one corpus over the other "
                             "silently leaves four of the six models with the wrong sentences)")
    args = parser.parse_args(argv)
    cases, dropped = all_cases(args.model == "moe")
    print(f"model kind {args.model}: {len(cases)} cases, D={D}")
    if dropped:
        print(f"dropped {len(dropped)} coincident cells: "
              + ", ".join(f"{name}={alias}" for name, alias in dropped))
    check_sound(cases)
    check_budgets(cases)
    audit(cases)
    if not args.list:
        out = args.out or (OUT_MOE if args.model == "moe" else OUT)
        write(cases, out)
        print(f"wrote {len(cases)} cases -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
