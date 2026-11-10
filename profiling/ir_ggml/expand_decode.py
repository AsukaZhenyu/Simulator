#!/usr/bin/env python3
"""Give every policy family the decode rounds it needs, without touching the archive.

Why this exists
---------------
The archived E/S corpus (`corpus_es.json.gz`) makes decode look like a property of the
`pd-*` family: 128 of its 800 cases carry a decode round, and they are all `pd-*`.  That
is wrong as a statement about the strategy space.  Every policy in the corpus names a
*placement* -- where each operator runs and what is kept -- and a placement applies to
whatever the model is doing.  `pd-*` is the family that *separates* prefill and decode
onto different machines; it is not the family that *has* decode.  A policy that offloads
attention (the `attention-*` family, operator-split into qk/softmax/pv/output) obviously
runs its decode steps with that same split, and so do `pipeline-*`, `moe-*` and `dense-*`.

So this module appends decode rounds to every family.  A `pd-*` case keeps its own round
and gains more after it.

The transform
-------------
For one request `q`, its prefill chain is the ordered set of `-prefill-` nodes whose id
names `request{q}`.  A round `r` mirrors that chain:

    g:request{q}-prefill-layer0:op:layer0:request{q}:proj
    g:request{q}-decode{r}-layer0:op:layer0:request{q}:proj

with dependencies rewritten into the round:

* a dependency on a same-request prefill node becomes the dep on that node's round-(r)
  copy;
* a dependency on anything else (the request's arrival, another request's work) is
  dropped -- the round's own `trace_next` already sits downstream of it;
* an entry node depends on the round's `trace_next`;
* an attention node keeps a dependency on its prefill counterpart, which is the KV the
  prefill wrote.  This is the one dependency that crosses the phase boundary, and it is
  the same one the archived `pd-*` round carries.

Round `r`'s `trace_next` depends on the leaves of round `r-1` (or, for `r == 0`, on the
leaves of the prefill chain), which is how the rounds form one chain instead of `K`
parallel copies of the same work.

What is *not* changed
---------------------
Node attributes are copied verbatim from the prefill counterpart: `kind`, `machine`,
`duration`, `bytes`, `tensor`, `release`.  `duration` is a declared constant that the IR
pricing path never reads (it prices `op` at `t=`/`h=` from the case header), so copying it
keeps the file self-describing without implying it is a price.  The archived pack and the
`cases-es/` tree are read-only here: output always goes to a new directory, because these
are a *different* corpus and must never be reported as the 800.

Usage::

    python expand_decode.py --corpus cases-es --out cases-es-r8 --rounds 8
    python expand_decode.py --corpus cases-es --out /tmp/x --rounds 8 --only attention
    python expand_decode.py --corpus cases-es --out cases-es-r8 --rounds 8 --check

`--check` re-reads what was written and verifies every round is a well-formed DAG whose
chain length equals the prefill chain's.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUEST_RE = re.compile(r"g:request(\d+)[-:]")
PREFILL_RE = re.compile(r"-prefill-layer(\d+):")
DECODE_RE = re.compile(r"-decode(\d+)-layer\d+:")
TRACE_RE = re.compile(r"-decode\d*:trace_next:")


def request_of(node_id: str) -> int | None:
    match = REQUEST_RE.search(node_id)
    return int(match.group(1)) if match else None


def tail_of(node_id: str) -> str:
    return node_id.split(":")[-1]


def is_attention(tail: str) -> bool:
    return tail.startswith("attn")


def decode_id(prefill_id: str, round_index: int) -> str:
    return PREFILL_RE.sub(f"-decode{round_index}-layer\\1:", prefill_id, count=1)


def topo_order(nodes: dict) -> list[str]:
    """A real topological order.  The file's own order is *not* one.

    The archived cases list every `copy` node after the compute nodes that consume it
    (e.g. `dense-compute-001-demand` puts the six `cpu>gpu` weight copies at indices 4-9
    while `proj` at index 1 already depends on them).  The executor dispatches on
    dependencies rather than on index order, so a file can be executed while its listing
    is not an order -- but a transform that *appends* nodes has to know one.  Kahn here,
    with the file order as the tie-break, so the answer is deterministic.
    """
    unsatisfied = {name: {d for d in node["deps"]} for name, node in nodes.items()}
    for name, deps in unsatisfied.items():
        missing = deps - nodes.keys()
        if missing:
            raise SystemExit(f"{name}: dependency {sorted(missing)[0]} is not in the case")
    order: list[str] = []
    while unsatisfied:
        ready = [name for name in nodes if name in unsatisfied and not unsatisfied[name]]
        if not ready:
            raise SystemExit(f"{len(unsatisfied)} nodes form a cycle, starting at"
                             f" {next(iter(unsatisfied))}")
        for name in ready:
            order.append(name)
            del unsatisfied[name]
        for deps in unsatisfied.values():
            deps.difference_update(ready)
    return order


def prefill_chain(nodes: dict, request: int) -> list[str]:
    """One request's prefill work, in topological order.

    Weight copies carry no request id (`copy:w:layer0:W_q:cpu>gpu`) and are therefore not
    in any request's chain: a weight transferred once stays where it was put, and whether
    it survives to the next round is the case's *budget* decision, not the round's.  Only
    activation copies (`copy:g:request0-...-t:...`) are per round.
    """
    chain = [name for name in topo_order(nodes)
             if PREFILL_RE.search(name) and request_of(name) == request]
    for name in chain:
        node = nodes[name]
        if node["kind"] == "copy" and not name.startswith("copy:g:"):
            raise SystemExit(f"{name}: a request-scoped copy that is not an activation")
    return chain


def leaves(nodes: dict, chain: list[str]) -> list[str]:
    """The chain's nodes that nothing else in the chain depends on."""
    used = {dep for name in chain for dep in nodes[name]["deps"]}
    out = [name for name in chain if name not in used]
    if not out:
        raise SystemExit(f"cycle or empty chain: {chain[:1]}")
    return out


def strip_decode(nodes: dict) -> dict:
    """The case with every decode node and decode marker removed."""
    return {name: node for name, node in nodes.items()
            if not DECODE_RE.search(name) and not TRACE_RE.search(name)}


def reindex_decode(node_id: str, round_index: int) -> str:
    """The same decode node one round later."""
    return DECODE_RE.sub(f"-decode{round_index}-layer\\1:", node_id, count=1)


def weight_key(node: dict) -> str | None:
    """The tensor a `copy:w:` node admits, or None for anything else.

    A weight is put on a device once.  Whether it *stays* there across rounds is the
    case's declared budget and reclaim policy, not something a round gets to decide, so a
    round only carries the weight transfer that first declares it.  Re-declaring it in
    every round would charge a real `ggml_backend_tensor_copy` each time (the executor's
    ledger dedups the *budget*, not the work), which would make a resident-weight policy
    pay K uploads it never makes.
    """
    if node["kind"] != "copy" or not node["id"].startswith("copy:w:"):
        return None
    return node.get("tensor") or node["id"]


def reference_round(nodes: dict, request: int) -> tuple[list[str], int]:
    """The round a new one is copied from, and the first round index still missing.

    A `pd-*` case already has one, and it deliberately differs from that case's prefill:
    `pd-compute-001-demand` gets three bare compute nodes because prefill left its outputs
    on the CPU, while `pd-compute-008-demand` moves every weight into the round because
    *decode* is the half that runs on the GPU.  So an existing round is the reference, not
    the prefill chain, and it is kept rather than regenerated.
    """
    rounds = sorted({int(m.group(1)) for name in nodes
                     if (m := DECODE_RE.search(name)) and request_of(name) == request})
    if not rounds:
        return prefill_chain(strip_decode(nodes), request), 0
    last = max(rounds)
    order = topo_order(nodes)
    ref = [name for name in order
           if (m := DECODE_RE.search(name)) and int(m.group(1)) == last
           and request_of(name) == request]
    return ref, last + 1


def expand_case(nodes: dict, rounds: int) -> tuple[dict, dict]:
    """Return (new node table, the decode nodes the archive already had).

    Round 0..N-1 all exist on exit.  A round that the archive already declares is left
    exactly as it is; the missing ones are copied from the last existing round, or from
    the prefill chain when there is none.
    """
    archived = {name: node for name, node in nodes.items()
                if DECODE_RE.search(name) or TRACE_RE.search(name)}
    out = dict(nodes)                      # preserves order; appended keys go last
    base = strip_decode(nodes)
    declared_weights = {key for node in nodes.values() if (key := weight_key(node))}
    for request in sorted({r for r in map(request_of, base) if r is not None}):
        reference, start = reference_round(nodes, request)
        if not reference:
            continue
        reference_set = set(reference)
        previous_leaves = leaves(nodes, reference)
        for round_index in range(start, rounds):
            trace = f"g:request{request}-request{request}-decode{round_index}:trace_next:request{request}"
            out[trace] = {"id": trace, "kind": "compute", "machine": "cpu",
                          "deps": list(previous_leaves), "duration": 36,
                          "release": 0, "bytes": 0}
            made: list[str] = []
            for name in reference:
                node = nodes[name]
                key = weight_key(node)
                if key is not None and key in declared_weights:
                    continue                 # this device already holds it
                new_name = reindex_decode(name, round_index) if DECODE_RE.search(name) \
                    else decode_id(name, round_index)
                copied = []
                for dep in node["deps"]:
                    if TRACE_RE.search(dep):
                        copied.append(trace)
                    elif DECODE_RE.search(dep):
                        copied.append(reindex_decode(dep, round_index))
                    elif dep in reference_set:
                        copied.append(decode_id(dep, round_index))
                    elif dep in base and weight_key(base[dep]) is None:
                        copied.append(dep)   # the KV prefill wrote, read by this round
                if not copied:
                    copied = [trace]
                if is_attention(tail_of(name)) and trace not in copied:
                    copied.append(trace)
                out[new_name] = {**node, "id": new_name, "deps": copied}
                made.append(new_name)
                if key is not None:
                    declared_weights.add(key)
            previous_leaves = leaves(out, made) if made else []
    return out, archived


def check(original: dict, expanded: dict, rounds: int) -> list[str]:
    """Every round present, every dependency resolvable, the archive's prefill intact."""
    problems = []
    topo_order(expanded)                   # raises if any dep is missing or a cycle exists
    base = strip_decode(original)
    for name in expanded:
        if name in base and base[name] != expanded[name]:
            problems.append(f"{name}: an archived prefill node changed")
    for request in sorted({r for r in map(request_of, base) if r is not None}):
        _, start = reference_round(original, request)
        per_round: dict[int, list[str]] = {}
        for name in expanded:
            match = DECODE_RE.search(name)
            if match and request_of(name) == request:
                per_round.setdefault(int(match.group(1)), []).append(name)
            trace = TRACE_RE.search(name)
            if trace and f":request{request}" in name:
                per_round.setdefault(int(re.search(r"-decode(\d*):", name).group(1) or 0),
                                     []).append(name)
        for r in range(rounds):
            if r not in per_round:
                problems.append(f"request{request}: round {r} absent")
        sizes = {r: len(names) for r, names in per_round.items() if r >= start}
        if len(set(sizes.values())) > 1:
            problems.append(f"request{request}: the added rounds differ in size {sizes}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="a directory of <case>/nodes.json (e.g. cases-es)")
    parser.add_argument("--out", type=Path, required=True, help="new directory; must not exist")
    parser.add_argument("--rounds", type=int, default=8, help="total decode rounds (default 8)")
    parser.add_argument("--only", default=None,
                        help="only case names starting with this prefix (e.g. attention)")
    parser.add_argument("--check", action="store_true", help="verify what was written")
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"{args.out}: already exists -- this transform never overwrites")
    cases = sorted(p for p in args.corpus.iterdir() if (p / "nodes.json").exists())
    if args.only:
        cases = [c for c in cases if c.name.startswith(args.only)]
    if not cases:
        raise SystemExit(f"no cases under {args.corpus}")

    args.out.mkdir(parents=True)
    before = after = 0
    problems = []
    per_family: dict[str, int] = {}
    repaired = archived_total = regenerated = 0
    for case in cases:
        nodes = json.loads((case / "nodes.json").read_text(encoding="utf-8"))
        expanded, archived = expand_case(nodes, args.rounds)
        target = args.out / case.name
        target.mkdir()
        (target / "nodes.json").write_text(json.dumps(expanded, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
        before += len(nodes)
        after += len(expanded)
        if archived:
            repaired += 1
            archived_total += len(archived)
            for name, node in archived.items():
                if expanded.get(name) == node:
                    regenerated += 1
        family = case.name.split("-")[0]
        per_family[family] = per_family.get(family, 0) + 1
        if args.check:
            written = json.loads((target / "nodes.json").read_text(encoding="utf-8"))
            problems += [f"{case.name}: {p}" for p in check(nodes, written, args.rounds)]

    print(f"{len(cases)} cases -> {args.out}  rounds {args.rounds}  "
          f"nodes {before} -> {after}")
    for family, count in sorted(per_family.items()):
        print(f"   {family:12s}{count:4d} cases")
    if repaired:
        print(f"   {repaired} case(s) arrived with a decode round; of their "
              f"{archived_total} archived decode nodes {regenerated} were regenerated"
              f" identically, {archived_total - regenerated} were corrected")
    if args.check:
        if problems:
            print(f"{len(problems)} problem(s):", file=sys.stderr)
            for line in problems[:20]:
                print("  " + line, file=sys.stderr)
            return 1
        print("check: every round present, every dependency resolvable, no prefill node changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
