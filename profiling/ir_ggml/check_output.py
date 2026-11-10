#!/usr/bin/env python3
"""Look at what the executor actually computed, rather than how long it took.

`ir_runner --dump-dir` writes every port of the executed shape as raw F32.  This reads them
and answers three questions, in increasing strength:

1. **Are the values real numbers?**  NaN or Inf anywhere means a broken dequantisation, a
   shape the kernel did not expect, or a tensor that was never written.
2. **Is anything degenerate?**  A port whose standard deviation is zero is a value that was
   copied from somewhere it did not come from -- the classic signature of a scatter offset
   that misses, or of a copy that never ran.
3. **Do softmax rows sum to one?**  `probs` is the one port with an arithmetic invariant the
   model imposes regardless of weights, so it is the one place a wrong axis shows up as a
   number instead of as a shadow.

With `--compare DIR2` it also runs the same checks on a second dump and reports, per port,
the largest absolute difference.  Two dumps of the *same* case on different backends must
agree up to float accumulation order; a difference of order 1 means the two paths are not
computing the same graph -- which is a plumbing bug, because ggml's own kernels are the same
code on both.

    python check_output.py --dump dgpu
    python check_output.py --dump dgpu --compare dcpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FINITE_ONLY = {"x", "query", "new_key", "new_value", "scores", "probs", "attended",
               "attention_hidden", "hidden_output", "normed", "moe_probs", "expert_out"}


def load(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read every port as a numpy array in the order the tensor's own axes name it.

    ggml lays a tensor out with `ne[0]` varying fastest and numpy with the *last* axis
    varying fastest, so a plain `reshape(ne)` silently transposes the tensor.  The first
    version of this file did that, and reported a correct causal mask, a correct softmax and
    a correct value-product as three separate defects -- every one of them a property of the
    reader.  `order="F"` is the whole fix, and the manifest carries `nb` so a tensor whose
    strides do not follow its shape is caught instead of read as if it did.
    """
    manifest = json.loads((directory / "ports.json").read_text(encoding="utf-8"))
    arrays = {}
    for name, meta in manifest["ports"].items():
        ne = meta["ne"] if isinstance(meta, dict) else meta
        raw = (directory / f"{name}.f32").read_bytes()
        n = int(np.prod(ne))
        if len(raw) != n * 4:
            raise SystemExit(f"{directory}/{name}.f32: {len(raw)} bytes, manifest says {n} floats")
        if isinstance(meta, dict):
            nb = meta.get("nb")
            if nb and nb[0] != 4:
                raise SystemExit(f"{name}: element stride {nb[0]}, not a contiguous F32 tensor")
        arrays[name] = np.frombuffer(raw, dtype=np.float32).reshape(ne, order="F")
    return manifest, arrays


def port(arrays: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    """A 3-D port without its trailing singleton axes, or None if the case never wrote it."""
    a = arrays.get(name)
    if a is None:
        return None
    return np.squeeze(a, axis=tuple(i for i, n in enumerate(a.shape) if n == 1 and i >= 2))


def describe(name: str, a: np.ndarray) -> list[str]:
    notes = []
    finite = np.isfinite(a).all()
    if not finite:
        bad = int((~np.isfinite(a)).sum())
        # A causal mask writes -inf on purpose: in `scores` that is the mask, not a fault.
        # A NaN, or a +inf, or a -inf anywhere else, is a fault.
        if name == "scores" and np.isneginf(a).sum() == bad:
            notes.append(f"{bad} -inf (causal mask)")
        else:
            notes.append(f"NOT FINITE ({bad} of {a.size})")
    f = a[np.isfinite(a)]
    std = float(f.std()) if f.size else 0.0
    if f.size and std == 0.0:
        notes.append("DEGENERATE (std 0)")
    if name == "probs":
        s = a.astype(np.float64).sum(axis=0)
        worst = float(np.abs(s - 1.0).max())
        notes.append(f"softmax over keys sums to 1 within {worst:.2e}"
                     + ("" if worst < 1e-4 else "  OVER"))
    return notes


def relations(arrays: dict[str, np.ndarray]) -> list[tuple[str, str, bool]]:
    """The arithmetic the layer imposes on its own ports, independent of the weights.

    These are the checks a wrong axis, a missing mask or a copy that never landed all fail,
    and they fail them in numbers rather than in shadows:

    * a softmax row is a distribution, so `probs` sums to one over keys;
    * a masked score is exactly -inf, and only above the diagonal;
    * attention has to change the hidden state, so `attention_hidden` cannot equal `x`;
    * the FFN has to change it again, so `hidden_output` cannot equal `attention_hidden`.
    """
    out = []
    def flat(name):
        a = arrays.get(name)
        if a is None:
            return None
        while a.ndim > 2 and a.shape[-1] == 1:
            a = a[..., 0]
        return a
    s, p = flat("scores"), flat("probs")
    if s is not None:
        # scores is [key, token, head] and `ggml_diag_mask_inf` masks every key past the
        # token, so the -inf region is the strict upper triangle in those axes.
        nk, nt = s.shape[0], s.shape[1]
        upper = np.arange(nk)[:, None] > np.arange(nt)[None, :]
        masked = ~np.isfinite(s)
        ok = bool((masked == upper[..., None]).all())
        out.append(("scores", "masked entries are exactly the causal mask", ok))
        out.append(("scores", "no NaN, only -inf as the mask value",
                    bool((np.isfinite(s) | np.isneginf(s)).all())))
    if p is not None:
        worst = float(np.abs(p.astype(np.float64).sum(axis=0) - 1.0).max())
        out.append(("probs", f"sums to 1 over keys (worst {worst:.2e})", worst < 1e-4))
    a, x = arrays.get("attention_hidden"), arrays.get("x")
    if a is not None and x is not None and a.shape == x.shape:
        d = float(np.abs(a - x).max())
        out.append(("attention_hidden", f"differs from x (max {d:.3g})", d > 0))
    h = arrays.get("hidden_output")
    if h is not None and a is not None and h.shape == a.shape:
        d = float(np.abs(h - a).max())
        out.append(("hidden_output", f"differs from attention_hidden (max {d:.3g})", d > 0))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None,
                    help="a second dump of the same case, e.g. the same run with --no-gpu")
    args = ap.parse_args(argv)

    manifest, arrays = load(args.dump)
    print(f"{args.dump}  shape {manifest['shape']}  ports {len(arrays)}")
    print(f"   {'port':18s}{'ne':18s}{'min':>11s}{'max':>11s}{'mean':>11s}{'std':>11s}"
          f"  notes")
    problems = 0
    for name in sorted(arrays):
        a = arrays[name]
        f = a[np.isfinite(a)]
        notes = describe(name, a)
        problems += sum(1 for n in notes if "OVER" in n or "NOT" in n or "DEGEN" in n)
        print(f"   {name:18s}{str(list(a.shape)):18s}"
              f"{(f.min() if f.size else float('nan')):11.4g}"
              f"{(f.max() if f.size else float('nan')):11.4g}"
              f"{(f.mean() if f.size else float('nan')):11.4g}"
              f"{(f.std() if f.size else float('nan')):11.4g}"
              f"  {'; '.join(notes)}")

    if args.compare:
        _, other = load(args.compare)
        print(f"\n   compare with {args.compare}")
        print(f"   {'port':18s}{'max |diff|':>14s}{'rms diff / rms':>17s}  verdict")
        worst = 0.0
        for name in sorted(set(arrays) & set(other)):
            a, b = arrays[name], other[name]
            if a.shape != b.shape:
                print(f"   {name:18s}shape {a.shape} vs {b.shape}  MISMATCH")
                worst = float("inf")
                continue
            d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
            ra = float(np.sqrt((a.astype(np.float64) ** 2).mean()))
            rel = d / ra if ra > 0 else d
            worst = max(worst, rel)
            print(f"   {name:18s}{d:14.4e}{rel:17.3e}")
        print(f"\n   worst relative difference {worst:.3e}")
        verdict = ("MATCH -- the two backends compute the same graph"
                   if worst < 1e-3 else "DIFFER -- the two paths are not the same computation")
        print(f"   verdict: {verdict}")
        return 0 if worst < 1e-3 else 1

    print("\n   relations the layer imposes on its own ports")
    rel = relations(arrays)
    for name, what, ok in rel:
        print(f"   {name:18s}{what:52s} {'ok' if ok else 'FAILED'}")
    problems += sum(1 for _, _, ok in rel if not ok)

    print("\nverdict: " + ("values look well formed"
                           if problems == 0 else f"{problems} problem(s) above"))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
