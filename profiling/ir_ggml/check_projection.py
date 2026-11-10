#!/usr/bin/env python3
"""Recompute one attention block's projections from the same weights and compare.

`check_output.py` checks the invariants the layer imposes on itself (a softmax is a
distribution, a mask is a mask).  Those hold for any input, so they cannot tell a correct
projection from a plausible one.  This does the arithmetic again, independently:

    x -> rms_norm(input_norm) -> W_v                       -> new_value
                              -> W_k -> rms_norm -> k_norm -> rope -> new_key
                              -> W_q -> rms_norm -> q_norm -> rope -> query

The weights are not re-read from the GGUF by a private dequantiser, because a hand-written
block decoder that disagrees with the loader would look exactly like a bug in the graph.
They are dumped by the executor itself through ggml's own `to_float`:

    ir_runner --case <case> --gguf <file> --dump-dir <dir> --dump-weight k:<dir>/k.f32
    # and the same for v and q; the option takes one role per invocation

The comparison is a relative L2 error over the whole tensor.  It has to be: a per-element
relative error explodes on entries that happen to be near zero, and reports a 40% error on
pairs that carry 1e-7 of the row's energy.  What the numbers mean is set by the gap between
the two outcomes -- a correct path lands at 1e-3, and the two ways this can be structurally
wrong (no rope, rope turning the other way) land at 0.2 to 0.9.

    python check_projection.py --dump C:/tmp/ref --gguf model.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_weights import read_gguf                                    # noqa: E402


def load_ports(directory: Path):
    manifest = json.loads((directory / "ports.json").read_text(encoding="utf-8"))["ports"]

    def port(name):
        # ggml lays ne[0] out fastest; numpy's reshape makes the last axis fastest.
        return (np.frombuffer((directory / f"{name}.f32").read_bytes(), np.float32)
                .reshape(manifest[name]["ne"], order="F"))
    return manifest, port


def gguf_f32(gguf: Path, name: str, shape) -> np.ndarray:
    """A tensor the GGUF stores as plain F32 -- the norms; the projections are quantised."""
    _, tensors, data_off = read_gguf(gguf)
    n = int(np.prod(shape))
    _, tt, off = tensors[name]
    if tt != 0:
        raise SystemExit(f"{name}: ggml type {tt}, this reader only takes F32")
    with open(gguf, "rb") as f:
        f.seek(data_off + off)
        raw = f.read(n * 4)
    if len(raw) != n * 4:
        raise SystemExit(f"{name}: short read")
    return np.frombuffer(raw, np.float32).reshape(shape, order="F").astype(np.float64)


def rms_norm(a: np.ndarray, eps: float, axis: int = 0) -> np.ndarray:
    """ggml's `ggml_rms_norm`: divide by the root mean square, no mean subtracted."""
    return a / np.sqrt((a * a).mean(axis=axis, keepdims=True) + eps)


def rope_neox(a: np.ndarray, base: float, sign: int = +1) -> np.ndarray:
    """Rotate (2p, 2p+1) by `sign * t * base**(-2p/head_dim)`, as ggml's NEOX mode does."""
    hd, tokens = a.shape[0], a.shape[1]
    ang = sign * np.arange(tokens)[:, None] * base ** (-2.0 * np.arange(hd // 2)[None, :] / hd)
    c, s = np.cos(ang).T[:, :, None], np.sin(ang).T[:, :, None]
    out = np.empty_like(a)
    out[0::2] = a[0::2] * c - a[1::2] * s
    out[1::2] = a[0::2] * s + a[1::2] * c
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True, help="--dump-dir of the run")
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--norm-eps", type=float, default=1e-6)
    ap.add_argument("--rope-base", type=float, default=1e6)
    ap.add_argument("--positions", type=int, default=0,
                    help="token index of the first row of the shape (history)")
    ap.add_argument("--qk-norm", default="rms", choices=("rms", "none"),
                    help="`none` for a model whose block has no q/k norm (granite)")
    args = ap.parse_args(argv)

    _, port = load_ports(args.dump)
    H, HD, KVH = args.hidden, args.head_dim, args.kv_heads
    W = {r: np.fromfile(args.dump / f"{r}.f32", np.float32).reshape(ne, order="F").astype(np.float64)
         for r, ne in (("k", (H, KVH * HD)), ("v", (H, KVH * HD)), ("q", (H, args.heads * HD)))}

    x = port("x").reshape(H, -1).astype(np.float64)
    inw = gguf_f32(args.gguf, "blk.0.attn_norm.weight", (H,))
    xn = rms_norm(x, args.norm_eps) * inw[:, None]
    if args.qk_norm == "rms":
        kn = gguf_f32(args.gguf, "blk.0.attn_k_norm.weight", (HD,))
        qn = gguf_f32(args.gguf, "blk.0.attn_q_norm.weight", (HD,))
    else:
        kn = qn = None

    def proj(w, n_heads, norm_w, do_rope):
        # Rows of the projection are one head each, so the norm is applied per head.
        # Normalising across the whole projection instead is a plausible-looking mistake that
        # makes every head's magnitude wrong at once, so it is worth naming here.
        a = (w.T @ xn).reshape(n_heads, HD, -1)
        if norm_w is not None:
            a = rms_norm(a, args.norm_eps, axis=1) * norm_w[None, :, None]
        a = a.transpose(1, 2, 0)                          # [head_dim, tokens, heads]
        return rope_neox(a, args.rope_base) if do_rope else a

    ref = {"query": proj(W["q"], args.heads, qn, True),
           "new_key": proj(W["k"], KVH, kn, True),
           "new_value": proj(W["v"], KVH, None, False)}
    # The same arithmetic without RoPE, and with RoPE turning the other way.  These are the
    # two mistakes this check exists to catch, so they are printed beside the honest number:
    # the verdict means nothing unless right and wrong are far apart.
    wrong = {}
    for name, w, n_heads, norm_w in (("query", W["q"], args.heads, qn),
                                     ("new_key", W["k"], KVH, kn)):
        plain = proj(w, n_heads, norm_w, False)
        wrong[name] = (plain, rope_neox(plain, args.rope_base, -1))

    def rel_l2(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    print(f"{'port':12s}{'|g-ref|/|ref|':>16s}{'no rope':>12s}{'rope -1':>12s}"
          f"{'pair-mag':>12s}  verdict")
    ok = True
    for name, want in ref.items():
        got = port(name)[:, :, :, 0].astype(np.float64)
        if got.shape != want.shape:
            print(f"{name:12s}shape {got.shape} vs reference {want.shape}  MISMATCH")
            ok = False
            continue
        good = rel_l2(got, want) < 3e-2
        ok = ok and good
        norope, flipped = wrong.get(name, (float("nan"), float("nan")))
        # Pair magnitudes survive any rotation, so this column needs no rope model at all.
        def pm(a):
            return np.sqrt(a[0::2] ** 2 + a[1::2] ** 2)
        pm_rel = float(np.linalg.norm(pm(got) - pm(want)) / np.linalg.norm(pm(want)))
        print(f"{name:12s}{rel_l2(got, want):16.3e}{rel_l2(got, norope):12.3e}"
              f"{rel_l2(got, flipped):12.3e}{pm_rel:12.3e}  {'ok' if good else 'MISMATCH'}")
    print("\nverdict: " + ("the projections reproduce from the same weights; what is left "
                           "is the spread of the quantised matmul"
                           if ok else "the projections do NOT reproduce"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
