#!/usr/bin/env python3
"""Recompute, from the GGUF alone, the checksum the executor reported for the weights it loaded.

"Loading the real weights did not change the time" is also what "loading nothing" looks like.
This closes that gap without trusting the loader: it opens the same file, picks the same
tensors in the same order, hashes the same byte ranges, and compares the result with the
`weights.fnv64` the executor wrote into its result JSON.

The order is the executor's: `std::map<std::string, ggml_tensor *>` iterates role names
alphabetically, and a role the model does not have is skipped when the graph never reads it
(`q_norm` / `k_norm` under `qk_norm none`, which is granite).

    python verify_weights.py --gguf model.gguf --result out.json [--case case.txt]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

FNV_OFFSET = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3

# ggml type -> (block elements, block bytes).  Only the ones a Q4_K_M file uses are listed;
# an unknown type is an error, not a guess.
TYPE_LAYOUT = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
}

# Same table as `weight_tensor_name` in ir_runner.cpp; a difference here would be caught by
# the checksum disagreeing, which is the point of keeping them independent.
ROLE_TO_TENSOR = {
    "input_norm": "blk.0.attn_norm.weight",
    "post_attention_norm": "blk.0.ffn_norm.weight",
    "final_norm": "output_norm.weight",
    "q_norm": "blk.0.attn_q_norm.weight",
    "k_norm": "blk.0.attn_k_norm.weight",
    "q": "blk.0.attn_q.weight",
    "k": "blk.0.attn_k.weight",
    "v": "blk.0.attn_v.weight",
    "output": "blk.0.attn_output.weight",
    "gate": "blk.0.ffn_gate.weight",
    "up": "blk.0.ffn_up.weight",
    "down": "blk.0.ffn_down.weight",
    "router": "blk.0.ffn_gate_inp.weight",
    "gate_exps": "blk.0.ffn_gate_exps.weight",
    "up_exps": "blk.0.ffn_up_exps.weight",
    "down_exps": "blk.0.ffn_down_exps.weight",
    "gate_inp_shexp": "blk.0.ffn_gate_inp_shexp.weight",
    "gate_shexp": "blk.0.ffn_gate_shexp.weight",
    "up_shexp": "blk.0.ffn_up_shexp.weight",
    "down_shexp": "blk.0.ffn_down_shexp.weight",
}

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
          10: "<Q", 11: "<q", 12: "<d"}


TYPE_NAME = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0",
             10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k"}


def tensor_name(key: str) -> str:
    """`L3:q` -> `blk.3.attn_q.weight`; a bare role stays the layer-0 name it always was.

    The executor's weight map is keyed exactly like this, so iterating the same keys in the
    same order (a string sort) reproduces its checksum; `L10:` really does sort before `L2:`,
    in both languages.
    """
    role = key.split(":", 1)[1] if ":" in key else key
    layer = int(key[1:key.index(":")]) if key.startswith("L") and ":" in key else 0
    name = ROLE_TO_TENSOR.get(role)
    if name is None:
        raise SystemExit(f"{key}: no gguf tensor for this role")
    if name.startswith("blk.0."):
        name = f"blk.{layer}." + name[len("blk.0."):]
    return name


def read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def read_val(f, t):
    if t == 8:
        return read_str(f)
    if t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_val(f, et) for _ in range(n)]
    return struct.unpack(SCALAR[t], f.read(struct.calcsize(SCALAR[t])))[0]


def read_gguf(path: Path) -> tuple[dict, dict, int]:
    """(key-values, {name: (ne, type, offset)}, offset of the tensor data blob)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a gguf file")
        struct.unpack("<I", f.read(4))                     # version
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        kv = {}
        for _ in range(n_kv):
            key = read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            kv[key] = read_val(f, t)
        tensors = {}
        for _ in range(n_tensors):
            name = read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            ne = list(struct.unpack("<" + "Q" * nd, f.read(8 * nd)))
            (tt,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            tensors[name] = (ne, tt, off)
        # The tensor blob starts at the next alignment boundary.
        align = int(kv.get("general.alignment", 32))
        data_off = (f.tell() + align - 1) // align * align
    return kv, tensors, data_off


def row_size(tt: int, ne0: int) -> int:
    if tt not in TYPE_LAYOUT:
        raise SystemExit(f"ggml type {tt}: not in the layout table, refusing to guess")
    blck, size = TYPE_LAYOUT[tt]
    if ne0 % blck:
        raise SystemExit(f"ne0 {ne0} is not a multiple of the block size {blck}")
    return size * ne0 // blck


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True, help="the executor's JSON")
    ap.add_argument("--case", type=Path, default=None,
                    help="the case that was run; a case that declares layers has one weight "
                         "per (layer, role) and the same file covers every block.  Without "
                         "it the layer-0 key set is assumed, which is what a case that "
                         "declares none allocates")
    args = ap.parse_args(argv)

    res = json.loads(args.result.read_text(encoding="utf-8"))
    w = res.get("weights") or {}
    if not w.get("real"):
        raise SystemExit(f"{args.result}: this run used the synthetic pattern, nothing to verify")
    expect = w["fnv64"]
    experts_loaded = int(w.get("experts_loaded") or 0)

    kv, tensors, data_off = read_gguf(args.gguf)
    arch = kv.get("general.architecture", "?")
    print(f"{args.gguf.name}  arch {arch}  tensors {len(tensors)}  data at {data_off}")

    fnv = FNV_OFFSET
    total = 0
    keys = weight_keys(args.case) if args.case else sorted(ROLE_TO_TENSOR)
    with open(args.gguf, "rb") as f:
        for key in keys:
            role = key.split(":", 1)[1] if ":" in key else key
            name = tensor_name(key)
            if name not in tensors:
                continue                      # the executor skips it too (norms it never reads)
            ne, tt, off = tensors[name]
            dims = list(ne) + [1] * (4 - len(ne))
            planes = dims[2]
            if expert_roles(role) and experts_loaded and dims[2] > 1:
                planes = min(experts_loaded, dims[2])
            n = row_size(tt, dims[0]) * dims[1] * planes * dims[3]
            f.seek(data_off + off)
            chunk = f.read(n)
            if len(chunk) != n:
                raise SystemExit(f"{name}: short read")
            for b in chunk:
                fnv ^= b
                fnv = (fnv * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
            total += n
            print(f"   {key:20s}{name:36s}{n:>12d} B")
    print(f"   bytes {total}   (executor said {w.get('bytes_real')})")
    got = f"{fnv:016x}"
    print(f"   fnv64 recomputed {got}   executor {expect}")
    ok = got == expect and total == w.get("bytes_real")
    print("verdict: " + ("MATCH -- the executor ran the file's bytes"
                         if ok else "MISMATCH -- the loader is not reading what it claims"))
    return 0 if ok else 1


def expert_roles(role: str) -> bool:
    return role.endswith("_exps") or role.endswith("_shexp")


def weight_keys(case: Path) -> list[str]:
    """The exact key set the executor allocated for this case, in its own sort order.

    A case that names no layers allocated one buffer per role; one that does allocated one
    per (layer, role) plus a single `final_norm`.  Reading the keys from the case rather
    than assuming them is what makes the checksum a test of the loader instead of a test of
    this script's idea of the loader.
    """
    layers, experts, shared = set(), 0, 0
    for line in case.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "experts":
            experts = int(words[1])
        elif words[0] == "shared":
            shared = int(words[1])
        elif words[0] == "node":
            for item in ",".join(words[5:]).split(","):
                if item.startswith("layer="):
                    layers.add(int(item.split("=", 1)[1]))
    if not layers:
        return sorted(ROLE_TO_TENSOR)
    roles = ["input_norm", "post_attention_norm", "q_norm", "k_norm", "q", "k", "v", "output"]
    roles += (["router", "gate_exps", "up_exps", "down_exps"] if experts
              else ["gate", "up", "down"])
    if experts and shared:
        roles += ["gate_inp_shexp", "gate_shexp", "up_shexp", "down_shexp"]
    return sorted([f"L{layer}:{role}" for layer in layers for role in roles] + ["final_norm"])


if __name__ == "__main__":
    sys.exit(main())
