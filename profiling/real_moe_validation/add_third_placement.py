# -*- coding: utf-8 -*-
"""Add the third placement (cpu-attn-gpu-other) to the tap probe COPY.

The original tap_native/real_moe_tap_probe.cpp is not touched: every prior
observer build records the receipt of the source it was compiled from, and a
fresh chain is built from tap_native_attn/ instead.
"""
import hashlib
import io

BS = chr(92)          # one backslash
Q = chr(34)           # one double quote
FFN = Q + BS + BS + ".ffn_(up|down|gate|gate_up)_(ch|)exps" + Q
ATTN = Q + "blk" + BS + BS + "." + BS + BS + "d+" + BS + BS + ".attn_.*" + Q

PATH = "tap_native_attn/real_moe_tap_probe.cpp"

PAIRS = [
    # (1) admission: accept the third value, and remember which one it is.
    ('require(placement==' + Q + 'cpu' + Q + '||placement==' + Q + 'cpu-experts-gpu-other' + Q
     + ',' + Q + 'unknown placement' + Q + ');',
     'require(placement==' + Q + 'cpu' + Q + '||placement==' + Q + 'cpu-experts-gpu-other' + Q
     + '||placement==' + Q + 'cpu-attn-gpu-other' + Q + ',' + Q + 'unknown placement' + Q
     + ');const bool attn_on_cpu=placement==' + Q + 'cpu-attn-gpu-other' + Q + ';'),

    # (2) which weights stay on the host.
    ("overrides={{" + FFN + ",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};"
     "mp.tensor_buft_overrides=overrides.data();",
     "overrides=attn_on_cpu?"
     "std::vector<llama_model_tensor_buft_override>{{" + ATTN + ",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}}:"
     "std::vector<llama_model_tensor_buft_override>{{" + FFN + ",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};"
     "mp.tensor_buft_overrides=overrides.data();"),

    # (3) KQ/V and the scheduler hand-off. op_offload must be off: ggml-backend
    # hands host-resident ops to a higher-priority backend, and CUDA claims them
    # at batch >= 32 (GGML_OP_OFFLOAD_MIN_BATCH), which would copy the CPU
    # attention weights to the GPU on every prefill evaluation.
    ('cp.offload_kqv=placement!=' + Q + 'cpu' + Q + ';cp.op_offload=placement!=' + Q + 'cpu' + Q + ';',
     'cp.offload_kqv=placement!=' + Q + 'cpu' + Q + '&&!attn_on_cpu;'
     'cp.op_offload=placement!=' + Q + 'cpu' + Q + '&&!attn_on_cpu;'),
]


def main():
    s = io.open(PATH, encoding="utf-8", newline="").read()
    for before, after in PAIRS:
        n = s.count(before)
        if n != 1:
            raise SystemExit("%d matches for %r" % (n, before[:70]))
        s = s.replace(before, after, 1)
    io.open(PATH, "w", encoding="utf-8", newline="").write(s)
    print("wrote", PATH, hashlib.sha256(open(PATH, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
