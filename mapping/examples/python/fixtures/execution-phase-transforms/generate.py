"""Author explicit synthetic contracts for the combined P/D regression.

Phase durations below are independent constants, never scaled decode costs.
Transfer additions are indexed by actual F32 payload bytes; existing exact
prices remain unchanged. This generator does not claim hardware calibration.
"""
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
SIMULATOR = HERE.parents[4]
MODEL_FIXTURES = SIMULATOR / "modeling/tests/fixtures/global_execution"

# Independent stage declarations. Prefill constants were not computed from
# Decode times; both are toy contracts used to check composition and charging.
ATTENTION_COSTS = {
    "prefill": {"attention_qk": {"cpu": 211, "gpu": 181},
                "attention_softmax": {"cpu": 97, "gpu": 83},
                "attention_pv": {"cpu": 233, "gpu": 197},
                "attention_output": {"cpu": 307, "gpu": 271}},
    "decode": {"attention_qk": {"cpu": 67, "gpu": 53},
               "attention_softmax": {"cpu": 29, "gpu": 23},
               "attention_pv": {"cpu": 71, "gpu": 59},
               "attention_output": {"cpu": 89, "gpu": 79}},
}
FFN_COSTS = {
    "prefill": {("ffn_norm", "gpu", 32): 109,
                ("ffn_gate", "cpu", 8): 127, ("ffn_gate", "gpu", 24): 113,
                ("ffn_up", "cpu", 8): 131, ("ffn_up", "gpu", 24): 117,
                ("ffn_swiglu", "cpu", 8): 43, ("ffn_swiglu", "gpu", 24): 41,
                ("ffn_down", "cpu", 8): 139, ("ffn_down", "gpu", 24): 123,
                ("ffn_sum", "gpu", 32): 47, ("ffn_residual", "gpu", 32): 37},
    "decode": {("ffn_norm", "gpu", 32): 31,
               ("ffn_gate", "cpu", 8): 59, ("ffn_gate", "gpu", 24): 53,
               ("ffn_up", "cpu", 8): 61, ("ffn_up", "gpu", 24): 47,
               ("ffn_swiglu", "cpu", 8): 19, ("ffn_swiglu", "gpu", 24): 17,
               ("ffn_down", "cpu", 8): 67, ("ffn_down", "gpu", 24): 57,
               ("ffn_sum", "gpu", 32): 23, ("ffn_residual", "gpu", 32): 13},
}


def write(name, document):
    (HERE / name).write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")


def main():
    original = json.loads((MODEL_FIXTURES / "dense.global.json").read_text(encoding="utf-8"))
    original["id"] = "dense-two-layer-PD-attention-ffn-preparation-combined"
    for round_ in original["rounds"]:
        for invocation in round_["invocations"]:
            source = MODEL_FIXTURES / invocation["invocation_file"]
            invocation["invocation_file"] = Path(os.path.relpath(source, HERE)).as_posix()
    write("dense.global.json", original)
    write("environment.json", dict(format_version="environment-trace-0.1", events=[]))

    # Read only exact transfer entries, not old macro durations. Keep every
    # existing route/byte key and add only the genuinely new materializations.
    source = json.loads((MODEL_FIXTURES / "dense.costs.json").read_text(encoding="utf-8"))
    transfers = {(row["route"], row["storage_bytes"]): row["duration_ns"] for row in source["transfer_entries"]}
    # 144 = 3 keys * 3 queries * 4 query heads * 4 bytes.
    # 288 = 24 FFN columns * 3 tokens * 4 bytes.
    # 1536 = 16 hidden * 24 FFN columns * 4 bytes (one GPU matrix shard).
    for byte_count, duration in ((144, 17), (288, 23), (1536, 31)):
        for route in ("dram_to_vram", "vram_to_dram"):
            transfers.setdefault((route, byte_count), duration)
    transfer_rows = [dict(route=route, storage_bytes=byte_count, duration_ns=duration)
                     for (route, byte_count), duration in sorted(transfers.items())]
    for phase, tokens, history in (("prefill", 3, 0), ("decode", 1, 3)):
        attention = [dict(kind=kind, device=device, phase=phase, batch=1, tokens=tokens,
                          visible_history=history, duration_ns=duration, workspace_bytes=32)
                     for kind, devices in ATTENTION_COSTS[phase].items() for device, duration in devices.items()]
        write(f"attention.{phase}.json", dict(format_version="attention-stages-0.1", variant="qk_softmax_pv",
            provenance="synthetic_execution", geometry=dict(hidden=16, heads=4, kv_heads=2, head_dim=4,
                attention_gate="sigmoid"), frontend=dict(contract="existing_materialized_post_rope_qkv"),
            entries=attention, transfer_entries=transfer_rows))
        ffn = [dict(kind=kind, device=device, phase=phase, batch=1, tokens=tokens, visible_history=history,
                    hidden_size=16, intermediate_size=width, duration_ns=duration, workspace_bytes=16)
               for (kind, device, width), duration in FFN_COSTS[phase].items()]
        write(f"ffn.{phase}.json", dict(format_version="ffn-execution-0.1", target_kinds=["dense_ffn"],
            intermediate_size=32, shards=[dict(id="host", start=0, end=8, device="cpu"),
                                        dict(id="accelerator", start=8, end=32, device="gpu")],
            merge_input_count=2, preprocess_device="gpu", merge_device="gpu", provenance="synthetic_execution",
            weight_storage_contract="materialized_partition_replaces_parent", entries=ffn,
            transfer_entries=transfer_rows))


if __name__ == "__main__":
    main()
