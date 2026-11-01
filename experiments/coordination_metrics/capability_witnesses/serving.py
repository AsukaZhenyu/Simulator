"""Counted stock interface binding plus explicit new execution/cache semantics."""
from serving.core.memory_model import calculate_sizes
from serving.core.config_builder import get_device
from serving.core.trace_generator import indexed_cols
from serving.core.gate_function import GateRouter
from serial_adapter import run_declared_graph


def strategy(case, expert_bytes):
    if case == "dense_gpu":
        return dict(default_device="gpu", placements={}, cache="none", budget_bytes=0)
    if case == "cpu_attention":
        return dict(default_device="gpu", placements={"attention": "cpu"}, cache="none", budget_bytes=0)
    if case == "expert_lru":
        return dict(default_device="cpu", placements={"expert_ffn": "gpu"}, cache="lru", budget_bytes=2 * expert_bytes)
    raise ValueError(case)


def run(graph, policy, hardware, model_path):
    placement = dict(default=dict(weights="REMOTE:0", kv_loc="LOCAL", kv_evict_loc="REMOTE:0"), block=[], layer={})

    def native_check(op, tensors, context, device):
        symbols = {"projection": "qkv_proj", "attention": "attention", "dense_ffn": "gate_up_proj", "expert_ffn": "moe"}
        sizes = None
        if op["semantic_op"] in symbols:
            sizes = calculate_sizes(model_path, symbols[op["semantic_op"]], context.get("n", 1),
                                    kv_len=context.get("past", 0), parallel=1, fp=4)
        weight_location = get_device(placement, context.get("l", 0), "moe", "weights")
        inputs = sum(tensors[t]["storage_bytes"] for t in op["inputs"] if tensors[t]["role"] != "weight")
        weights = sum(tensors[t]["storage_bytes"] for t in op["inputs"] if tensors[t]["role"] == "weight")
        outputs = sum(tensors[t]["storage_bytes"] for t in [op["output"], *op.get("extra_outputs", ())])
        row = (op["id"].replace(" ", "_"), "1", "LOCAL", str(inputs), weight_location,
               str(weights), "LOCAL", str(outputs), "NONE", "0", "NONE")
        parsed = list(indexed_cols([row]))[0]
        return dict(operation=op["id"], parsed_native_row=parsed, native_sizes=sizes, device=device,
                    device_source="new adapter policy; stock row is not a CPU placement declaration")

    return run_declared_graph(graph, policy, native_check, hardware["vram_bytes"])


def route_identity_control(expert_ids, experts):
    router = GateRouter("scope", 0, experts, num_experts_per_tok=1, routing_policy="CUSTOM", block_copy=False)
    router.routing_fn = lambda token, count, topk: list(expert_ids)
    result = router.route_ep(0, "current-disclosed-step", 1, 1)
    return dict(local_tokens=result.local_tokens, activated_experts=result.activated_experts,
                source_tokens=result.source_tokens)
