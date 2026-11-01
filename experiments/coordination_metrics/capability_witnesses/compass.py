"""Counted native-operator binding and strategy declarations, not stock scheduler."""
from software_model.matmul import Matmul
from software_model.utils import Tensor, data_type_dict
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
    dtype = data_type_dict["fp32"]

    def native_check(op, tensors, context, device):
        calls = []
        tokens = context.get("n", 1)
        for tid in op["inputs"]:
            t = tensors[tid]
            if t["role"] == "weight" and t["ne"][1] > 1:
                k, n = t["ne"][:2]
                mm = Matmul(dtype)
                output = mm(Tensor([tokens, k], dtype), Tensor([k, n], dtype))
                estimate = mm.roofline_model(hardware[device])
                calls.append(dict(symbol="Matmul.__call__+roofline_model", weight=tid,
                                  output_shape=output.shape, native_seconds=estimate))
        return dict(operation=op["id"], device=device, calls=calls,
                    native_scope="matrix work only; remaining macro semantics supplied by common DAG")

    return run_declared_graph(graph, policy, native_check, hardware["vram_bytes"])
