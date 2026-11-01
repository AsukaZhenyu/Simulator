"""Strategy-only declarations using the existing public library."""
from llm_simulator import Cache, Scheduling


def strategy(case, expert_bytes):
    if case == "dense_gpu":
        return Scheduling(default_device="gpu", release="last_use")
    if case == "cpu_attention":
        return Scheduling(default_device="gpu", placements={"attention": "cpu"}, release="last_use")
    if case == "expert_lru":
        return Scheduling(default_device="cpu", placements={"expert_ffn": "gpu"},
                          release="last_use", cache=Cache(rule="lru", unit="expert_group",
                                                       budget_bytes=2 * expert_bytes))
    raise ValueError(case)
