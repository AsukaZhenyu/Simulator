"""Explicit adapter for the already executed native preparation definitions."""
from llm_infer_model.performance.preparation_contract import PreparationContract


IMPLEMENTATION_SOURCES = {
    "refinement_probe.cpp": "a188e44503c5ed42d1072a96e1b9f4372594ce6b98a2adf0c2748c8793898c8f",
    "steady_costs/probe.cpp": "757f97bb37a27df65e5d7e4fa4ef5707e58391a576140971982529fdc2bf2d5a",
}


def refinement_preparation(request, *, source_hashes):
    # A schema or mode name alone does not prove which native code executed.
    if not isinstance(source_hashes, dict) or not all(
            source_hashes.get(k) == v for k, v in IMPLEMENTATION_SOURCES.items()):
        return PreparationContract()
    if not isinstance(request, dict) or request.get("schema") != "dac-refinement-request-1":
        return PreparationContract()
    sampling = request.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("schema") != "steady-cost-request-1":
        return PreparationContract()
    mode = request.get("mode")
    if mode == "steady" and sampling.get("mode") in {"full", "full-stamped"}:
        prime = 0  # steady_costs/probe.cpp:127,133-134
    elif mode == "observer" and sampling.get("mode") == "full":
        prime = 1  # refinement_probe.cpp:93, before begin tick
    else:
        return PreparationContract()
    return PreparationContract(reset_before_batch=True,
        untimed_complete_prime_passes=prime,
        timed_replays_per_batch=sampling.get("repeats"), fixed_input_replay=True,
        warmup_iterations_min=sampling.get("warmup"), warmup_ms_min=sampling.get("warmup_min_ms"))
