"""Hand-specified bad declarations/actions; no production implementation edits."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import traceback

from experiments import strategy_fidelity as ref
from llm_simulator import Decision, InvalidInput, ModelPlan, Scheduling, TensorSelect, Unsupported
from llm_infer_model.tensor.runtime import dispatch_command
from tensor_mapping.causal import compile_command
from tensor_mapping.unified_strategy import create_session

ROOT = Path(__file__).resolve().parents[2]


def _partition(folder, mutation, expected_message):
    spec = ref.ffn_spec()
    mutation(spec)
    ref.write_json(folder / "bad-transform.json", spec)
    def configure(plan):
        plan.transform("layer0-prefill", spec)
    try:
        compiled = ref.dense().compile(directory=folder / "inputs",
            scheduling=Scheduling(default_device="gpu"), configure=configure)
    except InvalidInput as error:
        ref.require(expected_message in str(error), "incorrect declaration-rejection reason")
        return dict(exception=type(error).__name__, message=str(error), expected_message=expected_message)
    else:
        compiled.close()
        raise AssertionError("invalid partition/merge declaration silently compiled")


def n01(folder):
    return _partition(folder, lambda spec: spec["shards"][1].update(start=3), "without gaps or overlap")


def n02(folder):
    return _partition(folder, lambda spec: spec["shards"][1].update(end=15), "complete declared intermediate_size")


def n03(folder):
    return _partition(folder, lambda spec: spec.update(merge_input_count=3), "merge_input_count must equal len(shards)")


def n04(folder):
    with ref.dense().compile(directory=folder / "inputs", scheduling=Scheduling(default_device="gpu")) as compiled:
        session = create_session(compiled.plan._compiled)
        observation = session.observe()
        # Attention consumes projection outputs. The producer has not started,
        # so this tests a graph dependency, not just a missing weight upload.
        target = "g:layer0-prefill:op:layer0:request0:attn"
        producer = next(op for op in observation.operations
                        if op.id == "g:layer0-prefill:op:layer0:request0:proj")
        consumer = next(op for op in observation.operations if op.id == target)
        ref.equal(producer.status, "NOT_STARTED", "producer has not run")
        ref.require(set(producer.outputs).intersection(consumer.inputs), "test setup lacks a producer-consumer edge")
        decision = Decision("COMPUTE", target, device="cpu")
        ref.write_json(folder / "before.json", asdict(observation))
        step = dispatch_command(session, compile_command(decision, observation))
        ref.write_json(folder / "step.json", asdict(step))
        ref.require(not step.ok, "early compute was accepted")
        ref.equal(step.error_code, "INPUT_NOT_READY", "dependency readiness diagnostic")
        return dict(decision=asdict(decision), error_code=step.error_code, message=step.error_message)


def n05(folder):
    with ref.dense().compile(directory=folder / "inputs", scheduling=Scheduling(default_device="gpu")) as compiled:
        # A successful prefix is only setup. The oracle for the injected action
        # is the handwritten rule: an active input in its computing domain is locked.
        setup = compiled.run()
        ref.require(setup.success, "setup execution did not complete")
        session = create_session(compiled.plan._compiled)
        prefix = []
        for record in setup.raw[0].decisions:
            step = dispatch_command(session, compile_command(record.decision, session.observe()))
            ref.require(step.ok, "setup prefix could not replay")
            prefix.append(asdict(record.decision))
            if record.decision.kind == "COMPUTE":
                break
        before = session.observe()
        target = "w:layer0:W_input_norm"
        ref.require(any(item.resource == "gpu_compute" for item in before.running), "setup lacks running GPU compute")
        op = next(item for item in before.operations if item.id == prefix[-1]["target_id"])
        ref.equal(op.kind, "projection", "setup first compute kind")
        ref.require(target in op.inputs, "handwritten norm weight is not a current projection input")
        ref.write_json(folder / "setup-prefix.json", prefix)
        ref.write_json(folder / "before.json", asdict(before))
        decision = Decision("EVICT", target, location="vram")
        step = dispatch_command(session, compile_command(decision, before))
        ref.write_json(folder / "step.json", asdict(step))
        ref.require(not step.ok, "eviction of running input was accepted")
        ref.equal(step.error_code, "TENSOR_IN_USE", "active input lock diagnostic")
        return dict(decision=asdict(decision), error_code=step.error_code, setup_decisions=len(prefix))


def n06(folder):
    simulator = ref.dense(declared_hardware=replace(ref.hardware(), vram_capacity_bytes=1))
    with simulator.compile(directory=folder / "inputs", scheduling=Scheduling(default_device="gpu")) as compiled:
        session = create_session(compiled.plan._compiled)
        before = session.observe()
        target = "w:layer0:W_input_norm"
        tensor = next(item for item in before.tensors if item.id == target)
        ref.equal(tensor.storage_bytes, 32, "H=8 F32 normalization weight size")
        decision = Decision("COPY", target, src="dram", dst="vram")
        ref.write_json(folder / "before.json", asdict(before))
        step = dispatch_command(session, compile_command(decision, before))
        ref.write_json(folder / "step.json", asdict(step))
        ref.require(not step.ok, "32-byte copy fit into one byte")
        ref.equal(step.error_code, "CAPACITY_EXCEEDED", "physical capacity diagnostic")
        return dict(decision=asdict(decision), error_code=step.error_code,
                    global_infeasibility_proven=False)


def n07(folder):
    source = ROOT / "mapping/examples/composition/block.strategy.json"
    plan = ModelPlan.from_strategy(source)
    plan.transfer(engine="ce", select=TensorSelect(layer_id="L1"))
    plan.transfer(engine="sm", select=TensorSelect(stage="dense_ffn"))
    ref.write_json(folder / "declaration.json", plan.declaration_document())
    try:
        plan.compile()
    except InvalidInput as error:
        ref.require("ambiguous scoped transport" in str(error), "wrong reason for conflicting-rule rejection")
        return dict(exception=type(error).__name__, message=str(error), source=str(source))
    raise AssertionError("conflicting incomparable rules silently compiled")


def n08(folder):
    with ref.moe_simulator(2).compile(directory=folder / "inputs",
            scheduling=Scheduling(default_device="cpu")) as compiled:
        before = create_session(compiled.plan._compiled).observe()
        ref.equal(before.route_history, (), "no route is initially public")
        ref.require(not any(op.kind == "expert_ffn" for op in before.operations), "expert task leaked into observation")
        decision = Decision("COMPUTE", "g:layer0-decode0:op:layer0:__moe_batch__:expert.2", device="cpu")
        ref.write_json(folder / "before.json", asdict(before))
        try:
            compile_command(decision, before)
        except Unsupported as error:
            # This specific Unsupported means an unrevealed target, not an
            # unimplemented strategy. The reason and public precondition matter.
            ref.require("has not been revealed" in str(error), "wrong information-boundary rejection")
            return dict(exception=type(error).__name__, message=str(error), decision=asdict(decision))
        raise AssertionError("future private expert execution was accepted")


CASES = dict(zip((f"N{i:02d}" for i in range(1, 9)), (n01, n02, n03, n04, n05, n06, n07, n08)))


def run_group(output):
    output = Path(output)
    requirements = json.loads((Path(__file__).with_name("manifest.json")).read_text(encoding="utf-8"))["negative"]
    rows = []
    for requirement in requirements:
        identity = requirement["id"]
        folder = output / identity
        folder.mkdir(parents=True, exist_ok=False)
        ref.write_json(folder / "expected.json", requirement)
        try:
            evidence = CASES[identity](folder)
            row = dict(id=identity, status="pass", reason="Rejected for the independently required semantic reason.", evidence=evidence)
        except Exception as error:
            row = dict(id=identity, status="fail", reason=f"{type(error).__name__}: {error}",
                       evidence=dict(traceback=traceback.format_exc()))
        row["artifacts"] = str(folder)
        ref.write_json(folder / "result.json", row)
        rows.append(row)
    return rows
