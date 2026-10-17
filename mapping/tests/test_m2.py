"""M2 (``format_version = "m2-layer-0.1"``) acceptance tests.

``M2_LAYER_ACCEPTANCE.md`` §2 is the table this file turns into assertions: one
class per section, one method per row, each naming its section. §1.1's refusals
are written as *pairs* -- the mutated document is refused **and** the same
document without that one field compiles -- so a refusal test can never pass
because the fixture itself rotted, and a check that quietly stopped firing would
have to take its accepted twin down with it.

Every number here is synthetic (``costs.source = "m2:synthetic_layer"``), so a
makespan in this file is a statement about the expansion and the scheduler, never
about any real hardware.

Three pieces of the contract carry most of the weight:

* the slice is ``full_causal + dense``, one request, one CPU. Every other
  combination is refused **by name** (``Unsupported``), not ignored, because a
  silently dropped field is how "we support SWA" becomes true by accident;
* the KV history is *structural*: the previous call's blocks enter as leaves with
  ``initial_locations`` and no producer, so "rewrite the whole cache on every
  decode" is not something the kernel can be talked into
  (``M2_LAYER_ACCEPTANCE.md`` §2's last sentence). The companion assertion is
  that the attention task *lists the blocks it reads* -- dropping them changes no
  makespan, no byte count and no ``next_position``, so only a structural
  assertion can catch it;
* ``state_out`` **is** a legal ``state_in`` -- the same type, and the prefill
  output is compared field by field against the decode fixture's ``state_in``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from llm_infer_model.m2 import (
    ATTENTION_SLIDING_WINDOW,
    FFN_MOE,
    PHASE_DECODE,
    PHASE_PREFILL,
    MacroShape,
    block_id,
    compile_invocation,
    expand_costs_document,
    hidden_bytes,
    load_and_compile,
    op_id,
    parse_cost_table,
    parse_state_document,
    plan_for,
    run_invocation,
    state_in_from_state_out,
    state_out_from_result,
    tensor_id,
    to_state_document,
)
from llm_infer_model.tensor.spec import build_scenario
from tensor_mapping import (
    Action,
    InvalidInput,
    MappingError,
    Unsupported,
    evaluate_mapping,
    initial_state,
    is_goal,
    scenario_fingerprint,
    transition,
    workload_fingerprint,
)
from tensor_mapping.engine import (
    ACTION_ADVANCE,
    ACTION_COMPUTE,
    ACTION_EVICT,
    CODE_INPUT_NOT_READY,
    CODE_LIVE_VALUE_LOSS,
    CODE_TENSOR_IN_USE,
    REASON_ACTIONS_REMAIN_AFTER_GOAL,
    REASON_GOAL_REACHED,
    STATUS_INVALID_MAPPING,
    STATUS_VALID,
    StateSnapshot,
    used_bytes,
)
from tensor_mapping.mapper import (
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    STATUS_UNKNOWN,
    search,
)
from tensor_mapping.spec import (
    COPY_ABSENT,
    COPY_READY,
    COPY_RESERVED_OUTPUT,
    LOC_DRAM,
    LOC_SSD,
    LOC_VRAM,
    ROLE_INPUT,
    ROLE_OUTPUT,
)
from tests.support import EXAMPLES, ms

M2_EXAMPLES = EXAMPLES / "m2"
LAYER_FILE = "fcd.layer.json"
COSTS_FILE = "fcd.costs.json"
PREFILL = "fcd-prefill-3"
DECODE = "fcd-decode-1"


# ---------------------------------------------------------------------------
# The three-document set, written to a temp dir so a mutation never touches the
# repo and the loader still resolves ``layer_file`` / ``costs_file`` from disk.
# ---------------------------------------------------------------------------


def _read(name: str) -> dict[str, Any]:
    return json.loads((M2_EXAMPLES / name).read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@contextmanager
def m2_documents(
    name: str = PREFILL,
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None] | None = None,
) -> Iterator[Path]:
    """A (layer, costs, invocation) triple in a temp dir; yields the invocation path.

    ``mutate`` receives *all three* documents, in that order: the M2 refusals are
    cross-document by construction (a block's bytes come from the layer, a cost
    entry from the table, the history from the invocation), and a helper that
    could only reach one of them would push those rows into near-identical JSON
    files.
    """
    layer_raw = _read(LAYER_FILE)
    costs_raw = _read(COSTS_FILE)
    invocation_raw = _read(f"{name}.json")
    if mutate is not None:
        mutate(layer_raw, costs_raw, invocation_raw)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
        directory = Path(raw_dir)
        _write(directory / LAYER_FILE, layer_raw)
        _write(directory / COSTS_FILE, costs_raw)
        target = directory / f"{name}.json"
        _write(target, invocation_raw)
        yield target


class M2Case(unittest.TestCase):
    """Compilation, replay and refusal helpers shared by the tables."""

    def compile(self, name: str = PREFILL, mutate=None):
        with m2_documents(name, mutate) as path:
            return load_and_compile(path)

    def invoke(self, name: str = PREFILL, mutate=None):
        """Compile and replay in one step (not ``run``: that is the runner's)."""
        return run_invocation(self.compile(name, mutate))

    def refuse(
        self,
        mutate,
        error_type: type[Exception],
        needle: str,
        name: str = PREFILL,
        accepted=None,
    ):
        """One mutation is refused, and an unmutated (or differently mutated) twin compiles.

        ``accepted`` is for rows that say "this *number* is wrong" rather than
        "this field is wrong": it is a mutation that must still compile, which is
        what proves the check is about the relation and not about the field's
        presence.
        """
        with self.assertRaises(error_type) as caught:
            self.compile(name, mutate)
        self.assertIn(needle, str(caught.exception))
        self.compile(name, accepted)
        return caught.exception

    # -- replay ------------------------------------------------------------

    def compute(self, compiled, task_id: str, request_id: str = "r0") -> Action:
        task = next(
            task
            for task in compiled.tasks_of_request(request_id)
            if task.task_id == task_id
        )
        return Action(ACTION_COMPUTE, task.operation_id, device=task.device)

    def replay(self, compiled, *actions: Action):
        state = initial_state(compiled.scenario)
        for action in actions:
            state = transition(compiled.scenario, state, action)
        return state

    def refusal(self, compiled, state, action: Action) -> MappingError:
        with self.assertRaises(MappingError) as caught:
            transition(compiled.scenario, state, action)
        return caught.exception

    # -- reading a replay ---------------------------------------------------

    def operation(self, compiled, request_id: str, task_id: str):
        operation_id = op_id(compiled.layer.id, request_id, task_id)
        for operation in compiled.scenario.workload.operations:
            if operation.id == operation_id:
                return operation
        raise AssertionError(f"展开里没有算子 {operation_id!r}")

    def tensor(self, compiled, tensor_id_value: str):
        return compiled.scenario.workload.tensor_by_id[tensor_id_value]

    def dram_status(self, state) -> dict[str, str]:
        return dict(state.dram_status)


# ---------------------------------------------------------------------------
# §1.1  The layer × invocation × history gates
# ---------------------------------------------------------------------------


class M2LayerValidation(M2Case):
    """``M2_LAYER_ACCEPTANCE.md`` §1.1: what is refused, and what it is refused by.

    Two of these rows are the *kernel's* gates rather than this layer's: the
    macro-task semantics and ``extra_outputs`` are only accepted under an M2
    ``origin.kind``. They are asserted here because the expansion is what puts
    those two things into a workload document, so this is the layer that would be
    the first to notice if the gate stopped firing.
    """

    def test_the_two_fixtures_compile(self):
        prefill = self.compile(PREFILL)
        decode = self.compile(DECODE)
        self.assertEqual(prefill.invocation.phase, PHASE_PREFILL)
        self.assertEqual(decode.invocation.phase, PHASE_DECODE)
        # The layer is a separate reusable document; both calls name the same one.
        self.assertEqual(prefill.layer.id, decode.layer.id)
        self.assertEqual(
            [task.task_id for task in prefill.tasks],
            ["projection", "attention", "dense_ffn"],
        )

    def test_a_sliding_window_layer_is_unsupported(self):
        def mutate(layer, costs, invocation):
            layer["attention"] = ATTENTION_SLIDING_WINDOW
            layer["attention_params"] = {"window": 4}

        self.refuse(mutate, Unsupported, "'full_causal'")

    def test_a_moe_layer_is_unsupported(self):
        def mutate(layer, costs, invocation):
            layer["ffn"] = FFN_MOE
            layer["ffn_params"] = {"experts": 4, "top_k": 1, "expert_weight_bytes": 16}

        self.refuse(mutate, Unsupported, "'dense'")

    def test_a_gpu_device_is_unsupported(self):
        def mutate(layer, costs, invocation):
            invocation["devices"] = ["gpu"]

        self.refuse(mutate, Unsupported, "'cpu'")

    def test_a_decode_with_more_than_one_token_is_refused(self):
        def mutate(layer, costs, invocation):
            invocation["requests"][0]["tokens"] = 2

        self.refuse(mutate, InvalidInput, "decode 的 q_i 必须是 1", name=DECODE)

    def test_a_position_start_that_disagrees_with_the_state_is_refused(self):
        def mutate(layer, costs, invocation):
            invocation["requests"][0]["position_start"] = 2

        self.refuse(mutate, InvalidInput, "不一致", name=DECODE)

    def test_a_call_without_a_state_record_is_refused(self):
        def mutate(layer, costs, invocation):
            invocation["state_in"]["requests"] = []

        self.refuse(mutate, InvalidInput, "没有 request 'r0' 的记录", name=DECODE)

    def test_a_state_record_for_an_absent_request_is_refused(self):
        def mutate(layer, costs, invocation):
            invocation["state_in"]["requests"].append(
                {"request_id": "r9", "next_position": 0, "blocks": []}
            )

        self.refuse(mutate, InvalidInput, "但这次调用没有它", name=DECODE)

    def test_a_discontinuous_history_is_refused(self):
        def mutate(layer, costs, invocation):
            invocation["state_in"]["requests"][0]["blocks"][1]["first_position"] = 5

        self.refuse(mutate, InvalidInput, "不连续", name=DECODE)

    def test_a_block_whose_bytes_disagree_with_the_layer_is_refused(self):
        def mutate(layer, costs, invocation):
            invocation["state_in"]["requests"][0]["blocks"][0]["storage_bytes"] = 8

        self.refuse(mutate, InvalidInput, "按每 token", name=DECODE)

    def test_the_same_block_id_twice_is_refused(self):
        def mutate(layer, costs, invocation):
            blocks = invocation["state_in"]["requests"][0]["blocks"]
            blocks[1]["id"] = blocks[0]["id"]

        # The id check fires before the coverage check, so this row is about the
        # duplicate name and not about the hole it leaves.
        self.refuse(mutate, InvalidInput, "出现了两次", name=DECODE)

    def test_a_task_graph_with_a_cycle_is_refused(self):
        def mutate(layer, costs, invocation):
            layer["tasks"][0]["depends_on"] = ["dense_ffn"]

        self.refuse(mutate, InvalidInput, "有环")

    def test_a_task_graph_without_the_three_required_kinds_is_refused(self):
        def mutate(layer, costs, invocation):
            # Relabelled, not deleted: deleting it would break the dependency
            # edge first, and this row is about the missing kind.
            layer["tasks"][1]["kind"] = "state_update"

        self.refuse(mutate, InvalidInput, "缺少")

    def test_an_invocation_naming_another_layer_is_refused(self):
        def mutate(layer, costs, invocation):
            invocation["layer"] = "other"

        self.refuse(mutate, InvalidInput, "而拿到的是")

    def test_a_missing_cost_entry_is_refused_with_the_shapes_that_exist(self):
        def mutate(layer, costs, invocation):
            costs["entries"] = [e for e in costs["entries"] if e["phase"] == "prefill"]

        error = self.refuse(
            mutate, Unsupported, "成本表里没有", name=DECODE, accepted=None
        )
        # A refusal has to say what does exist: "no entry" and "no entry *that I
        # could find*" lead to different next steps.
        self.assertIn("prefill/B1/q3/hist0", str(error))

    def test_a_workspace_total_that_disagrees_with_the_layer_is_refused(self):
        def disagree(layer, costs, invocation):
            layer["workspace_bytes"] = 8

        def agree(layer, costs, invocation):
            # The same declaration, with the table measuring it: 8 B on the
            # projection task of every request. This is the twin that proves the
            # check is about the serial scratch maximum, not a nonzero flag.
            layer["workspace_bytes"] = 8
            for entry in costs["entries"]:
                if entry["kind"] == "projection":
                    entry["workspace_bytes"] = 8

        self.refuse(disagree, InvalidInput, "相加会重复计费", accepted=agree)

    def test_a_macro_semantic_under_a_non_m2_origin_is_refused_by_the_kernel(self):
        compiled = self.compile(PREFILL)
        relabelled = dataclasses.replace(compiled.layer, provenance="synthetic_fixture")
        with self.assertRaises(Unsupported) as caught:
            compile_invocation(
                relabelled,
                compiled.invocation,
                compiled.cost_table,
                where="relabelled",
            )
        self.assertIn("is an M2 macro-task", str(caught.exception))

    def test_extra_outputs_on_a_non_macro_operation_is_refused_by_the_kernel(self):
        compiled = self.compile(PREFILL)
        workload = copy.deepcopy(compiled.workload_document)
        attention = next(
            operation
            for operation in workload["operations"]
            if operation["semantic_op"] == "attention"
        )
        self.assertTrue(attention["extra_outputs"])  # the premise of this row
        # A real, non-macro semantic op: lowercase would trip the "unknown
        # semantic_op" gate first, and this row is about the extras gate.
        attention["semantic_op"] = "MUL_MAT"
        attention["ggml_op"] = "MUL_MAT"
        with self.assertRaises(Unsupported) as caught:
            build_scenario(
                workload,
                copy.deepcopy(compiled.scenario_document),
                where="extra_outputs",
            )
        self.assertIn("extra_outputs is only meaningful", str(caught.exception))


# ---------------------------------------------------------------------------
# §2  Prefill: q = 3, zero history
# ---------------------------------------------------------------------------


class M2Prefill(M2Case):
    """``M2_LAYER_ACCEPTANCE.md`` §2: 10 ms, 48 B, hidden_out 3 tokens, next_position 3."""

    def test_the_makespan_is_ten_milliseconds(self):
        result = self.invoke(PREFILL)
        self.assertEqual(result.status, STATUS_VALID)
        self.assertEqual(result.makespan_ns, ms(10))
        self.assertEqual(result.evaluation.reason, REASON_GOAL_REACHED)

    def test_the_three_tasks_run_serially(self):
        result = self.invoke(PREFILL)
        # 0-2 / 2-5 / 5-10 ms. The plan alternates COMPUTE and ADVANCE: a COMPUTE
        # starts the task at the current time without moving it, an ADVANCE
        # completes it at the task's end time. So the three *end* times are 2, 5
        # and 10 ms, and the three intervals between them are the acceptance
        # table's three rows -- one CPU slot, no scheduling argument.
        self.assertEqual(
            [state.t for state in result.evaluation.states],
            [0, 0, ms(2), ms(2), ms(5), ms(5), ms(10)],
        )

    def test_the_three_tasks_end_at_two_five_and_ten(self):
        compiled = self.compile(PREFILL)
        ends = []
        state = initial_state(compiled.scenario)
        for task in compiled.tasks:
            state = transition(compiled.scenario, state, self.compute(compiled, task.task_id))
            state = transition(compiled.scenario, state, Action(ACTION_ADVANCE))
            ends.append(state.t)
        self.assertEqual(ends, [ms(2), ms(5), ms(10)])

    def test_the_three_new_blocks_total_48_bytes(self):
        result = self.invoke(PREFILL)
        blocks = result.compiled.new_blocks["r0"]
        self.assertEqual(
            [block.id for block in blocks],
            [block_id("fcd", "r0", position) for position in (0, 1, 2)],
        )
        self.assertEqual([block.storage_bytes for block in blocks], [16, 16, 16])
        self.assertEqual(result.state_out.total_kv_bytes("r0"), 48)

    def test_the_hidden_out_is_three_tokens(self):
        result = self.invoke(PREFILL)
        hidden = result.compiled.hidden_out_ids["r0"]
        self.assertEqual(hidden, tensor_id("fcd", "r0", "dense_ffn"))
        (ref,) = result.state_out.hidden
        self.assertEqual((ref.request_id, ref.first_position, ref.tokens), ("r0", 0, 3))
        self.assertEqual(ref.storage_bytes, hidden_bytes(result.compiled.layer, 3))
        self.assertEqual(ref.storage_bytes, 48)

    def test_next_position_is_three(self):
        result = self.invoke(PREFILL)
        (state,) = result.state_out.requests
        self.assertEqual(state.next_position, 3)
        self.assertEqual(
            [(block.first_position, block.tokens) for block in state.blocks],
            [(0, 1), (1, 1), (2, 1)],
        )

    def test_the_attention_reserves_its_hidden_and_every_block_atomically(self):
        compiled = self.compile(PREFILL)
        hidden_out = compiled.hidden_out_ids["r0"]
        after_projection = self.replay(
            compiled, self.compute(compiled, "projection"), Action(ACTION_ADVANCE)
        )
        before = self.dram_status(after_projection)
        # Nothing of this call's output exists yet: the blocks are materialised by
        # the attention task, not before it.
        for name in compiled.block_ids:
            self.assertEqual(before.get(name), COPY_ABSENT, name)
        self.assertEqual(before.get(hidden_out), COPY_ABSENT)

        running = self.replay(
            compiled,
            self.compute(compiled, "projection"),
            Action(ACTION_ADVANCE),
            self.compute(compiled, "attention"),
        )
        statuses = self.dram_status(running)
        # All three blocks plus the attention tensor appear together, as
        # RESERVED_OUTPUT: "reads history and appends" is one task with four
        # outputs, and half of them existing would mean it is two.
        for name in compiled.block_ids:
            self.assertEqual(statuses.get(name), COPY_RESERVED_OUTPUT, name)
        self.assertEqual(
            statuses.get(tensor_id("fcd", "r0", "attention")), COPY_RESERVED_OUTPUT
        )
        self.assertEqual(statuses.get(hidden_out), COPY_ABSENT)

    def test_the_peak_dram_ledger_is_the_whole_call(self):
        result = self.invoke(PREFILL)
        # 128 B of weights + 48 B of input + 48 B of output per task (three of
        # them) + 48 B of KV: nothing is evicted, so the peak is the sum.
        self.assertEqual(result.evaluation.peak_dram_bytes, 368)
        self.assertEqual(result.evaluation.peak_vram_bytes, 0)

    def test_nothing_is_transferred(self):
        result = self.invoke(PREFILL)
        evaluation = result.evaluation
        self.assertEqual((evaluation.h2d_bytes, evaluation.d2h_bytes), (0, 0))
        self.assertEqual(
            (evaluation.ssd_read_bytes, evaluation.ssd_write_bytes), (0, 0)
        )
        self.assertEqual(result.compiled.scenario.costs.transfer, {})

    def test_the_cost_provenance_never_claims_a_measurement(self):
        result = self.invoke(PREFILL)
        self.assertEqual(result.cost_provenance, "m2:synthetic_layer")
        self.assertEqual(
            result.compiled.scenario.workload.origin.kind, "synthetic_layer"
        )


# ---------------------------------------------------------------------------
# §2  Decode: q = 1, three blocks of history
# ---------------------------------------------------------------------------


class M2Decode(M2Case):
    """``M2_LAYER_ACCEPTANCE.md`` §2: the same 10 ms, but over a history of three."""

    def test_the_makespan_is_ten_milliseconds(self):
        result = self.invoke(DECODE)
        self.assertEqual(result.status, STATUS_VALID)
        self.assertEqual(result.makespan_ns, ms(10))

    def test_the_two_calls_cost_twenty_milliseconds_together(self):
        prefill = self.invoke(PREFILL)
        decode = self.invoke(DECODE)
        self.assertEqual(prefill.makespan_ns + decode.makespan_ns, ms(20))

    def test_the_state_covers_positions_zero_to_three(self):
        result = self.invoke(DECODE)
        (state,) = result.state_out.requests
        self.assertEqual(
            [(block.first_position, block.tokens) for block in state.blocks],
            [(0, 1), (1, 1), (2, 1), (3, 1)],
        )
        self.assertEqual(state.next_position, 4)
        self.assertEqual(result.state_out.total_kv_bytes("r0"), 64)

    def test_only_the_fourth_block_is_appended(self):
        result = self.invoke(DECODE)
        (block,) = result.compiled.new_blocks["r0"]
        self.assertEqual(block.id, block_id("fcd", "r0", 3))
        self.assertEqual(block.first_position, 3)
        self.assertEqual(block.storage_bytes, 16)
        known = {entry.id for entry in result.compiled.state_in.for_request("r0").blocks}
        self.assertEqual(known, {block_id("fcd", "r0", p) for p in (0, 1, 2)})
        self.assertNotIn(block.id, known)

    def test_the_attention_reads_the_first_three_blocks(self):
        compiled = self.compile(DECODE)
        operation = self.operation(compiled, "r0", "attention")
        for position in (0, 1, 2):
            self.assertIn(block_id("fcd", "r0", position), operation.inputs)
        # And it does not read the block it is about to produce.
        self.assertNotIn(block_id("fcd", "r0", 3), operation.inputs)

    def test_the_old_blocks_are_leaves_that_cannot_be_rewritten(self):
        compiled = self.compile(DECODE)
        produced = {
            name for operation in compiled.scenario.workload.operations
            for name in operation.outputs
        }
        for position in (0, 1, 2):
            name = block_id("fcd", "r0", position)
            tensor = self.tensor(compiled, name)
            self.assertEqual(tensor.role, ROLE_INPUT)
            self.assertEqual(tensor.initial_locations, (LOC_DRAM,))
            # A leaf has no producer, so "rewrite the whole 48 B" is structurally
            # impossible rather than merely forbidden.
            self.assertNotIn(name, produced)
        self.assertEqual(self.tensor(compiled, block_id("fcd", "r0", 3)).role, ROLE_OUTPUT)

    def test_deliverables_include_hidden_and_all_retained_full_causal_kv(self):
        compiled = self.compile(DECODE)
        self.assertEqual(
            set(compiled.scenario.goal.output_locations),
            {
                compiled.hidden_out_ids["r0"],
                *(block_id("fcd", "r0", position) for position in range(4)),
            },
        )

    def test_the_peak_dram_ledger_is_the_whole_decode(self):
        result = self.invoke(DECODE)
        # 128 B of weights + 16 B of input + 16 B of output per task (three of
        # them) + 64 B of KV.
        self.assertEqual(result.evaluation.peak_dram_bytes, 256)
        self.assertEqual(result.evaluation.peak_vram_bytes, 0)


# ---------------------------------------------------------------------------
# §2  Continuity: the output of one call is the input of the next
# ---------------------------------------------------------------------------


class M2StateContinuity(M2Case):
    """``M2_LAYER_ACCEPTANCE.md`` §2 last sentence, plus the three ledgers."""

    def test_the_prefill_output_is_exactly_the_decode_input(self):
        prefill = self.invoke(PREFILL)
        document = to_state_document(prefill.state_out)
        # The checked-in decode fixture *is* the prefill's golden output; if the
        # expansion ever stopped producing a legal state_in, this comparison is
        # where it would show up as a field-by-field difference.
        self.assertEqual(document["requests"], _read(f"{DECODE}.json")["state_in"]["requests"])
        self.assertEqual(document["layer"], "fcd")

    def test_the_replica_locations_pass_from_output_to_input(self):
        prefill = self.invoke(PREFILL)
        decode = self.compile(DECODE)
        carried = state_in_from_state_out(prefill.state_out)
        self.assertEqual(carried.requests, decode.state_in.requests)
        for state in carried.requests:
            for block in state.blocks:
                # Not "dram because the fixture says so": the locations the
                # kernel recorded are the ones the next call starts from.
                self.assertEqual(block.locations, (LOC_DRAM,))

    def test_the_version_counts_the_calls_that_extended_the_state(self):
        prefill = self.invoke(PREFILL)
        decode = self.invoke(DECODE)
        self.assertEqual(prefill.state_out.requests[0].version, 2)
        self.assertEqual(decode.state_out.requests[0].version, 3)
        self.assertEqual(prefill.state_out.version, 2)
        self.assertEqual(decode.state_out.version, 3)

    def test_the_state_out_round_trips_through_its_document(self):
        result = self.invoke(DECODE)
        document = to_state_document(result.state_out)
        parsed = parse_state_document(document)
        self.assertEqual(parsed.layer, result.state_out.layer)
        self.assertEqual(parsed.version, result.state_out.version)
        self.assertEqual(parsed.requests, result.state_out.requests)
        self.assertEqual(parsed.hidden, result.state_out.hidden)

    def test_the_three_ledgers_are_read_in_a_fixed_order(self):
        compiled = self.compile(DECODE)
        evaluation = run_invocation(compiled).evaluation
        hidden = compiled.hidden_out_ids["r0"]
        snapshot = StateSnapshot(
            t=ms(10),
            used_vram_bytes=0,
            op_status=(),
            # The VRAM ledger is the field called ``copy_status`` (that is what
            # 0.1 named it), which is exactly why a reader of the artefact needs
            # ``statuses(location)`` rather than the field names.
            copy_status=(
                (block_id("fcd", "r0", 0), COPY_READY),
                (block_id("fcd", "r0", 3), COPY_READY),
                (hidden, COPY_READY),
            ),
            used_dram_bytes=0,
            dram_status=(
                (block_id("fcd", "r0", 1), COPY_READY),
                (block_id("fcd", "r0", 3), COPY_READY),
            ),
            ssd_status=(
                (block_id("fcd", "r0", 1), COPY_READY),
                (block_id("fcd", "r0", 2), COPY_READY),
                (block_id("fcd", "r0", 3), COPY_READY),
            ),
        )
        state_out = state_out_from_result(
            compiled, dataclasses.replace(evaluation, states=(snapshot,))
        )
        locations = {
            block.id: block.locations for block in state_out.requests[0].blocks
        }
        self.assertEqual(locations[block_id("fcd", "r0", 0)], (LOC_VRAM,))
        self.assertEqual(locations[block_id("fcd", "r0", 1)], (LOC_DRAM, LOC_SSD))
        self.assertEqual(locations[block_id("fcd", "r0", 2)], (LOC_SSD,))
        self.assertEqual(
            locations[block_id("fcd", "r0", 3)], (LOC_DRAM, LOC_VRAM, LOC_SSD)
        )
        self.assertEqual(state_out.hidden[0].locations, (LOC_VRAM,))

    def test_a_block_that_is_ready_nowhere_is_refused(self):
        compiled = self.compile(DECODE)
        evaluation = run_invocation(compiled).evaluation
        snapshot = dataclasses.replace(
            evaluation.states[-1],
            copy_status=tuple(
                pair
                for pair in evaluation.states[-1].copy_status
                if pair[0] != block_id("fcd", "r0", 2)
            ),
            dram_status=tuple(
                pair
                for pair in evaluation.states[-1].dram_status
                if pair[0] != block_id("fcd", "r0", 2)
            ),
            ssd_status=tuple(
                pair
                for pair in evaluation.states[-1].ssd_status
                if pair[0] != block_id("fcd", "r0", 2)
            ),
        )
        with self.assertRaises(InvalidInput) as caught:
            state_out_from_result(
                compiled, dataclasses.replace(evaluation, states=(snapshot,))
            )
        self.assertIn("都不是 READY", str(caught.exception))

    def test_a_replay_that_missed_the_goal_has_no_state_out(self):
        compiled = self.compile(DECODE)
        plan = plan_for(compiled)
        unfinished = evaluate_mapping(
            compiled.scenario, plan[:-1]
        )  # the last task never advanced
        self.assertNotEqual(unfinished.status, STATUS_VALID)
        with self.assertRaises(InvalidInput) as caught:
            state_out_from_result(compiled, unfinished)
        self.assertIn("没有可读的 state_out", str(caught.exception))


# ---------------------------------------------------------------------------
# §2  Replay and search over the compiled scenario
# ---------------------------------------------------------------------------


class M2ReplayAndSearch(M2Case):
    """The hand plan, the search, and the read lock the first cut keeps open."""

    def test_the_hand_plan_reaches_the_goal(self):
        compiled = self.compile(PREFILL)
        plan = plan_for(compiled)
        result = evaluate_mapping(compiled.scenario, plan)
        self.assertEqual(result.status, STATUS_VALID)
        self.assertEqual(result.makespan_ns, ms(10))
        # ``is_goal`` is about a live state, not a snapshot: replay the same plan
        # by hand and ask the engine whether the ledger is where the goal says.
        self.assertTrue(is_goal(compiled.scenario, self.replay(compiled, *plan)))

    def test_an_action_after_the_goal_is_reported_as_such(self):
        compiled = self.compile(PREFILL)
        plan = (*plan_for(compiled), Action(ACTION_ADVANCE))
        result = evaluate_mapping(compiled.scenario, plan)
        self.assertEqual(result.status, STATUS_INVALID_MAPPING)
        self.assertIn(REASON_ACTIONS_REMAIN_AFTER_GOAL, result.reason)

    def test_the_serial_plan_is_optimal(self):
        compiled = self.compile(PREFILL)
        found = search(compiled.scenario)
        self.assertEqual(found.status, STATUS_OPTIMAL)
        self.assertEqual(found.makespan_ns, ms(10))
        self.assertEqual(found.makespan_ns, run_invocation(compiled).makespan_ns)

    def test_a_budgeted_search_only_reports_feasible_or_unknown(self):
        compiled = self.compile(DECODE)
        result = search(compiled.scenario, max_expanded_states=1)
        self.assertIn(result.status, (STATUS_FEASIBLE, STATUS_UNKNOWN))
        self.assertNotEqual(result.status, STATUS_INFEASIBLE)
        # A plan the search *did* return cannot beat the single CPU slot's
        # 2 + 3 + 5 ms, and a budget cut must not come back as a number.
        if result.makespan_ns is not None:
            self.assertEqual(result.makespan_ns, ms(10))
        else:
            self.assertEqual(result.status, STATUS_UNKNOWN)

    def test_no_transfer_route_is_open(self):
        compiled = self.compile(DECODE)
        self.assertEqual(
            compiled.scenario.mapspace.allowed_transfers,
            {"dram_to_vram": (), "vram_to_dram": (), "ssd_to_dram": (), "dram_to_ssd": ()},
        )

    def test_old_block_is_read_locked_and_then_retained_for_next_call(self):
        compiled = self.compile(DECODE)
        running = self.replay(
            compiled,
            self.compute(compiled, "projection"),
            Action(ACTION_ADVANCE),
            self.compute(compiled, "attention"),
        )
        block = block_id("fcd", "r0", 0)
        error = self.refusal(
            compiled, running, Action(ACTION_EVICT, block, location=LOC_DRAM)
        )
        self.assertEqual(error.code, CODE_TENSOR_IN_USE)
        # Attention no longer reads it, but full causal state must survive for
        # the next invocation. An unrelated consumed activation remains legally
        # evictable, proving that retention did not disable eviction globally.
        done = self.replay(
            compiled,
            self.compute(compiled, "projection"),
            Action(ACTION_ADVANCE),
            self.compute(compiled, "attention"),
            Action(ACTION_ADVANCE),
        )
        retained = self.refusal(compiled, done, Action(ACTION_EVICT, block, location=LOC_DRAM))
        self.assertEqual(retained.code, CODE_LIVE_VALUE_LOSS)
        freed = transition(
            compiled.scenario, done, Action(ACTION_EVICT, tensor_id("fcd", "r0", "in"), location=LOC_DRAM)
        )
        self.assertEqual(used_bytes(compiled.scenario, freed, LOC_DRAM) + 16,
                         used_bytes(compiled.scenario, done, LOC_DRAM))

    def test_a_block_is_needed_before_it_exists(self):
        compiled = self.compile(PREFILL)
        initial = initial_state(compiled.scenario)
        error = self.refusal(
            compiled, initial, self.compute(compiled, "attention")
        )
        self.assertEqual(error.code, CODE_INPUT_NOT_READY)

    def test_the_compilation_is_deterministic(self):
        first = self.compile(PREFILL)
        second = self.compile(PREFILL)
        self.assertEqual(first.workload_document, second.workload_document)
        self.assertEqual(
            workload_fingerprint(first.scenario.workload),
            workload_fingerprint(second.scenario.workload),
        )
        self.assertEqual(
            scenario_fingerprint(first.scenario), scenario_fingerprint(second.scenario)
        )


# ---------------------------------------------------------------------------
# The cost table's physical format: the one place it is allowed to appear
# ---------------------------------------------------------------------------


class M2CostTableFormat(M2Case):
    """``M2_LAYER_DESIGN.md`` §3: the parse layer, and 门 2 of the cost bridge.

    This is the module that will change when codex's measured table lands
    (``costs.py`` says so), so its refusals are asserted here rather than only
    through a compiling document: a shape that is *missing* must be refused
    rather than interpolated, and a duration must not survive a round trip
    through floating-point seconds as 0 ns -- "this task takes no time" is a
    false fact, not a small number.
    """

    def costs_document(self) -> dict[str, Any]:
        """A fresh copy of the fixture table, ready to be mutated."""
        return _read(COSTS_FILE)

    def prefill_shape(self) -> MacroShape:
        return MacroShape(phase=PHASE_PREFILL, batch=1, tokens=3, visible_history=0)

    def refuse_table(self, mutate, error_type: type[Exception], needle: str):
        """One field of the table is refused, and the table without it parses.

        The twin is the same table with only that one field restored, so a row
        cannot pass because the fixture stopped being a legal table.
        """
        raw = self.costs_document()
        mutate(raw)
        with self.assertRaises(error_type) as caught:
            parse_cost_table(raw, where="costs")
        self.assertIn(needle, str(caught.exception))
        parse_cost_table(self.costs_document(), where="costs")
        return caught.exception

    def test_a_duration_in_seconds_is_the_same_number(self):
        raw = self.costs_document()
        for entry in raw["entries"]:
            entry["duration_s"] = entry.pop("duration_ns") / 1e9
        table = parse_cost_table(raw, where="costs")
        # 2/3/5 ms, spelled in the other allowed unit: the two spellings are one
        # number, and this is the only conversion point in the repository.
        self.assertEqual(
            [table.lookup(kind, "cpu", self.prefill_shape()).duration_ns
             for kind in ("projection", "attention", "dense_ffn")],
            [ms(2), ms(3), ms(5)],
        )

    def test_a_duration_that_rounds_away_is_refused(self):
        def mutate(raw):
            raw["entries"][0].pop("duration_ns")
            raw["entries"][0]["duration_s"] = 4e-10

        # 门 2 of the cost bridge: rounding to 0 ns would make "this task takes no
        # time" a fact instead of an error.
        self.refuse_table(mutate, InvalidInput, "无法表达")

    def test_both_duration_spellings_at_once_are_refused(self):
        def mutate(raw):
            raw["entries"][0]["duration_s"] = 0.002

        self.refuse_table(mutate, InvalidInput, "恰好给一处")

    def test_a_missing_duration_is_refused(self):
        def mutate(raw):
            raw["entries"][0].pop("duration_ns")

        self.refuse_table(mutate, InvalidInput, "恰好给一处")

    def test_workspace_must_be_declared_per_entry(self):
        def mutate(raw):
            raw["entries"][0].pop("workspace_bytes")

        # Not "absent means zero": absent means nobody said, and a number nobody
        # said is how a memory account comes out too small.
        self.refuse_table(mutate, InvalidInput, "必须给出")

    def test_an_unknown_entry_field_is_refused(self):
        def mutate(raw):
            raw["entries"][0]["duration_ms"] = 2

        self.refuse_table(mutate, InvalidInput, "未知字段")

    def test_the_same_shape_twice_is_refused(self):
        def mutate(raw):
            raw["entries"].append(copy.deepcopy(raw["entries"][0]))

        self.refuse_table(mutate, InvalidInput, "重复")

    def test_a_number_without_a_provenance_is_refused(self):
        def mutate(raw):
            raw.pop("provenance")
            for entry in raw["entries"]:
                entry.pop("provenance", None)

        self.refuse_table(mutate, InvalidInput, "每个数字都要说明它从哪来")

    def test_the_source_label_is_the_union_of_provenances(self):
        raw = self.costs_document()
        # This unit test exercises provenance union only; integration tests
        # validate actual macro geometry and storage boundaries.
        signature = {"test_fixture": "source_union_only"}
        raw["validation_only"] = False
        from llm_infer_model.m2.costs import canonical_contract_hash
        raw["entries"][0].update(provenance="measured_macro", accounting="per_request",
            physical_signature=signature, contract_hash=canonical_contract_hash(signature),
            workspace_scope="macro_internal_scratch", workspace_source="unit_test_declared")
        table = parse_cost_table(raw, where="costs")
        shape = self.prefill_shape()
        source, document = expand_costs_document(
            table,
            [
                ("op:fcd:r0:projection", "projection", "cpu", shape),
                ("op:fcd:r0:attention", "attention", "cpu", shape),
            ],
            physical_signatures={"op:fcd:r0:projection": signature},
        )
        # One measured entry in a table of synthetic ones makes the whole label
        # mixed: a partly measured table must not be reported as measured, and a
        # synthetic table must not be able to hide behind it either.
        self.assertEqual(source, "m2:measured_macro+synthetic_layer")
        self.assertEqual(document["source"], source)
        self.assertEqual(document["transfer"], {})
        self.assertEqual(
            document["compute"]["op:fcd:r0:projection"]["cpu"]["duration_ns"], ms(2)
        )


if __name__ == "__main__":
    unittest.main()
