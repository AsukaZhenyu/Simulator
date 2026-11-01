"""Finite request ordering and P/D handoff declarations share one runtime."""
from dataclasses import replace
import unittest

from llm_infer_model.m2.causal import DevicePrice, PublicOperation
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.request_scheduling import (
    PhaseHandoff, RequestPipeline, normalize_handoffs, order_request_operations,
)


def work(identifier, request, phase):
    return PublicOperation(identifier, "attention", "NOT_STARTED", (), (),
        (DevicePrice("cpu", 17, 0, 0),), request_id=request, phase=phase)


class RequestPipelineDeclarationTests(unittest.TestCase):
    def test_exact_roundtrip_keeps_independent_kernels_and_shared_resource_scope(self):
        for rule in ("graph_order", "decode_first", "round_robin"):
            declaration = RequestPipeline(ready_rule=rule)
            document = declaration.document()
            self.assertEqual(document["batch_accounting"], "independent_request_kernels")
            self.assertEqual(document["resource_scope"], "shared_cpu_gpu")
            self.assertEqual(RequestPipeline.from_document(document), declaration)

    def test_no_implicit_fusion_or_private_arrival_ordering(self):
        for rule in ("future_shortest_request", True, None, ["decode_first"]):
            with self.subTest(rule=rule), self.assertRaises((InvalidInput, Unsupported)):
                RequestPipeline(ready_rule=rule)
        for accounting in ("fused_batch_kernel", "whole_batch_cost_per_request", None):
            with self.subTest(accounting=accounting), self.assertRaises((InvalidInput, Unsupported)):
                RequestPipeline(batch_accounting=accounting)

    def test_strict_parser_rejects_unknown_missing_version_and_resource_fields(self):
        good = RequestPipeline().document()
        variants = []
        for key in good:
            changed = dict(good)
            del changed[key]
            variants.append(changed)
        variants.append(dict(good, future_arrivals=["r3"]))
        variants.append(dict(good, format_version="request-pipeline-99"))
        variants.append(dict(good, resource_scope="separate_pool_per_request"))
        for document in variants:
            with self.subTest(document=document), self.assertRaises((InvalidInput, Unsupported)):
                RequestPipeline.from_document(document)


class PhaseHandoffDeclarationTests(unittest.TestCase):
    def test_roundtrip_and_all_requests_normalize_to_one_canonical_ownership_map(self):
        for destination in ("dram", "vram"):
            declaration = PhaseHandoff(destination)
            self.assertEqual(PhaseHandoff.from_document(declaration.document()), declaration)
            self.assertEqual(normalize_handoffs((declaration,), ("r1", "r0")),
                             (("r0", destination), ("r1", destination)))
        explicit = (PhaseHandoff("vram", ("r1",)), PhaseHandoff("dram", ("r0",)))
        self.assertEqual(normalize_handoffs(explicit, ("r0", "r1")), (("r0", "dram"), ("r1", "vram")))
        self.assertEqual(normalize_handoffs((), ("r0", "r1")), ())

    def test_unknown_and_overlapping_request_ownership_are_rejected(self):
        invalid = (
            (PhaseHandoff("dram", ("absent",)),),
            (PhaseHandoff("dram", ("r0",)), PhaseHandoff("vram", ("r0",))),
            (PhaseHandoff("dram"), PhaseHandoff("vram", ("r1",))),
            (PhaseHandoff("dram"), PhaseHandoff("vram")),
        )
        for declarations in invalid:
            with self.subTest(declarations=declarations), self.assertRaises(InvalidInput):
                normalize_handoffs(declarations, ("r0", "r1"))
        with self.assertRaises(InvalidInput):
            normalize_handoffs([PhaseHandoff("dram")], ("r0",))
        with self.assertRaises(InvalidInput):
            normalize_handoffs((object(),), ("r0",))

    def test_strict_identifiers_destination_and_boundary(self):
        for identities in (["r0"], ("r0", "r0"), ("",), ("r:0",), ("r\n0",), (True,)):
            with self.subTest(identities=identities), self.assertRaises(InvalidInput):
                PhaseHandoff("dram", identities)
        for destination in ("ssd", "cpu", "gpu", None, ["dram"]):
            with self.subTest(destination=destination), self.assertRaises((InvalidInput, Unsupported)):
                PhaseHandoff(destination)
        good = PhaseHandoff("dram", ("r0",)).document()
        for changed in (dict(good, destination="ssd"), dict(good, request_ids="r0"),
                        dict(good, boundary="kv_only_skip_hidden"), dict(good, unknown=0),
                        dict(good, format_version="phase-handoff-99")):
            with self.subTest(document=changed), self.assertRaises((InvalidInput, Unsupported)):
                PhaseHandoff.from_document(changed)


class PublicRequestOrderingTests(unittest.TestCase):
    def setUp(self):
        # This immutable tuple is all the caller knows. No future request list,
        # private tensor graph or new compute kernel is supplied to ordering.
        self.operations = (work("a.prefill.0", "a", "prefill"), work("b.decode.0", "b", "decode"),
                           work("a.prefill.1", "a", "prefill"), work("c.decode.0", "c", "decode"),
                           work("b.decode.1", "b", "decode"))

    def test_graph_order_is_identity_and_invalid_spec_is_a_contract_error(self):
        for declaration in (None, RequestPipeline()):
            ordered = order_request_operations(self.operations, declaration)
            self.assertEqual(ordered, self.operations)
            self.assertTrue(all(a is b for a, b in zip(ordered, self.operations)))
        with self.assertRaises(InvalidInput):
            order_request_operations(self.operations, object())

    def test_decode_first_is_stable_and_cannot_invent_or_mutate_public_work(self):
        ordered = order_request_operations(self.operations, RequestPipeline("decode_first"))
        self.assertEqual([operation.id for operation in ordered],
                         ["b.decode.0", "c.decode.0", "b.decode.1", "a.prefill.0", "a.prefill.1"])
        self.assertEqual(set(map(id, ordered)), set(map(id, self.operations)))
        self.assertEqual(tuple(op.id for op in self.operations),
                         ("a.prefill.0", "b.decode.0", "a.prefill.1", "c.decode.0", "b.decode.1"))
        # An unrevealed future request has no representation in the input. The
        # result consists exclusively of these exact same operation objects.
        self.assertNotIn("future", {operation.request_id for operation in ordered})

    def test_round_robin_is_deterministic_rotates_requests_and_keeps_internal_order(self):
        declaration = RequestPipeline("round_robin")
        cases = {
            None: ("a", "a", "b", "b", "c"),
            "a": ("b", "b", "c", "a", "a"),
            "b": ("c", "a", "a", "b", "b"),
            "c": ("a", "a", "b", "b", "c"),
            "absent-past-request": ("a", "a", "b", "b", "c"),
        }
        for last, expected in cases.items():
            with self.subTest(last=last):
                first = order_request_operations(self.operations, declaration, last_request=last)
                self.assertEqual(first, order_request_operations(self.operations, declaration, last_request=last))
                self.assertEqual(tuple(operation.request_id for operation in first), expected)
                self.assertEqual(set(map(id, first)), set(map(id, self.operations)))
                for request in ("a", "b", "c"):
                    self.assertEqual([op.id for op in first if op.request_id == request],
                                     [op.id for op in self.operations if op.request_id == request])

    def test_request_ordering_preserves_grouped_expert_priority_within_one_request(self):
        # Expert priorities are already sorted by the preceding expert hook.
        # Grouped experts share one public batch request, so stable request
        # ordering must not reverse that inner priority or duplicate its work.
        first = replace(work("expert-large", "__moe_batch__", "decode"), kind="expert_ffn", expert_id=1)
        second = replace(work("expert-small", "__moe_batch__", "decode"), kind="expert_ffn", expert_id=0)
        public = (self.operations[0], first, self.operations[1], second)
        for rule in ("graph_order", "decode_first", "round_robin"):
            ordered = order_request_operations(public, RequestPipeline(rule), last_request="a")
            self.assertLess(ordered.index(first), ordered.index(second))


class PublicCriticalPathTieBreakTests(unittest.TestCase):
    """A public bottom-level rank may order work only inside a rule's class."""

    def setUp(self):
        self.operations = (work("a.prefill.0", "a", "prefill"), work("b.decode.0", "b", "decode"),
                           work("a.prefill.1", "a", "prefill"), work("c.decode.0", "c", "decode"))
        self.priority = {"a.prefill.0": 1000, "a.prefill.1": 900,
                         "b.decode.0": 1, "c.decode.0": 2}

    def test_priority_never_promotes_a_prefill_past_a_ready_decode_class(self):
        ordered = order_request_operations(self.operations, RequestPipeline("decode_first"),
                                           priority=self.priority)
        self.assertEqual([operation.id for operation in ordered],
                         ["c.decode.0", "b.decode.0", "a.prefill.0", "a.prefill.1"])
        # Without the rank the declared class order is preserved exactly.
        plain = order_request_operations(self.operations, RequestPipeline("decode_first"))
        self.assertEqual([operation.id for operation in plain],
                         ["b.decode.0", "c.decode.0", "a.prefill.0", "a.prefill.1"])

    def test_round_robin_rotation_is_not_overridden_by_priority(self):
        ordered = order_request_operations(self.operations, RequestPipeline("round_robin"),
                                           last_request="a", priority=self.priority)
        self.assertEqual([operation.request_id for operation in ordered], ["b", "c", "a", "a"])
        identity = order_request_operations(self.operations, RequestPipeline("graph_order"),
                                            priority=self.priority)
        self.assertEqual(identity, self.operations)

    def test_rank_is_total_and_does_not_invent_or_drop_public_work(self):
        ordered = order_request_operations(self.operations, RequestPipeline("decode_first"),
                                           priority={})
        self.assertEqual(sorted(map(id, ordered)), sorted(map(id, self.operations)))
        self.assertEqual([operation.id for operation in ordered],
                         [operation.id for operation in self.operations
                          if operation.phase == "decode"] +
                         [operation.id for operation in self.operations
                          if operation.phase != "decode"])
        with self.assertRaises(InvalidInput):
            order_request_operations(self.operations, RequestPipeline("decode_first"), priority=[1])


    def test_a_zero_input_ready_gate_is_served_before_longer_decode_work(self):
        gate = PublicOperation("gate.decode", "state_update", "NOT_STARTED", (), (),
                               (DevicePrice("cpu", 1, 0, 0),), request_id="b", phase="decode")
        job = PublicOperation("job.decode", "attention", "NOT_STARTED", ("x",), ("y",),
                              (DevicePrice("cpu", 1, 0, 0),), request_id="b", phase="decode")
        ordered = order_request_operations((job, gate), RequestPipeline("decode_first"),
                                           priority={"gate.decode": 1, "job.decode": 5})
        self.assertEqual([operation.id for operation in ordered], ["gate.decode", "job.decode"])


if __name__ == "__main__":
    unittest.main()
