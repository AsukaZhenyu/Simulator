"""Contract tests: what M0 accepts, and how it refuses the rest.

The three refusal kinds are exercised separately, because the contract gives
them different statuses and conflating them would lose information the caller
needs (``DESIGN.md`` §8).
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tensor_mapping import (
    InitialCapacityExceeded,
    InvalidInput,
    SCHEMA_VERSION,
    Unsupported,
    aligned_size,
    load_and_validate,
    load_workload,
    scenario_fingerprint,
    workload_fingerprint,
)
from tensor_mapping.spec import check_initial_capacity

from .support import EXAMPLES, documents, load_example, with_capacity

EXAMPLES_SCENARIOS = ("chain-cap160", "residual-cap160", "fork-cap160", "matvec-cap160")


class ExampleFixtureTests(unittest.TestCase):
    """The checked-in examples load, and say honestly where they came from."""

    def test_all_examples_load(self) -> None:
        for name in EXAMPLES_SCENARIOS:
            with self.subTest(scenario=name):
                scenario = load_example(name)
                self.assertEqual(scenario.schema_version, SCHEMA_VERSION)
                self.assertEqual(scenario.id, name)
                self.assertTrue(scenario.workload.operations)

    def test_fixtures_are_marked_synthetic(self) -> None:
        # ACCEPTANCE.md §7 and DESIGN.md §9 both forbid letting a hand-written
        # graph pass as a real GGML export. "ggml" is reserved for a graph that
        # actually came out of the exporter, which does not exist yet.
        for name in EXAMPLES_SCENARIOS:
            with self.subTest(scenario=name):
                self.assertEqual(load_example(name).workload.origin.kind, "synthetic_fixture")

    def test_no_fixture_claims_to_be_a_ggml_export(self) -> None:
        for path in sorted(EXAMPLES.glob("*.workload.json")):
            with self.subTest(workload=path.name):
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["origin"]["kind"], "synthetic_fixture")

    def test_rectangular_matvec_is_not_square(self) -> None:
        # A square weight cannot catch a transposed ne/nb reading, so the suite
        # needs at least one weight where ne[0] != ne[1] (DESIGN.md §3).
        weight = load_example("matvec-cap160").workload.tensor_by_id["W"]
        self.assertNotEqual(weight.ne[0], weight.ne[1])
        self.assertEqual(weight.ne, (3, 4, 1, 1))

    def test_relu_keeps_its_raw_ggml_identity(self) -> None:
        # ggml_relu() is GGML_OP_UNARY with a unary-op tag, not a GGML_OP_RELU
        # of its own, so the exporter must normalise while keeping the original
        # (DESIGN.md §3, ACCEPTANCE.md §5).
        operation = load_example("residual-cap160").workload.operation_by_id["r"]
        self.assertEqual(operation.semantic_op, "RELU")
        self.assertEqual(operation.ggml_op, "GGML_OP_UNARY")
        self.assertEqual(operation.unary_op, "GGML_UNARY_OP_RELU")

    def test_nonzero_relu_params_are_not_mistaken_for_unsupported(self) -> None:
        # op_params[0] holds the ggml unary-op enum, which is legitimately
        # non-zero. Rejecting it would drop ReLU from every real export
        # (DESIGN.md §4.1).
        operation = load_example("residual-cap160").workload.operation_by_id["r"]
        self.assertEqual(operation.op_params, (6,))


class FingerprintTests(unittest.TestCase):
    """The canonicalisation is pinned here, as DESIGN.md §4.3 asks."""

    def test_fingerprints_are_sha256_hex(self) -> None:
        scenario = load_example("chain-cap160")
        for digest in (workload_fingerprint(scenario.workload), scenario_fingerprint(scenario)):
            with self.subTest(digest=digest[:8]):
                self.assertEqual(len(digest), 64)
                self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_tensor_order_does_not_change_the_fingerprint(self) -> None:
        # Reordering a JSON array changes nothing semantically, so it must not
        # invalidate every stored mapping.
        baseline = load_example("chain-cap160")
        payload = json.loads((EXAMPLES / "chain.workload.json").read_text(encoding="utf-8"))
        payload["tensors"] = list(reversed(payload["tensors"]))
        payload["operations"] = list(reversed(payload["operations"]))
        with documents(workload=payload) as path:
            shuffled = load_and_validate(path)
        self.assertEqual(
            workload_fingerprint(shuffled.workload),
            workload_fingerprint(baseline.workload),
        )

    def test_provenance_does_not_change_the_fingerprint(self) -> None:
        # The GGML version and sample name say where a graph came from, not what
        # it does. Including them would invalidate stored mappings on every
        # re-export (DESIGN.md §4.3).
        baseline = load_example("chain-cap160")
        payload = json.loads((EXAMPLES / "chain.workload.json").read_text(encoding="utf-8"))
        payload["origin"]["ggml_version"] = "b8705-some-other-build"
        payload["origin"]["sample"] = "a different name for the same graph"
        with documents(workload=payload) as path:
            relabelled = load_and_validate(path)
        self.assertEqual(
            workload_fingerprint(relabelled.workload),
            workload_fingerprint(baseline.workload),
        )

    def test_a_semantic_change_does_change_the_fingerprint(self) -> None:
        baseline = load_example("chain-cap160")
        payload = json.loads((EXAMPLES / "chain.workload.json").read_text(encoding="utf-8"))
        # Giving x a DRAM copy changes what eviction is allowed to do, so a
        # mapping searched against the old graph must not be accepted here.
        next(t for t in payload["tensors"] if t["id"] == "x")["initial_locations"] = [
            "dram",
            "vram",
        ]
        with documents(workload=payload) as path:
            altered = load_and_validate(path)
        self.assertNotEqual(
            workload_fingerprint(altered.workload),
            workload_fingerprint(baseline.workload),
        )

    def test_scenario_fingerprint_tracks_capacity(self) -> None:
        base = load_example("chain-cap160")
        self.assertNotEqual(
            scenario_fingerprint(base), scenario_fingerprint(with_capacity(base, 96))
        )

    def test_scenario_fingerprint_keeps_the_workload_bound(self) -> None:
        # workload_file is an external path and is excluded, so the graph is
        # pinned by embedding the workload's own fingerprint instead. Two
        # scenarios over different graphs must not collide.
        chain = load_example("chain-cap160")
        fork = load_example("fork-cap160")
        self.assertNotEqual(scenario_fingerprint(chain), scenario_fingerprint(fork))


class PathResolutionTests(unittest.TestCase):
    def test_workload_file_resolves_relative_to_the_scenario(self) -> None:
        # Not relative to the shell's cwd -- a scenario must stay valid wherever
        # it is checked out (DESIGN.md §4).
        scenario = load_example("chain-cap160")
        self.assertEqual(scenario.workload.id, "chain")

    def test_missing_workload_file_is_reported(self) -> None:
        with documents() as path:
            path.parent.joinpath("w.json").unlink()
            with self.assertRaises(InvalidInput) as caught:
                load_and_validate(path)
        self.assertIn("cannot read", str(caught.exception))


class InvalidInputTests(unittest.TestCase):
    """Structurally broken or semantically inconsistent documents."""

    def _chain(self) -> dict:
        return json.loads((EXAMPLES / "chain.workload.json").read_text(encoding="utf-8"))

    def _reject(self, workload: dict | None = None, scenario: dict | None = None) -> str:
        with documents(workload=workload, scenario=scenario) as path:
            with self.assertRaises(InvalidInput) as caught:
                load_and_validate(path)
        return str(caught.exception)

    def test_unknown_input_tensor(self) -> None:
        workload = self._chain()
        workload["operations"][0]["inputs"] = ["W1", "nope"]
        self.assertIn("unknown input tensor", self._reject(workload))

    def test_unknown_output_tensor(self) -> None:
        workload = self._chain()
        workload["outputs"] = ["nope"]
        self.assertIn("unknown tensor", self._reject(workload))

    def test_duplicate_tensor_id(self) -> None:
        workload = self._chain()
        workload["tensors"].append(copy.deepcopy(workload["tensors"][0]))
        self.assertIn("duplicate tensor id", self._reject(workload))

    def test_duplicate_operation_id(self) -> None:
        workload = self._chain()
        workload["operations"].append(copy.deepcopy(workload["operations"][0]))
        self.assertIn("duplicate operation id", self._reject(workload))

    def test_multiple_producers(self) -> None:
        workload = self._chain()
        workload["operations"].append(
            {
                "id": "c3",
                "semantic_op": "MUL_MAT",
                "ggml_op": "GGML_OP_MUL_MAT",
                "inputs": ["W2", "x"],
                "output": "h",
            }
        )
        self.assertIn("several producers", self._reject(workload))

    def test_cycle(self) -> None:
        workload = self._chain()
        # h = W1*y and y = W2*h: each has one producer, but neither is reachable
        # from an output, so the graph is cyclic.
        workload["operations"][0]["inputs"] = ["W1", "y"]
        workload["operations"][0]["output"] = "h"
        message = self._reject(workload)
        self.assertIn("cycle", message)

    def test_orphan_operation(self) -> None:
        workload = self._chain()
        workload["tensors"].append(
            {
                "id": "dead",
                "name": "dead",
                "role": "intermediate",
                "dtype": "GGML_TYPE_F32",
                "ne": [4, 1, 1, 1],
                "nb": [4, 16, 16, 16],
                "storage_bytes": 16,
            }
        )
        workload["operations"].append(
            {
                "id": "c_orphan",
                "semantic_op": "RELU",
                "ggml_op": "GGML_OP_UNARY",
                "unary_op": "GGML_UNARY_OP_RELU",
                "inputs": ["x"],
                "output": "dead",
            }
        )
        self.assertIn("do not contribute to any requested output", self._reject(workload))

    def test_missing_cost(self) -> None:
        scenario = json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))
        del scenario["costs"]["compute"]["c2"]
        self.assertIn("missing an entry for operation", self._reject(scenario=scenario))

    def test_storage_bytes_must_match_the_layout(self) -> None:
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "W1")["storage_bytes"] = 63
        self.assertIn("implies", self._reject(workload))

    def test_leaf_without_an_initial_location(self) -> None:
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "W1")["initial_locations"] = []
        self.assertIn("no initial copy", self._reject(workload))

    def test_produced_tensor_may_not_declare_an_initial_location(self) -> None:
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "h")["initial_locations"] = ["dram"]
        self.assertIn("also declares an initial copy", self._reject(workload))

    def test_unknown_key_is_refused_not_ignored(self) -> None:
        # A typo'd constraint silently dropped is worse than a loud failure.
        workload = self._chain()
        workload["operations"][0]["unary_opp"] = "GGML_UNARY_OP_RELU"
        self.assertIn("unknown key", self._reject(workload))

    def test_missing_schema_version(self) -> None:
        workload = self._chain()
        del workload["schema_version"]
        self.assertIn("schema_version", self._reject(workload))

    def test_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
            directory = Path(raw_dir)
            directory.joinpath("w.json").write_text("{ not json", encoding="utf-8")
            directory.joinpath("s.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": "s",
                        "workload_file": "w.json",
                        "architecture": {"vram_capacity_bytes": 160},
                        "costs": {"source": "x", "compute": {}, "h2d": {}},
                        "mapspace": {
                            "copy_tensor_ids": [],
                            "allow_copy_compute_overlap": True,
                            "allow_eviction": True,
                        },
                        "mapper": {"algorithm": "uniform_cost"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(InvalidInput) as caught:
                load_and_validate(directory / "s.json")
        self.assertIn("not valid JSON", str(caught.exception))

    def test_float_duration_is_refused(self) -> None:
        # Nanoseconds are integers; a float time would quietly break the
        # exactness the search relies on (DESIGN.md §4).
        scenario = json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))
        scenario["costs"]["compute"]["c1"]["duration_ns"] = 2.5
        self.assertIn("must be an integer", self._reject(scenario=scenario))

    def test_zero_duration_is_refused(self) -> None:
        scenario = json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))
        scenario["costs"]["compute"]["c1"]["duration_ns"] = 0
        self.assertIn(">= 1", self._reject(scenario=scenario))

    def test_boolean_is_not_an_integer(self) -> None:
        scenario = json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))
        scenario["architecture"]["vram_capacity_bytes"] = True
        self.assertIn("must be an integer", self._reject(scenario=scenario))

    def test_copy_tensor_must_be_a_leaf_with_a_dram_copy(self) -> None:
        # h is a computed result, so it has no DRAM copy to load from. This is
        # the check that actually fires for a non-leaf: requiring an initial
        # DRAM copy already implies leaf-ness, because only leaves may declare
        # one. See the comment in _parse_mapspace.
        scenario = json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))
        scenario["mapspace"]["copy_tensor_ids"] = ["W1", "h"]
        self.assertIn("no initial DRAM copy", self._reject(scenario=scenario))

    def test_a_produced_tensor_cannot_forge_an_initial_copy(self) -> None:
        # The other half of the same rule, enforced on the workload side: this
        # is what makes the mapspace check above cover non-leaves.
        workload = json.loads((EXAMPLES / "chain.workload.json").read_text(encoding="utf-8"))
        next(t for t in workload["tensors"] if t["id"] == "h")["initial_locations"] = ["dram"]
        scenario = json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))
        scenario["mapspace"]["copy_tensor_ids"] = ["W1", "h"]
        self.assertIn("also declares an initial copy", self._reject(workload, scenario))

    def test_copy_tensor_must_have_a_dram_copy(self) -> None:
        scenario = json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))
        scenario["mapspace"]["copy_tensor_ids"] = ["W1", "x"]
        self.assertIn("no initial DRAM copy", self._reject(scenario=scenario))

    def test_malformed_shape_is_refused(self) -> None:
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "x")["ne"] = [4, 1, 1]
        self.assertIn("exactly 4 entries", self._reject(workload))

    def test_mul_mat_mismatched_inner_dimension(self) -> None:
        workload = self._chain()
        # W1 is 4x4; make the activation 3-wide so the product is undefined.
        activation = next(t for t in workload["tensors"] if t["id"] == "x")
        activation["ne"] = [3, 1, 1, 1]
        activation["nb"] = [4, 12, 12, 12]
        activation["storage_bytes"] = 12
        self.assertIn("does not match", self._reject(workload))

    def test_add_requires_matching_shapes(self) -> None:
        workload = json.loads((EXAMPLES / "fork.workload.json").read_text(encoding="utf-8"))
        # Widen only y: its operands still agree with each other, but no longer
        # with the result. Resizing an operand instead would trip the MUL_MAT
        # check upstream and never reach the ADD rule.
        target = next(t for t in workload["tensors"] if t["id"] == "y")
        target["ne"] = [8, 1, 1, 1]
        target["nb"] = [4, 32, 32, 32]
        target["storage_bytes"] = 32
        self.assertIn("elementwise-identical shapes", self._reject(workload))

    def test_relu_must_preserve_shape(self) -> None:
        workload = json.loads((EXAMPLES / "residual.workload.json").read_text(encoding="utf-8"))
        next(t for t in workload["tensors"] if t["id"] == "a")["ne"] = [8, 1, 1, 1]
        next(t for t in workload["tensors"] if t["id"] == "a")["nb"] = [4, 32, 32, 32]
        next(t for t in workload["tensors"] if t["id"] == "a")["storage_bytes"] = 32
        self.assertIn("preserve the shape", self._reject(workload))

    def test_duplicate_output(self) -> None:
        workload = self._chain()
        workload["outputs"] = ["y", "y"]
        self.assertIn("twice", self._reject(workload))

    def test_weight_must_be_a_leaf(self) -> None:
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "W1")["role"] = "intermediate"
        next(t for t in workload["tensors"] if t["id"] == "h")["role"] = "weight"
        self.assertIn("must be a leaf", self._reject(workload))


class UnsupportedTests(unittest.TestCase):
    """Well-formed documents asking for semantics M0 deliberately lacks."""

    def _chain(self) -> dict:
        return json.loads((EXAMPLES / "chain.workload.json").read_text(encoding="utf-8"))

    def _scenario(self) -> dict:
        return json.loads((EXAMPLES / "chain-cap160.json").read_text(encoding="utf-8"))

    def _reject(self, workload: dict | None = None, scenario: dict | None = None) -> str:
        with documents(workload=workload, scenario=scenario) as path:
            with self.assertRaises(Unsupported) as caught:
                load_and_validate(path)
        return str(caught.exception)

    def test_non_f32_dtype(self) -> None:
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "W1")["dtype"] = "GGML_TYPE_Q4_K"
        self.assertIn("GGML_TYPE_F32", self._reject(workload))

    def test_non_contiguous_layout(self) -> None:
        # What a view looks like once its strides are recorded: the outer stride
        # is no longer the inner extent times the type size.
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "W1")["nb"] = [4, 16, 96, 96]
        next(t for t in workload["tensors"] if t["id"] == "W1")["storage_bytes"] = 96
        self.assertIn("non-contiguous", self._reject(workload))

    def test_view_src(self) -> None:
        workload = self._chain()
        next(t for t in workload["tensors"] if t["id"] == "h")["view_src"] = "x"
        self.assertIn("neither views", self._reject(workload))

    def test_in_place_shared_storage(self) -> None:
        workload = self._chain()
        workload["operations"][0]["inputs"] = ["x", "x"]
        workload["operations"][0]["output"] = "x"
        self.assertIn("in place", self._reject(workload))

    def test_dram_capacity_must_be_null(self) -> None:
        scenario = self._scenario()
        scenario["architecture"]["dram_capacity_bytes"] = 1 << 30
        self.assertIn("must be null", self._reject(scenario=scenario))

    def test_more_than_one_compute_slot(self) -> None:
        scenario = self._scenario()
        scenario["architecture"]["gpu_compute_slots"] = 2
        self.assertIn("exactly one resource", self._reject(scenario=scenario))

    def test_more_than_one_copy_slot(self) -> None:
        scenario = self._scenario()
        scenario["architecture"]["h2d_copy_slots"] = 4
        self.assertIn("exactly one resource", self._reject(scenario=scenario))

    def test_recomputation_must_be_false(self) -> None:
        scenario = self._scenario()
        scenario["mapspace"]["allow_recomputation"] = True
        self.assertIn("no recomputation", self._reject(scenario=scenario))

    def test_unknown_algorithm(self) -> None:
        scenario = self._scenario()
        scenario["mapper"]["algorithm"] = "simulated_annealing"
        self.assertIn("uniform_cost", self._reject(scenario=scenario))

    def test_unknown_semantic_op(self) -> None:
        workload = self._chain()
        workload["operations"][0]["semantic_op"] = "MUL_MAT_ID"
        self.assertIn("M0 models only", self._reject(workload))

    def test_unknown_schema_version(self) -> None:
        workload = self._chain()
        workload["schema_version"] = "0.2"
        self.assertIn("0.2", self._reject(workload))

    def test_multi_gpu_compute_device(self) -> None:
        scenario = self._scenario()
        scenario["mapspace"]["compute_device"] = "cpu"
        self.assertIn("gpu", self._reject(scenario=scenario))


class InitialCapacityTests(unittest.TestCase):
    """An impossible starting layout is a verdict, not an input error."""

    def test_x_over_budget_is_flagged(self) -> None:
        # x is 16 B, so a 8 B budget cannot even hold the initial layout
        # (ACCEPTANCE.md §5).
        scenario = with_capacity(load_example("chain-cap160"), 8)
        with self.assertRaises(InitialCapacityExceeded) as caught:
            check_initial_capacity(scenario)
        self.assertIn("initial layout needs", str(caught.exception))

    def test_initial_capacity_error_is_not_an_input_error(self) -> None:
        # It must not be catchable as InvalidInput: the scenario is legitimate
        # and the same document is fine once the capacity rises.
        scenario = with_capacity(load_example("chain-cap160"), 8)
        self.assertFalse(issubclass(InitialCapacityExceeded, InvalidInput))
        check_initial_capacity(with_capacity(scenario, 160))

    def test_runtime_reserve_counts_towards_the_initial_layout(self) -> None:
        from .support import with_runtime_reserve

        scenario = with_runtime_reserve(load_example("chain-cap160"), 160)
        with self.assertRaises(InitialCapacityExceeded):
            check_initial_capacity(scenario)

    def test_exactly_fitting_initial_layout_is_accepted(self) -> None:
        # x is the only initially resident tensor: 16 B exactly fits.
        check_initial_capacity(with_capacity(load_example("chain-cap160"), 16))


class AlignmentTests(unittest.TestCase):
    def test_aligned_size_rounds_up(self) -> None:
        self.assertEqual(aligned_size(16, 32), 32)
        self.assertEqual(aligned_size(32, 32), 32)
        self.assertEqual(aligned_size(33, 32), 64)
        self.assertEqual(aligned_size(16, 1), 16)
        self.assertEqual(aligned_size(0, 32), 0)

    def test_scenario_reports_the_aligned_allocation(self) -> None:
        from .support import with_alignment

        scenario = with_alignment(load_example("chain-cap160"), 32)
        self.assertEqual(scenario.size_alloc("W1"), 64)
        self.assertEqual(scenario.size_alloc("x"), 32)  # 16 B rounds up to 32


class LoadWorkloadTests(unittest.TestCase):
    def test_workload_can_be_loaded_without_a_scenario(self) -> None:
        workload = load_workload(EXAMPLES / "chain.workload.json")
        self.assertEqual(workload.id, "chain")
        self.assertEqual(workload.leaf_tensors, frozenset({"W1", "W2", "x"}))
        self.assertEqual(workload.producer_of["y"], "c2")
        self.assertEqual(workload.consumers_of["x"], ("c1",))


if __name__ == "__main__":
    unittest.main()
