"""Non-native checks of candidate identity, physical costs and causal scope."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import moe


def prices(k):
    config = moe.native_config("small", k)
    cells = {"LOAD": 20, "RELEASE": 1}
    for q in (1, 4, 8, 16, 32):
        cells[f"EXPERT:cpu:q={q}"] = 5 if q == 1 else (100 if q == 32 else 25)
        cells[f"EXPERT:gpu:q={q}"] = 20 if q == 32 else 10
    for q in (33, 36, 40, 48):
        cells[f"COMBINE:q={q}:topk=2"] = 3
    return dict(schema="moe-expert-policy-calibration-1", config=config,
                boundary=moe.BOUNDARY, prices_ns=cells)


class MoESpaceTests(unittest.TestCase):
    def setUp(self):
        self.geometry = patch.dict(moe.GEOMETRIES, {"small": (8, 16)})
        self.geometry.start()
        self.addCleanup(self.geometry.stop)

    def test_matrix_shapes_and_geometry_transfer_share_q(self):
        cases = moe.workload_specs(8)
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(moe.workload_specs(8, True)), 36)
        self.assertEqual({row["light_rows"] for row in cases if row["geometry"] == "real"}, {8})
        for spec in cases[:12]:
            steps = moe.make_steps(spec)
            self.assertEqual(len(steps), 8)
            for step in steps:
                counts = sorted(moe.Counter(e for row in step["routes"] for e in row).values())
                self.assertEqual(counts, [8, 8, 32, 32])

    def test_boundary_selection_and_fallback_do_not_invent_costs(self):
        profiles = {("small", k): prices(k) for k in (2, 4)}
        result = moe.select_boundary_rows(profiles)
        self.assertEqual(result["q_b"], 4)  # identical distances: smallest q wins
        self.assertTrue(result["found_shared_crossover"])
        for profile in profiles.values():
            for q in (4, 8, 16):
                profile["prices_ns"][f"EXPERT:cpu:q={q}"] = 1000
        result = moe.select_boundary_rows(profiles)
        self.assertEqual(result["q_b"], 16)
        self.assertFalse(result["found_shared_crossover"])

    def test_compute_only_decision_keeps_real_transfer_costs(self):
        spec = moe.workload_specs(4)[0]
        config, steps, physical = moe.native_config("small", 2), moe.make_steps(spec), prices(2)
        original = deepcopy(physical)
        shape = dict(id="shape", execution="E_SHAPE", storage="S_LRU", main=True)
        path = dict(id="path", execution="E_PATH", storage="S_LRU", main=True)
        a = moe._compile(config, steps, spec["seed"], shape, physical)
        b = moe._compile(config, steps, spec["seed"], path, physical)
        self.assertEqual(physical, original)
        self.assertEqual(a["compilation_evidence"]["decision_weight_transfer_price_ns"], 0)
        self.assertEqual(a["compilation_evidence"]["physical_weight_transfer_price_ns"], 20)
        self.assertNotEqual(a["program"]["actions"], b["program"]["actions"])
        prediction = moe.evaluate_expert_policy(a["program"], physical)
        self.assertGreater(prediction["weight_h2d_bytes"], 0)
        # Physical evaluation must include all actual LOADs at 20 ns each.
        self.assertGreaterEqual(prediction["duration_ns"], 20 * prediction["action_counts"]["LOAD"])

    def test_every_declared_policy_and_history_frequency_is_causal(self):
        spec = moe.workload_specs(4)[2]
        config, steps, physical = moe.native_config("small", 2), moe.make_steps(spec), prices(2)
        for candidate in moe.policy_candidates():
            result = moe._compile(config, steps, spec["seed"], candidate, physical)
            moe.evaluate_expert_policy(result["program"], physical)
            proof = moe.check_causal_prefix(config, steps, spec["seed"], candidate, physical, result)
            self.assertTrue(proof["records_equal"])
            if candidate["storage"] == "S_FREQ":
                self.assertEqual(result["compilation_evidence"]["normalized_policy"]["cache"]["rule"],
                                 "history_frequency")

    def test_alias_dedup_reconstruction_and_payload_tamper(self):
        profiles = {("small", 2): prices(2)}
        spec = moe.workload_specs(4)[0]
        # Keep disposable test artifacts inside the explicitly writable project.
        with tempfile.TemporaryDirectory(dir=moe.ROOT / "build", prefix="es-moe-test-") as temp:
            Path(temp).resolve().relative_to(moe.ROOT.resolve())
            out = Path(temp) / "case"
            result = moe.compile_all(profiles, out, q_b=4, workloads=[spec], verify_prefix=False)
            self.assertEqual(result["nominal_conditions"], 15)
            self.assertLess(result["unique_programs"], 15)
            cpu = [row for row in result["labels"] if row["execution"] == "E_CPU"]
            self.assertEqual(len({row["program_id"] for row in cpu}), 1)
            for row in result["programs"]:
                program = moe.load_program(out, row["id"])
                self.assertEqual(moe.value_hash(program), row["program_sha256"])
            row = result["programs"][0]
            file = out / row["actions_path"]
            file.write_text("[]\n", encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(ValueError, "differ"):
                moe.load_program(out, row["id"])


if __name__ == "__main__":
    unittest.main()
