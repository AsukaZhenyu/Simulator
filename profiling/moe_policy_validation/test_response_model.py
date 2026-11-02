"""Independent obligation/cost tests for the model-owned expert response path."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modeling"))

from llm_infer_model.performance.expert_policy_response import (
    CALIBRATION_SCHEMA, action_price_keys, evaluate_expert_policy,
)
from llm_infer_model.performance.parameters import InvalidModel, UnknownParameter


def program():
    return {
        "schema": "moe-expert-policy-native-1",
        "config": dict(hidden=4, intermediate=8, experts=3, gpu_slots=1,
                       threads=2, cuda_device=0),
        "seed": 91,
        "resource_guards": dict(host_reserve_bytes=1024, gpu_reserve_bytes=1024),
        "steps": [dict(input=[[0.1] * 4, [0.2] * 4],
                       routes=[[0, 1], [1, 2]], scores=[[0.6, 0.4], [0.3, 0.7]])],
        "actions": [dict(kind="LOAD", expert=1, slot=0),
                    dict(kind="EXPERT", step=0, expert=0, device="cpu"),
                    dict(kind="EXPERT", step=0, expert=1, device="gpu", slot=0),
                    dict(kind="RELEASE", expert=1, slot=0),
                    dict(kind="EXPERT", step=0, expert=2, device="cpu"),
                    dict(kind="COMBINE", step=0)],
    }


def calibration(p):
    return dict(schema=CALIBRATION_SCHEMA, config=deepcopy(p["config"]),
                prices_ns={"LOAD": 100.5, "RELEASE": 2,
                           "EXPERT:cpu:q=1": 200, "EXPERT:gpu:q=2": 90,
                           "COMBINE:q=2:topk=2": 30})


class ExpertResponseTests(unittest.TestCase):
    def test_exact_routed_rows_and_serial_inclusive_composition(self):
        p = program()
        original = deepcopy(p)
        result = evaluate_expert_policy(p, calibration(p))
        self.assertEqual(result["duration_ns"], 622.5)
        self.assertEqual(result["actions"][2]["q"], 2)
        self.assertEqual(action_price_keys(p)[1], "EXPERT:cpu:q=1")
        self.assertEqual(result["breakdown_ns"]["EXPERT:cpu"], 400)
        self.assertEqual(result["weight_h2d_bytes"], 384)
        self.assertEqual(result["total_h2d_bytes"], 416)
        self.assertEqual(result["total_d2h_bytes"], 32)
        self.assertEqual(result["peak_resident_weight_bytes"], 384)
        self.assertEqual(result["gpu_weight_logical_bytes"], 384)
        self.assertIsNone(result["workspace_bytes"])
        self.assertFalse(result["total_device_capacity_certified"])
        self.assertFalse(result["causal_policy_certified"])
        self.assertEqual(p, original)
        for first, second in zip(result["actions"], result["actions"][1:]):
            self.assertEqual(first["end_ns"], second["start_ns"])

    def test_allocated_slots_remain_after_release(self):
        p = program()
        p["config"]["gpu_slots"] = 2
        result = evaluate_expert_policy(p, calibration(p))
        self.assertEqual(result["peak_resident_weight_bytes"], 384)
        self.assertEqual(result["gpu_weight_logical_bytes"], 768)

    def test_missing_or_unknown_exact_price_never_falls_back(self):
        p = program()
        for missing in (True, False):
            c = calibration(p)
            if missing:
                del c["prices_ns"]["EXPERT:gpu:q=2"]
            else:
                c["prices_ns"]["EXPERT:gpu:q=2"] = None
            c["prices_ns"]["EXPERT:gpu:q=1"] = 45
            with self.assertRaisesRegex(UnknownParameter, "EXPERT:gpu:q=2"):
                evaluate_expert_policy(p, c)

    def test_wrong_implementation_config_is_unpriced(self):
        p = program()
        c = calibration(p)
        c["config"]["threads"] = 4
        with self.assertRaises(UnknownParameter):
            evaluate_expert_policy(p, c)

    def test_gpu_weights_required_and_slot_overwrite_forbidden(self):
        p = program()
        del p["actions"][0]
        with self.assertRaisesRegex(InvalidModel, "lacks its weights"):
            action_price_keys(p)
        p = program()
        p["actions"].insert(1, dict(kind="LOAD", expert=0, slot=0))
        with self.assertRaisesRegex(InvalidModel, "overwrites"):
            action_price_keys(p)

    def test_release_checks_owner_and_capacity(self):
        p = program()
        p["actions"][3]["expert"] = 0
        with self.assertRaisesRegex(InvalidModel, "ownership"):
            action_price_keys(p)
        p = program()
        p["actions"][0]["slot"] = 1
        with self.assertRaisesRegex(InvalidModel, "capacity"):
            action_price_keys(p)

    def test_required_experts_execute_once_before_combine(self):
        p = program()
        del p["actions"][4]
        with self.assertRaisesRegex(InvalidModel, "precedes"):
            action_price_keys(p)
        p = program()
        p["actions"].insert(2, deepcopy(p["actions"][1]))
        with self.assertRaisesRegex(InvalidModel, "more than once"):
            action_price_keys(p)
        p = program()
        p["actions"].pop()
        with self.assertRaisesRegex(InvalidModel, "omits"):
            action_price_keys(p)

    def test_no_unrouted_or_late_expert_actions(self):
        p = program()
        p["steps"][0]["routes"] = [[0, 1], [0, 1]]
        with self.assertRaisesRegex(InvalidModel, "not selected"):
            action_price_keys(p)
        p = program()
        p["actions"].append(deepcopy(p["actions"][1]))
        with self.assertRaisesRegex(InvalidModel, "after combined"):
            action_price_keys(p)

    def test_invalid_route_scores_and_hidden_dimensions(self):
        for field, replacement in (("input", [[1.0] * 3] * 2),
                                   ("routes", [[1, 1], [1, 2]]),
                                   ("scores", [[0.8, 0.8], [0.3, 0.7]])):
            p = program()
            p["steps"][0][field] = replacement
            with self.assertRaises(InvalidModel):
                action_price_keys(p)

    def test_variable_topk_is_not_priced_as_uniform_topk(self):
        p = program()
        p["steps"][0]["routes"][0] = [0]
        p["steps"][0]["scores"][0] = [1.0]
        with self.assertRaisesRegex(InvalidModel, "constant top-k"):
            action_price_keys(p)

    def test_invalid_or_negative_prices(self):
        p = program()
        for value in (-1, float("nan"), float("inf"), True):
            c = calibration(p)
            c["prices_ns"]["LOAD"] = value
            with self.assertRaises(InvalidModel):
                evaluate_expert_policy(p, c)

    def test_cpu_only_has_no_gpu_weight_allocation(self):
        p = program()
        p["config"]["gpu_slots"] = 0
        p["actions"] = [a for a in p["actions"] if a["kind"] not in ("LOAD", "RELEASE")]
        p["actions"][1] = dict(kind="EXPERT", step=0, expert=1, device="cpu")
        c = calibration(p)
        c["prices_ns"]["EXPERT:cpu:q=2"] = 300
        result = evaluate_expert_policy(p, c)
        self.assertEqual(result["duration_ns"], 730)
        self.assertEqual(result["gpu_weight_logical_bytes"], 0)
        self.assertEqual(result["total_h2d_bytes"], 0)
        self.assertEqual(result["total_d2h_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
