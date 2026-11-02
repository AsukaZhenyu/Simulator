import copy
import unittest

import numpy as np

from profiling.moe_policy_validation.reference import copy_traffic, reference_outputs, verify_result, weights


def fixture():
    return dict(config=dict(hidden=4, intermediate=8, experts=3, gpu_slots=1, threads=2, cuda_device=0), seed=71,
                steps=[dict(input=[[.1, -.3, .2, .4], [.4, .2, -.1, .2]], routes=[[0, 1], [1, 2]], scores=[[.2, .8], [.3, .7]])],
                actions=[dict(kind="LOAD", expert=0, slot=0), dict(kind="EXPERT", step=0, expert=0, device="gpu", slot=0),
                         dict(kind="RELEASE", expert=0, slot=0), dict(kind="EXPERT", step=0, expert=1, device="cpu"),
                         dict(kind="LOAD", expert=2, slot=0), dict(kind="EXPERT", step=0, expert=2, device="gpu", slot=0),
                         dict(kind="COMBINE", step=0)])


class ReferenceTests(unittest.TestCase):
    def test_distinct_weights_and_declared_shapes(self):
        a, b = weights(71, 0, 4, 8), weights(71, 1, 4, 8)
        self.assertEqual([x.shape for x in a], [(8, 4), (8, 4), (4, 8)])
        self.assertTrue(all(x.dtype == np.float32 for x in a))
        self.assertFalse(np.array_equal(a[0], a[1]))
        self.assertFalse(np.array_equal(a[0], b[0]))

    def test_copy_traffic_counts_grouped_rows_not_top_k_twice(self):
        self.assertEqual(copy_traffic(fixture()), dict(weight_h2d_bytes=768, total_h2d_bytes=800, total_d2h_bytes=32))

    def test_wrong_output_and_wrong_copy_accounting_are_detected(self):
        p = fixture()
        native = dict(config=p["config"], seed=p["seed"], actions=p["actions"], outputs=reference_outputs(p),
                      gpu_weight_logical_bytes=384, peak_resident_experts=1, **copy_traffic(p))
        self.assertTrue(verify_result(p, native)["numerically_equal"])
        bad = copy.deepcopy(native)
        bad["outputs"][0][0][0] += .01
        self.assertFalse(verify_result(p, bad)["numerically_equal"])
        bad = copy.deepcopy(native)
        bad["total_h2d_bytes"] += 4
        with self.assertRaisesRegex(ValueError, "physical"):
            verify_result(p, bad)

    def test_residual_included_once(self):
        p = fixture()
        p["steps"][0]["input"] = [[0.] * 4, [0.] * 4]
        np.testing.assert_array_equal(reference_outputs(p), np.zeros((1, 2, 4)))


if __name__ == "__main__":
    unittest.main()
