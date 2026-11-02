from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis as a
import run_native as r


def suite():
    return r.document(HERE / "transform_suite.json")


def payload(case, protocol, seed=17):
    result = {key: value for key, value in case.items() if key != "id"}
    result.update(protocol)
    result.update(schema="execution-transform-native-0.1", matmul_precision="GGML_PREC_F32_explicit", seed=seed, timer="steady_clock_completed_work",
                  boundary={"output_location": "dram", "merge_device": "cpu",
                            "input_location": "all_required_immutable_inputs_already_resident_in_both_cpu_and_gpu",
                            "weights": "generated_f32_not_model_weights"},
                  wall_ns={metric: [100] * protocol["samples"] for metric in r.expected_metrics(case)},
                  ports={name: {"bytes": size, "input_fixture_fnv1a64": "0123456789abcdef"} for name, size in r.expected_port_bytes(case).items()},
                  workspaces={"cpu/test": {"known_gallocr_plus_cplan_bytes": 32, "workspace_bytes": 32, "source": "gallocr_plus_explicit_cplan"},
                              "gpu/test": {"known_gallocr_plus_cplan_bytes": 32, "workspace_bytes": None, "source": "unknown_private_cuda_scratch"}},
                  all_checks_passed=True, gpu_capacity_proved=False, control_overhead_separately_identified=False)
    names = {"gpu/resident/output", "hybrid/resident/output"}
    if case["family"] == "ffn":
        names.add("hybrid/preload/output")
        names |= {device + "/shard/" + stage for device in ("cpu", "gpu") for stage in ("gate", "up", "activated", "output")}
    else:
        names |= {"gpu/pv/context", "gpu/pv/key_appended", "gpu/pv/value_appended", "cpu/cache_unchanged", "gpu/cache_unchanged", "cpu/pv/context"}
    result["checks"] = {name: {"passed": True, "max_abs_error": .000001, "max_relative_error": .000001,
                              "atol": .00001, "rtol": .0001, "elements": 1} for name in names}
    return result


class RunnerContracts(unittest.TestCase):
    def test_explicit_suite_and_disjoint_jobs(self):
        doc = r.validate_suite(suite())
        train, test = r.jobs(doc, "profile"), r.jobs(doc, "heldout")
        self.assertEqual(len(train), 14)
        self.assertEqual(len(test), 14)
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in test})
        self.assertFalse({row["seed"] for row in train} & {row["seed"] for row in test})

    def test_every_suite_key_required_and_unknown_refused(self):
        for key in suite():
            doc = suite()
            doc.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_suite(doc)
        doc = suite()
        doc["guess"] = 0
        with self.assertRaises(ValueError):
            r.validate_suite(doc)

    def test_no_silent_geometry_thread_precision_substitution(self):
        for field, value in (("dtype", "f16"), ("cpu_intermediate", 0), ("cpu_intermediate", 512), ("hidden", True)):
            doc = suite()
            doc["cases"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                r.validate_suite(doc)
        doc = suite()
        doc["protocol"]["threads"] = 2
        with self.assertRaises(ValueError):
            r.validate_suite(doc)

    def test_shared_seeds_refused(self):
        doc = suite()
        doc["heldout_seeds"] = doc["profile_seeds"]
        with self.assertRaises(ValueError):
            r.validate_suite(doc)

    def test_command_contains_every_explicit_field(self):
        doc = suite()
        cmd = r.command("probe", doc["cases"][0], doc["protocol"], 17)
        for name in r.CASE_FIELDS - {"id"} | r.PROTOCOL_FIELDS | {"seed"}:
            self.assertIn("--" + name.replace("_", "-"), cmd)

    def test_cuda_graphs_and_tf32_flags_explicit_and_observed(self):
        env, flags = r.child_context([])
        self.assertEqual(env["GGML_CUDA_DISABLE_GRAPHS"], "1")
        self.assertEqual(flags["GGML_CUDA_DISABLE_GRAPHS"], "1")
        self.assertEqual(env["NVIDIA_TF32_OVERRIDE"], "0")
        self.assertEqual(flags["NVIDIA_TF32_OVERRIDE"], "0")

    def test_explicit_matmul_precision_required(self):
        doc = suite()
        case = doc["cases"][0]
        native = payload(case, doc["protocol"])
        native.pop("matmul_precision")
        with self.assertRaisesRegex(ValueError, "precision"):
            r.validate_native({"case": case, "seed": 17}, native, doc)

    def test_both_native_contracts_accepted(self):
        doc = suite()
        for case in (doc["cases"][0], doc["cases"][-1]):
            r.validate_native({"case": case, "seed": 17}, payload(case, doc["protocol"]), doc)

    def test_bad_native_binding_or_numerics_rejected(self):
        doc = suite()
        case = doc["cases"][0]
        for field, value in (("seed", 29), ("all_checks_passed", False), ("dtype", "f16"), ("samples", 2), ("timer", "enqueue_only")):
            native = payload(case, doc["protocol"])
            native[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.validate_native({"case": case, "seed": 17}, native, doc)

    def test_unknown_workspace_not_replaced_by_zero(self):
        doc = suite()
        case = doc["cases"][0]
        native = payload(case, doc["protocol"])
        native["workspaces"]["gpu/test"]["workspace_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "remain unknown"):
            r.validate_native({"case": case, "seed": 17}, native, doc)

    def test_wrong_ports_or_missing_numerical_endpoints_rejected(self):
        doc = suite()
        case = doc["cases"][-1]
        for variant in ("port", "check", "tolerance", "metric"):
            native = payload(case, doc["protocol"])
            if variant == "port":
                native["ports"]["cpu/key_cache"]["bytes"] += 4
            elif variant == "check":
                native["checks"].pop("gpu/pv/key_appended")
            elif variant == "metric":
                native["wall_ns"].pop("gpu/pv")
            else:
                native["checks"]["gpu/pv/key_appended"]["rtol"] = 1
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                r.validate_native({"case": case, "seed": 17}, native, doc)

    def test_component_model_never_uses_parallel_total(self):
        case = suite()["cases"][0]
        recipes = r.calibration_recipes(case)
        for expression in recipes.values():
            self.assertNotIn("hybrid/parallel_branches", a.terms(expression))
        self.assertNotIn("hybrid/dispatch_barrier", a.terms(recipes["hybrid/resident"]))
        self.assertIn("hybrid/dispatch_barrier", a.terms(recipes["hybrid/resident@dispatch_model"]))
        self.assertEqual(r.observed_aliases(case)["hybrid/resident@dispatch_model"], "hybrid/resident")


if __name__ == "__main__":
    unittest.main()
