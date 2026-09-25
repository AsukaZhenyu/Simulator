from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("2b", "4b", "9b")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class GeneratedArtifactTests(unittest.TestCase):
    def test_manifest_accounting_matches_generated_configs(self) -> None:
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                manifest = read_json(
                    ROOT / "outputs" / "gguf" / f"qwen35_{model_id}_q4_k_m_manifest.json"
                )
                config = read_json(
                    ROOT / "configs" / "generated" / f"qwen35_{model_id}_q4_k_m.json"
                )
                layers = manifest["layers"]
                self.assertEqual(manifest["layer_count"], len(layers))
                self.assertEqual(
                    manifest["layer_tensor_bytes"],
                    sum(layer["weight_bytes"] for layer in layers),
                )
                self.assertEqual(
                    manifest["tensor_data_bytes"],
                    manifest["layer_tensor_bytes"] + manifest["global_tensor_bytes"],
                )
                self.assertEqual(len(config["layers"]), manifest["layer_count"])
                self.assertEqual(
                    config["hardware"]["global_bytes"], manifest["global_tensor_bytes"]
                )
                self.assertEqual(
                    [layer["weight_bytes"] for layer in config["layers"]],
                    [layer["weight_bytes"] for layer in layers],
                )
                self.assertIn("state", config)
                self.assertEqual(config["hardware"]["kv_bytes"], 0)
                self.assertGreater(config["state"]["ssm_state_size"], 0)
                self.assertGreater(config["state"]["attention_kv_head_count"], 0)

    def test_simulation_is_between_analytical_bounds(self) -> None:
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                analysis = read_json(
                    ROOT
                    / "outputs"
                    / "structure_calibrated"
                    / f"qwen35_{model_id}_analysis.json"
                )
                simulation = read_json(
                    ROOT
                    / "outputs"
                    / "structure_calibrated"
                    / f"qwen35_{model_id}_simulation.json"
                )
                self.assertGreaterEqual(
                    simulation["makespan_seconds"],
                    analysis["ideal_overlap_lower_bound_seconds"],
                )
                self.assertLessEqual(
                    simulation["makespan_seconds"],
                    analysis["no_overlap_upper_bound_seconds"],
                )

    def test_rtx4070_calibration_fit_is_within_five_percent(self) -> None:
        report = read_json(
            ROOT / "outputs" / "calibration" / "calibration_fit_rtx4070.json"
        )
        self.assertEqual(report["schema_version"], 6)
        self.assertLess(report["pipeline_service_fit"]["relative_rmse"], 0.03)
        self.assertLess(report["scheduler_overhead_fit"]["relative_spread"], 0.02)
        self.assertEqual(len(report["state_transfer_fit"]["rows"]), 3)
        self.assertEqual(len(report["kv_storage_fit"]["rows"]), 1)
        storage_fit = report["kv_storage_fit"]["rows"][0]
        self.assertEqual(storage_fit["model_id"], "2b")
        self.assertGreater(
            storage_fit["kv_storage_cached_bandwidth_bytes_per_s"],
            storage_fit["kv_storage_uncached_bandwidth_bytes_per_s"],
        )
        self.assertGreater(storage_fit["kv_storage_cache_bytes"], 0)
        for model in report["models"]:
            with self.subTest(model=model["model_id"]):
                self.assertLess(abs(model["simulation_relative_error"]), 0.05)
                config = read_json(
                    ROOT
                    / "configs"
                    / "calibrated_rtx4070"
                    / f"qwen35_{model['model_id']}_q4_k_m_rtx4070.json"
                )
                self.assertGreater(
                    config["policy"]["token_pipeline_overhead_s"], 0
                )
                self.assertGreater(
                    config["policy"]["token_embedding_fallback_s"], 0
                )
                self.assertLess(
                    config["policy"]["token_embedding_fallback_s"]
                    + config["policy"]["token_state_roundtrip_s"],
                    config["policy"]["token_pipeline_overhead_s"],
                )
                self.assertEqual(
                    config["policy"]["embedding_fallback_scale"], 1.0
                )
                self.assertEqual(
                    config["policy"]["staging_overlap_window"], 4
                )
                self.assertGreater(
                    config["hardware"]["host_staging_bandwidth_bytes_per_s"],
                    0,
                )
                self.assertGreater(
                    config["hardware"]["state_transfer_bandwidth_bytes_per_s"],
                    0,
                )
                self.assertEqual(config["policy"]["state_placement"], "roundtrip")
                self.assertGreater(config["policy"]["token_state_roundtrip_s"], 0)
                self.assertEqual(
                    config["policy"]["layer_scheduler_overhead_s"], 0
                )
                if model["model_id"] == "2b":
                    self.assertGreater(
                        config["hardware"][
                            "kv_storage_cached_bandwidth_bytes_per_s"
                        ],
                        config["hardware"][
                            "kv_storage_uncached_bandwidth_bytes_per_s"
                        ],
                    )

    def test_embedding_counterfactual_only_helps_after_streaming_ceases(self) -> None:
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                path = (
                    ROOT
                    / "outputs"
                    / "counterfactual_no_embedding"
                    / f"qwen35_{model_id}.csv"
                )
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                k4 = next(row for row in rows if int(row["window_size"]) == 4)
                resident_window = rows[-1]
                self.assertAlmostEqual(float(k4["speedup"]), 1.0)
                self.assertGreater(float(resident_window["speedup"]), 1.8)

    def test_state_residency_has_capacity_and_latency_boundaries(self) -> None:
        resident_windows = {"2b": 24, "4b": 32, "9b": 32}
        for model_id, resident_window in resident_windows.items():
            with self.subTest(model=model_id):
                path = ROOT / "outputs" / "state_residency" / f"qwen35_{model_id}.csv"
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                k4_zero = next(
                    row
                    for row in rows
                    if int(row["window_size"]) == 4
                    and int(row["context_tokens"]) == 0
                )
                full_zero = next(
                    row
                    for row in rows
                    if int(row["window_size"]) == resident_window
                    and int(row["context_tokens"]) == 0
                )
                self.assertAlmostEqual(float(k4_zero["speedup"]), 1.0)
                self.assertGreater(float(full_zero["speedup"]), 1.05)
                self.assertGreater(
                    int(k4_zero["latency_break_even_context_tokens"]), 0
                )
                self.assertGreater(int(k4_zero["max_resident_context_tokens"]), 0)


if __name__ == "__main__":
    unittest.main()
