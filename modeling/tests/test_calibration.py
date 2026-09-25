from __future__ import annotations

import unittest

from llm_infer_model.calibration import (
    H2D_PATTERN,
    fit_host_staging,
    fit_pipeline_service,
    fit_scheduler_overhead,
    fit_state_transfer,
    parse_llama_bench_payload,
)
from llm_infer_model.model import (
    HardwareSpec,
    LayerSpec,
    ModelConfig,
    PolicySpec,
    StateSpec,
)


class CalibrationTests(unittest.TestCase):
    def test_scheduler_fit_uses_pipeline_resident_delta(self) -> None:
        measurements = {
            "models": [
                {
                    "id": "a",
                    "layer_count": 4,
                    "resident": {"avg_tokens_per_second": 10.0},
                    "pipeline_resident": {"avg_tokens_per_second": 5.0},
                },
                {
                    "id": "b",
                    "layer_count": 8,
                    "resident": {"avg_tokens_per_second": 10.0},
                    "pipeline_resident": {"avg_tokens_per_second": 10.0 / 3.0},
                },
            ]
        }
        fit = fit_scheduler_overhead(measurements)
        self.assertAlmostEqual(fit["layer_scheduler_overhead_s"], 0.025)
        self.assertAlmostEqual(fit["mean_token_pipeline_overhead_s"], 0.15)
        self.assertAlmostEqual(fit["rows"][0]["token_pipeline_overhead_s"], 0.1)
        self.assertAlmostEqual(fit["rows"][1]["token_pipeline_overhead_s"], 0.2)

    def test_pipeline_fit_recovers_bandwidth_and_layer_latency(self) -> None:
        bandwidth = 1000.0
        latency = 0.25
        rows = []
        for model_id, weight_bytes, layer_count in (
            ("a", 100, 2),
            ("b", 250, 3),
            ("c", 400, 5),
        ):
            seconds = weight_bytes / bandwidth + layer_count * latency
            rows.append(
                {
                    "id": model_id,
                    "layer_weight_bytes": weight_bytes,
                    "layer_count": layer_count,
                    "pipeline": {"avg_tokens_per_second": 1.0 / seconds},
                }
            )
        fit = fit_pipeline_service({"models": rows})
        self.assertAlmostEqual(fit["effective_bandwidth_bytes_per_s"], bandwidth)
        self.assertAlmostEqual(fit["effective_layer_latency_s"], latency)
        self.assertAlmostEqual(fit["relative_rmse"], 0.0)

    def test_host_staging_fit_uses_k2_excess_latency(self) -> None:
        measurements = {
            "models": [
                {
                    "id": "a",
                    "layer_weight_bytes": 100,
                    "pipeline": {
                        "window_size": 4,
                        "avg_tokens_per_second": 10.0,
                    },
                    "window_validation": [
                        {"window_size": 2, "avg_tokens_per_second": 5.0}
                    ],
                }
            ]
        }
        fit = fit_host_staging(measurements)
        self.assertAlmostEqual(
            fit["host_staging_bandwidth_bytes_per_s"], 1000.0
        )
        self.assertEqual(fit["staging_overlap_window"], 4)
        self.assertAlmostEqual(
            fit["rows"][0]["exposed_host_staging_seconds"], 0.1
        )

    def test_cuda_bandwidth_output_pattern(self) -> None:
        line = (
            "bandwidthTest-H2D-Pinned, Bandwidth = 11659.1 MB/s, "
            "Time = 0.01098 s, Size = 134217728 bytes, NumDevsUsed = 1"
        )
        match = H2D_PATTERN.search(line)
        self.assertIsNotNone(match)
        self.assertEqual(match.group("memory"), "Pinned")
        self.assertEqual(int(match.group("size")), 134217728)

    def test_state_transfer_fit_uses_semantic_recurrent_bytes(self) -> None:
        config = ModelConfig(
            name="state-fit",
            layers=(LayerSpec("ssm", 100, 100, kind="ssm"),),
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                vram_capacity_bytes=1000,
            ),
            policy=PolicySpec(window_size=1),
            state=StateSpec(
                ssm_state_size=10,
                ssm_inner_size=10,
                recurrent_element_bytes=4,
            ),
        )
        decomposition = {
            "copy_summary": {
                "directions": [
                    {
                        "direction": "Host-to-Device",
                        "gpu_milliseconds_per_token": 1.0,
                    },
                    {
                        "direction": "Device-to-Host",
                        "gpu_milliseconds_per_token": 2.0,
                    },
                ]
            },
            "embedding_fallback": {
                "d2h_gpu_milliseconds_per_token": 1.0
            },
        }
        fit = fit_state_transfer(config, decomposition)
        self.assertEqual(fit["recurrent_state_bytes"], 400)
        self.assertEqual(fit["semantic_roundtrip_bytes"], 800)
        self.assertAlmostEqual(
            fit["effective_state_transfer_bandwidth_bytes_per_s"], 400000.0
        )

    def test_context_decode_uses_generation_at_cached_depth(self) -> None:
        payload = [
            {
                "n_prompt": 0,
                "n_gen": 8,
                "n_depth": 1024,
                "avg_ts": 10.0,
                "samples_ns": [800_000_000, 800_000_000],
            },
        ]
        parsed = parse_llama_bench_payload(
            payload, prompt_tokens=1024, generation_tokens=8
        )
        self.assertAlmostEqual(parsed["avg_tokens_per_second"], 10.0)
        self.assertAlmostEqual(parsed["avg_token_seconds"], 0.1)
        self.assertEqual(parsed["decode_samples_ns"], [800_000_000, 800_000_000])


if __name__ == "__main__":
    unittest.main()
