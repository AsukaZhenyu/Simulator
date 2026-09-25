from __future__ import annotations

import unittest

from llm_infer_model.analytical import analyze_decode
from llm_infer_model.gguf_extract import config_from_gguf_manifest
from llm_infer_model.model import (
    HardwareSpec,
    LayerSpec,
    ModelConfig,
    PolicySpec,
    StateSpec,
)
from llm_infer_model.simulator import simulate_decode


def make_config(window_size: int, vram_capacity_bytes: int = 1000) -> ModelConfig:
    return ModelConfig(
        name="two-layer-test",
        layers=(
            LayerSpec("layer_0", weight_bytes=100, flops=100),
            LayerSpec("layer_1", weight_bytes=100, flops=100),
        ),
        hardware=HardwareSpec(
            gpu_effective_flops=100,
            cpu_effective_flops=50,
            h2d_bandwidth_bytes_per_s=100,
            h2d_latency_s=0,
            vram_capacity_bytes=vram_capacity_bytes,
            activation_transfer_bytes=10,
        ),
        policy=PolicySpec(window_size=window_size, static_gpu_layers=1),
    )


class AnalyticalModelTests(unittest.TestCase):
    def test_streaming_bounds(self) -> None:
        result = analyze_decode(make_config(window_size=1))
        self.assertEqual(result.bytes_transferred_per_token, 200)
        self.assertEqual(result.compute_seconds, 2)
        self.assertEqual(result.transfer_seconds, 2)
        self.assertEqual(result.ideal_overlap_lower_bound_seconds, 2)
        self.assertEqual(result.no_overlap_upper_bound_seconds, 4)

    def test_all_layers_resident_in_steady_state(self) -> None:
        result = analyze_decode(make_config(window_size=2))
        self.assertEqual(result.bytes_transferred_per_token, 0)
        self.assertEqual(result.transfer_seconds, 0)
        self.assertEqual(result.ideal_overlap_lower_bound_seconds, 2)

    def test_capacity_check(self) -> None:
        result = analyze_decode(make_config(window_size=2, vram_capacity_bytes=150))
        self.assertFalse(result.capacity_feasible)

    def test_measured_layer_times_override_roofline_estimates(self) -> None:
        base = make_config(window_size=1)
        config = ModelConfig(
            name=base.name,
            layers=tuple(
                LayerSpec(
                    layer.name,
                    layer.weight_bytes,
                    layer.flops,
                    measured_compute_seconds=0.25,
                    measured_transfer_seconds=0.5,
                )
                for layer in base.layers
            ),
            hardware=base.hardware,
            policy=base.policy,
        )
        result = analyze_decode(config)
        self.assertEqual(result.compute_seconds, 0.5)
        self.assertEqual(result.transfer_seconds, 1.0)
        self.assertEqual(result.measured_compute_layer_count, 2)
        self.assertEqual(result.measured_transfer_layer_count, 2)
        self.assertIsNone(result.bandwidth_to_match_static_bytes_per_s)

    def test_pipeline_scheduler_overhead_is_charged_per_layer(self) -> None:
        base = make_config(window_size=2)
        config = ModelConfig(
            name=base.name,
            layers=base.layers,
            hardware=base.hardware,
            policy=PolicySpec(window_size=2, layer_scheduler_overhead_s=0.5),
        )
        analytical = analyze_decode(config)
        simulation = simulate_decode(config)
        self.assertEqual(analytical.scheduler_overhead_seconds, 1.0)
        self.assertEqual(analytical.compute_seconds, 3.0)
        self.assertEqual(simulation.scheduler_service_seconds, 1.0)
        self.assertEqual(simulation.makespan_seconds, 3.0)

    def test_pipeline_token_overhead_is_charged_once(self) -> None:
        base = make_config(window_size=2)
        config = ModelConfig(
            name=base.name,
            layers=base.layers,
            hardware=base.hardware,
            policy=PolicySpec(window_size=2, token_pipeline_overhead_s=0.5),
        )
        analytical = analyze_decode(config)
        simulation = simulate_decode(config)
        self.assertEqual(analytical.token_pipeline_overhead_seconds, 0.5)
        self.assertEqual(analytical.layer_scheduler_overhead_seconds, 0.0)
        self.assertEqual(analytical.scheduler_overhead_seconds, 0.5)
        self.assertEqual(simulation.scheduler_service_seconds, 0.5)
        self.assertEqual(simulation.makespan_seconds, 2.5)
        self.assertEqual(simulation.events[0].task, "PIPELINE_CONTROL")

    def test_embedding_fallback_counterfactual_removes_only_traced_component(self) -> None:
        base = make_config(window_size=2)
        config = ModelConfig(
            name=base.name,
            layers=base.layers,
            hardware=base.hardware,
            policy=PolicySpec(
                window_size=2,
                token_pipeline_overhead_s=0.5,
                token_embedding_fallback_s=0.2,
            ),
        )
        baseline = simulate_decode(config)
        no_embedding = simulate_decode(
            config.with_overrides(embedding_fallback_scale=0.0)
        )
        self.assertEqual(baseline.makespan_seconds, 2.5)
        self.assertEqual(no_embedding.makespan_seconds, 2.3)
        analysis = analyze_decode(
            config.with_overrides(embedding_fallback_scale=0.0)
        )
        self.assertEqual(analysis.token_pipeline_overhead_seconds, 0.3)
        self.assertEqual(analysis.residual_token_pipeline_overhead_seconds, 0.3)
        self.assertEqual(analysis.embedding_fallback_scale, 0.0)

    def test_embedding_component_cannot_exceed_total_overhead(self) -> None:
        with self.assertRaises(ValueError):
            PolicySpec(
                window_size=1,
                token_pipeline_overhead_s=0.1,
                token_embedding_fallback_s=0.2,
            )

    def test_short_window_exposes_host_staging_service(self) -> None:
        config = ModelConfig(
            name="three-layer-staging",
            layers=tuple(
                LayerSpec(f"layer_{index}", weight_bytes=100, flops=100)
                for index in range(3)
            ),
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                host_staging_bandwidth_bytes_per_s=100,
                vram_capacity_bytes=300,
            ),
            policy=PolicySpec(window_size=2, staging_overlap_window=4),
        )
        analytical = analyze_decode(config)
        simulation = simulate_decode(config)
        self.assertEqual(analytical.weight_stream_seconds, 3.0)
        self.assertEqual(analytical.host_staging_service_seconds, 3.0)
        self.assertEqual(analytical.staging_exposure_fraction, 1.0)
        self.assertEqual(analytical.exposed_host_staging_seconds, 3.0)
        self.assertEqual(analytical.transfer_seconds, 6.0)
        self.assertEqual(simulation.exposed_host_staging_seconds, 3.0)
        self.assertEqual(simulation.makespan_seconds, 7.0)

    def test_hybrid_state_bytes_and_context_scaling(self) -> None:
        config = ModelConfig(
            name="hybrid-state",
            layers=(
                LayerSpec("ssm", weight_bytes=100, flops=100, kind="ssm"),
                LayerSpec("attention", weight_bytes=100, flops=100, kind="attention"),
            ),
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                state_transfer_bandwidth_bytes_per_s=100,
                vram_capacity_bytes=1000,
            ),
            policy=PolicySpec(window_size=1),
            state=StateSpec(
                context_tokens=10,
                attention_kv_head_count=2,
                attention_key_length=4,
                attention_value_length=4,
                kv_element_bytes=2,
                ssm_conv_kernel=4,
                ssm_state_size=3,
                ssm_group_count=2,
                ssm_inner_size=8,
                recurrent_element_bytes=4,
            ),
        )
        result = analyze_decode(config)
        self.assertEqual(result.total_recurrent_state_bytes, 336)
        self.assertEqual(result.total_attention_kv_bytes, 320)
        self.assertEqual(result.total_runtime_state_bytes, 656)
        self.assertEqual(result.state_roundtrip_bytes_per_token, 1312)
        self.assertAlmostEqual(result.state_transfer_seconds, 13.12)

    def test_state_residency_trades_capacity_for_roundtrip_service(self) -> None:
        state = StateSpec(
            context_tokens=10,
            attention_kv_head_count=2,
            attention_key_length=4,
            attention_value_length=4,
            ssm_conv_kernel=4,
            ssm_state_size=3,
            ssm_group_count=2,
            ssm_inner_size=8,
        )
        config = ModelConfig(
            name="state-placement",
            layers=(
                LayerSpec("ssm", weight_bytes=100, flops=100, kind="ssm"),
                LayerSpec("attention", weight_bytes=100, flops=100, kind="attention"),
            ),
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                state_transfer_bandwidth_bytes_per_s=1000,
                vram_capacity_bytes=700,
            ),
            policy=PolicySpec(window_size=1, state_placement="roundtrip"),
            state=state,
        )
        roundtrip = analyze_decode(config)
        resident = analyze_decode(
            config.with_overrides(state_placement="resident")
        )
        self.assertTrue(roundtrip.capacity_feasible)
        self.assertFalse(resident.capacity_feasible)
        self.assertGreater(roundtrip.state_transfer_seconds, 0)
        self.assertEqual(resident.state_transfer_seconds, 0)
        self.assertEqual(resident.max_resident_state_context_tokens, 8)
        self.assertGreater(
            resident.required_window_bytes, roundtrip.required_window_bytes
        )

    def test_attention_kv_storage_is_window_aware_and_two_tier(self) -> None:
        config = ModelConfig(
            name="kv-storage",
            layers=(
                LayerSpec("attention_0", 100, 100, kind="attention"),
                LayerSpec("attention_1", 100, 100, kind="attention"),
            ),
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                state_transfer_bandwidth_bytes_per_s=100,
                kv_storage_cached_bandwidth_bytes_per_s=100,
                kv_storage_uncached_bandwidth_bytes_per_s=10,
                kv_storage_cache_bytes=100,
                vram_capacity_bytes=2000,
            ),
            policy=PolicySpec(window_size=1),
            state=StateSpec(
                context_tokens=30,
                attention_kv_head_count=1,
                attention_key_length=2,
                attention_value_length=2,
                kv_element_bytes=1,
            ),
        )
        streamed = analyze_decode(config)
        resident_window = analyze_decode(config.with_overrides(window_size=2))
        self.assertEqual(streamed.total_attention_kv_bytes, 240)
        self.assertEqual(streamed.attention_kv_roundtrip_bytes_per_token, 480)
        self.assertAlmostEqual(streamed.attention_kv_transfer_seconds, 4.8)
        self.assertAlmostEqual(streamed.attention_kv_storage_io_seconds, 30.0)
        self.assertFalse(streamed.attention_kv_window_resident)
        self.assertEqual(resident_window.attention_kv_roundtrip_bytes_per_token, 0)
        self.assertEqual(resident_window.attention_kv_storage_io_seconds, 0)
        self.assertTrue(resident_window.attention_kv_window_resident)


class SimulatorTests(unittest.TestCase):
    def test_window_one_is_serial(self) -> None:
        result = simulate_decode(make_config(window_size=1))
        self.assertEqual(result.makespan_seconds, 4)
        self.assertEqual(result.overlap_seconds, 0)
        self.assertEqual(result.bytes_transferred, 200)

    def test_two_slots_overlap_for_streamed_three_layer_case(self) -> None:
        config = ModelConfig(
            name="three-layer-overlap",
            layers=tuple(
                LayerSpec(f"layer_{index}", weight_bytes=100, flops=100)
                for index in range(3)
            ),
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                vram_capacity_bytes=200,
            ),
            policy=PolicySpec(window_size=2),
        )
        result = simulate_decode(config)
        self.assertEqual(result.makespan_seconds, 4)
        self.assertEqual(result.overlap_seconds, 2)
        analytical = analyze_decode(config)
        self.assertGreaterEqual(result.makespan_seconds, analytical.ideal_overlap_lower_bound_seconds)
        self.assertLessEqual(result.makespan_seconds, analytical.no_overlap_upper_bound_seconds)

    def test_all_resident_has_no_transfer(self) -> None:
        result = simulate_decode(make_config(window_size=2))
        self.assertEqual(result.makespan_seconds, 2)
        self.assertEqual(result.bytes_transferred, 0)
        self.assertEqual(result.gpu_utilization, 1)

    def test_infeasible_capacity_raises(self) -> None:
        with self.assertRaises(ValueError):
            simulate_decode(make_config(window_size=2, vram_capacity_bytes=150))

    def test_state_roundtrip_is_an_explicit_control_event(self) -> None:
        base = make_config(window_size=2)
        layers = tuple(
            LayerSpec(layer.name, layer.weight_bytes, layer.flops, kind="ssm")
            for layer in base.layers
        )
        config = ModelConfig(
            name="state-event",
            layers=layers,
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                state_transfer_bandwidth_bytes_per_s=160,
                vram_capacity_bytes=1000,
            ),
            policy=PolicySpec(window_size=2),
            state=StateSpec(ssm_state_size=1, ssm_inner_size=10),
        )
        roundtrip = simulate_decode(config)
        resident = simulate_decode(
            config.with_overrides(state_placement="resident")
        )
        self.assertEqual(roundtrip.events[0].task, "STATE_ROUNDTRIP")
        self.assertEqual(roundtrip.state_roundtrip_bytes, 160)
        self.assertEqual(roundtrip.makespan_seconds, 3.0)
        self.assertEqual(resident.makespan_seconds, 2.0)

    def test_streamed_attention_emits_storage_event(self) -> None:
        config = ModelConfig(
            name="kv-storage-event",
            layers=(
                LayerSpec("attention_0", 100, 100, kind="attention"),
                LayerSpec("attention_1", 100, 100, kind="attention"),
            ),
            hardware=HardwareSpec(
                gpu_effective_flops=100,
                h2d_bandwidth_bytes_per_s=100,
                state_transfer_bandwidth_bytes_per_s=100,
                kv_storage_cached_bandwidth_bytes_per_s=100,
                kv_storage_uncached_bandwidth_bytes_per_s=10,
                kv_storage_cache_bytes=100,
                vram_capacity_bytes=1000,
            ),
            policy=PolicySpec(window_size=1),
            state=StateSpec(
                context_tokens=10,
                attention_kv_head_count=1,
                attention_key_length=2,
                attention_value_length=2,
                kv_element_bytes=1,
            ),
        )
        result = simulate_decode(config)
        self.assertIn("ATTENTION_KV_PCIE_EXPOSED", [event.task for event in result.events])
        self.assertIn("KV_STORAGE_IO_EXPOSED", [event.task for event in result.events])
        self.assertGreater(result.kv_storage_service_seconds, 0)


class GGUFConfigTests(unittest.TestCase):
    def test_reference_manifest_calibrates_gpu_and_preserves_layer_structure(self) -> None:
        manifest = {
            "schema_version": 1,
            "source_file": "small.gguf",
            "layer_count": 2,
            "global_tensor_bytes": 300,
            "estimated_layer_matvec_flops_per_token": 600,
            "layers": [
                {
                    "index": 0,
                    "kind": "ssm",
                    "weight_bytes": 100,
                    "estimated_matvec_flops": 200,
                },
                {
                    "index": 1,
                    "kind": "attention",
                    "weight_bytes": 150,
                    "estimated_matvec_flops": 400,
                },
            ],
        }
        reference = {
            "source_file": "reference.gguf",
            "estimated_layer_matvec_flops_per_token": 1000,
        }
        template = {
            "name": "template",
            "layers": {"count": 4, "uniform_weight_bytes": 1, "uniform_flops": 1},
            "hardware": {
                "gpu_effective_flops": 10,
                "cpu_effective_flops": 5,
                "h2d_bandwidth_bytes_per_s": 100,
                "vram_capacity_bytes": 1000,
            },
            "policy": {"window_size": 4, "static_gpu_layers": 4},
        }
        raw = config_from_gguf_manifest(
            manifest,
            template,
            reference_manifest=reference,
            reference_compute_seconds=0.1,
        )
        self.assertEqual(raw["hardware"]["gpu_effective_flops"], 10000)
        self.assertEqual(raw["hardware"]["global_bytes"], 300)
        self.assertIsNone(raw["hardware"]["cpu_effective_flops"])
        self.assertEqual(raw["policy"]["window_size"], 2)
        self.assertEqual(raw["policy"]["static_gpu_layers"], 2)
        self.assertEqual([layer["kind"] for layer in raw["layers"]], ["ssm", "attention"])


if __name__ == "__main__":
    unittest.main()
