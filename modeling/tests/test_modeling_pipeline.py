"""端到端检查建模管线：只用仓库内的输入，现场算出所有断言对象。

这个文件取代了原来的 test_generated_artifacts.py。旧版直接读 modeling/outputs/ 下
的产物，而 outputs/ 不入库，所以干净克隆上必然报 13 个错误；而且即使 simulate_decode
写坏了，只要旧的产物还在，测试依然会绿。

现在分两类输入：
  * 结构/校准：仓库内的 configs/ 和 tests/data/ 下的示例测量数据
  * GGUF：测试内现场构造一个小 GGUF，不依赖 1.28 GB 的真实模型

示例数据 tests/data/measurements_rtx4070_20260821.json 是一次真实 RTX 4070 测量的
副本，只把 nsys/context_validation 的路径改写为同目录下的示例文件。重新测量本机
硬件请用 `regenerate --measure`，不要改这个文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from llm_infer_model.analytical import analyze_decode
from llm_infer_model.calibration import calibrate_configs
from llm_infer_model.cli import (
    command_compare_embedding_fallback,
    command_compare_state_residency,
)
from llm_infer_model.gguf_extract import (
    config_from_gguf_manifest,
    extract_gguf_manifest,
)
from llm_infer_model.model import load_config
from llm_infer_model.simulator import simulate_decode

try:
    import numpy as np
    from gguf import GGUFValueType, GGUFWriter
except ImportError as exc:  # pragma: no cover - only without the optional extra
    GGUF_IMPORT_ERROR = str(exc)
    GGUF_AVAILABLE = False
else:
    GGUF_IMPORT_ERROR = ""
    GGUF_AVAILABLE = True


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
MEASUREMENTS = DATA / "measurements_rtx4070_20260821.json"
MODEL_IDS = ("2b", "4b", "9b")
# 每个模型的层数，也是「整模型驻留」所需的窗口大小（2b=24 层，4b/9b=32 层）。
# 对照实验的窗口列表必须逐个模型取到该值，用统一的 4,8,16,32 会让 2b 取不到驻留点。
RESIDENT_WINDOWS = {"2b": 24, "4b": 32, "9b": 32}
STATE_RESIDENCY_CONTEXTS = "0,512,2048,8192,32768,131072,262144"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_synthetic_gguf(path: Path) -> None:
    """构造一个 2 层的 GGUF，覆盖 attention/ssm 两种层分类和矩形权重。

    GGML 的 ``ne`` 是 numpy shape 的逆序：写入 ``(16, 8)`` 会读回 ``(8, 16)``。
    所以这里刻意使用非方阵，方阵会让维度转置错误无法暴露。
    """
    writer = GGUFWriter(str(path), "qwen35")
    writer.add_name("synthetic-qwen35-2-layer")
    writer.add_block_count(2)
    writer.add_context_length(64)
    writer.add_embedding_length(8)
    for key, value in (
        ("qwen35.attention.head_count_kv", 2),
        ("qwen35.attention.key_length", 4),
        ("qwen35.attention.value_length", 4),
        ("qwen35.ssm.conv_kernel", 4),
        ("qwen35.ssm.state_size", 16),
        ("qwen35.ssm.group_count", 1),
        ("qwen35.ssm.inner_size", 16),
    ):
        writer.add_key_value(key, value, GGUFValueType.UINT32)

    # 非 blk.* 的张量算全局常驻，不参与流式层窗口
    writer.add_tensor("token_embd.weight", np.zeros((8, 32), dtype=np.float32))
    writer.add_tensor("output_norm.weight", np.zeros((8,), dtype=np.float32))
    # 第 0 层判定为 attention
    writer.add_tensor("blk.0.attn_q.weight", np.zeros((16, 8), dtype=np.float32))
    writer.add_tensor("blk.0.attn_output.weight", np.zeros((8, 16), dtype=np.float32))
    # 第 1 层判定为 ssm
    writer.add_tensor("blk.1.attn_qkv.weight", np.zeros((24, 8), dtype=np.float32))
    writer.add_tensor("blk.1.ssm_conv1d.weight", np.zeros((8, 4), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@unittest.skipUnless(
    GGUF_AVAILABLE,
    f'GGUF 测试需要可选的 gguf 依赖：pip install -e ".[gguf]"  ({GGUF_IMPORT_ERROR})',
)
class SyntheticGgufManifestTests(unittest.TestCase):
    """现场构造小 GGUF，验证结构提取的账目与适配后的配置。"""

    @classmethod
    def setUpClass(cls) -> None:
        # Windows 上 GGUFReader 用 memmap 持有文件句柄，清理可能失败
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.gguf_path = Path(cls._tmp.name) / "synthetic-qwen35.gguf"
        build_synthetic_gguf(cls.gguf_path)
        cls.manifest = extract_gguf_manifest(cls.gguf_path)
        cls.gguf_size_bytes = cls.gguf_path.stat().st_size

    def test_manifest_accounting_is_internally_consistent(self) -> None:
        manifest = self.manifest
        layers = manifest["layers"]
        self.assertEqual(manifest["layer_count"], len(layers))
        self.assertEqual(manifest["layer_count"], 2)
        self.assertEqual(manifest["tensor_count"], 6)
        self.assertEqual(
            manifest["layer_tensor_bytes"],
            sum(layer["weight_bytes"] for layer in layers),
        )
        self.assertEqual(
            manifest["tensor_data_bytes"],
            manifest["layer_tensor_bytes"] + manifest["global_tensor_bytes"],
        )
        self.assertEqual(manifest["source_size_bytes"], self.gguf_size_bytes)
        self.assertEqual(
            manifest["container_overhead_bytes"],
            manifest["source_size_bytes"] - manifest["tensor_data_bytes"],
        )
        self.assertGreater(manifest["container_overhead_bytes"], 0)

    def test_layer_kinds_are_classified_from_tensor_names(self) -> None:
        kinds = [layer["kind"] for layer in self.manifest["layers"]]
        self.assertEqual(kinds, ["attention", "ssm"])
        self.assertEqual(
            [layer["index"] for layer in self.manifest["layers"]], [0, 1]
        )

    def test_ggml_ne_is_reversed_from_numpy_shape(self) -> None:
        """矩形权重必须按 GGML 的 ne 顺序读回，不能套用 NumPy 维度顺序。"""
        by_name = {tensor["name"]: tensor for tensor in self.manifest["tensors"]}
        q_proj = by_name["blk.0.attn_q.weight"]
        # 写入 numpy (16, 8) -> GGML ne = [8, 16]
        self.assertEqual(q_proj["shape"], [8, 16])
        self.assertEqual(q_proj["n_elements"], 128)
        self.assertEqual(q_proj["n_bytes"], 128 * 4)
        self.assertEqual(q_proj["layer"], 0)
        # 非 blk.* 的张量归入全局常驻
        self.assertIsNone(by_name["token_embd.weight"]["layer"])
        self.assertEqual(len(self.manifest["global_tensors"]), 2)

    def test_manifest_metadata_and_flops_proxies(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["architecture"], "qwen35")
        self.assertEqual(
            manifest["metadata"]["qwen35.attention.head_count_kv"], 2
        )
        # 2*n_elements 只对 rank>=2 的张量生效；一维 norm 记 0
        by_name = {tensor["name"]: tensor for tensor in manifest["tensors"]}
        self.assertEqual(by_name["blk.0.attn_q.weight"]["estimated_matvec_flops"], 256)
        self.assertEqual(by_name["output_norm.weight"]["estimated_matvec_flops"], 0)
        self.assertGreater(manifest["estimated_layer_matvec_flops_per_token"], 0)

    def test_config_from_manifest_matches_manifest_bytes(self) -> None:
        template = read_json(ROOT / "configs" / "generated" / "qwen35_2b_q4_k_m.json")
        config = config_from_gguf_manifest(self.manifest, template)
        manifest = self.manifest

        self.assertEqual(len(config["layers"]), manifest["layer_count"])
        self.assertEqual(
            [layer["weight_bytes"] for layer in config["layers"]],
            [layer["weight_bytes"] for layer in manifest["layers"]],
        )
        self.assertEqual(
            config["hardware"]["global_bytes"], manifest["global_tensor_bytes"]
        )
        # 模板有 24 层，合成模型只有 2 层，窗口必须被夹到层数以内
        self.assertEqual(config["policy"]["window_size"], manifest["layer_count"])
        # state 几何来自 GGUF metadata，模板里不透明的 KV 预留被清零
        self.assertIn("state", config)
        self.assertEqual(config["hardware"]["kv_bytes"], 0)
        self.assertGreater(config["state"]["ssm_state_size"], 0)
        self.assertGreater(config["state"]["attention_kv_head_count"], 0)
        self.assertEqual(config["hardware"]["cpu_effective_flops"], None)


class CalibratedPipelineTests(unittest.TestCase):
    """用仓库内示例测量数据现场跑校准，再驱动两个对照实验。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.addClassCleanup(cls._tmp.cleanup)
        cls.calibration_dir = Path(cls._tmp.name)
        _, cls.report = calibrate_configs(MEASUREMENTS, cls.calibration_dir)

    def config_path(self, model_id: str) -> Path:
        return self.calibration_dir / f"qwen35_{model_id}_q4_k_m_rtx4070.json"

    def test_simulation_is_between_analytical_bounds(self) -> None:
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                config = load_config(
                    ROOT / "configs" / "generated" / f"qwen35_{model_id}_q4_k_m.json"
                )
                analysis = analyze_decode(config)
                self.assertTrue(analysis.capacity_feasible)
                simulation = simulate_decode(config)
                self.assertGreaterEqual(
                    simulation.makespan_seconds,
                    analysis.ideal_overlap_lower_bound_seconds,
                )
                self.assertLessEqual(
                    simulation.makespan_seconds,
                    analysis.no_overlap_upper_bound_seconds,
                )

    def test_calibration_fit_is_within_tolerance(self) -> None:
        report = self.report
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

    def test_calibrated_configs_carry_expected_policy(self) -> None:
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                config = read_json(self.config_path(model_id))
                self.assertGreater(config["policy"]["token_pipeline_overhead_s"], 0)
                self.assertGreater(config["policy"]["token_embedding_fallback_s"], 0)
                self.assertLess(
                    config["policy"]["token_embedding_fallback_s"]
                    + config["policy"]["token_state_roundtrip_s"],
                    config["policy"]["token_pipeline_overhead_s"],
                )
                self.assertEqual(config["policy"]["embedding_fallback_scale"], 1.0)
                self.assertEqual(config["policy"]["staging_overlap_window"], 4)
                self.assertEqual(config["policy"]["state_placement"], "roundtrip")
                self.assertGreater(config["policy"]["token_state_roundtrip_s"], 0)
                self.assertEqual(config["policy"]["layer_scheduler_overhead_s"], 0)
                self.assertGreater(
                    config["hardware"]["host_staging_bandwidth_bytes_per_s"], 0
                )
                self.assertGreater(
                    config["hardware"]["state_transfer_bandwidth_bytes_per_s"], 0
                )
                if model_id == "2b":
                    self.assertGreater(
                        config["hardware"]["kv_storage_cached_bandwidth_bytes_per_s"],
                        config["hardware"]["kv_storage_uncached_bandwidth_bytes_per_s"],
                    )

    def test_embedding_counterfactual_only_helps_after_streaming_ceases(self) -> None:
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                output = self.calibration_dir / f"counterfactual_{model_id}.csv"
                command_compare_embedding_fallback(
                    argparse.Namespace(
                        config=str(self.config_path(model_id)),
                        windows=f"1,2,4,8,16,{RESIDENT_WINDOWS[model_id]}",
                        output=str(output),
                    )
                )
                rows = read_csv(output)
                k4 = next(row for row in rows if int(row["window_size"]) == 4)
                resident_window = rows[-1]
                # 窗口放不下整模型时，去掉 embedding D2H 没有收益
                self.assertAlmostEqual(float(k4["speedup"]), 1.0)
                # 整模型驻留后才体现收益
                self.assertGreater(float(resident_window["speedup"]), 1.8)

    def test_state_residency_has_capacity_and_latency_boundaries(self) -> None:
        for model_id, resident_window in RESIDENT_WINDOWS.items():
            with self.subTest(model=model_id):
                output = self.calibration_dir / f"state_residency_{model_id}.csv"
                command_compare_state_residency(
                    argparse.Namespace(
                        config=str(self.config_path(model_id)),
                        contexts=STATE_RESIDENCY_CONTEXTS,
                        windows=f"4,8,16,{resident_window}",
                        output=str(output),
                    )
                )
                rows = read_csv(output)
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
