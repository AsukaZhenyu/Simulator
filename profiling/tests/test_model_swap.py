# -*- coding: utf-8 -*-
"""Model-swap skeleton generator: both entry points and the M2 hand-off.

The GGUF case is skipped when the local checkpoint directory is absent; it is
the only test that reads a real file, and it reads only the header/tensor
directory (never tensor data, never inference).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
for _name in ("src", "mapping", "modeling", "profiling"):
    sys.path.insert(0, str(ROOT / _name))

from model_swap import build_model  # noqa: E402
from llm_infer_model.m2.costs import (  # noqa: E402
    canonical_contract_hash,
    parse_cost_table,
)
from llm_infer_model.m2.spec import parse_layer_document  # noqa: E402
from llm_infer_model.tensor.spec import InvalidInput  # noqa: E402


GGUF = Path(
    os.environ.get(
        "LLM_SIM_MODEL_SWAP_GGUF",
        ROOT.parent
        / "llama.cpp-implement"
        / "Qwen-Series-GGUF-Q4_K_M"
        / "Qwen3.5-4B-Q4_K_M.gguf",
    )
)


def dense_metadata(**changes):
    raw = {
        "schema_version": build_model.METADATA_SCHEMA,
        "architecture": "toy",
        "model_name": "Toy-4B",
        "block_count": 8,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "heads": 16,
        "kv_heads": 4,
        "head_dim": 256,
        "rope_dims": 64,
        "rope_base": 10_000_000.0,
        "norm_epsilon": 1e-6,
        "full_attention_interval": 4,
    }
    raw.update(changes)
    return raw


def moe_metadata(**changes):
    raw = {
        "architecture": "toymoe",
        "block_count": 8,
        "hidden_size": 2048,
        "heads": 8,
        "kv_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "experts": 8,
        "top_k": 2,
        "attention_weight_bytes": 4096,
        "router_weight_bytes": 512,
        "expert_weight_bytes": 1024,
        "shared_expert_bytes": 256,
    }
    raw.update(changes)
    return raw


def write_metadata(directory: Path, raw: dict, name: str = "meta.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class MetadataEntryTests(unittest.TestCase):
    def test_dense_metadata_writes_geometry_layer_and_blank_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            meta_path = write_metadata(directory, dense_metadata())
            meta = build_model.load_metadata_json(meta_path)
            paths = build_model.write_model(
                meta, name="toy4b", out_dir=directory / "out"
            )
            self.assertEqual(
                set(paths),
                {
                    "toy4b.geometry.json",
                    "toy4b.layer.json",
                    "toy4b.costs.template.json",
                },
            )

            geometry = read_json(paths["toy4b.geometry.json"])
            self.assertEqual(geometry["block_count"], 8)
            # key_length/head_dim must win over hidden_size/heads (2560/16=160).
            self.assertEqual(geometry["attention"]["head_dim"], 256)
            self.assertEqual(geometry["attention"]["kv_heads"], 4)
            self.assertEqual(geometry["attention"]["heads_per_kv"], 4)
            self.assertEqual(
                geometry["attention"]["layer_kind_counts"],
                {"recurrent_linear": 6, "full_causal": 2},
            )
            self.assertEqual(geometry["m2_contract"]["representative_layer"], 3)
            self.assertEqual(geometry["m2_contract"]["kv_bytes_per_token"], 8192)

            layer = parse_layer_document(read_json(paths["toy4b.layer.json"]))
            self.assertEqual(layer.format_version, "m2-layer-0.1")
            self.assertEqual(layer.attention, "full_causal")
            self.assertEqual(layer.ffn, "dense")
            self.assertEqual(
                [task.kind for task in layer.tasks],
                ["projection", "attention", "dense_ffn"],
            )
            self.assertEqual(layer.hidden_bytes_per_token, 10240)

            costs = read_json(paths["toy4b.costs.template.json"])
            self.assertEqual(costs["provenance"], "measured_macro")
            self.assertEqual(len(costs["entries"]), 6)
            self.assertTrue(all(entry["duration_ns"] is None for entry in costs["entries"]))
            self.assertTrue(all(entry["workspace_bytes"] is None for entry in costs["entries"]))
            with self.assertRaises(InvalidInput):
                parse_cost_table(costs)

            # Filling the blanks (and nothing else) must yield a loadable table.
            signature = {"contract": "toy"}
            for entry in costs["entries"]:
                entry["duration_ns"] = 1
                entry["workspace_bytes"] = 0
                entry["workspace_source"] = "toy-measurement"
                entry["physical_signature"] = signature
                entry["contract_hash"] = canonical_contract_hash(signature)
            parse_cost_table(costs)

    def test_moe_metadata_writes_a_valid_m2_layer_0_6(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            meta = build_model.load_metadata_json(
                write_metadata(directory, moe_metadata())
            )
            paths = build_model.write_model(
                meta, name="toymoe", out_dir=directory / "out"
            )
            raw_layer = read_json(paths["toymoe.layer.json"])
            layer = parse_layer_document(raw_layer)  # runs validate_moe_layer
            self.assertEqual(layer.format_version, "m2-layer-0.6")
            self.assertEqual(layer.ffn, "moe")
            self.assertEqual(layer.ffn_params["experts"], 8)
            self.assertEqual(layer.ffn_params["top_k"], 2)
            self.assertEqual(len(layer.ffn_params["expert_weights"]), 8)
            self.assertEqual(layer.ffn_params["shared_expert_weight"], "W_shared")
            self.assertEqual(
                [task.kind for task in layer.tasks],
                ["projection", "attention", "router", "expert_ffn", "combine"],
            )
            self.assertEqual(len(layer.weights), 11)  # proj + router + 8 experts + shared

            costs = read_json(paths["toymoe.costs.template.json"])
            self.assertEqual(costs["format_version"], "m2-layer-0.6")
            self.assertEqual(costs["provenance"], "synthetic_layer")
            expert = [
                entry for entry in costs["entries"] if entry["kind"] == "expert_ffn"
            ]
            self.assertEqual(len(expert), 2)
            self.assertTrue(all(entry["visible_history"] == 0 for entry in expert))
            self.assertTrue(
                all(entry["accounting"] == "per_expert_group" for entry in expert)
            )
            self.assertIn("shared_expert_ffn", {entry["kind"] for entry in costs["entries"]})
            self.assertEqual(costs["transfer_entries"], [])
            for entry in costs["entries"]:
                entry["duration_ns"] = 1
                entry["workspace_bytes"] = 0
            parse_cost_table(costs)

    def test_moe_without_bytes_is_rejected_for_layer_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = moe_metadata()
            for key in ("attention_weight_bytes", "router_weight_bytes", "expert_weight_bytes"):
                raw.pop(key)
            meta = build_model.load_metadata_json(write_metadata(Path(tmp), raw))
            # Geometry is still useful without a weight inventory.
            geometry = build_model.build_geometry(meta, layer_index=None)
            self.assertEqual(geometry["ffn"]["kind"], "moe")
            with self.assertRaises(ValueError):
                build_model.build_layer(meta, layer_index=3)

    def test_model_without_full_causal_layer_is_structure_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw = dense_metadata(layer_types=["ssm"] * 8)
            meta = build_model.load_metadata_json(write_metadata(directory, raw))
            with self.assertRaises(ValueError):
                build_model.write_model(meta, name="linear", out_dir=directory / "out")
            paths = build_model.write_model(
                meta,
                name="linear",
                out_dir=directory / "out",
                structure_only=True,
            )
            self.assertEqual(set(paths), {"linear.geometry.json"})

    def test_key_and_value_length_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = dense_metadata(value_length=128)
            with self.assertRaises(ValueError):
                build_model.load_metadata_json(write_metadata(Path(tmp), raw))

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            meta = build_model.load_metadata_json(
                write_metadata(directory, dense_metadata())
            )
            build_model.write_model(meta, name="toy", out_dir=directory / "out")
            with self.assertRaises(FileExistsError):
                build_model.write_model(meta, name="toy", out_dir=directory / "out")


class ExpertWidthFallsBackToTheTensorShape(unittest.TestCase):
    """granitemoe and qwen2moe publish no expert_feed_forward_length key.

    For a pure-MoE block feed_forward_length is the expert width, so reporting
    None would make an exact value in the tensor directory look unknown.
    """

    layers = [
        {"tensors": [
            {"name": "blk.0.ffn_gate_exps.weight", "shape": [1536, 512, 40]},
            {"name": "blk.0.ffn_down_exps.weight", "shape": [512, 1536, 40]},
        ]},
    ]

    def test_a_rank_three_expert_tensor_yields_its_width(self):
        self.assertEqual(build_model._tensor_width(self.layers, "ffn_gate_exps", 3), 512)

    def test_a_shared_expert_tensor_is_rank_two(self):
        layers = [{"tensors": [{"name": "blk.0.ffn_gate_shexp.weight",
                                "shape": [2048, 5632]}]}]
        self.assertEqual(build_model._tensor_width(layers, "ffn_gate_shexp", 2), 5632)

    def test_the_wrong_rank_is_not_read_as_a_width(self):
        # ffn_down_exps is [width, hidden, experts]; treating it as a gate
        # tensor would silently report the hidden size as the expert width.
        self.assertIsNone(build_model._tensor_width(self.layers, "ffn_down_exps", 2))

    def test_an_absent_tensor_yields_none_rather_than_zero(self):
        self.assertIsNone(build_model._tensor_width(self.layers, "ffn_gate_shexp", 2))


@unittest.skipUnless(GGUF.is_file(), f"local GGUF not present: {GGUF}")
class GGUFTests(unittest.TestCase):
    def test_real_gguf_metadata_builds_a_parsable_skeleton(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = build_model.metadata_from_gguf(GGUF)
            self.assertEqual(meta["architecture"], "qwen35")
            self.assertEqual(meta["block_count"], 32)
            self.assertEqual(meta["hidden_size"], 2560)
            self.assertEqual(meta["heads"], 16)
            self.assertEqual(meta["kv_heads"], 4)
            self.assertEqual(meta["head_dim"], 256)

            paths = build_model.write_model(
                meta, name="qwen35-4b", out_dir=Path(tmp)
            )
            geometry = read_json(paths["qwen35-4b.geometry.json"])
            self.assertEqual(geometry["source"]["entry"], "gguf")
            full_attention = [
                layer["index"]
                for layer in geometry["layers"]
                if layer["attention"] == "full_causal"
            ]
            self.assertEqual(full_attention, [3, 7, 11, 15, 19, 23, 27, 31])
            self.assertEqual(
                geometry["weight_layout"]["q_gate_interleaved_layers"],
                full_attention,
            )
            self.assertGreater(geometry["global_weight_bytes"], 0)

            layer = parse_layer_document(read_json(paths["qwen35-4b.layer.json"]))
            self.assertEqual(layer.id, "qwen35-4b-layer3")
            self.assertIn("W_q_gate_packed", {weight.id for weight in layer.weights})
            self.assertEqual(sum(w.storage_bytes for w in layer.weights), geometry["layers"][3]["weight_bytes"])


if __name__ == "__main__":
    unittest.main()
