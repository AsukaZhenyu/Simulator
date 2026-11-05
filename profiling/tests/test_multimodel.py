"""The three gates, tested against arithmetic and hand-built documents.

No model file, no network and no GPU is needed.  A real GGUF appears only in the
skip-when-absent cases, because what has to be true here is that a *wrong* file
cannot pass -- not that the publisher's bytes are still where they were.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from profiling.multimodel import devices, fetch_models, run, suite

HERE = Path(__file__).resolve().parents[1] / "multimodel"


def model(**overrides) -> dict:
    base = {"id": "toy", "role": "dense", "repository": "o/r", "filename": "toy.gguf",
            "revision": "0" * 40, "bytes": 0, "sha256": "0" * 64}
    base.update(overrides)
    return base


def write(path: Path, payload: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return model(id=path.stem, filename=path.name, bytes=len(payload),
                 sha256=hashlib.sha256(payload).hexdigest())


class TheManifestDeclaresExactlyTheModelsThatCanBeRepresented(unittest.TestCase):
    def setUp(self):
        self.document = fetch_models.read_manifest(HERE / "models.json")

    def test_schema_and_a_unique_id_per_model(self):
        self.assertEqual(self.document["schema"], "multimodel-models-1")
        ids = [entry["id"] for entry in self.document["models"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_set_is_three_dense_and_three_sparse(self):
        # The published table is exactly this shape; a fourth model on either
        # side is a campaign decision, not an incidental edit.
        roles = [entry["role"] for entry in self.document["models"]]
        self.assertEqual(roles.count("dense"), 3)
        self.assertEqual(roles.count("moe"), 3)
        self.assertEqual(len(roles), 6)

    def test_every_model_pins_a_commit_not_a_branch(self):
        for entry in self.document["models"]:
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["bytes"], 0)

    def test_the_declared_layer_kinds_account_for_every_block(self):
        for entry in self.document["models"]:
            geometry = entry["geometry"]
            self.assertEqual(sum(geometry["attention"]["layer_kind_counts"].values()),
                             geometry["block_count"], entry["id"])

    def test_only_full_causal_attention_is_declared(self):
        # The cost table has one attention term.  A model whose blocks are not
        # all full_causal would need a term that does not exist, so it must never
        # be added to the manifest without changing the model first.
        for entry in self.document["models"]:
            self.assertEqual(set(entry["geometry"]["attention"]["layer_kind_counts"]),
                             {"full_causal"}, entry["id"])

    def test_moe_models_declare_their_expert_geometry(self):
        for entry in self.document["models"]:
            ffn = entry["geometry"]["ffn"]
            if entry["role"] != "moe":
                self.assertEqual(ffn["kind"], "dense")
                continue
            self.assertEqual(ffn["kind"], "moe")
            self.assertGreater(ffn["experts"], ffn["top_k"])
            self.assertGreater(ffn["expert_intermediate_size"], 0)


class BytesAreProvenRatherThanAssumed(unittest.TestCase):
    def test_absent_missing_size_and_hash_are_three_different_answers(self):
        with tempfile.TemporaryDirectory() as root:
            declared = model(bytes=4, sha256=hashlib.sha256(b"good").hexdigest())
            path = Path(root) / "toy.gguf"
            self.assertEqual(fetch_models.verify(path, declared), "missing")
            path.write_bytes(b"bad")
            self.assertTrue(fetch_models.verify(path, declared).startswith("size"))
            path.write_bytes(b"evil")
            self.assertTrue(fetch_models.verify(path, declared).startswith("sha256"))
            path.write_bytes(b"good")
            self.assertIsNone(fetch_models.verify(path, declared))

    def test_a_file_that_exists_but_differs_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            declared = model(filename="toy.gguf", bytes=4,
                             sha256=hashlib.sha256(b"good").hexdigest())
            (Path(root) / "toy.gguf").write_bytes(b"evil")
            with self.assertRaises(SystemExit) as raised:
                fetch_models.download(declared, Path(root))
            self.assertIn("move it away", str(raised.exception))
            self.assertEqual((Path(root) / "toy.gguf").read_bytes(), b"evil")

    def test_the_manifest_is_read_from_disk_not_guessed_from_filenames(self):
        with tempfile.TemporaryDirectory() as root:
            document = {"schema": "multimodel-models-1",
                        "models": [model(filename="a.gguf")]}
            path = Path(root) / "m.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(fetch_models.read_manifest(path)["models"][0]["id"], "toy")
            del document["models"][0]["sha256"]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(SystemExit):
                fetch_models.read_manifest(path)


class GeometryIsComparedFieldByField(unittest.TestCase):
    declared = {"block_count": 28,
                "attention": {"heads": 16, "kv_heads": 8, "layer_kind_counts": {"full_causal": 28}},
                "ffn": {"kind": "dense", "experts": None}}

    def observed(self, **overrides):
        document = json.loads(json.dumps(self.declared))
        document["vocab_size"] = 151936
        for path, value in overrides.items():
            node = document
            parts = path.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
        return document

    def test_a_matching_geometry_passes_and_extra_fields_are_reported_not_fatal(self):
        result = run.compare(self.declared, self.observed())
        self.assertTrue(result["ok"])
        self.assertIn("vocab_size", result["observed_only"])
        self.assertEqual(result["mismatches"], [])

    def test_a_mismatch_names_the_field_it_disagrees_on(self):
        result = run.compare(self.declared, self.observed(**{"attention.kv_heads": 4}))
        self.assertFalse(result["ok"])
        self.assertEqual(result["mismatches"][0]["field"], "attention.kv_heads")
        self.assertEqual(result["mismatches"][0]["observed"], 4)

    def test_null_in_the_manifest_means_not_asserted_not_must_be_null(self):
        result = run.compare(self.declared, self.observed(**{"ffn.experts": 40}))
        self.assertTrue(result["ok"])
        self.assertIn("ffn.experts", result["unasserted_null"])

    def test_a_field_the_generated_geometry_lacks_is_a_mismatch(self):
        observed = self.observed()
        del observed["attention"]["kv_heads"]
        result = run.compare(self.declared, observed)
        self.assertFalse(result["ok"])
        self.assertIn("absent", result["mismatches"][0]["reason"])

    def test_norm_epsilon_from_float32_is_not_a_mismatch(self):
        declared = {"attention": {"norm_epsilon": 1e-06}}
        observed = {"attention": {"norm_epsilon": 9.999999974752427e-07}}
        self.assertTrue(run.compare(declared, observed)["ok"])
        self.assertFalse(run.compare({"attention": {"norm_epsilon": 2e-06}}, observed)["ok"])


class ADeviceDescriptorCarriesFactsAndNoPrices(unittest.TestCase):
    def descriptor(self, **overrides) -> dict:
        document = {"schema": devices.SCHEMA, "device_id": "toy-windows",
                    "collected_utc": "2026-11-03T00:00:00+00:00",
                    "os": "Windows-11", "machine_architecture": "AMD64",
                    "logical_processors": 16, "memory_bytes": 34359738368,
                    "gpu": [], "unknown_fields_are_null_not_inferred": True}
        document.update(overrides)
        return document

    def test_a_complete_descriptor_checks_out(self):
        self.assertEqual(devices.check(self.descriptor()), [])

    def test_an_unobserved_machine_is_null_rather_than_guessed(self):
        document = self.descriptor(memory_bytes=None, logical_processors=None, gpu=[])
        self.assertEqual(devices.check(document), [])

    def test_a_descriptor_without_the_null_promise_is_rejected(self):
        document = self.descriptor()
        document["unknown_fields_are_null_not_inferred"] = False
        self.assertTrue(devices.check(document))

    def test_a_missing_field_or_foreign_schema_is_rejected(self):
        document = self.descriptor()
        del document["gpu"]
        self.assertTrue(devices.check(document))
        self.assertTrue(devices.check(self.descriptor(schema="something-else")))

    def test_no_measured_price_can_hide_in_a_descriptor(self):
        # devices.py only ever writes structural facts; this pins that the
        # survey's vocabulary has no place for a duration.
        document = devices.survey()
        text = json.dumps(document).lower()
        for banned in ("duration", "latency", "price", "ns\"", "throughput", "flops"):
            self.assertNotIn(banned, text)


class TheBoundaryMatricesComeFromTheTensorDirectory(unittest.TestCase):
    """A token step also reads the embedding and multiplies the LM head.

    Neither is part of the decoder block, and their storage types cannot be
    inferred from the quantization label: a Q4_K_M file stores them differently.
    """

    def meta(self, *, vocab=5, hidden=4, tensors=None):
        if tensors is None:
            tensors = [{"name": "token_embd.weight", "shape": [hidden, vocab], "dtype": "q4_k"},
                       {"name": "output.weight", "shape": [hidden, vocab], "dtype": "q6_k"}]
        return {"vocab_size": vocab, "hidden_size": hidden, "global_tensors": tensors}

    def test_the_two_matrices_keep_their_own_storage_types(self):
        actual = suite.boundary(self.meta())
        self.assertEqual(actual, {"vocab": 5, "token_embd_dtype": "q4_k",
                                  "lm_head_dtype": "q6_k", "tied_embeddings": False})

    def test_a_tied_checkpoint_has_no_separate_head_tensor(self):
        actual = suite.boundary(self.meta(tensors=[
            {"name": "token_embd.weight", "shape": [4, 5], "dtype": "q4_k"}]))
        self.assertTrue(actual["tied_embeddings"])
        self.assertEqual(actual["lm_head_dtype"], "q4_k")

    def test_an_unpriceable_boundary_is_an_error_not_a_default(self):
        with self.assertRaisesRegex(ValueError, "token_embd"):
            suite.boundary(self.meta(tensors=[]))
        with self.assertRaisesRegex(ValueError, "geometry implies"):
            suite.boundary(self.meta(tensors=[
                {"name": "token_embd.weight", "shape": [4, 6], "dtype": "f32"}]))
        with self.assertRaisesRegex(ValueError, "not one of"):
            suite.boundary(self.meta(tensors=[
                {"name": "token_embd.weight", "shape": [4, 5], "dtype": "q8_0"}]))
        with self.assertRaisesRegex(ValueError, "vocabulary size"):
            suite.boundary(self.meta(vocab=None))

    def test_boundary_cases_exist_only_where_the_token_step_is_counted_once(self):
        from llm_infer_model.m2.dense_contract import weight_shapes
        config = dict(hidden=256, intermediate=512, heads=2, kv_heads=1, head_dim=64,
                      attention_gate="none", qk_norm="none", rope_mode="normal",
                      rope_dims=64, rope_base=10000, norm_epsilon=1e-5)
        config["weight_dtypes"] = {role: "f32" for role in weight_shapes(config)}
        description = {"config": config, "boundary": {"vocab": 5, "token_embd_dtype": "q4_k",
                                                      "lm_head_dtype": "q6_k"}}
        shapes = {"batch": 1, "prefill": [{"tokens": 4}],
                  "decode": [{"tokens": 1, "visible_history": 0},
                             {"tokens": 1, "visible_history": 64}]}
        document = suite.suite_document(description, shapes, model_id="toy")
        kinds = [(case["kind"], case["phase"], case["tokens"], case["visible_history"])
                 for case in document["cases"]]
        # Four decoder stages on all three shapes, plus two boundaries on the two
        # shapes with no history: a cached step does not repeat the embedding.
        self.assertEqual(len(kinds), 4 * 3 + 2 * 2)
        self.assertIn(("embedding", "prefill", 4, 0), kinds)
        self.assertIn(("head", "decode", 1, 0), kinds)
        self.assertNotIn(("head", "decode", 1, 64), kinds)
        for case in document["cases"]:
            if case["kind"] in suite.BOUNDARY_STAGES:
                self.assertEqual(case["config"]["vocab"], 5)
                self.assertEqual(case["config"]["weight_dtypes"]["token_embd"], "q4_k")
                self.assertEqual(case["config"]["weight_dtypes"]["lm_head"], "q6_k")
                self.assertEqual(case["config"]["weight_dtypes"]["final_norm"], "f32")
            else:
                self.assertNotIn("vocab", case["config"])
        suite_plan = suite.dense_contract.build_plan(document, 2)
        self.assertEqual(len(suite_plan), len(kinds))


class TheQKNormWidthIsReadFromTheGainVector(unittest.TestCase):
    """OLMoE normalizes the head-concatenated projection, not each head.

    Its QK-norm gain is [hidden], and llama.cpp applies it before the reshape.
    Asking only "is there a q_norm tensor" would declare a per-head
    normalization the checkpoint does not perform, and the descriptor would then
    disagree with the tensor directory on the gain's shape.
    """

    def meta(self, *, q_width=None, k_width=None):
        hidden, heads, head_dim, experts, width, vocab = 4, 2, 2, 2, 2, 4

        def tensor(name, shape):
            return {"name": f"blk.0.{name}.weight", "shape": shape, "dtype": "f32",
                    "storage_bytes": 4 * math.prod(shape)}

        tensors = [tensor("attn_norm", [hidden]), tensor("attn_q", [hidden, hidden]),
                   tensor("attn_k", [hidden, hidden]), tensor("attn_v", [hidden, hidden]),
                   tensor("attn_output", [hidden, hidden]),
                   tensor("attn_q_norm", [q_width if q_width is not None else hidden]),
                   tensor("attn_k_norm", [k_width if k_width is not None else hidden]),
                   tensor("ffn_norm", [hidden]), tensor("ffn_gate_inp", [hidden, experts]),
                   tensor("ffn_gate_exps", [hidden, width, experts]),
                   tensor("ffn_up_exps", [hidden, width, experts]),
                   tensor("ffn_down_exps", [width, hidden, experts])]
        boundary = [{"name": "token_embd.weight", "shape": [hidden, vocab], "dtype": "f32",
                     "storage_bytes": 4 * hidden * vocab},
                    {"name": "output.weight", "shape": [hidden, vocab], "dtype": "f32",
                     "storage_bytes": 4 * hidden * vocab}]
        return {"architecture": "olmoe", "block_count": 1, "hidden_size": hidden,
                "intermediate_size": width, "heads": heads, "kv_heads": heads,
                "head_dim": head_dim, "rope_dims": None, "rope_base": 10000.0,
                "norm_epsilon": 1e-5, "experts": experts, "top_k": 1,
                "expert_intermediate_size": width, "shared_expert_intermediate_size": None,
                "vocab_size": vocab, "global_tensors": boundary,
                "layers": [{"index": 0, "attention": "full_causal", "ffn": "moe",
                            "tensors": tensors}]}

    def test_a_gain_of_hidden_width_is_one_normalization_per_token(self):
        description = suite.describe(self.meta())
        self.assertEqual(description["config"]["qk_norm"], "rms_hidden")
        shapes = {weight["id"]: weight["shape"] for weight in description["weights"]}
        self.assertEqual(shapes["W_q_norm"], [4])
        self.assertEqual(shapes["W_k_norm"], [4])

    def test_a_gain_of_head_width_is_still_the_per_head_normalization(self):
        self.assertEqual(suite.describe(self.meta(q_width=2, k_width=2))["config"]["qk_norm"],
                         "rms")

    def test_any_other_width_is_refused_rather_than_rounded_to_one_of_them(self):
        with self.assertRaisesRegex(ValueError, "QK norm gain"):
            suite.describe(self.meta(q_width=3, k_width=3))
        with self.assertRaisesRegex(ValueError, "different widths"):
            suite.describe(self.meta(q_width=4, k_width=2))


class StructureRefusesToRunWithoutTheDeclaredBytes(unittest.TestCase):
    def test_a_missing_model_is_skipped_and_counted_as_a_problem(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            args = type("Args", (), {"root": Path(root), "manifest": HERE / "models.json",
                                     "out": Path(out), "only": ["qwen3-0.6b"]})
            self.assertEqual(run.command_structure(args), 1)
            self.assertEqual(list(Path(out).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
