"""The pooled-median rule, and the tool that applies it to an existing table."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from history_invariant_prices import (MAX_HISTORY_REPLICATE_SPREAD, ReplicateError,
                                      pool_history_invariant)
from llm_infer_model.m2 import dense_contract
from llm_infer_model.m2.costs import canonical_contract_hash
from llm_infer_model.m2.dense_contract import validate_config
from llm_infer_model.tensor.spec import InvalidInput
import reprice_history_invariant as repricer


def config():
    return validate_config({"hidden": 16, "intermediate": 32, "heads": 4, "kv_heads": 2, "head_dim": 4,
        "attention_gate": "sigmoid", "qk_norm": "rms", "rope_mode": "normal", "rope_dims": 4,
        "rope_base": 10000, "norm_epsilon": 1e-6, "weight_dtypes": {role: "f32" for role in
            ("input_norm", "q", "k", "v", "gate_proj", "q_norm", "k_norm", "output",
             "post_attention_norm", "gate", "up", "down")}})


def entry(kind, history, duration_ns, *, workspace_bytes=128):
    signature = dense_contract.signature(config(), kind, "decode", 1, history, 2)
    return {"kind": kind, "device": "cpu", "phase": "decode", "batch": 1, "tokens": 1,
            "visible_history": history, "duration_ns": duration_ns,
            "workspace_bytes": workspace_bytes, "provenance": "measured_macro",
            "accounting": "per_request", "physical_signature": signature,
            "contract_hash": canonical_contract_hash(signature),
            "workspace_scope": "macro_internal_scratch",
            "workspace_source": "ggml_gallocr_internal_buffer_plus_cpu_graph_plan"}


PRICES = {"projection": {128: 1000, 512: 1100, 1024: 1050},
          "attention": {128: 500, 512: 900, 1024: 1300},
          "dense_ffn": {128: 2000, 512: 2100, 1024: 2200}}


def table():
    return {"format_version": "m2-layer-0.3", "source": "synthetic-reprice-fixture",
            "validation_only": False, "comment": "fixture",
            "entries": [entry(kind, history, price)
                        for kind, prices in PRICES.items() for history, price in prices.items()]}


class PoolTests(unittest.TestCase):
    def test_the_pooled_price_is_the_median_and_the_spread_is_recorded(self):
        pooled = pool_history_invariant([(128, "a", 1000), (512, "b", 1100), (1024, "c", 1050)])
        self.assertEqual(pooled["pooled_ns"], 1050)
        self.assertAlmostEqual(pooled["relative_spread"], 100/1050)
        self.assertEqual([item["visible_history"] for item in pooled["replicates"]], [128, 512, 1024])

    def test_two_readings_pool_to_their_rounded_midpoint(self):
        self.assertEqual(pool_history_invariant([(128, "a", 1000), (512, "b", 1001)])["pooled_ns"], 1001)

    def test_one_reading_is_not_a_set_of_replicates(self):
        with self.assertRaisesRegex(ReplicateError, "at least two"):
            pool_history_invariant([(128, "a", 1000)])

    def test_a_repeated_depth_is_not_a_replicate(self):
        with self.assertRaisesRegex(ReplicateError, "repeat a depth"):
            pool_history_invariant([(128, "a", 1000), (128, "b", 1100)])

    def test_a_spread_past_the_limit_is_not_pooled(self):
        self.assertEqual(MAX_HISTORY_REPLICATE_SPREAD, 0.5)
        with self.assertRaisesRegex(ReplicateError, "not replicate noise"):
            pool_history_invariant([(128, "a", 1000), (512, "b", 4000)])


class RepriceTests(unittest.TestCase):
    def setUp(self):
        self.original = table()

    def test_only_the_history_invariant_decode_prices_are_replaced(self):
        repriced, report = repricer.reprice(self.original, "median_of_history_replicates")
        by_kind = {}
        for item in repriced["entries"]:
            by_kind.setdefault(item["kind"], {}).setdefault(item["visible_history"], item["duration_ns"])
        self.assertEqual(by_kind["projection"], {128: 1050, 512: 1050, 1024: 1050})
        self.assertEqual(by_kind["dense_ffn"], {128: 2100, 512: 2100, 1024: 2100})
        self.assertEqual(by_kind["attention"], {128: 500, 512: 900, 1024: 1300})
        self.assertEqual(report["entries_changed"], 6)
        self.assertFalse(report["measurement_performed"])

    def test_a_repriced_entry_declares_which_estimator_priced_it(self):
        repriced, _ = repricer.reprice(self.original, "median_of_history_replicates")
        for item in repriced["entries"]:
            stored = item["physical_signature"]
            if item["kind"] == "attention":
                self.assertNotIn("history_invariant_price_policy", stored)
            else:
                self.assertEqual(stored["history_invariant_price_policy"],
                                 dense_contract.PRICE_POLICIES["median_of_history_replicates"])
                self.assertEqual(item["contract_hash"], canonical_contract_hash(stored))

    def test_the_input_table_is_never_mutated(self):
        before = copy.deepcopy(self.original)
        repricer.reprice(self.original, "median_of_history_replicates")
        self.assertEqual(before, self.original)

    def test_the_table_says_a_repriced_pipeline_produced_it(self):
        repriced, report = repricer.reprice(self.original, "median_of_history_replicates")
        self.assertTrue(repriced["source"].startswith("synthetic-reprice-fixture-replicated-median-"))
        self.assertIn("Re-priced by profiling/reprice_history_invariant.py", repriced["comment"])
        self.assertEqual(report["input_source"], "synthetic-reprice-fixture")

    def test_pooling_twice_is_refused_rather_than_pooled_again(self):
        repriced, _ = repricer.reprice(self.original, "median_of_history_replicates")
        with self.assertRaisesRegex(repricer.RepriceError, "not the one its own declarations produce"):
            repricer.reprice(repriced, "median_of_history_replicates")

    def test_entries_that_differ_in_their_workspace_are_not_replicates(self):
        self.original["entries"][1]["workspace_bytes"] = 4096
        with self.assertRaisesRegex(repricer.RepriceError, "not.*replicates"):
            repricer.reprice(self.original, "median_of_history_replicates")

    def test_a_table_whose_hash_disagrees_with_its_signature_is_refused(self):
        self.original["entries"][0]["contract_hash"] = "0" * 64
        with self.assertRaises(InvalidInput):
            repricer.reprice(self.original, "median_of_history_replicates")

    def test_the_cli_writes_both_documents_and_refuses_a_full_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory(prefix="reprice-cli-") as name:
            root = Path(name)
            source = root / "in.json"
            source.write_text(json.dumps(table()), encoding="utf-8")
            out = root / "out"
            self.assertEqual(repricer.main(["--table", str(source), "--output", str(out)]), 0)
            self.assertEqual(sorted(path.name for path in out.iterdir()),
                             ["costs.json", "reprice_report.json"])
            with self.assertRaisesRegex(repricer.RepriceError, "output directory must be empty"):
                repricer.main(["--table", str(source), "--output", str(out)])


if __name__ == "__main__":
    unittest.main()
