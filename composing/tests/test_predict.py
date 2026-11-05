"""A prediction with no declared terms is exactly the table's sum, untouched."""
import json
from pathlib import Path
import tempfile
import unittest

from composing.predict import predict
from composing.spec import SHAPES
from composing.terms import Term

MODELS = {"m": {"id": "m", "filename": "m.gguf",
                "geometry": {"block_count": 2, "vocab_size": 7, "ffn": {"kind": "dense"}}}}


def _word(kind, phase, tokens, median):
    return {"kind": kind, "shape": f"{phase}/B1/q{tokens}/hist0",
            "sample_summary": {"median_ns": median, "cv": 0.01},
            "physical_signature": {"config": {"vocab": 7, "token_embd_dtype": "q4_k",
                                              "lm_head_dtype": "q6_k"}}}


def report():
    entries = []
    for phase, tokens, history in SHAPES.values():
        shape = f"{phase}/B1/q{tokens}/hist{history}"
        for kind in ("projection", "attention", "dense_ffn"):
            entries.append({"kind": kind, "shape": shape,
                            "sample_summary": {"median_ns": 100, "cv": 0.01}})
    entries.append(_word("embedding", "prefill", 256, 10))
    entries.append(_word("embedding", "decode", 1, 10))
    entries.append(_word("head", "decode", 1, 10))
    return {"validation_only": False, "entries": entries}


class PredictTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        table = self.root / "table"
        table.mkdir()
        (table / "export_report.json").write_text(json.dumps(report()), encoding="utf-8")
        self.table = table

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_zero_term_prediction_is_the_table_sum_and_says_so(self):
        document = predict({"m": self.table}, MODELS, self.root)
        self.assertEqual(document["composition_terms"], [])
        shape = document["models"]["m"]["shapes"]["tg128"]
        self.assertEqual(shape["total_ns"],
                         shape["block_sum_ns"] * 2 + shape["embedding"]["median_ns"]
                         + shape["head"]["median_ns"])
        # No per-shape term record exists when nothing was declared: a zero-term
        # shape stays byte-for-byte the shape archived comparisons were made
        # against.
        self.assertNotIn("terms_ns", shape)

    def test_an_enabled_term_is_added_only_where_its_unit_says(self):
        terms = (Term(name="per_layer_marginal", phase="decode", value=500.0, enabled=True),)
        document = predict({"m": self.table}, MODELS, self.root, terms=terms)
        plain = predict({"m": self.table}, MODELS, self.root)
        decode = document["models"]["m"]["shapes"]["tg128"]
        prefill = document["models"]["m"]["shapes"]["pp256"]
        self.assertEqual(decode["total_ns"],
                         plain["models"]["m"]["shapes"]["tg128"]["total_ns"] + 500 * 2)
        self.assertIn("terms_ns", decode)
        # A decode term does not touch a prefill shape: the phase is part of the
        # declaration, not a hint.
        self.assertEqual(prefill["total_ns"],
                         plain["models"]["m"]["shapes"]["pp256"]["total_ns"])
        self.assertNotIn("terms_ns", prefill)
        self.assertEqual(document["composition_terms"][0]["enabled"], True)

    def test_a_disabled_term_is_recorded_but_not_priced(self):
        # The document keeps the declaration with its knob, so a reader can tell
        # "not priced" from "priced at zero".
        document = predict({"m": self.table}, MODELS, self.root,
                           terms=(Term(name="per_token_fixed", phase="prefill", value=2.0),))
        self.assertEqual(document["composition_terms"][0]["enabled"], False)
        self.assertIn("knob", document["composition_terms"][0])
        self.assertNotIn("terms_ns", document["models"]["m"]["shapes"]["pp256"])
        self.assertEqual(set(document["composition_terms"][0]),
                         {"name", "phase", "value", "unit", "enabled", "note", "knob",
                          "quantity"})


if __name__ == "__main__":
    unittest.main()
