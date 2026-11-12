"""The declared model contract a second model brings to the window cohort.

The published template declares OLMoE's contract because OLMoE was the model it
was written for. These tests pin the two properties the port depends on: the
contract comes from the pinned geometry rather than from the template, and a
model the template was written for still gets a byte-identical template.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from profiling.real_moe_validation import prepare_portable

HERE = Path(__file__).resolve().parent


def pinned(model_id):
    return prepare_portable.pinned_model(model_id)


class ContractTest(unittest.TestCase):
    def test_a_pinned_no_expert_model_declares_no_expert_dimension(self):
        contract = prepare_portable.contract_of(pinned("qwen3-0.6b"))
        self.assertEqual(contract, dict(architecture="qwen3", layers=28, experts=0, top_k=0))

    def test_a_pinned_mixture_model_declares_its_own_geometry(self):
        contract = prepare_portable.contract_of(pinned("olmoe-1b-7b"))
        self.assertEqual(contract, dict(architecture="olmoe", layers=16, experts=64, top_k=8))

    def test_a_ffn_kind_that_contradicts_its_geometry_is_refused(self):
        model = dict(pinned("qwen3-0.6b"))
        model["geometry"] = json.loads(json.dumps(model["geometry"]))
        model["geometry"]["ffn"] = dict(model["geometry"]["ffn"], kind="dense", experts=8, top_k=2)
        with self.assertRaisesRegex(ValueError, "FFN kind and expert geometry disagree"):
            prepare_portable.contract_of(model)

    def test_a_manifest_entry_without_geometry_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no pinned geometry block"):
            prepare_portable.contract_of(dict(id="mystery", role="dense"))


class TemplateTest(unittest.TestCase):
    def template(self):
        return json.loads(prepare_portable.SHIPPED_TEMPLATE.read_text(encoding="utf-8"))

    def test_the_model_the_template_was_written_for_keeps_its_contract(self):
        model = dict(path="C:/models/olmoe.gguf", sha256="a" * 64, bytes=4213513024)
        contract = prepare_portable.contract_of(pinned("olmoe-1b-7b"))
        out, changed = prepare_portable.template_for(model, self.template(), contract)
        self.assertEqual(changed, ["model_path", "model_sha256"])
        self.assertEqual(out["model_contract"], self.template()["model_contract"])

    def test_a_second_model_brings_its_own_contract_and_the_template_says_so(self):
        model = dict(path="C:/models/qwen3.gguf", sha256="b" * 64, bytes=396705472)
        contract = prepare_portable.contract_of(pinned("qwen3-0.6b"))
        out, changed = prepare_portable.template_for(model, self.template(), contract)
        self.assertIn("model_contract", changed)
        self.assertEqual(out["model_contract"], contract)
        self.assertTrue(all(key in changed for key in
                            ("model_path", "model_sha256", "model_bytes", "model_contract")))
        # Nothing else in the published template moves.
        untouched = {k: v for k, v in self.template().items()
                     if k not in ("model_path", "model_sha256", "model_bytes", "model_contract")}
        self.assertEqual({k: v for k, v in out.items() if k in untouched}, untouched)


if __name__ == "__main__":
    unittest.main()
