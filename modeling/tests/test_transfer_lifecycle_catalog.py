from __future__ import annotations
import unittest
from dataclasses import asdict

from llm_infer_model.performance.transfer_lifecycle_catalog import ScopedTransferLifecycle
from llm_infer_model.performance.parameters import InvalidModel,UnknownParameter
# `unittest discover -s modeling/tests` puts that directory on sys.path but
# imports each file as a top-level module, so this sibling import is absolute.
from test_transfer_lifecycle import model

class TransferLifecycleCatalogTests(unittest.TestCase):
    def kw(self):return dict(context_sha256='a'*64,preparation_sha256='b'*64)
    def scoped(self):return ScopedTransferLifecycle(model(first=False,window=True),((1024,2000000),(1048576,10000000)),('independent-evaluation#sha256='+'c'*64,))

    def test_validated_coordinate_works_without_overwriting_full_grid_failure(self):
        p=self.scoped();self.assertFalse(p.first_qualified)
        actual=p.first_response_ns(1024,2000000,**self.kw())
        expected=p.lifecycle.first_response_ns(1024,2000000,**self.kw(),require_qualified=False)
        self.assertEqual(actual,expected);self.assertFalse(p.describe()['formula_or_parameters_changed'])

    def test_failed_first_coordinate_refuses_even_when_its_complete_window_passes(self):
        p=self.scoped()
        with self.assertRaises(UnknownParameter):p.first_response_ns(1024,10000000,**self.kw())
        self.assertGreater(p.window_ns(1024,10000000,64,**self.kw()),0)

    def test_unseen_byte_idle_context_and_window_do_not_fallback(self):
        p=self.scoped()
        for size,idle in ((2048,2000000),(1024,3000000)):
            with self.assertRaises(UnknownParameter):p.first_response_ns(size,idle,**self.kw())
        with self.assertRaises(UnknownParameter):p.first_response_ns(1024,2000000,context_sha256='d'*64,preparation_sha256='b'*64)
        with self.assertRaises(UnknownParameter):p.window_ns(1024,2000000,32,**self.kw())

    def test_allowlist_cannot_create_new_measured_coordinates(self):
        base=model(first=False,window=True)
        with self.assertRaises(InvalidModel):ScopedTransferLifecycle(base,((2048,2000000),),('e',))
        with self.assertRaises(InvalidModel):ScopedTransferLifecycle(base,((1024,3000000),),('e',))
        with self.assertRaises(InvalidModel):ScopedTransferLifecycle(base,((1024,2000000),(1024,2000000)),('e',))

    def test_additive_profile_roundtrip_retains_both_original_flags(self):
        p=self.scoped();value=asdict(p.lifecycle)
        value.update(schema='completed-transfer-lifecycle-1',fit_uses_whole_LLM_target=False,future_target_observations_used=False,
                     qualified_first_coordinates=[list(c) for c in p.qualified_first_coordinates],first_coordinate_qualification_evidence=list(p.qualification_evidence))
        restored=ScopedTransferLifecycle.from_profile(value)
        self.assertFalse(restored.first_qualified);self.assertTrue(restored.window_qualified)
        self.assertEqual(restored.first_response_ns(1048576,10000000,**self.kw()),p.first_response_ns(1048576,10000000,**self.kw()))

if __name__=='__main__':unittest.main()
