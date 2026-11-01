"""Controller precision arithmetic; no timer/backend or hardware is created."""
from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from profiling.dynamic_validation.idle_v2 import precision,prepare
from profiling.dynamic_validation import idle_campaign as base


class IdlePrecisionTests(TestCase):
    def test_coarse_sleep_actuation_is_rejected_without_dropping_samples(self):
        result=precision([9000000,11000000,15000000],1)
        self.assertFalse(result['qualified']);self.assertEqual(result['samples'],3)
        self.assertFalse(result['observations_removed'])

    def test_precision_limits_have_separate_median_and_tail(self):
        self.assertTrue(precision([1100000]*16,1)['qualified'])
        result=precision([1100000]*13+[2500000]*3,1)
        self.assertFalse(result['qualified']);self.assertEqual(result['median_overshoot_ns'],100000)
        self.assertEqual(result['p90_overshoot_ns'],1500000)

    def test_zero_idle_is_not_a_scheduler_yield(self):
        result=precision([100,200,100],0)
        self.assertTrue(result['zero_control']);self.assertTrue(result['qualified'])

    def test_early_timer_receipt_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError,'before requested'):precision([500000],1)

    def test_prepare_preserves_base_manifest_with_real_exclusive_writer(self):
        def draft(out,binary,**kwargs):
            out=Path(out);out.mkdir()
            base.write(out/'manifest.json',dict(schema='dynamic-idle-campaign-1',sources=[],jobs=[dict(request={})],
                protocol=dict(initial_state='explicit prime')))
            return dict(out=str(out))
        temporary_parent=(base.ROOT/'build').resolve()
        self.assertTrue(temporary_parent.is_relative_to(base.ROOT.resolve()))
        with TemporaryDirectory(dir=temporary_parent) as temporary:
            out=Path(temporary)/'new'
            with patch('profiling.dynamic_validation.idle_v2.base.prepare',side_effect=draft):prepare(out,'unused.exe')
            original=base.read(out/'base-preparation-manifest.json');final=base.read(out/'manifest.json')
            self.assertEqual(original['schema'],'dynamic-idle-campaign-1')
            self.assertEqual(final['schema'],'dynamic-idle-campaign-v2')
            self.assertEqual(final['base_preparation_manifest']['sha256'],base.sha(out/'base-preparation-manifest.json'))
            with self.assertRaises(FileExistsError):base.write(out/'manifest.json',dict(overwritten=True))
