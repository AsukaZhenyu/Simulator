"""Offline invariants for AFTER-only collection adaptation; no actors."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import unittest
from unittest.mock import patch

from profiling import physical_posttarget_analysis as new
from profiling import physical_stage_analysis as frozen


class PostTargetAnalysisTest(unittest.TestCase):
    def test_frozen_predictor_is_the_identical_function(self):
        self.assertIs(new._predict,frozen._predict)
        self.assertIs(new._references,frozen._references)
        self.assertIs(new._components,frozen._components)

    def test_after_protocol_cannot_change_predictions(self):
        campaign=Path('.').resolve()
        component=dict(means=dict(task={'numerator':100,'denominator':1}),
                       template=dict(task=dict(reference='cpu_reference',category='other_host_wall',action_index=0)))
        model=dict(schema=frozen.SCHEMA,campaign=str(campaign),manifest_sha256='manifest',
            source_receipts=[],profile_identities=[dict(process_id=1,seed=1)],
            signature_sha256='sig',frozen_at_utc='2026-10-22T11:00:00+00:00',
            candidates=list(frozen.CANDIDATES),conditions=dict(idle=component),
            idle_inclusive=component,observer_bare_idle=None,
            profile_before_reference_rates={kind:dict(cpu_reference={'numerator':10,'denominator':1},gpu_reference={'numerator':10,'denominator':1}) for kind in ('activity_rates','completed_rates')},
            observer=dict(neutrality_qualified=False))
        job=dict(id='h',split='heldout_bare',condition='idle',instrumentation='bare')
        loaded=dict(execution=dict(started_at_utc='2026-10-22T12:00:00+00:00',program=dict(seed=2)),
            native=dict(process_id=2,iterations=[dict(begin_ns=1000+i*1000,end_ns=1100+i*1000) for i in range(32)]),
            program={},parsed=None)
        before=dict(activity_rates=dict(cpu_reference=Fraction(10),gpu_reference=Fraction(10)),
                    completed_rates=dict(cpu_reference=Fraction(10),gpu_reference=Fraction(10)))
        protocol=dict(posttarget_cleanup_job_ids=[],declared_at_utc='2026-10-22T11:30:00+00:00')
        def read(path,expected=None):
            return {'proof':'retained'} if str(path)=='cleanup.json' else {}
        def run():
            with patch.object(new,'_read',side_effect=read),patch.object(new,'_posttarget_jobs',return_value=([job],deepcopy(protocol))),patch.object(new,'_load_job',return_value=deepcopy(loaded)),patch.object(new,'_signature',return_value='sig'),patch.object(new,'_references',return_value=deepcopy(before)):
                return new.evaluate_campaign_heldout(campaign,model,split='heldout_bare')
        original=run()
        protocol['posttarget_cleanup_job_ids']=['h']
        loaded['execution'].update(posttarget_cleanup_applied=True,posttarget_after_reference_scope='unloaded_drift_only',posttarget_cleanup_receipt=dict(path='cleanup.json',sha256='retained'))
        changed=run()
        self.assertEqual(original['runs'][0]['predictions'],changed['runs'][0]['predictions'])
        self.assertTrue(original['runs'][0]['after_audit_only']['same_load_drift_valid'])
        self.assertFalse(changed['runs'][0]['after_audit_only']['same_load_drift_valid'])
        self.assertEqual(changed['runs'][0]['predictions']['reference_activity_candidate']['predicted_ns'],100)
        self.assertFalse(changed['hardware_accuracy_qualified'])
        self.assertTrue(changed['frozen_predictor_reused_without_refit'])


if __name__=='__main__':
    unittest.main()
