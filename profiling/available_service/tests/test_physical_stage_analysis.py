"""Tiny offline invariants; no native process or hardware invocation."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_infer_model.performance.reference_components import ReferenceComponent, evaluate_reference_components
from llm_infer_model.performance.parameters import UnknownParameter
from profiling import physical_stage_analysis as analysis


class ReferenceModelTest(unittest.TestCase):
    def test_ratio_is_model_owned_and_not_clipped(self):
        result=evaluate_reference_components((ReferenceComponent('kernel',100,'gpu_reference'),ReferenceComponent('transfer',20,'static')),
            dict(gpu_reference=Fraction(10)),dict(gpu_reference=Fraction(5)),evidence=('separate reference',),scope='diagnostic')
        self.assertEqual(result['execution'].duration_ns,220)
        self.assertFalse(result['hardware_accuracy_qualified'])
        self.assertEqual(result['components'][0]['ratio'],2)

    def test_missing_reference_unknown(self):
        with self.assertRaises(UnknownParameter):
            evaluate_reference_components((ReferenceComponent('host',10,'cpu_reference'),),{}, {}, evidence=('x',),scope='diagnostic')


class PhysicalAnalysisTest(unittest.TestCase):
    def test_overlapping_API_device_not_added_twice(self):
        span=lambda a,b:dict(begin_ns=a,end_ns=b)
        action=dict(begin_ns=0,end_ns=100,kernel_union=[span(20,50)],device_activity_union=[span(20,60)],
                    runtime_api_union=[span(0,40)],driver_api_union=[span(30,80)])
        parts=analysis.action_components(action)
        self.assertEqual(parts,dict(kernel_activity=30,copy_activity_outside_kernel=10,host_api_outside_device=40,other_host_wall=20))
        self.assertEqual(sum(parts.values()),100)

    def test_CPU_counter_overwall_stays_unknown_completed_response_remains(self):
        clock=dict(clock_id='windows_qpc_floor_ns',clock_origin='absolute_qpc_zero',frequency_hz=10_000_000)
        raw=dict(mode='target',target_action_index=39,program_sha256='p',clock=clock,
            iterations=[{}]*4,samples_requested=4,samples_completed=4,sampling_begin_ns=100,sampling_end_ns=200,
            process_id=2,batch_cpu=dict(main_thread_active_cpu_ns=120))
        main=dict(program_sha256='p',clock=clock,iterations=[dict(begin_ns=300,end_ns=400)])
        result=analysis._reference(dict(raw={'before-cpu':raw},native=main),'before-cpu',before=True)
        self.assertIsNone(result['main_thread_active_share'])
        self.assertEqual(result['rates']['completed_reference_response'],40_000_000)
        self.assertEqual(result['staleness_ns'],100)

    def test_profile_only_frozen_and_no_heldout_offset(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
            campaign=Path(directory)
            jobs=[dict(id='profile-'+str(i),split='profile',program_key='p',condition='idle',instrumentation='cupti',references=True,samples=32) for i in range(3)]
            jobs += [dict(id='heldout-'+str(i),split='heldout',program_key='h',condition='idle',instrumentation='cupti',references=True,samples=32) for i in range(3)]
            manifest=dict(schema='physical-service-campaign-1',jobs=jobs,source_inventory_sha256='s',programs={})
            (campaign/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
            program={key:{} for key in ('boundary_contract','graph','physical','bindings')}
            program['actions']=[dict(kind='COMPUTE',device='gpu')]
            refs=dict(activity_rates=dict(cpu_reference=Fraction(10),gpu_reference=Fraction(10)),
                      completed_rates=dict(cpu_reference=Fraction(10),gpu_reference=Fraction(10)))
            heldout_duration=[100]
            loaded_jobs=[]
            def loader(campaign,manifest,job,**kwargs):
                loaded_jobs.append(job['id'])
                profile=job['split']=='profile'
                duration=100 if profile else heldout_duration[0]
                iterations=[]
                actions=[]
                for i in range(32):
                    begin=1000+i*1000
                    action=dict(iteration=i,action_index=0,begin_ns=begin,end_ns=begin+duration,
                        kernel_union=[dict(begin_ns=begin,end_ns=begin+duration//2)],
                        device_activity_union=[dict(begin_ns=begin,end_ns=begin+duration//2)],runtime_api_union=[],driver_api_union=[])
                    actions.append(action)
                    iterations.append(dict(iteration=i,begin_ns=begin,end_ns=begin+duration,actions=[action]))
                return dict(job=job,program=program,sources=[],
                    execution=dict(program=dict(seed=1 if profile else 2),started_at_utc='2026-10-22T12:00:00+00:00',ended_at_utc='2026-10-22T10:00:00+00:00'),
                    native=dict(iterations=iterations,process_id=(10 if profile else 20)+int(job['id'][-1])),
                    parsed=dict(actions=actions,capture_qualified=True,capture_issues=[],selected_action_issues=[]))
            with patch.object(analysis,'_load_job',side_effect=loader),patch.object(analysis,'_references',return_value=refs),patch.object(analysis,'_source_code_receipts',return_value=[]):
                frozen=analysis.freeze_campaign_profile(campaign,frozen_at_utc='2026-10-22T11:00:00+00:00')
                self.assertEqual(loaded_jobs,['profile-0','profile-1','profile-2'])
                first=analysis.evaluate_campaign_heldout(campaign,frozen)
                heldout_duration[0]=200
                second=analysis.evaluate_campaign_heldout(campaign,frozen)
            for name in analysis.CANDIDATES:
                a=first['runs'][0]['predictions'][name]
                b=second['runs'][0]['predictions'][name]
                self.assertEqual(a.get('predicted_ns'),b.get('predicted_ns'))
            self.assertFalse(first['hardware_accuracy_qualified'])
            self.assertFalse(first['candidate_selection_after_heldout'])
            self.assertEqual(first['runs'][0]['predictions']['reference_activity_candidate']['predicted_ns'],100)


if __name__=='__main__':
    unittest.main()
