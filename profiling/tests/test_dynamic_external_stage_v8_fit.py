"""Synthetic source/forecast checks; none are hardware accuracy measurements."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_infer_model.performance.parameters import UnknownParameter
from llm_infer_model.performance.runtime_response import (
    HostPlacement, RuntimeCell, RuntimeProfile, RuntimeProgram, RuntimeScope, RuntimeStage,
)
from profiling.dynamic_validation import external_stage_v8_fit as fit


class RuntimeForecastTests(unittest.TestCase):
    def models(self,two_APIs=False):
        scope=RuntimeScope('a'*64,HostPlacement(4,4,(1,),(1,),'b'*64))
        sequence=[('c'*64,'gap'),('d'*64,'api'),('e'*64,'gap')]
        if two_APIs:sequence += [('f'*64,'api'),('0'*64,'gap')]
        stages=tuple(RuntimeStage(f'{role}_{i}',key,role) for i,(key,role) in enumerate(sequence))
        cells=[]
        for key,role in sequence:
            for state in fit.STATES:
                value=0 if role=='gap' else 1000000
                if key=='d'*64 and state[0]:value=4000000 if two_APIs else 2000000
                cells.append(RuntimeCell(key,role,state,Fraction(value),('declared software fixture',)))
        return RuntimeProgram(scope,stages,('ordered software fixture',)),RuntimeProfile(scope,tuple(cells),('local software fixture',),False)

    def request(self,replays=4,block=2):
        return dict(cpu_actor=dict(intervals_ms=[[1,3]]),gpu_actor=dict(intervals_ms=[]),
                    replays=replays,block_replays=block)

    def test_static_and_memoryless_forecasts_integrate_planned_state_on_own_clock(self):
        program,profile=self.models();request=self.request()
        static=fit.predict_provider(program,profile,request,'static00')
        ordered=fit.predict_provider(program,profile,request,'ordered_API_gap_four_state')
        macro=fit.predict_provider(program,profile,request,'macro_four_state_sum_of_same_stage_prices')
        self.assertEqual(static['total_ns'],4000000)
        self.assertEqual(ordered['total_ns'],5000000)
        self.assertEqual(macro['total_ns'],ordered['total_ns'])
        self.assertFalse(ordered['observed_target_boundaries_used'])
        self.assertTrue(ordered['own_predicted_completion_clock'])
        self.assertEqual(ordered['planned_entry_ns'],0)

    def test_ordered_and_coarse_macro_baselines_can_differ_without_target_fitting(self):
        program,profile=self.models(two_APIs=True);request=self.request(1,1)
        request['cpu_actor']['intervals_ms']=[[1,2]]
        ordered=fit.predict_provider(program,profile,request,'ordered_API_gap_four_state')
        macro=fit.predict_provider(program,profile,request,'macro_four_state_sum_of_same_stage_prices')
        # The CPU-sensitive first API completes before the pulse; only the
        # unaffected second API overlaps it. A single coarse response loses
        # that distinction even though its prices sum the same local cells.
        self.assertEqual(ordered['total_ns'],2000000)
        self.assertEqual(macro['total_ns'],2600000)
        self.assertFalse(macro['complete_target_duration_fitted'])
        self.assertFalse(macro['internal_GPU_or_DMA_time_added'])

    def test_missing_exact_state_cell_is_unknown_for_both_dynamic_providers(self):
        program,profile=self.models();profile=RuntimeProfile(profile.scope,profile.cells[:-1],profile.evidence,False)
        for provider in ('ordered_API_gap_four_state','macro_four_state_sum_of_same_stage_prices'):
            with self.assertRaises((UnknownParameter,KeyError)):
                fit.predict_provider(program,profile,self.request(),provider)

    def test_frozen_forecast_function_accepts_no_observed_target_labels(self):
        program,profile=self.models()
        with self.assertRaises(TypeError):
            fit.predict_provider(program,profile,self.request(),'ordered_API_gap_four_state',observed_target_ns=1)
        with self.assertRaises(ValueError):
            fit.predict_provider(program,profile,self.request(),'unknown-fallback')


class StreamProvenanceTests(unittest.TestCase):
    def native(self,handles=(0,)):
        return dict(main_thread_id=9,runtime_callbacks=dict(enabled=True,target_thread_id=9,
            foreign_actor_threads_filtered_before_recording=True,record_serialization_after_target_and_actor_join=True,
            metadata=[dict(stream_handle_opaque_process_local=h) for h in handles]))

    def test_multiple_actual_streams_are_not_silently_collapsed_to_one_role(self):
        with self.assertRaises(ValueError):fit.stream_binding(self.native((0,123)),{})

    def test_foreign_actor_filter_is_an_explicit_prerequisite(self):
        native=self.native();native['runtime_callbacks']['foreign_actor_threads_filtered_before_recording']=False
        with self.assertRaises(ValueError):fit.stream_binding(native,{})

    def test_unique_observed_zero_handle_is_bound_to_source_without_PTDS_guess(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=[]
            for suffix,text in (
                ('native/v6/runtime_callbacks.hpp','GetCurrentThreadId()!=self->thread_; if(!self->phase_)return;'),
                ('native/v8/external_stage_probe.cpp','callbacks.arm(uint64_t(index+1));e.action(actions.at(attention));')):
                path=root/suffix;path.parent.mkdir(parents=True);path.write_text(text,encoding='utf-8',newline='\n');paths.append(path)
            manifest=dict(sources=[dict(source=fit.receipt(p),archived=fit.receipt(p)) for p in paths])
            native=self.native();native.update(callback_source_sha256=fit.sha(paths[0]),source_sha256=fit.sha(paths[1]))
            roles,evidence=fit.stream_binding(native,manifest)
            self.assertEqual(roles,{0:'sole_observed_target_Attention_runtime_stream'})
            self.assertFalse(evidence['PTDS_or_default_stream_semantics_inferred'])
            native['callback_source_sha256']='a'*64
            with self.assertRaises(ValueError):fit.stream_binding(native,manifest)


class FreezeRefusalTests(unittest.TestCase):
    def test_existing_heldout_job_directory_refuses_late_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'jobs'/'heldout-r0').mkdir(parents=True)
            with self.assertRaises(ValueError):
                fit.no_heldout(root,dict(jobs=[dict(id='heldout-r0',split='heldout')]))

    def test_failed_OFF_gate_cannot_produce_local_prices_or_a_forecast_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);fit.write(root/'manifest.json',dict(jobs=[]))
            fit.write(root/'00_observer_gate-observer-gate.json',dict(passed=False,missing_or_failed_jobs=[]))
            with patch.object(fit,'check_sources'):
                with self.assertRaises(ValueError):fit.fit_and_freeze(root)
            self.assertFalse((root/'local-runtime-profile.json').exists())
            self.assertFalse((root/'frozen-barrier.json').exists())


if __name__=='__main__':unittest.main()
