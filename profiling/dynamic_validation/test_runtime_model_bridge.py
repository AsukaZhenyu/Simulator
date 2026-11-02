"""Tiny source-level bridge checks, never a hardware qualification."""
from fractions import Fraction
import unittest

from .runtime_stage_profile import to_model_objects
from llm_infer_model.performance.parameters import UnknownParameter


class RuntimeBridgeTests(unittest.TestCase):
    def record(self):
        scope=dict(context_sha256='1'*64,runtime_binding_sha256='2'*64,native_source_sha256='3'*64,callback_source_sha256='4'*64,
                   actor_workload_sha256='5'*64,actor_workloads={'cpu':{'component':'fixture'},'gpu':{'component':'fixture'}},init_policy='pinned_init',
                   boundary='inclusive_host_response_partition',placement=dict(target_thread_mask=4,CPU_actor_thread_mask=16,
                       target_physical_cores=(1,),CPU_actor_physical_cores=(2,),GPU_actor_physical_cores=(8,),GPU_actor_submission_thread_mask=65536,topology_sha256='6'*64))
        sequence=[('a'*64,'gap'),('b'*64,'api'),('c'*64,'gap')]
        cells=[dict(coordinate_sha256=c,role=r,state=s,mean_ns=Fraction(0 if r=='gap' else 11))
               for c,r in sequence for s in ((0,0),(1,0),(0,1),(1,1))]
        return dict(schema='profile-local-runtime-response-cells-1',scope=scope,cells=cells,qualified=False,
                    whole_target_duration_used=False,heldout_data_used=False),sequence

    def test_zero_known_gaps_and_API_responses_bridge_without_GPU_charge(self):
        record,sequence=self.record();program,profile=to_model_objects(record,sequence,profile_evidence=('profile fixture',),program_evidence=('ordered partition',))
        self.assertEqual([s.role for s in program.stages],['gap','api','gap'])
        self.assertFalse(profile.qualified)
        self.assertEqual(profile.cells[0].mean_ns,0)
        self.assertFalse(program.scope.placement.physical_core_overlap)

    def test_unqualified_prices_do_not_start_default_prediction(self):
        record,sequence=self.record();program,profile=to_model_objects(record,sequence,profile_evidence=('profile fixture',),program_evidence=('ordered partition',))
        with self.assertRaises(UnknownParameter):
            program.predict(profile,cpu_intervals_ns=(),gpu_intervals_ns=(),provider='static',origin_ns=0,horizon_ns=1000,replays=1,block_replays=1,start_ns=0)

    def test_full_runtime_or_actor_scope_changes_model_scope(self):
        record,sequence=self.record();one,_=to_model_objects(record,sequence,profile_evidence=('profile fixture',),program_evidence=('ordered partition',))
        record['scope']['placement']['GPU_actor_submission_thread_mask']=131072
        two,_=to_model_objects(record,sequence,profile_evidence=('profile fixture',),program_evidence=('ordered partition',))
        self.assertNotEqual(one.scope,two.scope)


if __name__=='__main__':unittest.main()
