"""Software-only causal completed-component calibration contracts."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling.dynamic_validation import bare_macro_fit as fit


class ComponentFitTests(unittest.TestCase):
    def native(self,state=(0,0),value=100):
        return dict(clock=dict(frequency_hz=10**9),continuous_replays=1,
            blocks=[dict(actions=[dict(action_index=22,begin_tick=10,end_tick=10+value,wall_ns=value)])],
            target_wall_ns=99999999,instrumentation='stamped',request=dict(offered_state=list(state)),
            runtime_callbacks=dict(enabled=False,record_count=0))

    def test_component_label_does_not_use_complete_target_wall(self):
        value=fit.component_label(self.native(value=123))
        self.assertEqual(value,Fraction(123))

    def test_incomplete_component_or_wrong_clock_is_not_filled(self):
        native=self.native();native['blocks'][0]['actions'].append(native['blocks'][0]['actions'][0])
        with self.assertRaises(ValueError):fit.component_label(native)
        native=self.native();native['blocks'][0]['actions'][0]['wall_ns']=500
        with self.assertRaises(ValueError):fit.component_label(native)

    def contexts(self,root):
        jobs=[];records={}
        for state_index,state in enumerate(fit.STATES):
            for index in range(3):
                number=state_index*3+index;name=f'state{state_index}-r{index}'
                job=dict(id=name,split='profile',offered_state=list(state),request=dict(offered_state=list(state)),imported=state==(0,0));jobs.append(job)
                path=root/(name+'.json');path.write_text(name,encoding='utf-8',newline='\n')
                execution=dict(process_id=100+number,process_creation_time_100ns=1000+number,
                               native=dict(path=str(root/(name+'-native.json')),sha256=f'{number:064x}'))
                records[name]=(self.native(state,100*state_index+10+index),execution,path)
        return jobs,records

    def test_all_three_component_labels_per_state_define_mean_without_target_correction(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);jobs,records=self.contexts(root)
            with patch.object(fit.campaign,'verified_job',side_effect=lambda out,j,m:records[j['id']]):
                cells,evidence=fit.calibrated_cells(root,dict(jobs=jobs))
            self.assertEqual([c['mean_ns'] for c in cells],[11,111,211,311])
            self.assertEqual(len(evidence),12)
            self.assertTrue(all(c['fresh_contexts']==3 for c in cells))
            self.assertTrue(all(not c['three_component_observations_establish_low_variance'] for c in cells))

    def test_duplicate_process_across_states_or_missing_context_cannot_supply_prices(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);jobs,records=self.contexts(root)
            records[jobs[3]['id']][1].update(process_id=100,process_creation_time_100ns=1000)
            with patch.object(fit.campaign,'verified_job',side_effect=lambda out,j,m:records[j['id']]):
                with self.assertRaises(ValueError):fit.calibrated_cells(root,dict(jobs=jobs))
            with self.assertRaises(ValueError):fit.calibrated_cells(root,dict(jobs=jobs[:-1]))


class ComponentForecastTests(unittest.TestCase):
    def inputs(self):
        actors={d:dict(device=d,component=dict(kind='software_GEMM'),main_thread_mask=4 if d=='cpu' else 65536) for d in ('cpu','gpu')}
        context=dict(actors=actors);semantic=dict(kind='complete_Attention_fixture')
        scope=dict(context=context,context_sha256=fit.digest(context),semantic_coordinate=semantic,semantic_sha256=fit.digest(semantic))
        request=dict(replays=4,block_replays=2,cpu_actor=dict(actors['cpu'],intervals_ms=[[1,3]]),gpu_actor=dict(actors['gpu'],intervals_ms=[]))
        cells=[dict(state=state,mean_ns=1000000*(2 if state[0] else 1)) for state in fit.STATES]
        return cells,scope,request

    def test_static_and_dynamic_use_same_calibration_and_different_planned_clock(self):
        cells,scope,request=self.inputs()
        static=fit.predict(cells,scope,request,'static00',('software fixture',))
        dynamic=fit.predict(cells,scope,request,'memoryless_four_state_macro',('software fixture',))
        self.assertEqual(static['total_ns'],4000000)
        self.assertEqual(dynamic['total_ns'],5000000)
        self.assertEqual(dynamic['planned_entry_ns'],0)
        self.assertTrue(dynamic['own_predicted_completion_clock'])
        self.assertFalse(dynamic['API_or_gap_callback_prices_used'])
        self.assertFalse(dynamic['whole_LLM_target_time_used_in_fit'])

    def test_missing_state_or_changed_actor_workload_is_rejected(self):
        cells,scope,request=self.inputs()
        with self.assertRaises(ValueError):fit.predict(cells[:-1],scope,request,'static00',('fixture',))
        request['cpu_actor']['component']=dict(kind='unmeasured')
        with self.assertRaises(ValueError):fit.predict(cells,scope,request,'memoryless_four_state_macro',('fixture',))

    def test_no_actual_target_latency_or_transition_constant_can_enter_forecast(self):
        cells,scope,request=self.inputs()
        with self.assertRaises(TypeError):fit.predict(cells,scope,request,'memoryless_four_state_macro',('fixture',),observed_target_ns=1)
        with self.assertRaises(ValueError):fit.predict(cells,scope,request,'stateful-guessed-tau',('fixture',))


if __name__=='__main__':unittest.main()
