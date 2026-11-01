"""Revised profile classification preserves immutable failure and strict holdout."""
from pathlib import Path
import tempfile
import unittest

from profiling.dynamic_validation import bare_macro_fit as shared
from profiling.dynamic_validation import bare_macro_v2_fit as fit


class RevisedComponentLabelsTests(unittest.TestCase):
    def contexts(self,root):
        jobs=[];records={}
        for state_index,state in enumerate(fit.STATES):
            for index in range(3):
                n=state_index*3+index;name=f'state{state_index}-r{index}'
                jobs.append(dict(id=name,split='profile',offered_state=list(state),
                    request=dict(offered_state=list(state)),imported=n<4))
                path=root/(name+'.json');path.write_text(name,encoding='utf-8',newline='\n')
                old_failed=n==3
                execution=dict(status='failed' if old_failed else 'completed',process_id=100+n,
                    process_creation_time_100ns=1000+n,native=dict(path=str(root/(name+'-native.json')),sha256=f'{n:064x}'))
                native=dict(clock=dict(frequency_hz=10**9),continuous_replays=1,instrumentation='stamped',
                    blocks=[dict(actions=[dict(action_index=22,begin_tick=10,end_tick=110+n,wall_ns=100+n)])],
                    target_wall_ns=99999999,request=dict(offered_state=list(state)),
                    runtime_callbacks=dict(enabled=False,record_count=0),
                    target_entry_controller_passed=not old_failed,
                    initial_start_lateness_ns=25864000 if old_failed else 1000)
                records[name]=(native,execution,path)
        return jobs,records

    def test_verified_constant_state_label_retains_old_failure_and_lateness(self):
        with tempfile.TemporaryDirectory() as td:
            jobs,records=self.contexts(Path(td));before=records[jobs[3]['id']][1].copy()
            cells,evidence=fit._calibrated_cells(Path(td),dict(jobs=jobs),lambda out,j,m:records[j['id']])
            retained=cells[1]['observations'][0]
            self.assertEqual(retained['original_execution_status'],'failed')
            self.assertFalse(retained['original_target_entry_controller_passed'])
            self.assertEqual(retained['original_start_lateness_ns'],25864000)
            self.assertFalse(retained['original_rejection_or_lateness_record_modified'])
            self.assertEqual(records[jobs[3]['id']][1],before)
            self.assertEqual(cells[1]['mean_ns'],104)
            self.assertEqual(len(evidence),12)

    def test_v2_verification_is_required_before_label_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            jobs,records=self.contexts(Path(td))
            def rejected(out,job,manifest):
                raise ValueError('actual steady actor window not established')
            with self.assertRaises(ValueError):fit._calibrated_cells(Path(td),dict(jobs=jobs),rejected)

    def test_missing_context_and_duplicate_identity_are_not_relaxed(self):
        with tempfile.TemporaryDirectory() as td:
            jobs,records=self.contexts(Path(td))
            with self.assertRaises(ValueError):
                fit._calibrated_cells(Path(td),dict(jobs=jobs[:-1]),lambda out,j,m:records[j['id']])
            records[jobs[3]['id']][1].update(process_id=100,process_creation_time_100ns=1000)
            with self.assertRaises(ValueError):
                fit._calibrated_cells(Path(td),dict(jobs=jobs),lambda out,j,m:records[j['id']])


class UnchangedHeldoutTests(unittest.TestCase):
    def native(self,lateness):
        start=lateness;blocks=[]
        for block in range(16):
            begin=start+block*1600
            actions=[dict(action_index=22,begin_tick=begin+i*100,end_tick=begin+(i+1)*100,wall_ns=100) for i in range(16)]
            blocks.append(dict(begin_tick=begin,end_tick=begin+1600,wall_ns=1600,actions=actions))
        return dict(clock=dict(frequency_hz=10**9),target_begin_tick=start,target_end_tick=start+25600,
            target_wall_ns=25600,planned_target_entry_tick=0,initial_start_lateness_ns=lateness,blocks=blocks)

    def test_late_dynamic_target_is_still_rejected_by_same_frozen_helper(self):
        with self.assertRaises(ValueError):shared.measured_intervals(self.native(10000001))
        total,blocks,details=shared.measured_intervals(self.native(10000000))
        self.assertEqual(total,25600)
        self.assertEqual(len(blocks),16)
        self.assertFalse(details['outside_action_shell_used_for_prediction'])


if __name__=='__main__':unittest.main()
