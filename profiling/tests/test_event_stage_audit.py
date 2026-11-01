"""Small pure-data event-audit checks; no process or GPU work is launched."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from profiling import event_stage_audit as audit


class EventAuditTests(unittest.TestCase):
    def native(self):
        return dict(thread_cycles_enabled=True,iterations=[dict(iteration=0,begin_ns=0,end_ns=30,
            actions=[dict(action_index=0,kind='COMPUTE',device='cpu',begin_ns=10,end_ns=20,
                thread_cycles=dict(enabled=True,unit='raw_QueryThreadCycleTime_cycles',cycles_converted_to_time=False,
                    usable_as_heldout_resource_input=False,running_ready_blocked_separated=False,
                    before_query_begin_ns=1,before_query_end_ns=3,after_query_begin_ns=21,after_query_end_ns=25,
                    before_cycles=7,after_cycles=17,delta_cycles=10,monotonic=True))])])

    def test_raw_cycle_work_retains_unknown_and_never_converts_to_time(self):
        value=self.native();r=audit._cycles(value,True)
        self.assertEqual(r['rows'][0]['delta_cycles'],10)
        self.assertFalse(r['counters_are_active_nanoseconds'])
        self.assertFalse(r['counters_are_exogenous_available_service'])
        c=value['iterations'][0]['actions'][0]['thread_cycles'];c.update(before_cycles=None,delta_cycles=None,monotonic=None)
        r=audit._cycles(value,True);self.assertEqual(r['unknown_records'],1)
        self.assertIsNone(r['raw_cycle_delta']['mean'])
        c['delta_cycles']=0
        with self.assertRaisesRegex(ValueError,'repaired'):audit._cycles(value,True)

    def test_bad_counter_brackets_and_target_cycle_switch_are_rejected(self):
        value=self.native()
        with self.assertRaisesRegex(ValueError,'switch'):audit._cycles(value,False)
        value['iterations'][0]['actions'][0]['thread_cycles']['after_query_begin_ns']=19
        with self.assertRaisesRegex(ValueError,'brackets'):audit._cycles(value,True)

    def parking(self):
        value=dict(mode='parent_owned_Windows_events',go_event='Local\\go1',stop_event='Local\\stop1',
            parent_manual_reset_contract=True,native_event_set_or_reset_performed=False,
            post_READY_periodic_file_poll=False,post_sampling_periodic_wakeup=False,global_timer_settings_changed=False)
        process=dict(ready=dict(event_parking=deepcopy(value)),event_controller=dict(manual_reset=True,names=dict(go='Local\\go1',stop='Local\\stop1')))
        return dict(event_parking=value),process

    def test_parking_flags_and_unique_owned_names_are_required(self):
        native,process=self.parking();names=set()
        result=audit._parking(native,process,True,names)
        self.assertTrue(result['flags_and_controller_identity_verified'])
        self.assertFalse(result['continuous_quiescence_measured'])
        with self.assertRaisesRegex(ValueError,'reused'):audit._parking(native,process,True,names)
        native['event_parking']['post_sampling_periodic_wakeup']=True
        process['ready']['event_parking']=deepcopy(native['event_parking'])
        with self.assertRaisesRegex(ValueError,'contract'):audit._parking(native,process,True,set())

    def test_observer_cycle_overhead_is_separate_from_parking_effect(self):
        rows=[]
        for repeat in range(3):
            for parking,cycles,mean in (('poll',False,110),('event',False,100),('event',True,104)):
                rows.append(dict(status='included',job=dict(split='observer',program_key=f'profile-{repeat}',parking=parking,cycles=cycles),
                    wall_ns=dict(mean=mean,median=mean,p90=mean,worst=mean,n=32,cv=.01)))
        result=audit._observer(rows,.05)
        self.assertEqual(result['matched_repeats'],3)
        self.assertTrue(result['cycle_observer_overhead_pass'])
        self.assertFalse(result['parking_change_is_observer_overhead'])
        self.assertAlmostEqual(result['controls'][0]['relative_changes']['event_vs_poll']['mean'],-1/11)
        rows[-1]['wall_ns']['mean']=130
        self.assertFalse(audit._observer(rows,.05)['cycle_observer_overhead_pass'])
        self.assertFalse(audit._observer([],.05)['stable_control_processes'])

    def test_exact_file_hash_is_checked_before_parsing(self):
        with tempfile.TemporaryDirectory(prefix='event-audit-hash-',dir=Path(__file__).resolve().parent) as name:
            p=Path(name)/'source.json';p.write_text('{}\n',encoding='utf-8',newline='\n')
            expected=hashlib.sha256(p.read_bytes()).hexdigest()
            audit._checked(p,expected)
            p.write_text('{"changed":true}\n',encoding='utf-8',newline='\n')
            with self.assertRaisesRegex(ValueError,'hash differs'):audit._checked(p,expected)

    def test_partial_audit_requires_explicit_permission_and_never_claims_pass(self):
        with tempfile.TemporaryDirectory(prefix='event-audit-partial-',dir=Path(__file__).resolve().parent) as name:
            root=Path(name);parent=dict(sources={},source_inventory_sha256=audit.digest({}))
            (root/'manifest.json').write_text(json.dumps(parent),encoding='utf-8',newline='\n')
            protocol=dict(schema='event-stage-campaign-1',parent_manifest_sha256=audit._sha(root/'manifest.json'),
                source_receipts={},jobs=[dict(id='missing',split='profile')],observer_error_max=.05)
            (root/'event-protocol.json').write_text(json.dumps(protocol),encoding='utf-8',newline='\n')
            with self.assertRaisesRegex(ValueError,'incomplete'):audit.audit_campaign(root)
            value=audit.audit_campaign(root,allow_incomplete=True)
            self.assertEqual(value['counts']['pending'],1);self.assertFalse(value['passed'])
            self.assertFalse(value['hardware_accuracy_qualified'])


if __name__=='__main__':unittest.main()
