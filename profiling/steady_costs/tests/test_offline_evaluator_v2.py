"""The offline correction changes process identity, never measured prices."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from profiling.steady_costs.campaign import receipt
from profiling.steady_costs.offline_evaluator_v2 import optional_receipt,process_identity,validate_fresh_process,verify_correction_sources


def execution(pid=123,start='2026-10-22T10:00:00+00:00',end='2026-10-22T10:00:01+00:00'):
    return dict(process_id=pid,started_at_utc=start,completed_at_utc=end,status='completed',returncode=0,
        forced_termination=False,process_exited_verified=True)


class ProcessLaunchIdentityTests(unittest.TestCase):
    def test_recycled_numeric_pid_after_verified_normal_exit_is_fresh(self):
        old=execution();new=execution(start='2026-10-22T10:00:02+00:00',end='2026-10-22T10:00:03+00:00')
        result=validate_fresh_process(new,[old])
        self.assertTrue(result['numeric_pid_recycled'])
        self.assertNotEqual(process_identity(old),result['identity'])
        self.assertTrue(result['prior_same_pid_executions'][0]['verified_normal_exit_before_new_launch'])

    def test_same_pid_and_launch_epoch_is_rejected(self):
        old=execution()
        with self.assertRaisesRegex(ValueError,'same process launch identity'):validate_fresh_process(deepcopy(old),[old])

    def test_recycled_pid_with_overlapping_interval_is_rejected(self):
        old=execution(end='2026-10-22T10:00:10+00:00')
        new=execution(start='2026-10-22T10:00:02+00:00',end='2026-10-22T10:00:03+00:00')
        with self.assertRaisesRegex(ValueError,'intervals overlap'):validate_fresh_process(new,[old])

    def test_unknown_or_forced_prior_exit_does_not_allow_recycling(self):
        new=execution(start='2026-10-22T10:00:02+00:00',end='2026-10-22T10:00:03+00:00')
        for field,value in (('process_exited_verified',False),('forced_termination',True),('returncode',1),('completed_at_utc',None)):
            old=execution();old[field]=value
            with self.assertRaises(ValueError):validate_fresh_process(new,[old])

    def test_epoch_timezone_is_required_and_utc_equivalence_is_detected(self):
        old=execution();new=execution(start='2026-10-22T18:00:00+08:00',end='2026-10-22T18:00:01+08:00')
        # Even different textual offsets cannot disguise the same process epoch.
        with self.assertRaisesRegex(ValueError,'same process launch identity'):validate_fresh_process(new,[old])
        bad=execution(start='2026-10-22T10:00:00')
        with self.assertRaisesRegex(ValueError,'timezone'):process_identity(bad)

    def test_binding_detects_changed_evaluator_and_unexpected_inventory(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source=Path(directory)/'source.py';source.write_text('# initial\n',encoding='utf-8',newline='\n')
            saved=[receipt(source)];verify_correction_sources(saved,[source])
            with self.assertRaisesRegex(ValueError,'inventory'):verify_correction_sources(saved,[])
            source.write_text('# changed\n',encoding='utf-8',newline='\n')
            with self.assertRaisesRegex(ValueError,'changed'):verify_correction_sources(saved,[source])

    def test_future_campaign_without_prior_rejected_attempt_is_supported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/'no-prior-attempt.json'
            self.assertEqual(optional_receipt(path)['status'],'not_applicable')
            path.write_text('{}\n',encoding='utf-8',newline='\n')
            saved=optional_receipt(path)
            self.assertEqual(saved['status'],'present');self.assertEqual(saved['receipt'],receipt(path))


if __name__=='__main__':unittest.main()
