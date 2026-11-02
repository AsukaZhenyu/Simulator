from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from profiling.event_stage_recovery import bounded_completion_reads
from profiling.available_service import campaign


class CompletionReadTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        (self.folder/'sampling-complete.json').write_text('{}',encoding='utf-8')
        self.child=SimpleNamespace(name='main',directory=self.folder,
            process=SimpleNamespace(pid=77,poll=lambda:None))
        self.original=campaign.wait_complete

    def test_sharing_race_retries_without_restarting_actor(self):
        events=[]
        with patch.object(campaign.run,'read',side_effect=[PermissionError(13,'Permission denied'),{'process_id':77}]) as read:
            with patch('profiling.event_stage_recovery.time.sleep'):
                with bounded_completion_reads(events):
                    result=campaign.wait_complete(self.child,None,1)
        self.assertEqual(result,{'process_id':77})
        self.assertEqual(read.call_count,2)
        self.assertEqual(len(events),1)
        self.assertIs(campaign.wait_complete,self.original)

    def test_identity_failure_is_not_retried(self):
        events=[]
        with patch.object(campaign.run,'read',return_value={'process_id':78}) as read:
            with self.assertRaisesRegex(ValueError,'sampling identity'):
                with bounded_completion_reads(events):campaign.wait_complete(self.child,None,1)
        self.assertEqual(read.call_count,1)
        self.assertFalse(events)
        self.assertIs(campaign.wait_complete,self.original)

    def test_persistent_permission_error_keeps_original_deadline(self):
        with patch.object(campaign.run,'read',side_effect=PermissionError(13,'Permission denied')):
            with patch('profiling.event_stage_recovery.time.monotonic',side_effect=[0,.1,.6,1.1]):
                with patch('profiling.event_stage_recovery.time.sleep'):
                    with self.assertRaisesRegex(TimeoutError,'sampling deadline'):
                        with bounded_completion_reads([]):campaign.wait_complete(self.child,None,1)
        self.assertIs(campaign.wait_complete,self.original)

    def test_unrelated_io_error_propagates(self):
        with patch.object(campaign.run,'read',side_effect=OSError(5,'device error')):
            with self.assertRaises(OSError):
                with bounded_completion_reads([]):campaign.wait_complete(self.child,None,1)
        self.assertIs(campaign.wait_complete,self.original)


if __name__=='__main__':unittest.main()
