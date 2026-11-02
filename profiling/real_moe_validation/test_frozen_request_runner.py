"""Software-only byte identity and finite old-binary campaign safeguards."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import frozen_request_runner as f
from .dataset import read,write

def temporary():
    root=(f.ROOT/'build/real-moe-validation-20261027/frozen-request-runner-test-fixtures').resolve()
    if f.ROOT.resolve() not in root.parents:raise ValueError('fixture outside checked workspace')
    root.mkdir(parents=True,exist_ok=True);return tempfile.TemporaryDirectory(prefix='case-',dir=root)

class FrozenRequestRunnerTests(unittest.TestCase):
    def test_request_bytes_preserve_CRLF_spacing_and_SHA(self):
        with temporary() as td:
            p=Path(td);source=p/'source.json';source.write_bytes(b'{ "schema" : "unchanged" }\r\n');target=p/'copied.json'
            r=f.copy_exact_request(source,target,f.sha(source));self.assertEqual(source.read_bytes(),target.read_bytes());self.assertEqual(r['sha256'],f.sha(source))

    def test_existing_destination_is_never_overwritten(self):
        with temporary() as td:
            p=Path(td);a=p/'a';b=p/'b';a.write_bytes(b'a');b.write_bytes(b'b')
            with self.assertRaises(ValueError):f.copy_exact_request(a,b,f.sha(a))
            self.assertEqual(b.read_bytes(),b'b')

    def test_changed_source_is_rejected_before_copy(self):
        with temporary() as td:
            p=Path(td);a=p/'a';b=p/'b';a.write_bytes(b'a')
            with self.assertRaises(ValueError):f.copy_exact_request(a,b,'0'*64)
            self.assertFalse(b.exists())

    def test_fixed_four_private_two_controls_interleaved(self):
        self.assertEqual(f.PLAN,(('private-00','private_v5'),('original-00','original_v4'),('private-01','private_v5'),('private-02','private_v5'),('original-01','original_v4'),('private-03','private_v5')))

    def test_prepare_preserves_all_old_runtime_and_request_bindings_without_launch(self):
        with temporary() as td,patch.object(f.common,'launch') as launch:
            out=Path(td)/'cohort';old=f.ROOT/'build/real-moe-validation-20261027/tap-smoke-v5b-gpu-consumer-v1';f.prepare(out,old);m=read(out/'manifest.json');original=read(old/'manifest.json')
            self.assertEqual(m['bindings'],original['bindings']);self.assertEqual(len(m['jobs']),6)
            self.assertEqual([j['job_id'] for j in m['jobs']],[j[0] for j in f.PLAN]);self.assertFalse(m['native_source_changed']);self.assertFalse(m['native_sidecar_added'])
            for j in m['jobs']:
                self.assertEqual(Path(j['request']['path']).read_bytes(),Path(j['original_executed_request']['path']).read_bytes())
                self.assertEqual(j['request']['sha256'],j['original_executed_request']['sha256'])
            self.assertEqual(m['parent_preflight_bytes'],6076332864);self.assertEqual(m['native_guard_bytes'],5824125760);self.assertEqual(m['resident_peak_limit_bytes'],5287254848)
            launch.assert_not_called();self.assertFalse((out/'jobs').exists())

    def test_collect_refuses_without_root_receipt_before_model_hash_or_launch(self):
        with temporary() as td,patch.object(f.common,'launch') as launch,patch.object(f,'file_sha') as model_hash:
            out=Path(td);write(out/'FREEZE_BARRIER.json',dict(manifest=dict(sha256='a'*64)))
            with self.assertRaises(FileNotFoundError):f.collect(out,'private-00')
            launch.assert_not_called();model_hash.assert_not_called()

    def test_pipeline_stops_at_first_failure_without_retry(self):
        with temporary() as td:
            out=Path(td);write(out/'manifest.json',dict(fixed_order=[x[0] for x in f.PLAN]))
            with patch.object(f,'authorize'),patch.object(f,'collect',side_effect=[dict(status='completed'),dict(status='failed')]) as collect:
                result=f.collect_all(out)
            self.assertFalse(result['completed']);self.assertEqual(collect.call_count,2);self.assertEqual(read(out/'ACQUISITION_STOPPED.json')['failed_job'],'original-00')
            self.assertFalse((out/'ACQUISITION_COMPLETE.json').exists())

if __name__=='__main__':unittest.main()
