from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from profiling.depth_validation.preacquisition import amend, RESPONSE


class PreacquisitionTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2]/'build/test-depth-preacquisition'
        root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.out=Path(self.temp.name).resolve()
        self.assertTrue(self.out.is_relative_to(root.resolve()))
        self.manifest=dict(schema='dense-depth-validation-1',pilot=False,
            context=dict(response_context='independent hot action mean',threads=1),
            programs=dict(example=dict(program='bound input')),sources=[dict(sha256='unchanged')],
            jobs=[dict(id=split,split=split,request=dict(memory_mode='pageable',mode='full',
                instrumentation='stamped' if split=='profile' else 'bare')) for split in ('profile','heldout')],
            protocol=dict(profile_processes=36,heldout_processes=72))
        self.save()

    def tearDown(self): self.temp.cleanup()

    def save(self):
        self.original=(json.dumps(self.manifest)+'\n').encode()
        (self.out/'manifest.json').write_bytes(self.original)

    def test_exact_two_field_amendment_preserves_original_and_bindings(self):
        record=amend(self.out)
        self.assertEqual((self.out/'PREACQUISITION_ORIGINAL.json').read_bytes(),self.original)
        updated=json.loads((self.out/'manifest.json').read_text())
        expected=deepcopy(self.manifest)
        expected['context'].update(response_context=RESPONSE,memory_mode='pageable')
        self.assertEqual(updated,expected)
        self.assertEqual({r['field'] for r in record['changes']},{'context.response_context','context.memory_mode'})
        self.assertFalse(record['hardware_acquisition_started'])
        self.assertFalse(record['source_hash_bindings_changed'])

    def test_any_job_or_acquisition_or_freeze_prevents_amendment(self):
        for marker in ('jobs','profile-started.json','heldout-started.json','model-frozen.json'):
            with self.subTest(marker=marker):
                path=self.out/marker
                if marker=='jobs':path.mkdir()
                else:path.write_text('{}')
                try:
                    with self.assertRaisesRegex(ValueError,'forbidden'):
                        amend(self.out)
                    self.assertEqual((self.out/'manifest.json').read_bytes(),self.original)
                    self.assertFalse((self.out/'PREACQUISITION_ORIGINAL.json').exists())
                finally:
                    if marker=='jobs':path.rmdir()
                    else:path.unlink()

    def test_registered_request_cannot_be_relabelled_pageable(self):
        self.manifest['jobs'][0]['request']['memory_mode']='registered';self.save()
        with self.assertRaisesRegex(ValueError,'actual planned acquisition'):
            amend(self.out)
        self.assertEqual((self.out/'manifest.json').read_bytes(),self.original)

    def test_pilot_cannot_be_amended(self):
        self.manifest['pilot']=True;self.save()
        with self.assertRaisesRegex(ValueError,'formal unmeasured'):
            amend(self.out)

    def test_second_amendment_cannot_overwrite_original_evidence(self):
        amend(self.out)
        old=(self.out/'PREACQUISITION_ORIGINAL.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'already exists'):
            amend(self.out)
        self.assertEqual((self.out/'PREACQUISITION_ORIGINAL.json').read_bytes(),old)


if __name__=='__main__': unittest.main()
