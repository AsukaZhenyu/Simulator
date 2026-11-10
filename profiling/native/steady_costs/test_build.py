"""Data/provenance checks only: no target execution and no GPU work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest import mock


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('steady_generate_test',HERE/'generate_probe.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
record_spec=importlib.util.spec_from_file_location('steady_record_test',HERE/'record_build.py')
recorder=importlib.util.module_from_spec(record_spec);record_spec.loader.exec_module(recorder)


class BackendClaimTests(unittest.TestCase):
    """A build receipt may only claim what the binary it describes can do.

    The parent receipt was written for the published Win32 CUDA build. Two of its
    lines are false of a Metal build -- that event instrumentation exists, and
    that a CUDA stream schedule was queried -- and a campaign reading them would
    cite a capability the probe refuses at request construction.
    """

    def test_a_metal_build_does_not_claim_cuda_events_or_a_stream_schedule(self):
        fields=recorder.backend_fields('metal','arm64')
        self.assertEqual(fields['device_memory'],'shared')
        self.assertEqual(fields['supported_instrumentation'],['bare'])
        self.assertIn('not applicable',fields['cuda_schedule_observation'])
        self.assertIsNone(fields['cuda_schedule_override_environment'])
        self.assertEqual(fields['cuda_schedule_override_values'],[])
        self.assertIn('steady clock',fields['timing_events_scope'])

    def test_a_cuda_build_overrides_nothing_but_the_memory_mode(self):
        # A discrete card is decided by the host's shape. An aarch64 CUDA host is not:
        # a Grace-Blackwell box can carry a card over PCIe, so the receipt says
        # `unknown` rather than pricing every copy at zero, and the owner's
        # declaration is the only thing that turns it into `shared`.
        self.assertEqual(recorder.backend_fields('cuda','x86_64')['device_memory'],
                         'separate')
        self.assertEqual(recorder.backend_fields('cuda','aarch64')['device_memory'],
                         'unknown')
        with mock.patch.dict(os.environ,{'IR_DEVICE_MEMORY':'shared'}):
            fields=recorder.backend_fields('cuda','aarch64')
        self.assertEqual(fields['device_memory'],'shared')
        self.assertNotIn('supported_instrumentation',fields)
        self.assertNotIn('cuda_schedule_observation',fields)


class SteadyBuildTests(unittest.TestCase):
    def test_only_unused_main_declaration_is_renamed(self):
        source=b'int main(int argc,char **argv) { return 0; }\n'
        result=generator.state_source(source)
        self.assertEqual(result,b'static int steady_costs_unused_state_main(int argc,char **argv) { return 0; }\n')
        self.assertEqual(source,b'int main(int argc,char **argv) { return 0; }\n')

    def test_ambiguous_or_missing_main_is_rejected(self):
        for value in (b'none',b'int main(int argc,char **argv) int main(int argc,char **argv)'):
            with self.assertRaises(ValueError):generator.state_source(value)

    def test_pinned_executor_sources_unchanged(self):
        path=HERE.parent/'state_response/generate_probe.py'
        loader=importlib.util.spec_from_file_location('steady_parent_generator_test',path)
        parent=importlib.util.module_from_spec(loader);loader.loader.exec_module(parent)
        originals={name:(HERE.parent/name).read_bytes() for name in parent.PINS}
        parent.generated_sources(originals)
        for name,content in originals.items():self.assertEqual(hashlib.sha256(content).hexdigest(),parent.PINS[name])
        name=next(iter(originals));originals[name]+=b'\n'
        with self.assertRaises(ValueError):parent.generated_sources(originals)

    def test_native_main_does_not_add_cycle_queries_or_inferred_device_residual(self):
        source=(HERE/'probe.cpp').read_text(encoding='utf-8')
        self.assertNotIn('QueryThreadCycleTime(',source)
        self.assertIn('"stream_elapsed_is_kernel_active_time",false',source)
        self.assertIn('executor.reset(); // Outside the entire R-repeat price interval.',source)
        self.assertIn('"component_output",final',source)

    def test_event_stream_abi_remains_bound(self):
        path=HERE/'build-release-v1/source-manifest.json'
        if not path.exists():self.skipTest('native source manifest not built locally')
        source=json.loads(path.read_text(encoding='utf-8'))
        self.assertIn('ggml/src/ggml-backend-impl.h',source['event_abi_sha256'])
        self.assertIn('ggml/src/ggml-cuda/ggml-cuda.cu',source['event_abi_sha256'])
        event_file=HERE/'build-release-v1/timing_events.inc'
        self.assertIn('ggml_backend_event_record(&first_wrapper,backend)',event_file.read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
