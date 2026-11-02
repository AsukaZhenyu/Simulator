"""Data/provenance checks only: no target execution and no GPU work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('steady_generate_test',HERE/'generate_probe.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)


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
