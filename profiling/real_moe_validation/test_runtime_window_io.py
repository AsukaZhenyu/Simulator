"""Hardware-free collector memory and integrity contracts."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from profiling import host
from profiling.real_moe_validation.runtime_window_io import available, child_environment, file_sha, receipt, PLACEMENT_LABELS, LABELS
from profiling.real_moe_validation.runtime_window_campaign import verify_prediction_freeze,select
from profiling.real_moe_validation.dataset import write


class RuntimeWindowIOTest(unittest.TestCase):
    def test_complete_streaming_hash_crosses_buffer_boundaries(self):
        content = bytes(range(256)) * (1024 * 9) + b'last non-aligned byte'
        workspace_build = Path(__file__).resolve().parents[2] / 'build'
        workspace_build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_build) as folder:
            path = Path(folder) / 'binary.bin'
            path.write_bytes(content)
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(file_sha(path), expected)
            self.assertEqual(receipt(path), dict(path=str(path.resolve()), sha256=expected))
            with path.open('r+b') as stream:
                stream.seek(-1, 2)
                stream.write(b'!')
            self.assertNotEqual(file_sha(path), expected)

    def test_collect_import_does_not_load_numerical_or_plotting_runtime(self):
        program = "import sys; import profiling.real_moe_validation.runtime_window_campaign; assert not ({'numpy','torch','matplotlib'} & set(sys.modules))"
        result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True, env=dict(os.environ))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_environment_preserves_declared_runtime_policy(self):
        environment = child_environment(('private-runtime',), dict(cuda_graph_policy='runtime_default_enabled', tf32_override='0'))
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'], '0')
        self.assertNotIn('GGML_CUDA_DISABLE_GRAPHS', environment)
        self.assertEqual(environment['PATH'].split(os.pathsep)[0], 'private-runtime')
        with self.assertRaises(ValueError):
            child_environment((), dict(cuda_graph_policy='disabled', tf32_override='0'))

    def test_declared_disabled_policy_leaves_the_shell_environment_untouched(self):
        # The policy is applied by the observer to its own process. A parent that
        # exported the variable instead would make the ambient shell a hidden
        # coordinate of the measurement, and would break the cohort's requirement
        # that every placement run under one explicitly declared policy.
        environment = child_environment(('private-runtime',),
                                        dict(cuda_graph_policy='runtime_graphs_disabled_declared', tf32_override='0'))
        self.assertNotIn('GGML_CUDA_DISABLE_GRAPHS', environment)
        self.assertEqual(environment['NVIDIA_TF32_OVERRIDE'], '0')
        with self.assertRaises(ValueError):
            child_environment((), dict(cuda_graph_policy='runtime_graphs_disabled', tf32_override='0'))

    def test_environment_tells_the_loader_where_the_libraries_are(self):
        # PATH finds a probe's DLLs on Windows and nothing at all on Linux or
        # macOS, where the loader reads a variable of its own.
        environment = child_environment(('/private/runtime',),
                                        dict(cuda_graph_policy='runtime_default_enabled', tf32_override='0'))
        search = environment[host.shared_library_variable()].split(os.pathsep)
        self.assertEqual(search[0], '/private/runtime')
        self.assertEqual(environment['PATH'].split(os.pathsep)[0], '/private/runtime')

    def test_the_collectors_own_environment_names_that_variable_too(self):
        # The check above cannot see the defect on Windows, where both variables
        # are PATH. The collector that runs the published model jobs builds its
        # environment through `run.child_environment`, and that one is asked
        # here under a loader that has a variable of its own.
        from profiling.real_moe_validation.run import child_environment as collector_environment
        with mock.patch.object(host, 'shared_library_variable', return_value='LD_LIBRARY_PATH'):
            environment = collector_environment(('/private/runtime',),
                                                dict(cuda_graph_policy='runtime_default_enabled',
                                                     tf32_override='0'))
        self.assertEqual(environment['LD_LIBRARY_PATH'].split(os.pathsep)[0], '/private/runtime')

    def test_available_memory_is_asked_of_the_host(self):
        self.assertGreater(available(), 0)

    def test_bare_holdout_requires_prior_prediction_and_immutable_inventory(self):
        workspace_build = Path(__file__).resolve().parents[2] / 'build'
        with tempfile.TemporaryDirectory(dir=workspace_build) as folder:
            out=Path(folder)
            with self.assertRaisesRegex(ValueError,'freeze predictions'):
                verify_prediction_freeze(out)
            for name in ('manifest','predictions','prices','price-freeze','inventory'):
                write(out/(name+'.json'),dict(id=name))
            write(out/'PREDICTION_FREEZE.json',dict(manifest=receipt(out/'manifest.json'),
                predictions=receipt(out/'predictions.json'),prices=receipt(out/'prices.json'),
                price_freeze=receipt(out/'price-freeze.json'),inventories=[receipt(out/'inventory.json')]))
            self.assertEqual(verify_prediction_freeze(out),receipt(out/'PREDICTION_FREEZE.json'))
            (out/'inventory.json').write_text('{}\n',encoding='utf-8',newline='\n')
            with self.assertRaisesRegex(ValueError,'changed after freeze'):
                verify_prediction_freeze(out)

    def test_each_placement_has_a_distinct_non_prefix_label(self):
        # Job files are `<label>-<split>.json`; a label that repeats another
        # placement's label, or prefixes it, would silently overwrite published
        # evidence from a two-placement cohort.
        placements = [placement for placement, _ in PLACEMENT_LABELS]
        labels = [label for _, label in PLACEMENT_LABELS]
        self.assertEqual(len(set(placements)), len(placements))
        self.assertEqual(len(set(labels)), len(labels))
        self.assertEqual(tuple(LABELS), tuple(labels))
        for left in labels:
            for right in labels:
                if left != right:
                    self.assertFalse(right.startswith(left + '-'), left + ' prefixes ' + right)
        self.assertIn('cpu-attn-gpu-other', placements)

    def test_resident_peak_amendment_cannot_lower_the_budget_or_skip_its_reason(self):
        # A cohort that needs a larger resident peak must declare it in its own
        # manifest; lowering a frozen budget, or raising it silently, is refused
        # before any directory is created.
        #
        # `prepare` reads the published cohort's descriptor before it validates
        # anything, and that descriptor lives in the first machine's build tree.
        # The test supplies it, exactly as `prepare_portable` does for a second
        # machine -- otherwise this test only passes on the machine that built
        # the archive.
        from profiling.real_moe_validation import runtime_window_campaign as campaign
        descriptor = dict(resident_peak_limit_bytes=5287254848, parent_preflight_bytes=6076332864)
        original = campaign.read
        campaign.read = lambda path: descriptor if Path(path) == campaign.LEGACY_DESCRIPTOR else original(path)
        try:
            workspace_build = Path(__file__).resolve().parents[2] / 'build'
            with tempfile.TemporaryDirectory(dir=workspace_build) as folder:
                with self.assertRaisesRegex(ValueError, 'only raise the frozen budget'):
                    campaign.prepare(Path(folder) / 'cohort', Path(folder) / 'unused-build', resident_peak_limit_bytes=1, resident_peak_reason='x' * 40)
                with self.assertRaisesRegex(ValueError, 'stated reason'):
                    campaign.prepare(Path(folder) / 'cohort', Path(folder) / 'unused-build', resident_peak_limit_bytes=6000000000)
                self.assertFalse((Path(folder) / 'cohort').exists())
        finally:
            campaign.read = original

    def test_second_fixed_stratum_ID_is_selected_without_performance_labels(self):
        rows=[]
        for domain in ('dialogue','code','math'):
            for length in ('short','long'):
                for suffix in ('b','a'):
                    rows.append(dict(id=domain+'-'+length+'-'+suffix,domain=domain,length_class=length,split='heldout',measured_time=-100 if suffix=='a' else 100))
        suite=dict(requests=rows)
        self.assertTrue(all(r['id'].endswith('-a') for r in select(suite,'heldout')))
        self.assertTrue(all(r['id'].endswith('-b') for r in select(suite,'heldout',index=1)))
        with self.assertRaises(ValueError):select(suite,'heldout',index=-1)
        with self.assertRaises(ValueError):select(suite,'heldout',index=2)


if __name__ == '__main__':
    unittest.main()
