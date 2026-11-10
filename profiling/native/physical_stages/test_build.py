"""Native boundary/provenance checks without loading CUDA or starting workloads."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def module():
    spec = importlib.util.spec_from_file_location("physical_stages_build_record_test", HERE / "record_build.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class PhysicalStageSourceTests(unittest.TestCase):
    def test_only_the_function_declaration_is_renamed(self):
        spec=importlib.util.spec_from_file_location("physical_stage_generator_test",HERE/'generate_probe.py')
        generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
        original=b'struct CpuSnapshot { int main; }; int main(int argc,char **argv) { return 0; }'
        generated=generator.generated_state_source(original)
        self.assertIn(b'int main;',generated)
        self.assertIn(b'static int physical_stages_unused_state_response_main(',generated)
        self.assertEqual(generated.replace(b'static int physical_stages_unused_state_response_main(',b'int main('),original)

    def test_ambiguous_old_main_is_rejected(self):
        spec=importlib.util.spec_from_file_location("physical_stage_generator_ambiguous_test",HERE/'generate_probe.py')
        generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
        for source in (b'no main',b'int main(int argc,char **argv) int main(int argc,char **argv)'):
            with self.assertRaisesRegex(ValueError,'ambiguous'):
                generator.generated_state_source(source)

    def test_executor_is_reused_not_reimplemented(self):
        source=(HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn('#include "state_response_generated.inc"',source)
        self.assertNotIn('#define main',source)
        self.assertIn("executor.action(actions.at(index));",source)
        self.assertNotIn("ggml_backend_graph_compute_async(",source)
        self.assertIn('"historical_Executor_action_unchanged",true',source)

    def test_cupti_is_optional_observer_control(self):
        source=(HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn('instrument=="cupti"||instrument=="bare"',source)
        self.assertIn('if (instrument=="cupti") capture=',source)
        self.assertIn('"bare_control_scope"',source)
        self.assertIn('"available_resource_trajectory_measured",false',source)

    def test_timestamp_callback_precedes_activity_enable(self):
        source=(HERE / "cupti_capture.hpp").read_text(encoding="utf-8")
        self.assertLess(source.index('checked(cuptiActivityRegisterTimestampCallback(timestamp)'),
                        source.index('checked(cuptiActivityEnable(kind)'))
        self.assertIn('(value / hz) * 1000000000ULL + (value % hz) * 1000000000ULL / hz',source)
        # The header is shared with the dense column, so the clock it names comes
        # from the portable macros; on Windows those are the published strings.
        self.assertIn('{"clock_id",PORTABLE_PHYSICAL_CLOCK_ID},{"clock_origin",CLOCK_ORIGIN}',source)
        platform=(HERE.parent/'portable/platform.hpp').read_text(encoding="utf-8")
        self.assertIn('#define PORTABLE_PHYSICAL_CLOCK_ID "windows_qpc_floor_ns"',platform)
        self.assertIn('#define CLOCK_ORIGIN "absolute_qpc_zero"',platform)

    def test_kernel_memcpy_runtime_and_external_records_are_retained(self):
        source=(HERE / "cupti_capture.hpp").read_text(encoding="utf-8")
        for name in ('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY',
                     'CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER','CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION'):
            self.assertIn(name,source)
        for field in ('cupti_dropped_records','local_record_overflow','callback_errors',
                      'timestamp_callback_failed','all_requested_records_retained'):
            self.assertIn('"'+field+'"',source)
        self.assertIn('"host_and_device_times_are_additive",false',source)
        self.assertIn('"available_resource_rate_inferred",false',source)

    def test_ready_keeps_raw_qpc_and_output_uses_common_physical_clock(self):
        source=(HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn('{"clock",state_response::clock_json()},{"physical_clock",clock_json()}',source)
        self.assertIn('"clock_id","windows_qpc_floor_ns"',source)
        self.assertIn('"post_sampling_work_deferred_until_STOP",true',source)
        self.assertIn('write_new(argv[11],result)',source)

    def test_batched_counters_are_not_claimed_per_action_service(self):
        source=(HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn('"per_action_active_cpu_time_measured",false',source)
        self.assertIn('"running_ready_blocked_separated",false',source)
        self.assertIn('"counter_quantization_retained",true',source)
        self.assertIn('"independent_numeric_oracle_applied",false',source)

    def test_nullable_action_fields_are_serialized_as_json(self):
        source=(HERE / "probe.cpp").read_text(encoding="utf-8")
        # ADVANCE/control actions can legitimately carry target_id=null.
        # A string default would make nlohmann::value<string> throw after timing.
        self.assertIn('action.value("target_id",json(nullptr))',source)
        self.assertIn('action.value("device",json(nullptr))',source)
        self.assertNotIn('action.value("target_id","")',source)
        self.assertIn('stages.push_back(action_row(action,interval))',source)
        self.assertIn('--serialization-self-test',source)

    def test_build_has_explicit_local_sdk_and_no_downloads(self):
        source=(HERE / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('set(CUPTI_ROOT "" CACHE PATH',source)
        self.assertIn('NO_DEFAULT_PATH REQUIRED',source)
        self.assertNotIn('FetchContent',source)
        self.assertIn('"${CUPTI_ROOT}/lib64/cupti*.dll"',source)


class PhysicalStageBuildRecordTests(unittest.TestCase):
    def fixture(self, directory):
        directory=Path(directory)
        paths={name:directory/name for name in ('compiler.exe','cache.txt','native.exe','bound.cpp','helper.inc')}
        for name,path in paths.items():path.write_text(name,encoding="utf-8",newline="\n")
        def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
        manifest=directory/'source-manifest.json'
        manifest.write_text(json.dumps(dict(format_version='state-response-generated-v1',
            bound_files={str(paths['bound.cpp']):sha(paths['bound.cpp'])},
            generated_files={str(paths['helper.inc']):sha(paths['helper.inc'])},
            immutable_executors_sha256={'fixture':'0'*64},event_abi_sha256={'fixture':'1'*64})),
            encoding='utf-8',newline='\n')
        return paths,manifest

    def test_build_receipt_is_physical_stage_scope_without_execution_claim(self):
        with tempfile.TemporaryDirectory(prefix='physical-stage-build-') as directory:
            paths,manifest=self.fixture(directory)
            value=module().build_record(paths['native.exe'],manifest,paths['compiler.exe'],'fixture',
                paths['cache.txt'],directory,directory)
            self.assertEqual(value['format_version'],'physical-stage-native-build-1')
            self.assertEqual(value['supported_instrumentation'],['bare','cupti'])
            self.assertEqual(value['supported_modes'],['full','target'])
            self.assertFalse(value['hardware_executed'])
            self.assertFalse(value['runtime_loader_resolution_proven'])
            self.assertFalse(value['available_resource_rate_inferred'])
            self.assertTrue(value['bound_inputs_unchanged'])

    def test_changed_generated_or_original_input_rejected(self):
        for name in ('bound.cpp','helper.inc'):
            with self.subTest(name=name),tempfile.TemporaryDirectory(prefix='physical-stage-build-') as directory:
                paths,manifest=self.fixture(directory)
                paths[name].write_text('changed',encoding='utf-8',newline='\n')
                with self.assertRaisesRegex(ValueError,'changed during compile'):
                    module().build_record(paths['native.exe'],manifest,paths['compiler.exe'],'fixture',
                        paths['cache.txt'],directory,directory)


if __name__=='__main__':
    unittest.main()
