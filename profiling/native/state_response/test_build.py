"""Provenance and measurement-boundary checks; never launch native hardware."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("state_response_native_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN, RECORD = load("generate_probe"), load("record_build")


class ProvenanceTests(unittest.TestCase):
    def originals(self):
        return {name: (HERE.parent / name).read_bytes() for name in GEN.PINS}

    def test_exact_pins_and_only_main_symbols_changed(self):
        originals = self.originals()
        generated = GEN.generated_sources(originals)
        for output, original, replacement in (
            ("highlevel_generated.inc", "highlevel_accuracy/highlevel_probe.cpp", b"static int single_layer_historical_main(int argc,char **argv)"),
            ("single_layer_generated.inc", "single_layer_calibration/probe.cpp", b"static int state_response_historical_main(int argc,char **argv)"),
        ):
            self.assertEqual(hashlib.sha256(originals[original]).hexdigest(), GEN.PINS[original])
            self.assertEqual(generated[output].replace(replacement, GEN.MAIN), originals[original])

    def test_changed_and_missing_executor_rejected(self):
        originals = self.originals()
        originals["state_calibration/compute.cpp"] += b"\n"
        with self.assertRaisesRegex(ValueError, "pinned executor"):
            GEN.generated_sources(originals)
        originals.pop("state_calibration/compute.cpp")
        with self.assertRaisesRegex(ValueError, "executor set"):
            GEN.generated_sources(originals)

    def test_numeric_and_actual_stream_event_helpers_byte_exact(self):
        original = self.originals()
        generated = GEN.generated_sources(original)
        state = original["state_calibration/compute.cpp"]
        for key, first, last in (
            ("numeric_helpers.inc", b"static json compare_ports(", b"static json graph_description("),
            ("timing_events.inc", b"static void cuda_ok(", b"struct TargetTiming {"),
        ):
            self.assertEqual(generated[key], state[state.index(first):state.index(last)])
        self.assertIn(b"ggml_backend_event_record", generated["timing_events.inc"])


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (HERE / "probe.cpp").read_text(encoding="utf-8")

    def test_primary_uses_immutable_actions_diagnostics_separate(self):
        self.assertIn("if(!record){executor.action(action);return;}", self.source)
        self.assertIn('"legacy_double_sync"', self.source)
        self.assertIn('"explicit_async_double_sync_diagnostic"', self.source)
        self.assertIn('if(instrumentation=="diagnostic")events=', self.source)
        self.assertIn('stamped=instrumentation!="bare"', self.source)

    def test_sampling_has_no_json_allocation_or_port_queries(self):
        loop = self.source.split("while(true){const auto now=tick();", 1)[1].split("const auto end=tick();", 1)[0]
        for forbidden in ("json", "push_back", "resize(", "capture_verification", "cpu_point("):
            self.assertNotIn(forbidden, loop)
        self.assertIn("executor.reset();++reset_count;", loop)
        self.assertLess(loop.index("executor.reset()"), loop.index("iteration.begin=tick()"))
        self.assertIn("iteration.end=tick();++count;", loop)
        self.assertNotIn("schedule.end", loop.split("iteration.begin=tick();", 1)[1])

    def test_117_action_strategy_fits_same_bare_and_stamped_pod_envelope(self):
        self.assertIn("maximum_iterations=std::min<size_t>(main?20000:200000,2000000/actions_per_iteration)", self.source)
        actual_actions = 117
        iterations = min(20000, 2000000 // actual_actions)
        self.assertEqual(iterations, 17094)
        self.assertLessEqual(iterations * actual_actions, 2000000)
        self.assertGreater((iterations + 1) * actual_actions, 2000000)
        self.assertIn("stamps(maximum_iterations*actions_per_iteration)", self.source)
        self.assertIn('require(count<maximum_iterations,"POD iteration capacity reached; no observations dropped")', self.source)

    def test_worker_inputs_checked_and_post_replay_verified(self):
        self.assertIn('inputs(executor,actions.at(target))', self.source)
        self.assertIn("verify_inputs(executor,snapshots)", self.source)
        self.assertIn('executor.validate();const auto replay=single_layer::capture_verification(executor)', self.source)
        self.assertIn('{"post_replay_consistency",post_consistency}', self.source)
        self.assertIn("single_target_compute_not_full_layer_or_pure_resource_occupancy", self.source)

    def test_post_sampling_cuda_work_waits_for_owned_stop_release(self):
        after = self.source.split("const auto end=tick();const auto cpu_last=cpu_point();", 1)[1]
        self.assertLess(after.index('write_new(complete,'), after.index("while(!exists(stop))"))
        self.assertLess(after.index("while(!exists(stop))"), after.index("auto final=single_layer::capture_verification"))
        barrier = after.split('const auto release_deadline=', 1)[1].split("const auto release_tick=tick();", 1)[0]
        self.assertNotIn("executor.", barrier)
        self.assertIn("tick()+120*frequency()", barrier)

    def test_clocks_private_waits_and_diagnostics_limits_explicit(self):
        for marker in ("QueryThreadCycleTime", "GetCurrentThread()", "CREATE_NEW", "CreateWaitableTimerExW",
                       '"global_timer_settings_changed",false', '"stream_span_is_active_kernel_time",false',
                       '"host_CPU_and_GPU_intervals_are_additive",false', "GO phases do not cover end",
                       "POD iteration capacity reached; no observations dropped"):
            self.assertIn(marker, self.source)

    def test_thread_diagnostic_keeps_normal_action_between_separate_counter_queries(self):
        helper = self.source.split("static void perform_thread_diagnostic(", 1)[1].split("static json thread_diagnostic_json", 1)[0]
        helper = helper.split("{", 1)[1]
        ordered = ("record.before=cpu_point()", "stamp.begin=tick()", "executor.action(action)",
                   "stamp.end=tick()", "record.after=cpu_point()")
        positions = [helper.index(marker) for marker in ordered]
        self.assertEqual(positions, sorted(positions))
        for forbidden in ("compute_async", "TimingEvents", "json", "push_back", "resize("):
            self.assertNotIn(forbidden, helper)
        self.assertIn('require(!thread_diagnostic||(main&&target<0)', self.source)

    def test_thread_diagnostics_cannot_become_formal_prices_or_invent_thread_states(self):
        for marker in ('"diagnostic_only",diagnostic||thread_diagnostic',
                       '"usable_for_parameter_fit",main&&!diagnostic&&!thread_diagnostic',
                       '"action_timer_excludes_counter_queries",true',
                       '"invocation_timer_includes_counter_queries",true',
                       '"cpu_counter_bracket_is_exact_action_cpu_time",false',
                       '"cycles_converted_to_nanoseconds",false',
                       '"running_ready_blocked_separated_by_these_counters",false',
                       '"runtime_wait_or_spin_identified",false'):
            self.assertIn(marker, self.source)
        self.assertIn('thread_diagnostics(thread_diagnostic?maximum_iterations*actions_per_iteration:0)', self.source)
        self.assertIn('"thread_diagnostic_bytes",thread_diagnostics.size()*sizeof(ThreadDiagnostic)', self.source)

    def test_native_main_thread_identity_and_action_alignment_are_retained(self):
        self.assertIn('static json main_thread_id()', self.source)
        self.assertIn('return static_cast<uint64_t>(GetCurrentThreadId())', self.source)
        self.assertEqual(self.source.count('{"main_thread_id",main_thread_id()}'), 4)
        self.assertIn('"thread_action_diagnostics",thread_rows', self.source)
        self.assertIn('thread_diagnostics[i*actions_per_iteration+j],stamps[i*actions_per_iteration+j],i,j', self.source)

    def test_cuda_schedule_unset_does_not_call_setter_or_infer_an_actual_default(self):
        body=self.source.split('static CudaScheduleRequest configure_cuda_schedule_from_environment()',1)[1].split('static json observe_cuda_schedule',1)[0]
        self.assertLess(body.index('if(value==nullptr)return result'),body.index('cudaSetDeviceFlags'))
        for value in ('auto','spin','yield','blocking'):
            self.assertIn('value=="'+value+'"',self.source)
        self.assertIn('throw std::runtime_error("SIMULATOR_CUDA_SCHEDULE',self.source)
        self.assertIn('"auto_wait_behavior_resolved",false',self.source)

    def test_cuda_schedule_queries_are_outside_sampling_and_verified_at_two_endpoints(self):
        start=self.source.index('const auto schedule_before=observe_cuda_schedule')
        ready=self.source.index('const auto ready_tick=tick()')
        sample=self.source.index('const auto cpu_first=cpu_point()')
        released=self.source.index('const auto release_tick=tick()')
        after=self.source.index('const auto schedule_after=observe_cuda_schedule')
        self.assertLess(start,ready);self.assertLess(ready,sample);self.assertLess(released,after)
        sampling=self.source[sample:self.source.index('const auto end=tick();const auto cpu_last=cpu_point()')]
        for api in ('cudaGetDevice(', 'cudaGetDeviceFlags(', 'cudaSetDeviceFlags('): self.assertNotIn(api,sampling)
        self.assertIn('"cuda_schedule",schedule_before',self.source)
        self.assertIn('"before_ready",schedule_before',self.source)
        self.assertIn('"after_release",schedule_after',self.source)
        self.assertIn('require(!request.overridden||preserved',self.source)

    def test_cuda_override_precedes_executor_and_unknown_flags_are_nullable(self):
        main=self.source.split('int main(int argc,char **argv)',1)[1]
        self.assertLess(main.index('configure_cuda_schedule_from_environment()'),main.index('Executor executor('))
        observer=self.source.split('static json observe_cuda_schedule',1)[1].split('static bool exists',1)[0]
        self.assertIn('cudaGetDevice(&device)',observer);self.assertIn('cudaGetDeviceFlags(&flags)',observer)
        self.assertIn('flags_status==cudaSuccess?json(flags):json(nullptr)',observer)
        self.assertIn('device_status==cudaSuccess?json(device):json(nullptr)',observer)
        self.assertIn('request.overridden?json(preserved):json(nullptr)',observer)


class RecordTests(unittest.TestCase):
    def fixture(self, root):
        files = {}
        for name in ("binary.exe", "compiler.exe", "cache.txt", "input.lib", "helper.inc"):
            files[name] = root / name
            files[name].write_bytes(name.encode())
        manifest = root / "source.json"
        manifest.write_text(json.dumps(dict(format_version="state-response-generated-v1",
            bound_files={str(files["input.lib"]): RECORD.sha(files["input.lib"])},
            generated_files={str(files["helper.inc"]): RECORD.sha(files["helper.inc"])},
            immutable_executors_sha256=GEN.PINS, event_abi_sha256=GEN.EVENT_ABI_FILES)), encoding="utf-8")
        return files, manifest

    def test_build_record_binds_generated_inputs_and_never_runs_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            record = RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root)
            self.assertEqual(record["returncode"], 0)
            self.assertFalse(record["hardware_executed"])
            self.assertEqual(record["primary_execution_boundary"], "legacy_double_sync")
            self.assertIn(str(files["helper.inc"]), record["bound_files"])
            self.assertEqual(record["supported_roles"], ["main", "cpu-worker", "gpu-worker"])
            self.assertIn("thread-diagnostic", record["supported_instrumentation"])
            self.assertIn("diagnostic only", record["thread_diagnostic_execution_boundary"])
            self.assertIn("GetCurrentThreadId", record["main_thread_id_provider"])
            self.assertEqual(record["cuda_schedule_override_values"],["auto","spin","yield","blocking"])
            self.assertTrue(record["cuda_schedule_requested_mode_must_be_observed"])
            self.assertFalse(record["cuda_auto_wait_behavior_inferred"])

    def test_changed_input_rejects_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            files["input.lib"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed during compile"):
                RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root)

    def test_changed_generated_file_rejects_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            files["helper.inc"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed during compile"):
                RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root)


if __name__ == "__main__":
    unittest.main()
