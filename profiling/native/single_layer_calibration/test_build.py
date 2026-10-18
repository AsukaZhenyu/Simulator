"""Source/provenance guards; never execute GGML or start hardware activity."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("single_layer_native_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = load("generate_probe")
RECORD = load("record_build")


class ImmutableExecutionTests(unittest.TestCase):
    def test_helper_pinned(self):
        original = (HERE.parent / "highlevel_accuracy/highlevel_probe.cpp").read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), GEN.HELPER_SHA256)

    def test_generation_changes_only_main(self):
        original = (HERE.parent / "highlevel_accuracy/highlevel_probe.cpp").read_bytes()
        self.assertEqual(GEN.generated_helper(original).replace(
            b"static int single_layer_historical_main(int argc,char **argv)",
            b"int main(int argc,char **argv)"), original)

    def test_changed_executor_rejected(self):
        original = (HERE.parent / "highlevel_accuracy/highlevel_probe.cpp").read_bytes()
        for altered in (original + b"\n", original.replace(b"GGML_PREC_F32", b"GGML_PREC_DEFAULT")):
            with self.subTest(size=len(altered)), self.assertRaisesRegex(ValueError, "immutable historical"):
                GEN.generated_helper(altered)

    def test_context_inner_loop_contains_only_pod_clocks_and_unchanged_executor(self):
        source = (HERE / "probe.cpp").read_text(encoding="utf-8")
        begin = source.index("for(auto &iteration:iterations) {")
        end = source.index("envelope.end=tick();", begin)
        loop = source[begin:end]
        self.assertIn("executor.reset();", loop)
        self.assertLess(loop.index("executor.reset();"), loop.index("iteration.whole.begin=tick();"))
        self.assertIn("stamp.begin=tick();executor.action(action);stamp.end=tick();", loop)
        for forbidden in ("json", "push_back", "resize", "reserve", "new ", "graph_compute_async"):
            self.assertNotIn(forbidden, loop)

    def test_instrumentation_does_not_infer_active_gpu_time(self):
        source = (HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn('{"stream_active_time_measured",false}', source)
        self.assertIn('{"host_and_device_times_may_be_added",false}', source)
        self.assertNotIn("cudaEvent", source)

    def test_pool_override_is_explicit_and_does_not_change_threads(self):
        source = (HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn('requested=="default"||requested=="0"', source)
        self.assertIn("params.poll=0", source)
        self.assertIn('{"configured_n_threads",threads}', source)
        self.assertIn('{"observed_n_threads",nullptr}', source)
        self.assertNotIn("ggml_threadpool_get_n_threads(executor.pool)", source)
        self.assertLess(source.index("ggml_backend_cpu_set_threadpool(executor.cpu,replacement)"),
                        source.index("ggml_threadpool_free(previous)"))

    def test_cpu_counter_claim_and_unknown_are_explicit(self):
        source = (HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn("GetProcessTimes", source)
        self.assertIn("GetThreadTimes", source)
        self.assertIn("non-Windows unknown", source)
        self.assertIn("not proof of GGML polling", source)
        self.assertIn("available?json(process):json(nullptr)", source)


class BuildRecordTests(unittest.TestCase):
    def fixture(self, directory):
        paths = {}
        for name in ("probe.exe", "compiler.exe", "cache.txt", "helper.inc", "input.lib"):
            path = directory / name
            path.write_bytes(name.encode())
            paths[name] = path
        manifest = directory / "source.json"
        manifest.write_text(json.dumps(dict(format_version="single-layer-generated-source-v1",
            bound_files={str(paths["input.lib"]): RECORD.sha(paths["input.lib"])},
            generated_file=str(paths["helper.inc"]), generated_sha256=RECORD.sha(paths["helper.inc"]),
            immutable_helper_sha256=GEN.HELPER_SHA256)), encoding="utf-8")
        return paths, manifest

    def test_successful_build_binds_inputs_binary_and_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, manifest = self.fixture(root)
            result = RECORD.build_record(paths["probe.exe"], manifest, paths["compiler.exe"], "test", paths["cache.txt"], root)
            self.assertEqual(result["binary_sha256"], RECORD.sha(paths["probe.exe"]))
            self.assertTrue(result["bound_inputs_unchanged"])
            self.assertFalse(result["runtime_loader_resolution_proven"])
            self.assertEqual(result["supported_pool_poll"], ["default", "0"])
            for path in (manifest, paths["compiler.exe"], paths["cache.txt"], paths["input.lib"], paths["helper.inc"]):
                self.assertEqual(result["bound_files"][str(path.resolve())], RECORD.sha(path))

    def test_modified_input_blocks_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, manifest = self.fixture(root)
            paths["input.lib"].write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "changed during compile"):
                RECORD.build_record(paths["probe.exe"], manifest, paths["compiler.exe"], "test", paths["cache.txt"], root)

    def test_modified_generated_helper_blocks_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, manifest = self.fixture(root)
            paths["helper.inc"].write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "changed during compile"):
                RECORD.build_record(paths["probe.exe"], manifest, paths["compiler.exe"], "test", paths["cache.txt"], root)


if __name__ == "__main__":
    unittest.main()
