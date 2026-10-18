"""NVTX/provenance guards without CUDA, compilation or native execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("wait_diag_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = load("generate_probe")
RECORD = load("record_build")


class DiagnosticGenerationTests(unittest.TestCase):
    def test_three_original_sources_match_exact_pins(self):
        for relative, digest in GEN.PINS.items():
            self.assertEqual(GEN.sha(HERE.parent / relative), digest)

    def test_generated_changes_are_exactly_reversible(self):
        renames = {"highlevel_accuracy/highlevel_probe.cpp": b"static int single_layer_historical_main(int argc,char **argv)",
                   "single_layer_calibration/probe.cpp": b"static int context_pair_single_layer_main(int argc,char **argv)",
                   "single_layer_context_pair/paired.cpp": b"static int wait_diag_pair_main(int argc,char **argv)"}
        for relative in GEN.PINS:
            original = (HERE.parent / relative).read_bytes()
            generated = GEN.generate(relative, original)
            restored = generated.replace(renames[relative], GEN.MAIN)
            restored = restored.replace(GEN.ACTION_INSERTION, b"")
            restored = restored.replace(GEN.BLOCK_INSERTION, b"")
            self.assertEqual(restored, original)

    def test_altered_source_never_gets_instrumented(self):
        for relative in GEN.PINS:
            with self.assertRaisesRegex(ValueError, "pinned executor"):
                GEN.generate(relative, (HERE.parent / relative).read_bytes() + b"\n")

    def test_every_action_is_wrapped_once_without_replacing_GGML_calls(self):
        source = (HERE.parent / "highlevel_accuracy/highlevel_probe.cpp").read_bytes()
        generated = GEN.generate("highlevel_accuracy/highlevel_probe.cpp", source)
        self.assertEqual(generated.count(GEN.ACTION_INSERTION), 1)
        self.assertEqual(generated.count(b"ggml_backend_graph_compute(g.backend,g.graph)"), source.count(b"ggml_backend_graph_compute(g.backend,g.graph)"))
        self.assertEqual(generated.count(b"ggml_backend_tensor_copy(source,target)"), source.count(b"ggml_backend_tensor_copy(source,target)"))

    def test_labels_borrow_existing_JSON_strings_without_label_allocation(self):
        source = (HERE / "probe.cpp").read_text(encoding="utf-8")
        section = source[source.index("class ActionRange"):source.index("class BlockRange")]
        self.assertIn('action.at("target_id").get_ref<const std::string&>().c_str()', section)
        self.assertNotIn("get<std::string>", section)
        self.assertNotIn("std::to_string", section)
        self.assertNotIn("json(", section)

    def test_sampling_selection_is_checked_not_processwide(self):
        source = (HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn('{"target_range_count_per_block",1+warmup+samples}', source)
        self.assertIn("select the last samples ranges as timed", source)
        self.assertIn("Reject missing, extra or dropped ranges", source)
        self.assertIn("exclude them from action attribution", source)
        self.assertIn("do not sum them as disjoint costs", source)
        self.assertIn('{"usable_for_parameter_fit",false}', source)
        self.assertIn('{"usable_for_heldout_acceptance",false}', source)

    def test_NVTX_is_header_only_and_cannot_silently_be_disabled(self):
        source = (HERE / "probe.cpp").read_text(encoding="utf-8")
        self.assertIn("#include <nvtx3/nvToolsExt.h>", source)
        self.assertIn("defined(NVTX_DISABLE) || defined(NVTX_NO_IMPL)", source)
        cmake = (HERE / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("${NVTX_INCLUDE_DIR}/nvtx3/*.h", cmake)
        self.assertNotIn("FetchContent", cmake)
        self.assertNotIn("find_library(NVTX", cmake)


class ProvenanceTests(unittest.TestCase):
    def fixture(self, root):
        files = {}
        for name in ("binary.exe", "compiler.exe", "cache.txt", "helper.inc", "input.lib"):
            files[name] = root / name
            files[name].write_bytes(name.encode())
        manifest = root / "source.json"
        manifest.write_text(json.dumps(dict(format_version="single-layer-wait-diagnosis-generated-v1",
            bound_files={str(files["input.lib"]): RECORD.sha(files["input.lib"])},
            generated_files={str(files["helper.inc"]): RECORD.sha(files["helper.inc"])},
            immutable_executors_sha256=GEN.PINS, allowed_changes=["instrumentation only"])), encoding="utf-8")
        return files, manifest

    def test_successful_build_is_source_bound_and_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            result = RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root, root)
            self.assertTrue(result["nvtx_header_only"])
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["usable_for_parameter_fit"])
            self.assertFalse(result["usable_for_heldout_acceptance"])
            self.assertFalse(result["runtime_loader_resolution_proven"])

    def test_changed_generated_source_blocks_build_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            files["helper.inc"].write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "changed during compilation"):
                RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root, root)


if __name__ == "__main__":
    unittest.main()
