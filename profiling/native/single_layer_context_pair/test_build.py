"""Counterbalance/provenance tests without starting a measurement process."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("context_pair_native_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = load("generate_probe")
RECORD = load("record_build")


class ImmutableExecutorTests(unittest.TestCase):
    def test_both_originals_are_pinned_and_only_main_symbols_change(self):
        for relative, expected, replacement in (
            ("highlevel_accuracy/highlevel_probe.cpp", GEN.HIGHLEVEL_SHA256, b"static int single_layer_historical_main(int argc,char **argv)"),
            ("single_layer_calibration/probe.cpp", GEN.SINGLE_LAYER_SHA256, b"static int context_pair_single_layer_main(int argc,char **argv)"),
        ):
            original = (HERE.parent / relative).read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), expected)
            generated = GEN.rename_main(original, expected, replacement)
            self.assertEqual(generated.replace(replacement, b"int main(int argc,char **argv)"), original)

    def test_changed_original_rejected(self):
        original = (HERE.parent / "single_layer_calibration/probe.cpp").read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable executor"):
            GEN.rename_main(original + b"\n", GEN.SINGLE_LAYER_SHA256, b"unused")

    def test_same_executor_is_reused_in_fixed_counterbalanced_order(self):
        source = (HERE / "paired.cpp").read_text(encoding="utf-8")
        self.assertEqual(source.count("Executor executor("), 1)
        self.assertIn('order=="ABBA"||order=="BAAB"', source)
        self.assertIn("const char label=order.at(index)", source)
        self.assertIn('single_layer::run(executor,"context-trace",warmup,samples,pool)', source)
        self.assertIn('single_layer::run(executor,"bare-replay",warmup,samples,pool)', source)
        self.assertIn("isolated_target(executor,target,warmup,samples,pool)", source)

    def test_only_exact_pv_target_isolated_and_subset_explicit(self):
        source = (HERE / "paired.cpp").read_text(encoding="utf-8")
        self.assertIn('at("kind")=="attention_pv"', source)
        self.assertIn('{"component_subset",true}', source)
        self.assertIn("not_a_complete_component_profile", source)
        self.assertIn("for(auto &stamp:observations){stamp.begin=tick();executor.action(selected);stamp.end=tick();}", source)

    def test_every_block_resets_validation_records_and_never_exports_fit_acceptance(self):
        source = (HERE / "paired.cpp").read_text(encoding="utf-8")
        self.assertIn("clear_validation_records(executor);", source)
        self.assertIn("executor.copies=json::array()", source)
        self.assertIn('{"usable_for_parameter_fit",false}', source)
        self.assertIn('{"usable_for_heldout_acceptance",false}', source)
        self.assertIn("uncontrolled time-varying device state", source)


class BuildRecordTests(unittest.TestCase):
    def fixture(self, root):
        files = {}
        for name in ("binary.exe", "compiler.exe", "cache.txt", "first.inc", "second.inc", "input.lib"):
            path = root / name
            path.write_bytes(name.encode())
            files[name] = path
        manifest = root / "source.json"
        manifest.write_text(json.dumps(dict(format_version="single-layer-context-pair-generated-v1",
            bound_files={str(files["input.lib"]): RECORD.sha(files["input.lib"])},
            generated_files={str(files[name]): RECORD.sha(files[name]) for name in ("first.inc", "second.inc")},
            immutable_executors_sha256={"historical": GEN.HIGHLEVEL_SHA256, "single": GEN.SINGLE_LAYER_SHA256})), encoding="utf-8")
        return files, manifest

    def test_build_manifest_binds_both_generated_executors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            result = RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root)
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["usable_for_parameter_fit"])
            self.assertFalse(result["usable_for_heldout_acceptance"])
            for name in ("first.inc", "second.inc", "input.lib", "compiler.exe", "cache.txt"):
                self.assertEqual(result["bound_files"][str(files[name])], RECORD.sha(files[name]))
            self.assertEqual(result["supported_pair_kinds"], ["context_vs_isolated", "instrumentation"])

    def test_v2_pair_kind_and_A_B_meanings_are_explicit(self):
        source = (HERE / "paired.cpp").read_text(encoding="utf-8")
        self.assertIn('pair_kind=="context_vs_isolated"||pair_kind=="instrumentation"', source)
        self.assertIn('{"format_version","single-layer-context-pair-native-v2"}', source)
        self.assertIn('{"A_label",pair_kind=="instrumentation"?"bare-replay":"isolated-components"}', source)
        self.assertIn('{"B_label","context-trace"}', source)
        self.assertIn('require(argc==9', source)

    def test_input_change_blocks_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            files["input.lib"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed during compile"):
                RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root)

    def test_generated_executor_change_blocks_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, manifest = self.fixture(root)
            files["second.inc"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed during compile"):
                RECORD.build_record(files["binary.exe"], manifest, files["compiler.exe"], "test", files["cache.txt"], root)


if __name__ == "__main__":
    unittest.main()
