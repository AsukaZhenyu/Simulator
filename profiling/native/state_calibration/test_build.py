"""Provenance guards that run without a compiler, CUDA, or hardware activity."""

import hashlib
import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("state_native_build", HERE / "build.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class HistoricalHelperBindingTests(unittest.TestCase):
    def test_original_is_exactly_pinned(self):
        content = (HERE.parent / "highlevel_accuracy/highlevel_probe.cpp").read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), BUILD.HELPER_SHA256)

    def test_generated_helper_changes_only_main_symbol(self):
        original = (HERE.parent / "highlevel_accuracy/highlevel_probe.cpp").read_bytes()
        generated = BUILD.generated_helper(original)
        self.assertEqual(generated.replace(b"static int state_historical_main(int argc,char **argv)", b"int main(int argc,char **argv)"), original)
        self.assertEqual(generated.count(b"static int state_historical_main(int argc,char **argv)"), 1)

    def test_changed_helper_is_rejected(self):
        original = (HERE.parent / "highlevel_accuracy/highlevel_probe.cpp").read_bytes()
        for changed in (original + b"\n", original.replace(b"GGML_PREC_F32", b"GGML_PREC_DEFAULT"), b""):
            with self.subTest(size=len(changed)), self.assertRaisesRegex(ValueError, "historical helper bytes differ"):
                BUILD.generated_helper(changed)

    def test_execution_boundary_enum_is_explicit_and_shared_with_cpp(self):
        self.assertEqual(BUILD.execution_boundaries(), ("legacy_double_sync", "async_one_sync", "async_events"))
        header = (HERE / "execution_boundary.h").read_text(encoding="utf-8")
        self.assertEqual(header.count('#include "execution_boundaries.def"'), 3)
        self.assertIn("unknown STATE_PROBE_EXECUTION_BOUNDARY; no fallback", header)

    def test_execution_boundary_declarations_reject_duplicate_or_invalid_names(self):
        for content in ("", "STATE_BOUNDARY(legacy_double_sync)\nSTATE_BOUNDARY(legacy_double_sync)", "STATE_BOUNDARY(bad-name)", "silently_use_default", "STATE_BOUNDARY()"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                BUILD.execution_boundaries(content)

    def test_event_wrapper_abi_and_callback_sources_are_pinned(self):
        self.assertEqual(set(BUILD.EVENT_ABI_FILES), {"ggml/src/ggml-backend-impl.h", "ggml/src/ggml-backend.cpp", "ggml/src/ggml-cuda/ggml-cuda.cu"})
        self.assertTrue(all(len(value) == 64 for value in BUILD.EVENT_ABI_FILES.values()))


if __name__ == "__main__":
    unittest.main()
