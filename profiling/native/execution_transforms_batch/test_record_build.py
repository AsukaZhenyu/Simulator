"""CPU-only metadata-copy tests with a fake executable; no compilation claims."""
import json
from pathlib import Path
import tempfile
import unittest

from generate_probe import HERE, generate
from record_build import record_build, sha


class BatchBuildMetadataTests(unittest.TestCase):
    def prepare(self, root, *, separate_binary_directory):
        generated, manifest = generate(HERE.parent / "execution_transforms_dual/compute_context.cpp",
                                       HERE.parent / "matched_mechanisms/compute.cpp")
        build = root / "build"
        build.mkdir()
        source = build / "compute_batch.cpp"
        source.write_bytes(generated)
        manifest_path = build / "source-manifest.json"
        manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        target_directory = build / "Release" if separate_binary_directory else build
        target_directory.mkdir(exist_ok=True)
        binary = target_directory / "execution-transform-batch-probe.exe"
        binary.write_bytes(b"FAKE EXECUTABLE FOR METADATA TEST ONLY\x00\xff")
        return dict(binary=binary, generated_source=source, source_manifest=manifest_path,
                    cmake_source_dir=HERE, cmake_build_dir=build, config="Release")

    def test_multiconfig_layout_copies_exact_bytes_beside_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.prepare(Path(temporary), separate_binary_directory=True)
            before_source = args["generated_source"].read_bytes()
            before_manifest = args["source_manifest"].read_bytes()
            destination, record = record_build(**args)
            self.assertEqual(destination.parent, args["binary"].parent)
            self.assertEqual((destination.parent / "compute_batch.cpp").read_bytes(), before_source)
            self.assertEqual((destination.parent / "source-manifest.json").read_bytes(), before_manifest)
            self.assertEqual(args["generated_source"].read_bytes(), before_source)
            self.assertEqual(args["source_manifest"].read_bytes(), before_manifest)
            self.assertEqual(record["binary_sha256"], sha(args["binary"]))
            self.assertEqual(record["source_manifest"], json.loads(before_manifest))
            self.assertFalse(record["compile_verification_performed"])
            self.assertEqual(record["build_method"], "manual_metadata_only")
            self.assertIsNone(record["compile_success_reported_by"])
            self.assertFalse(record["hardware_measurement_performed"])

    def test_single_config_layout_skips_source_and_manifest_samefile(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.prepare(Path(temporary), separate_binary_directory=False)
            destination, record = record_build(**args)
            self.assertTrue(record["generated_source_copy"]["same_file_skipped"])
            self.assertTrue(record["source_manifest_copy"]["same_file_skipped"])
            self.assertEqual(record["generated_source_sha256"], sha(args["generated_source"]))
            self.assertEqual(json.loads(destination.read_text())["binary_sha256"], sha(args["binary"]))

    def test_declared_cmake_hook_context_is_not_compile_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.prepare(Path(temporary), separate_binary_directory=True)
            _, record = record_build(**args, cmake_post_build=True, cmake_generator="Fake test generator")
            self.assertEqual(record["build_method"], "cmake_post_build")
            self.assertEqual(record["compile_success_reported_by"], "cmake_post_build_hook")
            self.assertFalse(record["compile_verification_performed"])
            self.assertTrue(record["invocation_context_is_caller_declared"])
            self.assertEqual(record["command"][-2:], ["--target", "execution-transform-batch-probe"])

    def test_stale_generated_source_is_rejected_before_metadata_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.prepare(Path(temporary), separate_binary_directory=True)
            args["generated_source"].write_bytes(args["generated_source"].read_bytes() + b"// drift\n")
            with self.assertRaisesRegex(ValueError, "identity differs for generated"):
                record_build(**args)
            self.assertFalse((args["binary"].parent / "compute-batch-build.json").exists())

    def test_ambiguous_binary_adjacent_cpp_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.prepare(Path(temporary), separate_binary_directory=True)
            (args["binary"].parent / "unrelated.cpp").write_text("// unrelated\n")
            with self.assertRaisesRegex(ValueError, "ambiguous extra C[+][+] sources"):
                record_build(**args)


if __name__ == "__main__":
    unittest.main()
