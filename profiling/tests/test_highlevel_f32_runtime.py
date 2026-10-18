"""Offline guards for the q16 custom-MMF precision regression."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"profiling/native/highlevel_accuracy/build_f32_runtime.py"
spec=importlib.util.spec_from_file_location("highlevel_f32_builder",SOURCE)
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class StrictF32Runtime(unittest.TestCase):
    def test_unpinned_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"unsupported GGML router bytes"):
            builder.patch_router(b"different source")

    def test_precise_single_context_guard(self):
        raw=("// mma router\n"+builder.PATCH_OLD+"\n").encode()
        with patch.object(builder,"PINNED_ROUTER_SHA256",hashlib.sha256(raw).hexdigest()):
            result=builder.patch_router(raw).decode()
        self.assertEqual(result.count("ggml_get_op_params_i32(dst, 0) != GGML_PREC_F32"),1)
        self.assertEqual(result,"// mma router\n"+builder.PATCH_NEW+"\n")

    def test_duplicate_router_context_is_rejected(self):
        raw=("// mma\n"+builder.PATCH_OLD+builder.PATCH_OLD).encode()
        with patch.object(builder,"PINNED_ROUTER_SHA256",hashlib.sha256(raw).hexdigest()):
            with self.assertRaisesRegex(ValueError,"context differs"):
                builder.patch_router(raw)

    def test_original_pinned_source_and_tf32_trigger(self):
        upstream=ROOT.parent/"llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml/src/ggml-cuda"
        if not upstream.exists():self.skipTest("pinned external GGML checkout unavailable")
        router=(upstream/"ggml-cuda.cu").read_bytes()
        self.assertEqual(hashlib.sha256(router).hexdigest(),builder.PINNED_ROUTER_SHA256)
        self.assertIn("src1_ncols > 16",(upstream/"mmf.cu").read_text())
        self.assertIn("f32.tf32.tf32.f32",(upstream/"mma.cuh").read_text())
        self.assertNotIn("GGML_PREC_F32",builder.PATCH_OLD)
        patched=builder.patch_router(router)
        self.assertIn(b"ggml_get_op_params_i32(dst, 0) != GGML_PREC_F32",patched)

    def test_runtime_record_binds_current_inputs_and_cuda_dll(self):
        for folder in ("profiling","modeling","mapping"):
            if str(ROOT/folder) not in sys.path:sys.path.insert(0,str(ROOT/folder))
        from highlevel_accuracy.run import private_runtime_builds
        from highlevel_accuracy.bridge import BridgeError
        with tempfile.TemporaryDirectory(dir=ROOT/"profiling/highlevel_accuracy") as temporary:
            directory=Path(temporary);dll=directory/"ggml-cuda.dll";dll.write_bytes(b"private-binary")
            source=directory/"copied-source.cu";source.write_bytes(b"pinned-input")
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            record=dict(format_version="ggml-private-strict-f32-runtime-v1",patch_hunks=1,
                original_source_sha256={str(source):sha(source)},inputs_sha256={str(source):sha(source)},
                output_sha256={str(dll):sha(dll)},evidence_sha256={str(source):sha(source)})
            path=directory/"strict-f32-runtime.json";path.write_text(json.dumps(record))
            self.assertEqual(private_runtime_builds([directory])[str(path.resolve())]["record"],record)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(BridgeError,"input/output changed"):
                private_runtime_builds([directory])
            source.write_bytes(b"pinned-input");dll.write_bytes(b"changed")
            with self.assertRaisesRegex(BridgeError,"input/output changed"):
                private_runtime_builds([directory])
            dll.write_bytes(b"private-binary");record["output_sha256"]={str(source):sha(source)}
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(BridgeError,"does not bind supplied CUDA DLL"):
                private_runtime_builds([directory])


if __name__=="__main__":unittest.main()
