"""Physical GGML storage is opt-in for M2 macros, never ordinary M0/M1 ops."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_infer_model.tensor.spec import (
    DTYPE_F16, DTYPE_F32, DTYPE_Q4_K, DTYPE_Q6_K,
    InvalidInput, Unsupported, ggml_contiguous_layout, load_workload,
)


def tensor(tensor_id, dtype, ne, role, *, resident=False):
    nb, storage_bytes = ggml_contiguous_layout(dtype, ne)
    result = {
        "id": tensor_id, "name": tensor_id, "role": role,
        "dtype": dtype, "ne": list(ne), "nb": list(nb),
        "storage_bytes": storage_bytes,
    }
    if resident:
        result["initial_locations"] = ["dram"]
    return result


def macro_workload(dtype=DTYPE_Q4_K):
    return {
        "schema_version": "0.2", "id": "physical-storage-test",
        "origin": {"kind": "synthetic_layer"},
        "tensors": [
            tensor("w", dtype, (256, 2, 1, 1), "weight", resident=True),
            tensor("x", DTYPE_F32, (256, 1, 1, 1), "input", resident=True),
            tensor("y", DTYPE_F32, (2, 1, 1, 1), "output"),
        ],
        "operations": [{
            "id": "ffn", "semantic_op": "dense_ffn", "ggml_op": "M2_DENSE_FFN",
            "unary_op": None, "op_params": [], "inputs": ["w", "x"], "output": "y",
        }],
        "outputs": ["y"],
    }


def load_document(raw):
    # Public file entry point exercises origin gates and tensor/workload checks.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        path = Path(directory) / "workload.json"
        path.write_text(json.dumps(raw) + "\n", encoding="utf-8", newline="\n")
        return load_workload(path)


class ContiguousStorageTests(unittest.TestCase):
    def test_exact_block_sizes_and_outer_strides(self):
        # All four dimensions matter. The reduction dimension is first, and
        # nb[0] is one block's size for K quantization, not bytes per element.
        for dtype, block_bytes in ((DTYPE_Q4_K, 144), (DTYPE_Q6_K, 210)):
            with self.subTest(dtype=dtype):
                nb, size = ggml_contiguous_layout(dtype, (512, 3, 2, 5))
                self.assertEqual(nb, (block_bytes, block_bytes * 2, block_bytes * 6, block_bytes * 12))
                self.assertEqual(size, block_bytes * 60)

    def test_f16_and_f32_exact_extents(self):
        self.assertEqual(ggml_contiguous_layout(DTYPE_F16, (3, 5, 2, 7)), ((2, 6, 30, 60), 420))
        self.assertEqual(ggml_contiguous_layout(DTYPE_F32, (3, 5, 2, 7)), ((4, 12, 60, 120), 840))

    def test_quantized_rows_require_whole_blocks(self):
        for dtype in (DTYPE_Q4_K, DTYPE_Q6_K):
            for reduction in (1, 255, 257):
                with self.subTest(dtype=dtype, reduction=reduction):
                    with self.assertRaisesRegex(Unsupported, "ne\\[0\\] divisible by 256"):
                        ggml_contiguous_layout(dtype, (reduction, 2, 1, 1))

    def test_reject_invalid_extents_and_unknown_dtype(self):
        for ne in ((1, 2, 3), (True, 1, 1, 1), (0, 1, 1, 1), (1.5, 1, 1, 1)):
            with self.subTest(ne=ne), self.assertRaises(InvalidInput):
                ggml_contiguous_layout(DTYPE_F16, ne)
        with self.assertRaises(Unsupported):
            ggml_contiguous_layout("GGML_TYPE_Q8_0", (256, 1, 1, 1))


class M2StorageGateTests(unittest.TestCase):
    def test_m2_macro_preserves_real_datatypes_and_bytes(self):
        for origin in ("synthetic_layer", "measured_macro"):
            for dtype, expected in ((DTYPE_F16, 1024), (DTYPE_Q4_K, 288), (DTYPE_Q6_K, 420)):
                with self.subTest(origin=origin, dtype=dtype):
                    raw = macro_workload(dtype)
                    raw["origin"]["kind"] = origin
                    weight = load_document(raw).tensor_by_id["w"]
                    self.assertEqual(weight.dtype, dtype)
                    self.assertEqual(weight.ne, (256, 2, 1, 1))
                    self.assertEqual(weight.storage_bytes, expected)

    def test_non_m2_origins_remain_f32_only_for_both_schemas(self):
        for version in ("0.1", "0.2"):
            for origin in ("ggml", "synthetic_fixture"):
                for dtype in (DTYPE_F16, DTYPE_Q4_K, DTYPE_Q6_K):
                    with self.subTest(version=version, origin=origin, dtype=dtype):
                        raw = macro_workload(dtype)
                        raw["schema_version"] = version
                        raw["origin"]["kind"] = origin
                        raw["operations"][0]["semantic_op"] = "MUL_MAT"
                        with self.assertRaisesRegex(Unsupported, "non-M2 workloads model only"):
                            load_document(raw)

    def test_ordinary_matmul_cannot_use_quantized_weight_even_in_m2(self):
        raw = macro_workload()
        raw["operations"][0]["semantic_op"] = "MUL_MAT"
        with self.assertRaisesRegex(Unsupported, "ordinary MUL_MAT operations require F32"):
            load_document(raw)
        # Same geometry with F32 still exercises the existing ordinary op path.
        accepted = macro_workload(DTYPE_F32)
        accepted["operations"][0]["semantic_op"] = "MUL_MAT"
        self.assertEqual(load_document(accepted).operation_by_id["ffn"].semantic_op, "MUL_MAT")

    def test_ordinary_add_and_relu_reject_non_f32_input_or_output(self):
        for semantic, non_f32_output in (("ADD", False), ("RELU", True)):
            with self.subTest(semantic=semantic):
                raw = macro_workload(DTYPE_F32)
                raw["tensors"] = [
                    tensor("x", DTYPE_F32 if non_f32_output else DTYPE_F16,
                           (256, 1, 1, 1), "input", resident=True),
                    tensor("y", DTYPE_F16 if non_f32_output else DTYPE_F32,
                           (256, 1, 1, 1), "output"),
                ]
                raw["operations"][0].update(semantic_op=semantic, inputs=["x", "x"] if semantic == "ADD" else ["x"])
                with self.assertRaisesRegex(Unsupported, "ordinary .* operations require F32"):
                    load_document(raw)

    def test_reject_noncontiguous_strides_in_each_dimension(self):
        for dtype in (DTYPE_F16, DTYPE_Q4_K, DTYPE_Q6_K):
            for dimension in range(4):
                with self.subTest(dtype=dtype, dimension=dimension):
                    raw = macro_workload(dtype)
                    raw["tensors"][0]["nb"][dimension] += 1
                    with self.assertRaisesRegex(Unsupported, "non-contiguous"):
                        load_document(raw)

    def test_reject_wrong_storage_byte_claim(self):
        raw = macro_workload()
        raw["tensors"][0]["storage_bytes"] = 1024
        with self.assertRaisesRegex(InvalidInput, "layout implies 288 bytes"):
            load_document(raw)

    def test_reject_quantized_partial_row_through_public_loader(self):
        raw = macro_workload()
        raw["tensors"][0]["ne"][0] = 255
        with self.assertRaisesRegex(Unsupported, "row padding is not modeled"):
            load_document(raw)

    def test_unknown_dtype_and_views_are_still_unsupported(self):
        raw = macro_workload()
        raw["tensors"][0]["dtype"] = "GGML_TYPE_Q8_0"
        with self.assertRaisesRegex(Unsupported, "unsupported M2 GGML storage dtype"):
            load_document(raw)
        raw = macro_workload()
        raw["tensors"][0]["view_src"] = "x"
        with self.assertRaisesRegex(Unsupported, "aliasing"):
            load_document(raw)


if __name__ == "__main__":
    unittest.main()
