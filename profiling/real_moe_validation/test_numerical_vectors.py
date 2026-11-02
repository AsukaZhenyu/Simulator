"""Bounded workspace fixtures only; no NumPy, BLAS, model or GPU calls."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from .numerical_vectors import FORMAT, compare, inspect

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "build/real-moe-validation-20261027/numerical-vector-test-fixtures"


class NumericalVectors(unittest.TestCase):
    def setUp(self):
        FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=FIXTURE_ROOT, prefix="bounded-")
        self.directory = Path(self.temporary.name).resolve()
        self.assertTrue(self.directory.is_relative_to(FIXTURE_ROOT.resolve()))

    def tearDown(self):
        self.assertTrue(self.directory.is_relative_to(FIXTURE_ROOT.resolve()))
        self.temporary.cleanup()

    def vector(self, name, rows):
        self.assertTrue(rows and rows[0] and all(len(row) == len(rows[0]) for row in rows))
        raw = b"".join(struct.pack("<f", value) for row in rows for value in row)
        path = self.directory / name
        path.write_bytes(raw)
        columns = len(rows[0])
        meta = dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),format=FORMAT,rows=len(rows),columns=columns)
        observations = [dict(output_index=index,vocabulary_size=columns,
                             full_f32_logits_sha256=hashlib.sha256(raw[index * columns * 4:(index + 1) * columns * 4]).hexdigest())
                        for index in range(len(rows))]
        return meta, observations

    def test_file_and_all_output_rows_are_content_bound(self):
        meta, observations = self.vector("full.f32le", [[1., -2., 3.], [0., 4., 5.]])
        saved = copy.deepcopy((meta, observations))
        result = inspect(meta, observations)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["finite_entries"], 6)
        self.assertEqual(result["output_rows_bound_to_observations"], 2)
        self.assertEqual((meta, observations), saved)

    def test_wrong_whole_file_hash_is_rejected(self):
        meta, observations = self.vector("hash.f32le", [[1.]])
        meta["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sha256"):
            inspect(meta, observations)

    def test_wrong_bytes_or_actual_file_extent_is_rejected(self):
        meta, observations = self.vector("bytes.f32le", [[1., 2.]])
        changed = dict(meta,bytes=4)
        with self.assertRaisesRegex(ValueError, "bytes"):
            inspect(changed, observations)
        Path(meta["path"]).write_bytes(Path(meta["path"]).read_bytes() + b"\0")
        with self.assertRaisesRegex(ValueError, "bytes"):
            inspect(meta, observations)

    def test_bad_shape_format_and_boolean_dimensions_are_rejected(self):
        meta, observations = self.vector("shape.f32le", [[1., 2.]])
        for change in (dict(rows=True),dict(rows=9),dict(columns=50305),dict(format="native float"),dict(rows=2)):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    inspect(dict(meta,**change), observations)

    def test_nan_and_infinity_are_rejected_even_when_hashes_match(self):
        for index, value in enumerate((float("nan"),float("inf"),-float("inf"))):
            meta, observations = self.vector(str(index) + ".f32le", [[value]])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                inspect(meta, observations)

    def test_output_row_hashes_and_exact_inventory_are_required(self):
        meta, observations = self.vector("rows.f32le", [[1.], [2.]])
        changed = copy.deepcopy(observations)
        changed[1]["full_f32_logits_sha256"] = observations[0]["full_f32_logits_sha256"]
        with self.assertRaisesRegex(ValueError, "row hash"):
            inspect(meta, changed)
        for ledger in (observations[:1], [observations[0],observations[0]]):
            with self.assertRaises(ValueError):
                inspect(meta, ledger)

    def test_comparison_covers_every_vocab_value_and_output(self):
        left, _ = self.vector("left.f32le", [[1., 2.], [0., -4.]])
        right, _ = self.vector("right.f32le", [[1., 2.5], [0., -3.5]])
        result = compare(left, right)
        self.assertFalse(result["bitwise_equal"])
        self.assertEqual(result["entries_total"], 4)
        self.assertEqual(result["entries_equal_bits"], 2)
        self.assertEqual(result["entries_different_bits"], 2)
        self.assertEqual(result["first_difference"]["output_index"], 0)
        self.assertEqual(result["first_difference"]["vocab_index"], 1)
        self.assertEqual(result["max_abs"], .5)
        self.assertEqual(result["mean_abs"], .25)
        self.assertAlmostEqual(result["relative_l2"], math.sqrt(.5 / 21))
        self.assertEqual(len(result["per_output"]), 2)
        json.dumps(result,allow_nan=False)

    def test_identical_bits_have_zero_distances(self):
        meta, _ = self.vector("same.f32le", [[1., -2.]])
        result = compare(meta, meta)
        self.assertTrue(result["bitwise_equal"])
        self.assertIsNone(result["first_difference"])
        self.assertEqual((result["max_abs"],result["mean_abs"],result["relative_l2"],result["max_ulp"]), (0.,0.,0.,0))

    def test_signed_zero_is_bitwise_different_but_numeric_ULP_zero(self):
        left, _ = self.vector("plus-zero.f32le", [[0.]])
        right, _ = self.vector("minus-zero.f32le", [[-0.]])
        result = compare(left, right)
        self.assertFalse(result["bitwise_equal"])
        self.assertEqual(result["entries_different_bits"], 1)
        self.assertEqual(result["max_ulp"], 0)
        self.assertEqual(result["max_abs"], 0.)
        self.assertEqual(result["relative_l2"], 0.)
        self.assertEqual(result["relative_l2_status"], "both_numerically_zero_distance_convention")
        self.assertEqual(result["first_difference"]["right_bits"], "0x80000000")

    def test_adjacent_positive_and_negative_F32_values_have_ULP_one(self):
        for index, bits in enumerate((0x3f800000,0xbf800000)):
            one = struct.unpack("<f",struct.pack("<I",bits))[0]
            two = struct.unpack("<f",struct.pack("<I",bits + 1))[0]
            left, _ = self.vector(f"adjacent-left-{index}.f32le", [[one]])
            right, _ = self.vector(f"adjacent-right-{index}.f32le", [[two]])
            self.assertEqual(compare(left, right)["max_ulp"], 1)

    def test_zero_reference_with_nonzero_error_has_undefined_relative_L2(self):
        left, _ = self.vector("zero.f32le", [[0.]])
        right, _ = self.vector("nonzero.f32le", [[1.]])
        result = compare(left, right)
        self.assertIsNone(result["relative_l2"])
        self.assertEqual(result["relative_l2_status"], "undefined_zero_reference_norm")
        json.dumps(result,allow_nan=False)

    def test_reshaping_same_flat_inventory_cannot_change_row_correspondence(self):
        left, _ = self.vector("matrix.f32le", [[1.,2.],[3.,4.]])
        right, _ = self.vector("flat.f32le", [[1.,2.,3.,4.]])
        with self.assertRaisesRegex(ValueError, "correspondence"):
            compare(left, right)


if __name__ == "__main__":
    unittest.main()
