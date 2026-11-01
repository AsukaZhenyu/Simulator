"""Small integer fixtures only.  No model, GPU, NumPy, or BLAS dependency."""
import copy
import unittest

from .mmid_reference import INT32_MAX, reference


TAILS = (1, 3, 31, 32, 33, 125, 256)
SPARSE_EXPERTS = (0, 1, 3, 4, 7, 8, 12, 15, 19, 24, 31, 37, 45, 52, 60, 63)


def fixture(tokens, *, si1=11):
    storage = []
    rows = []
    for token in range(tokens):
        start = token * 3 % len(SPARSE_EXPERTS)
        row = [SPARSE_EXPERTS[(start + slot) % len(SPARSE_EXPERTS)]
               for slot in range(8)]
        rows.append(row)
        storage.extend(row + [-1234567] * (si1 - 8))
    return storage, rows


class MMIDReference(unittest.TestCase):
    def call(self, storage, tokens=1, **changes):
        parameters = dict(n_experts=64, n_tokens=tokens, si1=8,
                          sis1=8, nchannels_y=8)
        parameters.update(changes)
        return reference(storage, **parameters)

    def test_one_token_has_hand_checked_expert_sorted_permutation(self):
        result = self.call([7, 0, 6, 1, 5, 2, 4, 3], n_experts=8, sis1=13)
        self.assertEqual(result["ids_dst"], [1, 3, 5, 7, 6, 4, 2, 0])
        self.assertEqual(result["ids_src1"], [1, 3, 5, 7, 6, 4, 2, 0])
        self.assertEqual(result["expert_bounds"], list(range(9)))

    def test_all_fixed_tails_and_channel_layouts_match_sorted_tuple_spec(self):
        for tokens in TAILS:
            storage, rows = fixture(tokens)
            triples = sorted((expert, token, slot)
                             for token, row in enumerate(rows)
                             for slot, expert in enumerate(row))
            for channels, source_stride in ((1, 3), (8, 11)):
                with self.subTest(tokens=tokens, channels=channels):
                    result = self.call(storage, tokens, si1=11,
                                       sis1=source_stride, nchannels_y=channels)
                    self.assertEqual(result["ids_src1"],
                                     [token * source_stride + slot % channels
                                      for _, token, slot in triples])
                    self.assertEqual(result["ids_dst"],
                                     [token * 8 + slot for _, token, slot in triples])
                    self.assertEqual(result["expert_bounds"],
                                     [sum(expert < boundary for expert, _, _ in triples)
                                      for boundary in range(65)])
                    self.assertEqual(sorted(result["ids_dst"]), list(range(tokens * 8)))

    def test_padding_is_ignored_and_inputs_are_not_modified(self):
        padded, rows = fixture(3)
        saved = copy.deepcopy(padded)
        compact = [expert for row in rows for expert in row]
        one = self.call(padded, 3, si1=11)
        two = self.call(compact, 3)
        self.assertEqual(padded, saved)
        for key in ("ids_src1", "ids_dst", "expert_bounds"):
            self.assertEqual(one[key], two[key])

    def test_empty_experts_have_repeated_boundaries(self):
        storage, _ = fixture(1)
        result = self.call(storage, si1=11)
        self.assertEqual(result["expert_bounds"][2], result["expert_bounds"][3])
        self.assertEqual(result["expert_bounds"][-1], 8)
        self.assertEqual(len(result["expert_bounds"]), 65)

    def test_duplicate_top8_is_rejected_instead_of_repaired(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.call([0, 1, 2, 3, 4, 5, 6, 6])

    def test_negative_and_too_large_expert_ids_are_rejected(self):
        for bad in (-1, 64):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "bounds"):
                self.call([0, 1, 2, 3, 4, 5, 6, bad])

    def test_missing_and_extra_storage_elements_are_rejected(self):
        for storage in (list(range(7)), list(range(9))):
            with self.assertRaisesRegex(ValueError, "strided rows"):
                self.call(storage)

    def test_boolean_float_and_out_of_range_storage_are_rejected(self):
        for bad in (True, 1.0, INT32_MAX + 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "int32"):
                self.call([0, 1, 2, 3, 4, 5, 6, bad])

    def test_boolean_zero_and_negative_parameters_are_rejected(self):
        for name in ("n_experts", "n_tokens", "si1", "sis1", "nchannels_y"):
            for bad in (True, 0, -1):
                with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                    self.call(list(range(8)), **{name: bad})

    def test_unsupported_scope_is_explicitly_rejected(self):
        for changes in (dict(n_experts=7), dict(n_experts=1025), dict(n_tokens=257)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "scope"):
                self.call(list(range(8)), **changes)

    def test_overlapping_rows_and_source_channel_rows_are_rejected(self):
        for changes in (dict(si1=7), dict(sis1=7, nchannels_y=8)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(list(range(8)), **changes)

    def test_source_index_int32_overflow_is_rejected(self):
        storage, _ = fixture(3, si1=8)
        with self.assertRaisesRegex(ValueError, "exceed int32"):
            self.call(storage, 3, sis1=INT32_MAX)
        storage, _ = fixture(2, si1=8)
        with self.assertRaisesRegex(ValueError, "exceed int32"):
            self.call(storage, 2, sis1=INT32_MAX - 1, nchannels_y=3)


if __name__ == "__main__":
    unittest.main()
