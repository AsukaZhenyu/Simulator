"""Immutable transfer membership caching preserves direct-Python semantics."""
from dataclasses import asdict
import unittest

from llm_infer_model.tensor.spec import M1Mapspace


def mapspace(rows):
    return M1Mapspace({}, rows, True, False, False)


class MapspaceMembershipTests(unittest.TestCase):
    def test_plain_tuple_membership_matches_sequence_and_reuses_index(self):
        members = tuple(["a", "b", "a"])
        space = mapspace({"route": members})
        for value in ("a", "b", "c"):
            self.assertEqual(space.allows("route", value), value in members)
        first = space.__dict__["_allowed_transfer_membership_cache"]["route"]
        self.assertIs(first[0], members)
        space.allows("route", "a")
        self.assertIs(space.__dict__["_allowed_transfer_membership_cache"]["route"], first)

    def test_tuple_identity_replacement_invalidates_index(self):
        rows = {"route": tuple(["a", "b"])}
        space = mapspace(rows)
        self.assertTrue(space.allows("route", "a"))
        first = space.__dict__["_allowed_transfer_membership_cache"]["route"]
        rows["route"] = tuple(["a", "b"])
        self.assertTrue(space.allows("route", "a"))
        same_values = space.__dict__["_allowed_transfer_membership_cache"]["route"]
        self.assertIsNot(same_values, first)
        self.assertIs(same_values[0], rows["route"])
        rows["route"] = ("c",)
        self.assertFalse(space.allows("route", "a"))
        self.assertTrue(space.allows("route", "c"))

    def test_mutable_replacement_is_live_and_never_uses_old_index(self):
        rows = {"route": ("a",)}
        space = mapspace(rows)
        self.assertTrue(space.allows("route", "a"))
        rows["route"] = ["b"]
        self.assertFalse(space.allows("route", "a"))
        self.assertTrue(space.allows("route", "b"))
        rows["route"].append("c")
        self.assertTrue(space.allows("route", "c"))

    def test_unknown_none_and_unhashable_queries_keep_sequence_semantics(self):
        space = mapspace({"route": ("a",), None: ("a",)})
        self.assertFalse(space.allows("unknown", "a"))
        self.assertTrue(space.allows(None, "a"))
        self.assertFalse(space.allows("route", ["a"]))
        self.assertFalse(space.allows("unknown", []))
        self.assertNotIn("_allowed_transfer_membership_cache", space.__dict__)
        with self.assertRaises(TypeError):
            space.allows([], "a")

    def test_none_members_preserve_original_type_error(self):
        space = mapspace({"route": None})
        with self.assertRaises(TypeError):
            space.allows("route", "a")
        self.assertNotIn("_allowed_transfer_membership_cache", space.__dict__)

    def test_nonstring_tuple_members_and_custom_equality_are_not_cached(self):
        class Equivalent:
            def __eq__(self, other):
                return other == "match"
            __hash__ = None
        space = mapspace({"route": (Equivalent(),)})
        self.assertTrue(space.allows("route", "match"))
        self.assertNotIn("_allowed_transfer_membership_cache", space.__dict__)
        mutable = mapspace({"route": (["a"],)})
        self.assertTrue(mutable.allows("route", ["a"]))
        self.assertNotIn("_allowed_transfer_membership_cache", mutable.__dict__)

    def test_string_subclass_query_keeps_custom_sequence_equality(self):
        class Query(str):
            def __eq__(self, other):
                return other == "a"
            __hash__ = None
        space = mapspace({"route": ("a",)})
        self.assertTrue(space.allows("route", Query("different")))
        self.assertNotIn("_allowed_transfer_membership_cache", space.__dict__)

    def test_cache_is_absent_from_dataclass_value_and_serialization(self):
        first = mapspace({"route": ("a",)})
        second = mapspace({"route": ("a",)})
        original = asdict(first)
        first.allows("route", "a")
        self.assertEqual(asdict(first), original)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
