from copy import deepcopy
import unittest

from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.policy_ir import parse_policy, policy_document, parse_history, history_document


def policy_document_fixture(rule="fixed", cache="none", prefetch=False):
    return {"format_version": "causal-policy-0.1",
            "placements": {kind: "cpu" for kind in (
                "projection", "attention", "router", "expert_ffn", "shared_expert_ffn", "combine",
                "gather_rows", "scatter_rows", "expert_gate", "expert_up", "expert_swiglu",
                "expert_down", "combine_rows")},
            "experts": {"device_rule": rule}, "cache": {"rule": cache},
            "release": {"rule": "none"},
            "prefetch": {"enabled": prefetch, "max_count": 1 if prefetch else 0,
                         "cache_budget_bytes": 256 if prefetch else 0}}


def history_fixture(experts=(3,), layer="moe-small"):
    return {"format_version": "causal-history-0.1", "entries": [
        {"request_id": "old-request", "layer_id": layer, "absolute_position": 10,
         "expert_ids": list(experts), "scores": None}]}


class PolicyIRTests(unittest.TestCase):
    def test_round_trip_keeps_explicit_stage_placements(self):
        document = policy_document_fixture()
        document["placements"]["expert_gate"] = "gpu"
        document["placements"]["expert_up"] = "gpu"
        document["placements"]["ffn_gate"] = "gpu"
        self.assertEqual(policy_document(parse_policy(document)), document)

    def test_rejects_implicit_or_unknown_fields(self):
        documents = []
        for key in ("format_version", "placements", "experts", "cache", "prefetch", "release"):
            document = policy_document_fixture()
            del document[key]
            documents.append(document)
        document = policy_document_fixture()
        document["prefetch"]["future_route"] = True
        documents.append(document)
        document = policy_document_fixture()
        document["experts"]["fixed_device"] = "gpu"
        documents.append(document)
        for document in documents:
            with self.subTest(document=document), self.assertRaises(InvalidInput):
                parse_policy(document)

    def test_rejects_bad_devices_and_rule_names(self):
        for section, key, value in (("placements", "attention", "CUDA0"),
                                    ("placements", "unknown_operation", "gpu"),
                                    ("experts", "device_rule", []),
                                    ("cache", "rule", "offline_farthest")):
            document = policy_document_fixture()
            document[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(Unsupported):
                parse_policy(document)

    def test_explicit_prefetch_budget_and_integer_types(self):
        for key, value in (("enabled", 1), ("max_count", True),
                           ("cache_budget_bytes", -1), ("cache_budget_bytes", 1.0)):
            document = policy_document_fixture(prefetch=True)
            document["prefetch"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(InvalidInput):
                parse_policy(document)
        document = policy_document_fixture(prefetch=True)
        document["prefetch"]["max_count"] = 0
        with self.assertRaises(InvalidInput):
            parse_policy(document)
        document = policy_document_fixture()
        document["prefetch"]["max_count"] = 1
        with self.assertRaises(InvalidInput):
            parse_policy(document)
        document = policy_document_fixture()
        document["prefetch"]["cache_budget_bytes"] = 1
        with self.assertRaises(InvalidInput):
            parse_policy(document)

    def test_history_round_trip_and_duplicate_keys(self):
        document = history_fixture()
        self.assertEqual(history_document(parse_history(document)), document)
        duplicate = deepcopy(document)
        duplicate["entries"].append(deepcopy(duplicate["entries"][0]))
        with self.assertRaises(InvalidInput):
            parse_history(duplicate)

    def test_history_rejects_bad_route_evidence(self):
        for field, value in (("absolute_position", True), ("expert_ids", [True]),
                             ("expert_ids", [1, 1]), ("scores", [float("nan")]),
                             ("scores", [10 ** 1000]),
                             ("scores", []), ("request_id", "")):
            document = history_fixture()
            document["entries"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(InvalidInput):
                parse_history(document)
