"""Offline P1/P2 prediction-versus-measurement verdict tool.

All fixtures are synthetic. No GPU, benchmark, simulator or repository data is
read; only the new ``tools/verdict`` package is exercised.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verdict import analyze_crossover, analyze_ranking, load, validate
from verdict.schema import ValidationError, order, relative_gap

MISSING = object()


def row(name, predicted=MISSING, measured=MISSING):
    entry = {"name": name}
    if predicted is not MISSING:
        entry["predicted_latency_ns"] = predicted
    if measured is not MISSING:
        entry["measured_latency_ns"] = measured
    return entry


def condition(label, *configs, value=None):
    entry = {"label": label, "configs": list(configs)}
    if value is not None:
        entry["value"] = value
    return entry


def document(*conditions):
    return {"schema": "verdict-input-1", "conditions": list(conditions)}


class ToleranceTests(unittest.TestCase):
    def test_relative_gap_is_symmetric_and_order_uses_it(self):
        self.assertAlmostEqual(relative_gap(100.0, 80.0), 20.0 / 90.0, places=12)
        self.assertAlmostEqual(relative_gap(80.0, 100.0), 20.0 / 90.0, places=12)
        self.assertEqual(order(100.0, 101.0), 0)       # 1 % tie at default 2 %
        self.assertEqual(order(100.0, 103.0), -1)      # 3 % apart: first is faster
        self.assertEqual(order(103.0, 100.0), 1)
        self.assertEqual(order(100.0, 100.0, 0.0), 0)  # exact equality stays a tie
        self.assertEqual(order(100.0, 101.0, 0.0), -1)

    def test_tolerance_must_be_in_unit_interval(self):
        for bad in (-0.1, 1.0, 2.0, float("nan")):
            with self.subTest(tolerance=bad):
                with self.assertRaises(ValueError):
                    order(100.0, 101.0, bad)


class CrossoverTests(unittest.TestCase):
    def test_predicted_and_measured_flip_in_same_interval_confirms(self):
        result = analyze_crossover(document(
            condition("8GB",
                      row("cpu", 100.0, 100.0),
                      row("hybrid", 80.0, 90.0)),
            condition("16GB",
                      row("cpu", 100.0, 100.0),
                      row("hybrid", 120.0, 110.0)),
        ))
        self.assertEqual(result["verdict"], "CONFIRMED")
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["config_pair"], ["cpu", "hybrid"])
        self.assertEqual(event["relation"], "same_interval")
        self.assertEqual(event["reason_code"], "same_interval")
        self.assertEqual(event["predicted"]["from"]["faster"], "hybrid")
        self.assertEqual(event["predicted"]["to"]["faster"], "cpu")
        self.assertAlmostEqual(event["predicted"]["from"]["gap"], 20.0 / 90.0, places=12)
        self.assertAlmostEqual(event["predicted"]["to"]["gap"], 20.0 / 110.0, places=12)
        self.assertAlmostEqual(event["measured"]["from"]["gap"], 10.0 / 95.0, places=12)
        self.assertEqual(result["summary"]["confirmed_same_interval"], 1)
        self.assertEqual(result["summary"]["confirmed_adjacent_interval"], 0)
        self.assertEqual(result["reason_code"], "same_interval")
        self.assertEqual(len(result["predicted_crossovers"]), 1)
        self.assertEqual(len(result["measured_crossovers"]), 1)

    def test_predicted_flip_with_constant_measurement_is_coverage_unresolved(self):
        result = analyze_crossover(document(
            condition("8GB",
                      row("cpu", 100.0, 100.0),
                      row("hybrid", 80.0, 90.0)),
            condition("16GB",
                      row("cpu", 100.0, 100.0),
                      row("hybrid", 120.0, 95.0)),
        ))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["reason_code"], "coverage_unresolved")
        event = result["events"][0]
        self.assertEqual(event["relation"], "no_flip")
        self.assertEqual(event["reason_code"], "coverage_unresolved")
        self.assertIn("coverage_unresolved", event["reason"])
        self.assertEqual(result["summary"]["not_confirmed"], 0)
        self.assertEqual(result["summary"]["coverage_unresolved"], 1)

    def test_monotonic_measurement_within_range_is_coverage_unresolved(self):
        result = analyze_crossover(document(
            condition("c0", row("cpu", 100.0, 100.0), row("hybrid", 80.0, 90.0)),
            condition("c1", row("cpu", 100.0, 100.0), row("hybrid", 120.0, 85.0)),
            condition("c2", row("cpu", 100.0, 100.0), row("hybrid", 140.0, 80.0)),
        ))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["reason_code"], "coverage_unresolved")
        self.assertEqual([event["interval"]["from_index"] for event in result["events"]], [0])
        self.assertEqual(result["events"][0]["reason_code"], "coverage_unresolved")
        self.assertEqual(result["summary"]["coverage_unresolved"], 1)
        self.assertEqual(result["summary"]["not_confirmed"], 0)
        self.assertEqual(result["summary"]["confirmed_same_interval"], 0)
        self.assertEqual(result["summary"]["confirmed_adjacent_interval"], 0)
        self.assertEqual(result["measured_crossovers"], [])

    def test_measured_flip_only_far_away_is_not_confirmed(self):
        result = analyze_crossover(document(
            condition("c0", row("cpu", 100.0, 100.0), row("hybrid", 80.0, 90.0)),
            condition("c1", row("cpu", 100.0, 100.0), row("hybrid", 120.0, 85.0)),
            condition("c2", row("cpu", 100.0, 100.0), row("hybrid", 130.0, 80.0)),
            condition("c3", row("cpu", 100.0, 100.0), row("hybrid", 140.0, 120.0)),
            condition("c4", row("cpu", 100.0, 100.0), row("hybrid", 150.0, 130.0)),
        ))
        self.assertEqual(result["verdict"], "NOT_CONFIRMED")
        event = result["events"][0]
        self.assertEqual(event["interval"]["from_index"], 0)
        self.assertEqual(event["relation"], "other_interval")
        self.assertEqual(event["reason_code"], "other_interval")
        self.assertEqual(result["summary"]["not_confirmed"], 1)
        self.assertEqual(result["summary"]["coverage_unresolved"], 0)
        self.assertEqual({flip["interval"]["from_index"] for flip in result["measured_crossovers"]}, {2})

    def test_measured_flip_in_adjacent_interval_confirms(self):
        result = analyze_crossover(document(
            condition("c0", row("cpu", 100.0, 100.0), row("hybrid", 80.0, 90.0)),
            condition("c1", row("cpu", 100.0, 100.0), row("hybrid", 90.0, 110.0)),
            condition("c2", row("cpu", 100.0, 100.0), row("hybrid", 120.0, 115.0)),
        ))
        self.assertEqual(result["verdict"], "CONFIRMED")
        event = result["events"][0]
        self.assertEqual(event["interval"]["from_index"], 1)
        self.assertEqual(event["relation"], "adjacent_interval")
        self.assertEqual(event["reason_code"], "adjacent_interval")
        self.assertEqual(event["measured_flip_interval"]["from_index"], 0)
        self.assertEqual(result["summary"]["confirmed_same_interval"], 0)
        self.assertEqual(result["summary"]["confirmed_adjacent_interval"], 1)
        self.assertEqual(result["reason_code"], "adjacent_interval")

    def test_all_ties_are_inconclusive_not_contradiction(self):
        result = analyze_crossover(document(
            condition("a", row("cpu", 100.0, 100.0), row("hybrid", 100.0, 100.0)),
            condition("b", row("cpu", 120.0, 120.0), row("hybrid", 120.0, 120.0)),
        ))
        self.assertEqual(result["events"], [])
        self.assertEqual(result["predicted_crossovers"], [])
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertIn("no predicted ranking crossover", result["reason"])

    def test_missing_measured_value_makes_the_event_inconclusive(self):
        result = analyze_crossover(document(
            condition("8GB",
                      row("cpu", 100.0, 100.0),
                      row("hybrid", 80.0, 90.0)),
            condition("16GB",
                      row("cpu", 100.0, 100.0),
                      row("hybrid", 120.0)),
        ))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        event = result["events"][0]
        self.assertEqual(event["relation"], "missing")
        self.assertEqual(event["reason_code"], "missing_values")
        self.assertIn("conditions[1].configs[hybrid].measured_latency_ns", event["reason"])
        self.assertEqual(len(result["measured_skipped_pairs"]), 1)

    def test_measured_tie_is_inconclusive(self):
        result = analyze_crossover(document(
            condition("8GB", row("cpu", 100.0, 100.0), row("hybrid", 80.0, 99.0)),
            condition("16GB", row("cpu", 100.0, 100.0), row("hybrid", 120.0, 101.0)),
        ))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["events"][0]["relation"], "tie")
        self.assertEqual(result["events"][0]["reason_code"], "tied_measurement")

    def test_missing_adjacent_interval_prevents_a_no_flip_verdict(self):
        result = analyze_crossover(document(
            condition("c0", row("cpu", 100.0), row("hybrid", 80.0, 90.0)),
            condition("c1", row("cpu", 100.0, 100.0), row("hybrid", 90.0, 110.0)),
            condition("c2", row("cpu", 100.0, 100.0), row("hybrid", 120.0, 120.0)),
        ))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        event = result["events"][0]
        self.assertEqual(event["interval"]["from_index"], 1)
        self.assertEqual(event["relation"], "missing")
        self.assertEqual(event["reason_code"], "missing_values")

    def test_single_configuration_condition_is_reported_not_dropped(self):
        result = analyze_crossover(document(
            condition("only-cpu", row("cpu", 100.0, 100.0)),
            condition("both", row("cpu", 100.0, 100.0), row("hybrid", 120.0, 110.0)),
        ))
        self.assertEqual(result["verdict"], "INCONCLUSIVE")
        self.assertEqual(result["summary"]["pairs_considered"], 1)
        self.assertEqual(result["summary"]["skipped_pairs"], 1)
        self.assertEqual(result["skipped_pairs"][0]["config_pair"], ["cpu", "hybrid"])

    def test_within_tolerance_one_endpoint_is_ambiguous_not_a_flip(self):
        result = analyze_crossover(document(
            condition("8GB", row("cpu", 100.0, 100.0), row("hybrid", 80.0, 90.0)),
            condition("16GB", row("cpu", 100.0, 100.0), row("hybrid", 99.0, 99.0)),
        ))
        self.assertEqual(result["events"], [])
        self.assertEqual(len(result["ambiguous_crossovers"]), 1)
        self.assertEqual(result["verdict"], "INCONCLUSIVE")


class RankingTests(unittest.TestCase):
    def test_pairwise_tau_top1_and_regret(self):
        result = analyze_ranking(document(
            condition("mixed",
                      row("a", 10.0, 10.0),
                      row("b", 20.0, 25.0),
                      row("c", 30.0, 22.0),
                      row("d", 40.0, 40.0)),
            condition("wrong",
                      row("a", 10.0, 15.0),
                      row("b", 20.0, 12.0)),
        ))
        pairs = result["pairs"]
        self.assertEqual(pairs["total"], 7)
        self.assertEqual(pairs["comparable"], 7)
        self.assertEqual(pairs["concordant"], 5)
        self.assertEqual(pairs["discordant"], 2)
        self.assertEqual(pairs["predicted_tie"], 0)
        self.assertEqual(pairs["measured_tie"], 0)
        self.assertAlmostEqual(pairs["agreement"], 5.0 / 7.0, places=12)
        self.assertAlmostEqual(pairs["kendall_tau_b"], 3.0 / 7.0, places=12)
        self.assertAlmostEqual(pairs["kendall_tau_b_exact"], 3.0 / 7.0, places=12)
        self.assertAlmostEqual(result["conditions"][0]["pairs"]["kendall_tau_b"], 2.0 / 3.0, places=12)
        self.assertAlmostEqual(result["conditions"][1]["pairs"]["kendall_tau_b"], -1.0, places=12)
        top1 = result["top1"]
        self.assertEqual((top1["evaluated"], top1["hits"], top1["misses"]), (2, 1, 1))
        self.assertAlmostEqual(top1["hit_rate"], 0.5, places=12)
        self.assertEqual(top1["details"][0]["outcome"], "hit")
        self.assertEqual(top1["details"][1]["outcome"], "miss")
        regret = result["regret"]
        self.assertEqual(regret["evaluated"], 2)
        self.assertAlmostEqual(regret["mean"], 0.125, places=12)
        self.assertAlmostEqual(regret["median"], 0.125, places=12)
        self.assertAlmostEqual(regret["max"], 0.25, places=12)

    def test_ties_are_counted_and_excluded_from_agreement(self):
        result = analyze_ranking(document(
            condition("ties",
                      row("a", 100.0, 100.0),
                      row("b", 102.0, 120.0),
                      row("c", 300.0, 110.0)),
        ))
        pairs = result["pairs"]
        self.assertEqual(pairs["predicted_tie"], 1)
        self.assertEqual(pairs["measured_tie"], 0)
        self.assertEqual(pairs["both_tie"], 0)
        self.assertEqual(pairs["comparable"], 2)
        self.assertEqual(pairs["concordant"], 1)
        self.assertEqual(pairs["discordant"], 1)
        self.assertAlmostEqual(pairs["agreement"], 0.5, places=12)
        self.assertAlmostEqual(pairs["kendall_tau_b"], 0.0, places=12)
        self.assertAlmostEqual(pairs["kendall_tau_b_exact"], 1.0 / 3.0, places=12)
        top1 = result["top1"]
        self.assertEqual(top1["evaluated"], 0)
        self.assertEqual(top1["predicted_tie"], 1)
        self.assertIsNone(top1["hit_rate"])
        self.assertIsNone(result["regret"]["mean"])
        self.assertEqual(result["regret"]["predicted_tie_conditions"], ["ties"])

    def test_all_pairs_tied_gives_null_agreement_and_tau(self):
        result = analyze_ranking(document(
            condition("same", row("a", 100.0, 100.0), row("b", 100.0, 100.0)),
        ))
        pairs = result["pairs"]
        self.assertEqual(pairs["both_tie"], 1)
        self.assertEqual(pairs["comparable"], 0)
        self.assertIsNone(pairs["agreement"])
        self.assertIsNone(pairs["kendall_tau_b"])
        self.assertIsNone(pairs["kendall_tau_b_exact"])
        self.assertEqual(result["top1"]["predicted_tie"], 1)
        self.assertIsNone(result["regret"]["mean"])

    def test_missing_single_and_empty_conditions_are_counted(self):
        result = analyze_ranking(document(
            condition("single", row("a", 100.0, 100.0)),
            condition("empty"),
            condition("partial", row("a", 100.0, 100.0), row("b", 80.0), row("c", 90.0, 95.0)),
            condition("unmeasured", row("a", 100.0), row("b", 80.0)),
            condition("ok", row("a", 100.0, 100.0), row("b", 90.0, 95.0)),
        ))
        self.assertEqual(result["conditions_single_config"], ["single"])
        self.assertEqual(result["conditions_empty"], ["empty"])
        self.assertEqual([item["label"] for item in result["conditions_incomplete"]],
                         ["partial", "unmeasured"])
        self.assertEqual(result["conditions_evaluated"], ["ok"])
        partial = result["conditions"][2]
        self.assertEqual(partial["kind"], "incomplete")
        self.assertEqual(partial["missing_values"],
                         [{"config": "b", "missing": ["measured_latency_ns"]}])
        pairs = result["pairs"]
        self.assertEqual(pairs["total"], 5)
        self.assertEqual(pairs["missing"], 3)
        self.assertEqual(pairs["missing_measured"], 3)
        self.assertEqual(pairs["missing_predicted"], 0)
        self.assertEqual(pairs["missing_both"], 0)
        self.assertEqual(pairs["comparable"], 2)
        self.assertEqual(pairs["concordant"], 2)
        self.assertAlmostEqual(pairs["agreement"], 1.0, places=12)
        self.assertEqual(result["top1"]["evaluated"], 1)
        self.assertEqual(result["top1"]["hits"], 1)
        self.assertAlmostEqual(result["top1"]["hit_rate"], 1.0, places=12)
        self.assertEqual(result["regret"]["evaluated"], 1)
        self.assertTrue(any("missing values" in note for note in result["notes"]))

    def test_non_monotonic_noise_is_reported(self):
        result = analyze_ranking(document(
            condition("noise1",
                      row("a", 10.0, 10.0), row("b", 20.0, 30.0), row("c", 30.0, 20.0)),
            condition("noise2",
                      row("a", 10.0, 20.0), row("b", 20.0, 30.0), row("c", 30.0, 10.0)),
        ))
        pairs = result["pairs"]
        self.assertEqual(pairs["concordant"], 3)
        self.assertEqual(pairs["discordant"], 3)
        self.assertAlmostEqual(pairs["agreement"], 0.5, places=12)
        self.assertAlmostEqual(pairs["kendall_tau_b"], 0.0, places=12)
        self.assertAlmostEqual(pairs["kendall_tau_b_exact"], 0.0, places=12)
        self.assertAlmostEqual(result["top1"]["hit_rate"], 0.5, places=12)
        self.assertAlmostEqual(result["regret"]["mean"], 0.5, places=12)
        self.assertAlmostEqual(result["regret"]["median"], 0.5, places=12)
        self.assertAlmostEqual(result["regret"]["max"], 1.0, places=12)


class SchemaTests(unittest.TestCase):
    def assert_field_error(self, value, path, fragment):
        with self.assertRaises(ValidationError) as context:
            validate(value)
        matches = [error for error in context.exception.errors
                   if error["path"] == path and fragment in error["message"]]
        self.assertTrue(matches, "no error at %s containing %r in %s"
                        % (path, fragment, context.exception))

    def test_unknown_field_names_the_exact_path(self):
        value = document(condition("c", row("a", 100.0, 100.0)))
        value["conditions"][0]["configs"][0]["predicted_latency"] = 1
        self.assert_field_error(value, "conditions[0].configs[0].predicted_latency", "unknown field")

    def test_non_positive_and_boolean_latencies_are_rejected(self):
        for bad in (0, -1.0, True, "100"):
            with self.subTest(value=bad):
                value = document(condition("c", row("a", bad, 100.0)))
                self.assert_field_error(value, "conditions[0].configs[0].predicted_latency_ns",
                                        "finite positive")

    def test_duplicate_names_identify_the_later_field(self):
        duplicate_config = document(condition("c", row("a", 100.0, 100.0),
                                              row("a", 80.0, 90.0)))
        self.assert_field_error(duplicate_config, "conditions[0].configs[1].name", "duplicate")
        duplicate_label = document(condition("c", row("a", 100.0, 100.0)),
                                   condition("c", row("b", 100.0, 100.0)))
        self.assert_field_error(duplicate_label, "conditions[1].label", "duplicate")

    def test_missing_and_malformed_required_fields(self):
        value = document({"configs": []})
        self.assert_field_error(value, "conditions[0].label", "required field is missing")
        self.assert_field_error({"schema": "verdict-input-1"}, "conditions", "required field is missing")
        self.assert_field_error({"conditions": []}, "conditions", "at least one")
        self.assert_field_error({"conditions": "nope"}, "conditions", "expected an array")
        self.assert_field_error({"schema": "other", "conditions": []}, "schema", "expected")
        value = document(condition("c", row("a", 100.0, 100.0)))
        value["conditions"][0]["configs"] = "nope"
        self.assert_field_error(value, "conditions[0].configs", "expected an array")

    def test_null_latency_means_missing_and_is_accepted(self):
        value = document(condition("c", row("a", 100.0, 100.0), row("b", 80.0, None)))
        self.assertIs(validate(value), value)
        result = analyze_ranking(value)
        self.assertEqual(result["pairs"]["missing_measured"], 1)
        self.assertEqual(result["conditions_incomplete"][0]["missing_values"],
                         [{"config": "b", "missing": ["measured_latency_ns"]}])

    def test_load_rejects_duplicate_keys_non_finite_and_bad_json(self):
        cases = [
            ('{"conditions": [], "conditions": []}', "duplicate JSON key"),
            ('{"conditions": [{"label": "c", "configs": []}], "value": NaN}', "non-finite JSON"),
            ("{not json", "invalid JSON"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            for index, (text, fragment) in enumerate(cases):
                with self.subTest(case=index):
                    path = Path(folder) / ("case%d.json" % index)
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValidationError) as context:
                        load(path)
                    self.assertEqual(context.exception.errors[0]["path"], "<document>")
                    self.assertIn(fragment, context.exception.errors[0]["message"])
            with self.assertRaises(ValidationError) as context:
                load(Path(folder) / "does-not-exist.json")
            self.assertIn("cannot read", context.exception.errors[0]["message"])

    def test_load_accepts_a_complete_document(self):
        value = document(condition("c", row("a", 100.0, 100.0)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(load(path), value)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, payload=None, text=None, *arguments):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text(text if text is not None else json.dumps(payload), encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONIOENCODING"] = "utf-8"
            completed = subprocess.run(
                [sys.executable, str(TOOLS / "verdict" / "cli.py"), str(path), *arguments],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=environment)
            return completed

    @staticmethod
    def crossing_document():
        return document(
            condition("8GB", row("cpu", 100.0, 100.0), row("hybrid", 80.0, 90.0)),
            condition("16GB", row("cpu", 100.0, 100.0), row("hybrid", 120.0, 110.0)),
        )

    def test_json_output_contains_both_analyses(self):
        completed = self.run_cli(self.crossing_document(), None, "--json")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["schema"], "verdict-output-1")
        self.assertEqual(payload["tolerance"], 0.02)
        self.assertIn("(|a|+|b|)/2", payload["tie_rule"])
        self.assertEqual(set(payload["tau_b_variants"]),
                         {"kendall_tau_b", "kendall_tau_b_exact"})
        self.assertEqual(payload["p1"]["verdict"], "CONFIRMED")
        self.assertEqual(payload["p1"]["summary"]["confirmed_same_interval"], 1)
        self.assertEqual(payload["p1"]["summary"]["confirmed_adjacent_interval"], 0)
        self.assertEqual(payload["p2"]["conditions_total"], 2)
        self.assertIn("kendall_tau_b", payload["p2"]["pairs"])
        self.assertIn("kendall_tau_b_exact", payload["p2"]["pairs"])
        self.assertEqual(payload["p2"]["top1"]["hits"], 2)

    def test_human_output_and_kind_filter(self):
        completed = self.run_cli(self.crossing_document())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("P1 crossover", completed.stdout)
        self.assertIn("P2 ranking fidelity", completed.stdout)
        self.assertIn("CONFIRMED", completed.stdout)
        self.assertIn("same-interval confirmed 1", completed.stdout)
        only_p1 = self.run_cli(self.crossing_document(), None, "--json", "--kind", "p1")
        self.assertEqual(only_p1.returncode, 0, only_p1.stderr)
        self.assertIn("p1", json.loads(only_p1.stdout))
        self.assertNotIn("p2", json.loads(only_p1.stdout))

    def test_invalid_input_exits_two_with_the_field_path(self):
        payload = document(condition("c", row("a", 100.0, 100.0)))
        payload["conditions"][0]["configs"][0]["oops"] = 1
        completed = self.run_cli(payload)
        self.assertEqual(completed.returncode, 2)
        self.assertIn("conditions[0].configs[0].oops", completed.stderr)
        self.assertEqual(completed.stdout, "")

    def test_tolerance_flag_changes_tie_membership(self):
        payload = document(
            condition("c", row("a", 100.0, 100.0), row("b", 102.0, 130.0)),
        )
        default = json.loads(self.run_cli(payload, None, "--json", "--kind", "p2").stdout)
        strict = json.loads(self.run_cli(payload, None, "--json", "--kind", "p2",
                                         "--tol", "0.01").stdout)
        self.assertEqual(default["p2"]["pairs"]["predicted_tie"], 1)
        self.assertEqual(strict["p2"]["pairs"]["predicted_tie"], 0)
        self.assertEqual(strict["p2"]["pairs"]["concordant"], 1)
        self.assertEqual(strict["p2"]["pairs"]["discordant"], 0)


if __name__ == "__main__":
    unittest.main()
