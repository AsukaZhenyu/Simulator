"""Synthetic-only model tests; no retained heldout evidence or native execution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "model.py"
SPEC = importlib.util.spec_from_file_location("effective_composition_model_test", SOURCE)
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def case(family="ffn", identifier=None):
    result = dict(id=identifier or family, family=family, backend="hybrid", dtype="f32",
                  hidden=8, intermediate=16, cpu_intermediate=4, tokens=1,
                  history=0, heads=0, kv_heads=0, head_dim=0)
    if family == "attention":
        result.update(intermediate=0, cpu_intermediate=0, history=8,
                      heads=2, kv_heads=1, head_dim=4)
    return result


def row(family="ffn", identifier=None, component_us=1, totals=None):
    geometry = case(family)
    rid = identifier or "profile-" + family
    metrics = {name: [round(component_us * 1000)] * 3 for name in model.component_metrics(geometry)}
    # Hand-derived equal-price recipes: FFN full=5p, resident=6p,
    # preload=9p; attention full=5p, hybrid=6p.
    baseline = {"gpu/resident": 5 * component_us, "hybrid/resident": 6 * component_us}
    if family == "ffn":
        baseline["hybrid/preload"] = 9 * component_us
    for strategy, total in (totals or baseline).items():
        metrics[strategy] = [round(total * 1000)] * 3
    return dict(id=rid, split="profile", status="measured", case=geometry, metrics=metrics,
                batch_repetitions=16, seed=11, repeat=0,
                raw_sha256=hashlib.sha256(rid.encode("utf-8")).hexdigest())


def known_fit_rows():
    # Shared coefficients: 2 us/boundary, 0.25 * critical branch, 7 us/preload.
    return [
        row("ffn", totals={"gpu/resident": 13, "hybrid/resident": 17, "hybrid/preload": 33}),
        row("attention", totals={"gpu/resident": 13, "hybrid/resident": 16}),
    ]


class CompositionModelTests(unittest.TestCase):
    def test_recovers_shared_coefficients_and_exports_complete_provenance(self):
        rows = known_fit_rows()
        original = deepcopy(rows)
        fitted = model.fit(rows)
        self.assertEqual(rows, original)
        for name, expected in zip(model.FEATURE_NAMES, (2, .25, 7)):
            self.assertAlmostEqual(fitted["coefficients"][name], expected, places=11)
        self.assertEqual(fitted["fit_diagnostics"]["weighted_design_rank"], 3)
        self.assertEqual(fitted["fit_diagnostics"]["observation_count"], 5)
        self.assertTrue(fitted["profile_complete_strategy_totals_used_for_calibration"])
        self.assertFalse(fitted["heldout_used_for_calibration"])
        self.assertFalse(fitted["accepted_capacity_costs"])
        self.assertEqual({x["row_id"] for x in fitted["training"]["rows"]}, {r["id"] for r in rows})
        for source in fitted["training"]["rows"]:
            self.assertRegex(source["row_sha256"], "^[0-9a-f]{64}$")
            self.assertRegex(source["raw_sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(set(fitted["training"]["target_metrics"]),
                         {"gpu/resident", "hybrid/resident", "hybrid/preload"})
        self.assertEqual(json.loads(json.dumps(fitted, allow_nan=False)), fitted)
        self.assertEqual(model.fit(list(reversed(rows))), fitted)

    def test_predictions_account_for_every_term_and_three_features(self):
        fitted = model.fit(known_fit_rows())
        prices = model.pooled_component_prices(known_fit_rows())
        expected = {
            ("ffn", "gpu/resident"): (5, 8, 13, (4, 0, 0)),
            ("ffn", "hybrid/resident"): (6, 11, 17, (5, 4, 0)),
            ("ffn", "hybrid/preload"): (9, 24, 33, (8, 4, 1)),
            ("attention", "gpu/resident"): (5, 8, 13, (4, 0, 0)),
            ("attention", "hybrid/resident"): (6, 10, 16, (5, 0, 0)),
        }
        for (family, strategy), (baseline, correction, total, features) in expected.items():
            with self.subTest(family=family, strategy=strategy):
                prediction = model.predict(case(family), strategy, prices[family], fitted)
                self.assertAlmostEqual(prediction["baseline_us"], baseline)
                self.assertAlmostEqual(prediction["correction_us"], correction)
                self.assertAlmostEqual(prediction["total_us"], total)
                self.assertAlmostEqual(prediction["baseline_ns"], 1000 * baseline)
                self.assertAlmostEqual(prediction["correction_ns"], 1000 * correction)
                self.assertAlmostEqual(prediction["predicted_ns"], 1000 * total)
                self.assertEqual(prediction["feature_units"]["hybrid_ffn_critical_branch_us"], "microseconds")
                self.assertEqual(tuple(prediction["features"].values()), features)
                self.assertAlmostEqual(math.fsum(prediction["correction_contributions_us"].values()), correction)
                self.assertAlmostEqual(prediction["baseline_us"] + prediction["correction_us"], prediction["total_us"])

    def test_critical_feature_uses_isolated_branch_max_and_preload_prices_are_independent(self):
        fitted = model.fit(known_fit_rows())
        prices = {name: 1 for name in model.component_metrics(case())}
        prices.update({"cpu/shard_gate": 10, "gpu/shard_down": 20,
                       "hybrid/preload_gate_weight_h2d": 2,
                       "hybrid/preload_up_weight_h2d": 3,
                       "hybrid/preload_down_weight_h2d": 4,
                       "hybrid/preload_weights": 9999,
                       "hybrid/parallel_branches": 9999, "hybrid/dispatch_barrier": 9999,
                       "hybrid/preload": 9999, "hybrid/resident": 9999})
        prediction = model.predict(case(), "hybrid/preload", prices, fitted)
        self.assertEqual(prediction["features"]["hybrid_ffn_critical_branch_us"], 23)
        self.assertEqual(prediction["baseline_us"], 34)  # max(13,23)+delivery+merge+2+3+4.
        self.assertAlmostEqual(prediction["correction_us"], 16 + .25 * 23 + 7)

    def test_each_training_residual_uses_its_own_process_metric_medians(self):
        first = row(identifier="profile-first")
        second = row(identifier="profile-second", component_us=2)
        for samples in first["metrics"].values():
            samples[:] = [samples[0] - 10, samples[0], samples[0] + 500]
        fitted = model.fit([first, second])
        by_row = {rid: [x for x in fitted["training"]["observations"] if x["row_id"] == rid]
                  for rid in (first["id"], second["id"])}
        self.assertEqual([x["row_component_baseline_us"] for x in by_row[first["id"]]], [5, 6, 9])
        self.assertEqual([x["row_component_baseline_us"] for x in by_row[second["id"]]], [10, 12, 18])
        self.assertTrue(all(x["signed_residual_us"] == 0 for x in fitted["training"]["observations"]))

    def test_pooled_prices_pool_samples_rather_than_process_medians(self):
        first, second = row(identifier="profile-a"), row(identifier="profile-b")
        for name in model.component_metrics(case()):
            first["metrics"][name] = [1000, 1000, 9000]
            second["metrics"][name] = [2000, 2000, 2000, 2000, 2000]
        prices = model.pooled_component_prices([first, second])["ffn"]
        self.assertTrue(all(x == 2 for x in prices.values()))
        # Component-only pooling does not inspect or require complete targets.
        for r in (first, second):
            for strategy in model.primary_strategies(r["case"]):
                r["metrics"].pop(strategy)
        self.assertEqual(model.pooled_component_prices([first, second])["ffn"], prices)
        with self.assertRaisesRegex(ValueError, "missing profile metric"):
            model.fit([first, second])

    def test_nonprofile_input_is_rejected_before_reading_its_metrics(self):
        class UnreadableHeldout(dict):
            def __getitem__(self, key):
                if key == "metrics":
                    raise AssertionError("heldout metrics must never be read")
                return super().__getitem__(key)

        heldout = UnreadableHeldout(split="heldout", metrics=object())
        for function in (model.fit, model.pooled_component_prices):
            with self.assertRaisesRegex(ValueError, "profile-only"):
                function([row(), heldout])
            with self.assertRaisesRegex(ValueError, "profile-only"):
                function([heldout])
        with self.assertRaisesRegex(ValueError, "profile absence"):
            model.fit([])

    def test_missing_failed_or_unbound_evidence_is_rejected(self):
        for mutation, pattern in (
            (lambda r: r.update(status="failed"), "failed or unmeasured"),
            (lambda r: r.pop("metrics"), "component metrics"),
            (lambda r: r["metrics"].pop("gpu/full_gate"), "missing profile metric"),
            (lambda r: r["metrics"].pop("hybrid/preload"), "missing profile metric"),
            (lambda r: r.update(raw_sha256=""), "SHA256"),
            (lambda r: r["metrics"].update({"gpu/full_gate": [0, 1, 1]}), "positive integer"),
        ):
            with self.subTest(pattern=pattern):
                malformed = row()
                mutation(malformed)
                with self.assertRaisesRegex(ValueError, pattern):
                    model.fit([malformed])
        second = row(identifier="profile-other")
        second["batch_repetitions"] = 8
        with self.assertRaisesRegex(ValueError, "mixed batch"):
            model.fit([row(), second])
        second = row(identifier="profile-other")
        second["case"]["hidden"] = 16
        with self.assertRaisesRegex(ValueError, "conflicting geometry"):
            model.fit([row(), second])

    def test_negative_residuals_are_signed_and_cannot_be_fitted_by_negative_corrections(self):
        fitted = model.fit([row("attention", totals={"gpu/resident": 4, "hybrid/resident": 5})])
        self.assertEqual(list(fitted["coefficients"].values()), [0, 0, 0])
        observations = fitted["training"]["observations"]
        self.assertEqual([x["signed_residual_us"] for x in observations], [-1, -1])
        self.assertEqual(fitted["fit_diagnostics"]["negative_residual_count"], 2)
        self.assertAlmostEqual(fitted["fit_diagnostics"]["squared_relative_error_objective"], 1 / 16 + 1 / 25)

    def test_signed_negative_observation_changes_relative_weighted_solution(self):
        fitted = model.fit([row("attention", totals={"gpu/resident": 9, "hybrid/resident": 5.5})])
        expected = (4 * 4 / 9 ** 2 + 5 * (-.5) / 5.5 ** 2) / (4 ** 2 / 9 ** 2 + 5 ** 2 / 5.5 ** 2)
        self.assertGreater(expected, 0)
        self.assertAlmostEqual(fitted["coefficients"]["composition_boundaries"], expected, places=12)
        self.assertEqual(fitted["fit_diagnostics"]["negative_residual_count"], 1)

    def test_identity_and_rank_deficient_inputs_return_finite_deterministic_models(self):
        identity = model.fit([row()])
        self.assertEqual(list(identity["coefficients"].values()), [0, 0, 0])
        self.assertEqual(identity["fit_diagnostics"]["weighted_design_rank"], 3)
        attention = model.fit([row("attention", totals={"gpu/resident": 13, "hybrid/resident": 16})])
        self.assertTrue(attention["fit_diagnostics"]["rank_deficient"])
        self.assertEqual(attention["fit_diagnostics"]["weighted_design_rank"], 1)
        self.assertEqual(attention["fit_diagnostics"]["zero_feature_columns"], list(model.FEATURE_NAMES[1:]))
        self.assertAlmostEqual(attention["coefficients"]["composition_boundaries"], 2)
        self.assertEqual(list(attention["coefficients"].values())[1:], [0, 0])
        json.dumps(attention, allow_nan=False)

    def test_scaled_qr_handles_dependent_and_ill_scaled_columns(self):
        coefficients, diagnostics = model._nnls([[1, 2, 0], [2, 4, 0]], [3, 6])
        self.assertEqual(diagnostics["weighted_design_rank"], 1)
        self.assertAlmostEqual(coefficients[0] + 2 * coefficients[1], 3)
        self.assertTrue(all(x >= 0 for x in coefficients))
        coefficients, diagnostics = model._nnls(
            [[1e-9, 0, 0], [0, 1e9, 0], [0, 0, 1]], [2e-9, 3e9, 4])
        for actual, expected in zip(coefficients, (2, 3, 4)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(diagnostics["weighted_design_rank"], 3)

    def test_unknown_strategy_missing_price_and_invalid_model_are_rejected(self):
        fitted = model.fit(known_fit_rows())
        prices = fitted["profile_component_prices_us"]["ffn"]
        for strategy in ("hybrid/resident@dispatch_model", "cpu/resident", "hybrid/cached"):
            with self.subTest(strategy=strategy), self.assertRaisesRegex(ValueError, "unsupported"):
                model.predict(case(), strategy, prices, fitted)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            model.predict(case("attention"), "hybrid/preload", prices, fitted)
        incomplete = dict(prices)
        incomplete.pop("gpu/full_up")
        with self.assertRaisesRegex(ValueError, "missing component"):
            model.predict(case(), "gpu/resident", incomplete, fitted)
        with self.assertRaisesRegex(ValueError, "unsupported effective"):
            model.predict(case(), "gpu/resident", prices, {})
        altered = deepcopy(fitted)
        altered["coefficients"]["composition_boundaries"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model.predict(case(), "gpu/resident", prices, altered)


if __name__ == "__main__":
    unittest.main()
