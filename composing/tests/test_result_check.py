"""A result from another machine is accepted only if it says how it was taken."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from composing.result_check import inspect_result, require_result


def document(prediction_sha: str, **overrides) -> dict:
    value = {
        "schema_version": "dense_endtoend_measurement_v1",
        "created_at_utc": "2026-11-05T00:00:00+00:00",
        "llama_bench": "bench.exe", "llama_bench_sha256": "a" * 64,
        "threads": 2, "cpu_mask": [2, 4],
        "thread_placement": "explicit_strict_cpu_mask",
        "op_offload": False, "decode_history_knob": "llama-bench -d",
        "bench_protocol": {"echoed_and_checked": {"flash_attn": False},
                           "pinned_not_echoed": {"-mmp": "1"}},
        "decode_grouping": "one_process_per_shape",
        "warmup": {"discarded": None},
        "prediction": {"path": "predicted.json", "sha256": prediction_sha, "mtime_ns": 1},
        "measured_at_utc": {"m": "2026-11-05T00:00:01+00:00"},
        "models": {"m": {"tg128": {
            "avg_ns": 10, "samples_ns": [10, 11], "n_prompt": 0, "n_gen": 1, "n_depth": 128,
            "llama_bench": {"n_threads": 2, "cpu_mask": "0x14", "n_depth": 128}}}},
    }
    value.update(overrides)
    return value


class ResultCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.prediction = self.root / "predicted.json"
        self.prediction.write_text('{"models": {}}\n', encoding="utf-8")
        self.sha = hashlib.sha256(self.prediction.read_bytes()).hexdigest()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, **overrides):
        (self.root / "measured.json").write_text(
            json.dumps(document(self.sha, **overrides)), encoding="utf-8")

    def test_a_complete_result_is_accepted_and_described(self):
        self._write()
        manifest = require_result(self.root)
        self.assertTrue(manifest["ok"])
        self.assertEqual(manifest["prediction"]["sha256"], self.sha)
        self.assertEqual(manifest["environment"]["threads"], 2)
        self.assertEqual(manifest["models"]["m"]["tg128"]["samples"], 2)

    def test_rows_from_another_prediction_are_refused(self):
        self._write(prediction={"path": "x", "sha256": "b" * 64, "mtime_ns": 1})
        manifest = inspect_result(self.root)
        self.assertFalse(manifest["ok"])
        self.assertTrue(any("pre-registration" in problem for problem in manifest["problems"]))

    def test_a_header_that_forgets_its_protocol_is_refused(self):
        self._write(warmup=None, decode_grouping=None)
        manifest = inspect_result(self.root)
        # The keys are present but null: the value has to be there, not just the
        # name, because "which warm-up was dropped" is part of the number.
        self.assertFalse(manifest["ok"])
        self.assertTrue(any("does not declare" in problem for problem in manifest["problems"]))

    def test_another_binary_is_another_implementation(self):
        self._write()
        manifest = inspect_result(self.root, expect_llama_bench_sha256="c" * 64)
        self.assertFalse(manifest["ok"])
        self.assertTrue(any("another binary" in problem for problem in manifest["problems"]))

    def test_a_row_that_ran_on_other_cpus_is_refused(self):
        bad = document(self.sha)
        bad["models"]["m"]["tg128"]["llama_bench"]["cpu_mask"] = "0x3"
        (self.root / "measured.json").write_text(json.dumps(bad), encoding="utf-8")
        manifest = inspect_result(self.root)
        self.assertFalse(manifest["ok"])
        self.assertTrue(any("cpu_mask" in problem for problem in manifest["problems"]))

    def test_the_manifest_says_where_the_weights_were_read(self):
        # A prediction may be measured on several devices; the manifest keeps
        # the directory this run resolved it to, so "which file was timed" is
        # answerable after the fact.
        self._write(models_directory="/data/models")
        manifest = require_result(self.root)
        self.assertEqual(manifest["environment"]["models_directory"], "/data/models")

    def test_a_directory_without_a_measurement_is_not_a_result(self):
        with self.assertRaisesRegex(ValueError, "not a result directory"):
            inspect_result(self.root)


if __name__ == "__main__":
    unittest.main()
