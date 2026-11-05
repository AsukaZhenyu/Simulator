"""End-to-end prediction/measurement wiring; no hardware is executed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from composing.compare import compare
from composing.predict import block_cost, boundary_cost, predict
from composing.spec import SHAPES
from run_dense_endtoend import (WARMUP_GROUPED, bench_command, parse_bench,
                                parse_bench_rows)

BOUNDARY_NS = 10


def word_entry(kind, phase, tokens, *, median=BOUNDARY_NS, vocab=7, embd="q4_k",
               head="q6_k"):
    return {"kind": kind, "shape": f"{phase}/B1/q{tokens}/hist0",
            "sample_summary": {"median_ns": median, "cv": 0.01},
            "physical_signature": {"config": {"vocab": vocab, "token_embd_dtype": embd,
                                              "lm_head_dtype": head}}}


def full_report(*, median=1000, vocab=7):
    """A table complete enough for a boundary-complete prediction."""
    entries = []
    for phase, tokens, history in SHAPES.values():
        shape = f"{phase}/B1/q{tokens}/hist{history}"
        for kind in ("projection", "attention", "dense_ffn"):
            entries.append({"kind": kind, "shape": shape,
                            "sample_summary": {"median_ns": median, "cv": 0.01}})
    # The embedding is measured per token count; the head at the single position
    # llama.cpp evaluates for a batch.
    entries.append(word_entry("embedding", "prefill", 256, vocab=vocab))
    entries.append(word_entry("embedding", "decode", 1, vocab=vocab))
    entries.append(word_entry("head", "decode", 1, vocab=vocab))
    return {"validation_only": False, "entries": entries}


def table_with(root: Path, document: dict) -> Path:
    table = root / "table"
    table.mkdir()
    (table / "export_report.json").write_text(json.dumps(document), encoding="utf-8")
    return table


def report(*, validation_only=False, stages=("projection", "attention", "dense_ffn"),
           block=True):
    entries = []
    for index, kind in enumerate(stages):
        entries.append({"kind": kind, "shape": "decode/B1/q1/hist128",
                        "sample_summary": {"median_ns": 100 * (index + 1), "cv": 0.02}})
    if block:
        entries.append({"kind": "block", "shape": "decode/B1/q1/hist128",
                        "sample_summary": {"median_ns": 590, "cv": 0.02}})
    return {"validation_only": validation_only, "entries": entries}


class CostTests(unittest.TestCase):
    def test_only_the_three_stages_are_summed(self):
        cost = block_cost(report(), "decode", 1, 128)
        self.assertEqual(cost["block_sum_ns"], 600)
        self.assertNotIn("block", cost)

    def test_a_missing_stage_is_never_filled(self):
        with self.assertRaisesRegex(ValueError, "no interpolation"):
            block_cost(report(stages=("projection", "attention")), "decode", 1, 128)

    def test_a_validation_only_table_cannot_back_a_prediction(self):
        with self.assertRaisesRegex(ValueError, "validation_only"):
            block_cost(report(validation_only=True), "decode", 1, 128)


def row(**overrides):
    base = {"n_prompt": 256, "n_gen": 0, "n_depth": 0, "avg_ns": 10, "samples_ns": [10],
            "n_threads": 2, "cpu_mask": "0x14", "cpu_strict": True, "no_op_offload": 1,
            "n_gpu_layers": 0, "flash_attn": False, "type_k": "f16", "type_v": "f16"}
    base.update(overrides)
    return base


class CommandTests(unittest.TestCase):
    def test_prefill_and_decode_each_ask_for_one_row_on_the_declared_cpus(self):
        prefill = bench_command(Path("bench.exe"), Path("m.gguf"), "pp256", 2, [2, 4], 5)
        self.assertEqual(prefill[prefill.index("-p") + 1], "256")
        self.assertEqual(prefill[prefill.index("-n") + 1], "0")
        decode = bench_command(Path("bench.exe"), Path("m.gguf"), "tg512", 2, [2, 4], 3)
        self.assertEqual(decode[decode.index("-p") + 1], "0")
        self.assertEqual(decode[decode.index("-n") + 1], "1")
        self.assertEqual(decode[decode.index("-d") + 1], "512")
        for command in (prefill, decode):
            self.assertEqual(command[command.index("-C") + 1], "0x14")
            self.assertEqual(command[command.index("-nopo") + 1], "1")
            self.assertEqual(command[command.index("--cpu-strict") + 1], "1")

    def test_a_generation_row_at_the_wrong_depth_is_not_interchangeable(self):
        shallow = row(n_prompt=0, n_gen=1, n_depth=0, avg_ns=20)
        deep = row(n_prompt=0, n_gen=1, n_depth=512, avg_ns=30)
        self.assertEqual(parse_bench(json.dumps([deep]), "tg512", 2, [2, 4])["avg_ns"], 30)
        # The first attempt accepted exactly this: a depth-0 row standing in for 512.
        with self.assertRaisesRegex(ValueError, "n_prompt, n_gen, n_depth"):
            parse_bench(json.dumps([shallow]), "tg512", 2, [2, 4])
        with self.assertRaisesRegex(ValueError, "n_prompt, n_gen, n_depth"):
            parse_bench(json.dumps([shallow]), "pp256", 2, [2, 4])

    def test_a_row_from_another_protocol_is_rejected(self):
        for field, value in (("no_op_offload", 0), ("cpu_mask", "0x0"), ("cpu_strict", False),
                             ("n_threads", 16), ("n_gpu_layers", 99),
                             ("flash_attn", True), ("type_k", "q8_0"), ("type_v", "q8_0")):
            with self.assertRaisesRegex(ValueError, "not the requested"):
                parse_bench(json.dumps([row(**{field: value})]), "pp256", 2, [2, 4])

    def test_the_attention_implementation_and_cache_type_are_pinned_in_the_command(self):
        # Both change the wall clock without changing the protocol name: a fused
        # kernel moved the decode slope by 1.76x on this machine, and a
        # quantized cache reads a different byte count than the table priced.
        command = bench_command(Path("bench.exe"), Path("m.gguf"), "tg512", 2, [2, 4], 5)
        for flag, value in (("-fa", "0"), ("-ctk", "f16"), ("-ctv", "f16"),
                            ("-mmp", "1"), ("-b", "2048"), ("-ub", "512"),
                            ("--poll", "50")):
            self.assertEqual(command[command.index(flag) + 1], value, flag)


class BoundaryTests(unittest.TestCase):
    def test_the_boundary_is_priced_from_its_own_measured_entry(self):
        actual = boundary_cost(full_report(), "head", "decode", 1, 7)
        self.assertEqual(actual["median_ns"], BOUNDARY_NS)
        self.assertEqual(actual["token_embd_dtype"], "q4_k")
        self.assertEqual(actual["lm_head_dtype"], "q6_k")

    def test_a_table_measured_at_another_vocabulary_is_not_interchangeable(self):
        with self.assertRaisesRegex(ValueError, "not the declared"):
            boundary_cost(full_report(), "head", "decode", 1, 151936)

    def test_a_missing_or_duplicated_boundary_is_never_averaged(self):
        with self.assertRaisesRegex(ValueError, "found 0"):
            boundary_cost(full_report(), "head", "prefill", 256, 7)
        document = full_report()
        document["entries"].append(word_entry("head", "decode", 1))
        with self.assertRaisesRegex(ValueError, "found 2"):
            boundary_cost(document, "head", "decode", 1, 7)


class OrderTests(unittest.TestCase):
    def test_the_prediction_prices_a_complete_token_step(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, full_report())
            models = {"m": {"filename": "m.gguf",
                            "geometry": {"block_count": 4, "vocab_size": 7, "ffn": "dense"}}}
            document = predict({"m": table}, models, root)
            self.assertEqual(set(document["models"]["m"]["shapes"]), set(SHAPES))
            # 3 stages x 1000 ns x 4 layers, plus both boundary stages.
            self.assertEqual(document["models"]["m"]["shapes"]["pp256"]["total_ns"],
                             3000 * 4 + 2 * BOUNDARY_NS)
            self.assertEqual(document["models"]["m"]["shapes"]["pp256"]["embedding"]["median_ns"],
                             BOUNDARY_NS)
            self.assertTrue({"sampling", "graph_construction"} <= set(document["excluded_terms"]))
            self.assertNotIn("lm_head", document["excluded_terms"])

    def test_the_table_regime_is_recorded_and_cannot_be_mixed(self):
        models = {"m": {"filename": "m.gguf",
                        "geometry": {"block_count": 4, "vocab_size": 7, "ffn": "dense"}}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cold = full_report()
            for entry in cold["entries"]:
                entry.setdefault("physical_signature", {})["residency_regime"] = \
                    "streamed_cold_cache_before_each_sample"
            prediction = predict({"m": table_with(root, cold)}, models, root)
            self.assertEqual(prediction["models"]["m"]["residency_regime"],
                             "streamed_cold_cache_before_each_sample")
            # A resident table carries no such key and must be recorded as resident.
            (root / "resident").mkdir()
            resident = predict({"m": table_with(root / "resident", full_report())},
                               models, root)
            self.assertEqual(resident["models"]["m"]["residency_regime"],
                             "resident_warm_cache_shared_across_samples")
            # One table that mixes the two protocols cannot be predicted at all:
            # half its rows would answer a question the other half did not ask.
            mixed = full_report()
            mixed["entries"][0]["physical_signature"] = {
                "residency_regime": "streamed_cold_cache_before_each_sample"}
            (root / "mixed").mkdir()
            with self.assertRaisesRegex(ValueError, "mixes residency regimes"):
                predict({"m": table_with(root / "mixed", mixed)}, models, root)

    def test_a_model_without_a_declared_vocabulary_cannot_be_predicted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, full_report())
            with self.assertRaisesRegex(ValueError, "vocab_size"):
                predict({"m": table}, {"m": {"filename": "m.gguf",
                                             "geometry": {"block_count": 4, "ffn": "dense"}}}, root)

    def test_compare_reports_signed_error_without_editing_the_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, full_report(median=1000))
            prediction = predict({"m": table},
                                 {"m": {"filename": "m.gguf",
                                        "geometry": {"block_count": 1, "vocab_size": 7,
                                                     "ffn": "dense"}}}, root)
            predicted = prediction["models"]["m"]["shapes"]["pp256"]["total_ns"]
            measured = {"models": {"m": {"pp256": {"avg_ns": predicted * 1.1,
                                                   "samples_ns": [predicted * 1.1]}}}}
            result = compare(prediction, measured)
            self.assertAlmostEqual(result["models"]["m"]["pp256"]["signed_relative_error"], 0.1)
            self.assertFalse(result["calibration_claimed"])
            self.assertEqual(result["models"]["m"]["pp256"]["predicted_boundary_ns"],
                             2 * BOUNDARY_NS)
            self.assertAlmostEqual(result["models"]["m"]["pp256"]["boundary_fraction"],
                                   2 * BOUNDARY_NS / predicted)


class SparseTableTests(unittest.TestCase):
    """A sparse block is predicted exactly like a dense one, with its own stage."""

    def sparse_report(self, *, contract="ggml_cpu_moe_decoder_block_v1"):
        document = full_report()
        for entry in document["entries"]:
            if entry["kind"] == "dense_ffn":
                entry["kind"] = "moe_ffn"
        document["contract"] = contract
        document["decoder_stages"] = ["projection", "attention", "moe_ffn", "block"]
        return document

    def test_the_third_stage_comes_from_the_table_not_from_a_constant(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, self.sparse_report())
            models = {"m": {"filename": "m.gguf",
                            "geometry": {"block_count": 2, "vocab_size": 7, "ffn": "moe"}}}
            document = predict({"m": table}, models, root)
            entry = document["models"]["m"]
            self.assertEqual(entry["decoder_stages"],
                             ["projection", "attention", "moe_ffn", "block"])
            self.assertEqual(entry["shapes"]["pp256"]["total_ns"], 3000 * 2 + 2 * BOUNDARY_NS)
            self.assertNotIn("dense_ffn", entry["shapes"]["pp256"])

    def test_a_dense_table_cannot_price_a_sparse_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, full_report())
            models = {"m": {"filename": "m.gguf",
                            "geometry": {"block_count": 2, "vocab_size": 7, "ffn": "moe"}}}
            with self.assertRaisesRegex(ValueError, "declared feed-forward"):
                predict({"m": table}, models, root)

    def test_a_sparse_table_cannot_price_a_dense_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, self.sparse_report())
            models = {"m": {"filename": "m.gguf",
                            "geometry": {"block_count": 2, "vocab_size": 7, "ffn": "dense"}}}
            with self.assertRaisesRegex(ValueError, "declared feed-forward"):
                predict({"m": table}, models, root)

    def test_an_unknown_feed_forward_kind_is_refused_rather_than_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, full_report())
            with self.assertRaisesRegex(ValueError, "feed-forward kind"):
                predict({"m": table}, {"m": {"filename": "m.gguf",
                                             "geometry": {"block_count": 1, "vocab_size": 7}}}, root)

    def test_a_decode_price_that_falls_with_history_is_recorded_not_corrected(self):
        # The history-independent stages are priced per shape, so noise between
        # two of those rows can make the deeper shape look cheaper.  Only
        # attention depends on history and it only grows, so the prediction is
        # internally inconsistent -- but it must still be the one that was
        # pre-registered, so the inconsistency is recorded and left alone.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            document = full_report()
            for entry in document["entries"]:
                if entry["kind"] == "dense_ffn" and entry["shape"] == "decode/B1/q1/hist1024":
                    entry["sample_summary"]["median_ns"] = 1
            table = table_with(root, document)
            models = {"m": {"filename": "m.gguf",
                            "geometry": {"block_count": 1, "vocab_size": 7, "ffn": "dense"}}}
            prediction = predict({"m": table}, models, root)
            conflicts = prediction["declared_boundary_conflicts"]
            self.assertTrue(any("falls from tg512" in line and "tg1024" in line
                                for line in conflicts), conflicts)
            # Not silently "fixed": the priced total still contains the cheap row.
            self.assertEqual(prediction["models"]["m"]["shapes"]["tg1024"]["total_ns"],
                             1000 + 1000 + 1 + 2 * BOUNDARY_NS)


class KvDtypeTests(unittest.TestCase):
    """The cache the table priced and the cache the run read are one quantity."""

    MODELS = {"m": {"filename": "m.gguf",
                    "geometry": {"block_count": 1, "vocab_size": 7, "ffn": "dense"}}}

    def with_kv(self, document, dtype):
        for entry in document["entries"]:
            if entry["kind"] == "attention":
                entry.setdefault("physical_signature", {})["kv_dtype"] = dtype
        return document

    def prediction(self, folder, document):
        root = Path(folder)
        return predict({"m": table_with(root, document)}, self.MODELS, root)

    def test_the_declared_cache_type_is_carried_into_the_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            document = self.prediction(folder, self.with_kv(full_report(), "f16"))
            self.assertEqual(document["models"]["m"]["kv_dtype"], "f16")

    def test_a_table_pricing_two_cache_types_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            mixed = self.with_kv(full_report(), "f32")
            attention = [entry for entry in mixed["entries"] if entry["kind"] == "attention"]
            attention[0]["physical_signature"]["kv_dtype"] = "f16"
            with self.assertRaisesRegex(ValueError, "several KV dtypes"):
                self.prediction(folder, mixed)

    def test_a_run_with_another_cache_type_is_a_different_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            document = self.prediction(folder, self.with_kv(full_report(), "f16"))
            measured = {"models": {"m": {"pp256": {
                "avg_ns": 10, "samples_ns": [10], "llama_bench": {"type_k": "q8_0"}}}}}
            with self.assertRaisesRegex(ValueError, "different byte counts"):
                compare(document, measured)

    def test_a_run_with_the_declared_cache_type_compares(self):
        with tempfile.TemporaryDirectory() as folder:
            document = self.prediction(folder, self.with_kv(full_report(), "f16"))
            predicted = document["models"]["m"]["shapes"]["pp256"]["total_ns"]
            measured = {"models": {"m": {"pp256": {
                "avg_ns": predicted, "samples_ns": [predicted],
                "llama_bench": {"type_k": "f16"}}}}}
            result = compare(document, measured)
            self.assertAlmostEqual(result["models"]["m"]["pp256"]["signed_relative_error"], 0.0)


class GroupedDecodeTests(unittest.TestCase):
    """Several depths in one process: one machine state for all slope points."""

    DEPTHS = ("tg128", "tg512", "tg1024")

    def test_one_invocation_can_ask_for_several_depths(self):
        command = bench_command(Path("bench.exe"), Path("m.gguf"), self.DEPTHS, 2, [2, 4], 5)
        self.assertEqual(command[command.index("-d") + 1], "128,512,1024")
        self.assertEqual(command[command.index("-p") + 1], "0")
        self.assertEqual(command[command.index("-n") + 1], "1")

    def test_shapes_with_different_work_cannot_share_one_invocation(self):
        with self.assertRaisesRegex(ValueError, "several phases or token counts"):
            bench_command(Path("bench.exe"), Path("m.gguf"), ("pp256", "tg128"), 2, [2, 4], 5)
        with self.assertRaisesRegex(ValueError, "asked for twice"):
            bench_command(Path("bench.exe"), Path("m.gguf"), ("tg128", "tg128"), 2, [2, 4], 5)

    def grouped_stdout(self, depths=(128, 512, 1024)):
        return json.dumps([row(n_prompt=0, n_gen=1, n_depth=depth, avg_ns=depth)
                           for depth in depths])

    def test_every_declared_depth_is_read_back_by_its_own_triple(self):
        parsed = parse_bench_rows(self.grouped_stdout(), self.DEPTHS, 2, [2, 4], 1)
        self.assertEqual(sorted(parsed), sorted(self.DEPTHS))
        self.assertEqual(parsed["tg1024"]["avg_ns"], 1024)
        # A departure that silently dropped one depth is not a partial success.
        with self.assertRaisesRegex(ValueError, "found 0"):
            parse_bench_rows(self.grouped_stdout((128, 512)), self.DEPTHS, 2, [2, 4], 1)

    def test_one_wrong_row_rejects_the_whole_invocation(self):
        stdout = json.dumps([row(n_prompt=0, n_gen=1, n_depth=128),
                             row(n_prompt=0, n_gen=1, n_depth=512, type_k="q8_0"),
                             row(n_prompt=0, n_gen=1, n_depth=1024)])
        with self.assertRaisesRegex(ValueError, "not the requested"):
            parse_bench_rows(stdout, self.DEPTHS, 2, [2, 4], 1)

    def test_a_row_with_another_sample_count_is_not_the_run_the_command_describes(self):
        stdout = json.dumps([row(n_prompt=0, n_gen=1, n_depth=128,
                                 avg_ns=3, samples_ns=[1, 2, 3])])
        self.assertEqual(parse_bench_rows(stdout, "tg128", 2, [2, 4], 3)["tg128"]["avg_ns"], 3)
        with self.assertRaisesRegex(ValueError, "asked for 5"):
            parse_bench_rows(stdout, "tg128", 2, [2, 4], 5)

    def test_the_warm_up_sample_is_dropped_only_where_the_protocol_names_it(self):
        # The first shape timed in a grouped process pays its first decode; that
        # sample is on the record but must not move the mean it is dropped from.
        stdout = json.dumps([row(n_prompt=0, n_gen=1, n_depth=128, avg_ns=99,
                                 samples_ns=[62_000_000, 48_000_000, 49_000_000]),
                             row(n_prompt=0, n_gen=1, n_depth=512, avg_ns=51,
                                 samples_ns=[51_000_000, 52_000_000, 50_000_000])])
        parsed = parse_bench_rows(stdout, ("tg128", "tg512"), 2, [2, 4], 3,
                                 discard_warmup_on="tg128")
        self.assertEqual(parsed["tg128"]["discarded_warmup_ns"], 62_000_000)
        self.assertEqual(parsed["tg128"]["reported_avg_ns"], 99)
        self.assertEqual(parsed["tg128"]["samples_ns"], [48_000_000, 49_000_000])
        self.assertEqual(parsed["tg128"]["avg_ns"], 48_500_000)
        # The later shape keeps every sample: its first one was inside the spread.
        self.assertNotIn("discarded_warmup_ns", parsed["tg512"])
        self.assertEqual(parsed["tg512"]["avg_ns"], 51)

    def test_discarding_must_leave_something_measured(self):
        stdout = json.dumps([row(n_prompt=0, n_gen=1, n_depth=128, samples_ns=[7])])
        with self.assertRaisesRegex(ValueError, "leaves nothing measured"):
            parse_bench_rows(stdout, "tg128", 2, [2, 4], 1, discard_warmup_on="tg128")

    def fake_group_run(self, command, **_):
        def value(flag):
            return command[command.index(flag) + 1]
        repetitions = int(value("-r"))
        rows = [row(n_prompt=int(value("-p")), n_gen=int(value("-n")), n_depth=int(depth),
                    avg_ns=100 + int(depth),
                    samples_ns=[100 + int(depth)] * repetitions)
                for depth in value("-d").split(",")]
        return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")

    def test_grouping_is_one_process_for_all_depths_and_is_recorded(self):
        from unittest import mock
        from run_dense_endtoend import measure
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = table_with(root, full_report(median=1000))
            binary = root / "llama-bench.exe"
            binary.write_bytes(b"stub")
            models = {"m": {"filename": "m.gguf",
                            "geometry": {"block_count": 1, "vocab_size": 7, "ffn": "dense"}}}
            prediction = predict({"m": table}, models, root)
            (root / "m.gguf").write_bytes(b"stub weights")
            (root / "predicted.json").write_text(json.dumps(prediction), encoding="utf-8")
            with mock.patch("subprocess.run", side_effect=self.fake_group_run) as run:
                measured = measure(root, binary, {"pp256": 1, **{name: 1 for name in self.DEPTHS}},
                                   2, [2, 4], 60, None, group_decode=True)
        # One prefill invocation plus one invocation for all three depths.
        self.assertEqual(run.call_count, 2)
        self.assertEqual(measured["decode_grouping"], "one_process_per_model_decode")
        self.assertEqual(set(measured["models"]["m"]), {"pp256", *self.DEPTHS})
        # The grouped command asks for one sample more than it keeps, and only
        # the first depth timed gives one back as warm-up.
        grouped = run.call_args_list[1].args[0]
        self.assertEqual(grouped[grouped.index("-r") + 1], "2")
        self.assertEqual(measured["warmup"]["discarded"], WARMUP_GROUPED)
        first = measured["models"]["m"]["tg128"]
        self.assertEqual(first["samples_ns"], [228])
        self.assertEqual(first["discarded_warmup_ns"], 228)
        self.assertEqual(len(measured["models"]["m"]["tg512"]["samples_ns"]), 2)

    def test_grouping_needs_one_repetition_count(self):
        from run_dense_endtoend import measure
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "predicted.json").write_text(json.dumps({"models": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one repetition count"):
                measure(root, root / "b.exe", {"tg128": 1, "tg512": 2}, 2, [2, 4], 60, None,
                        group_decode=True)


class ModelDirectoryTests(unittest.TestCase):
    """A prediction is one document with one sha256; the weights are wherever
    each machine keeps them."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        table = table_with(self.root, full_report(median=1000))
        self.binary = self.root / "llama-bench.exe"
        self.binary.write_bytes(b"stub")
        self.elsewhere = self.root / "elsewhere"
        self.elsewhere.mkdir()
        models = {"m": {"filename": "m.gguf",
                        "geometry": {"block_count": 1, "vocab_size": 7, "ffn": "dense"}}}
        # Predicted on this machine, with the weights in a directory of its own.
        self.recorded_dir = self.root / "recorded"
        self.recorded_dir.mkdir()
        (self.recorded_dir / "m.gguf").write_bytes(b"stub weights")
        self.document = predict({"m": table}, models, self.recorded_dir)
        self.path = self.root / "predicted.json"
        self.path.write_text(json.dumps(self.document), encoding="utf-8")

    def tearDown(self):
        self.folder.cleanup()

    def fake_run(self, command, **_):
        def value(flag):
            return int(command[command.index(flag) + 1])
        row = {"n_prompt": value("-p"), "n_gen": value("-n"), "n_depth": value("-d"),
               "avg_ns": 100, "samples_ns": [100], "n_threads": 2, "cpu_mask": "0x14",
               "cpu_strict": True, "no_op_offload": 1, "n_gpu_layers": 0,
               "flash_attn": False, "type_k": "f16", "type_v": "f16"}
        return SimpleNamespace(returncode=0, stdout=json.dumps([row]), stderr="")

    def test_the_same_document_is_measured_where_this_machine_keeps_the_weights(self):
        from unittest import mock
        from run_dense_endtoend import measure
        (self.elsewhere / "m.gguf").write_bytes(b"stub weights")
        with mock.patch("subprocess.run", side_effect=self.fake_run) as run:
            measured = measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60,
                               models_dir=self.elsewhere)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-m") + 1],
                         str(self.elsewhere / "m.gguf"))
        self.assertEqual(measured["models_directory"], str(self.elsewhere))
        # The prediction was read, never rewritten: the registration stands.
        self.assertEqual(measured["prediction"]["sha256"],
                         hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), self.document)

    def test_a_recorded_path_that_is_absent_names_the_flag_instead_of_llama_bench(self):
        from run_dense_endtoend import measure
        (self.recorded_dir / "m.gguf").unlink()
        with self.assertRaisesRegex(ValueError, "not on this machine.*--models-dir"):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60)

    def test_a_models_dir_without_the_named_file_is_refused(self):
        from run_dense_endtoend import measure
        with self.assertRaisesRegex(ValueError, "holds no m.gguf"):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60,
                    models_dir=self.elsewhere)


class PartialCampaignTests(unittest.TestCase):
    """One campaign may span several processes; a second process must not
    silently delete what the first one measured, and the file has to say which
    row came from which process (measured latency is not one machine state)."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        table = table_with(self.root, full_report(median=1000))
        self.binary = self.root / "llama-bench.exe"
        self.binary.write_bytes(b"stub")
        models = {name: {"filename": f"{name}.gguf",
                         "geometry": {"block_count": 1, "vocab_size": 7, "ffn": "dense"}}
                  for name in ("a", "b")}
        for name in models:
            (self.root / f"{name}.gguf").write_bytes(b"stub weights")
        document = predict({name: table for name in models}, models, self.root)
        (self.root / "predicted.json").write_text(json.dumps(document), encoding="utf-8")

    def tearDown(self):
        self.folder.cleanup()

    def fake_run(self, command, **_):
        def value(flag):
            return int(command[command.index(flag) + 1])
        row = {"n_prompt": value("-p"), "n_gen": value("-n"), "n_depth": value("-d"),
               "avg_ns": 100, "samples_ns": [100], "n_threads": 2, "cpu_mask": "0x14",
               "cpu_strict": True, "no_op_offload": 1, "n_gpu_layers": 0,
               "flash_attn": False, "type_k": "f16", "type_v": "f16"}
        return SimpleNamespace(returncode=0, stdout=json.dumps([row]), stderr="")

    def test_a_second_process_keeps_the_first_models_rows(self):
        from unittest import mock
        from run_dense_endtoend import measure
        with mock.patch("subprocess.run", self.fake_run):
            first = measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["a"])
            second = measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["b"])
        self.assertEqual(sorted(first["models"]), ["a"])
        self.assertEqual(sorted(second["models"]), ["a", "b"])
        self.assertEqual(sorted(second["measured_at_utc"]), ["a", "b"])
        # The per-shape path keeps its historical -r meaning: every archived
        # campaign asked for exactly the samples it reports, none were dropped.
        self.assertIsNone(second["warmup"]["discarded"])

    def test_a_partial_run_cannot_be_mixed_with_another_prediction(self):
        from run_dense_endtoend import measure
        (self.root / "measured.json").write_text(json.dumps(
            {"prediction": {"sha256": "0" * 64}, "models": {"a": {}}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "another prediction"):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["b"])

    def test_an_undeclared_model_id_is_refused_before_anything_runs(self):
        from run_dense_endtoend import measure
        with self.assertRaisesRegex(ValueError, "declares no model named"):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["nope"])

    def _first_run_then_edit(self, field, value):
        from unittest import mock
        from run_dense_endtoend import measure
        with mock.patch("subprocess.run", self.fake_run):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["a"])
        path = self.root / "measured.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored[field] = value
        path.write_text(json.dumps(stored), encoding="utf-8")

    def test_rows_from_another_thread_placement_cannot_share_the_file(self):
        from run_dense_endtoend import measure
        self._first_run_then_edit("threads", 4)
        with self.assertRaisesRegex(ValueError, "threads=4"):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["b"])

    def test_rows_from_another_affinity_cannot_share_the_file(self):
        from run_dense_endtoend import measure
        self._first_run_then_edit("cpu_mask", [0, 1])
        with self.assertRaisesRegex(ValueError, "cpu_mask=\\[0, 1\\]"):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["b"])

    def test_rows_from_another_bench_binary_cannot_share_the_file(self):
        from unittest import mock
        from run_dense_endtoend import measure
        with mock.patch("subprocess.run", self.fake_run):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["a"])
        self.binary.write_bytes(b"another stub")     # same path, different binary
        with self.assertRaisesRegex(ValueError, "llama_bench_sha256"):
            measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, ["b"])

    def test_a_full_re_run_may_change_the_protocol_because_no_row_is_kept(self):
        from unittest import mock
        from run_dense_endtoend import measure
        self._first_run_then_edit("threads", 4)
        with mock.patch("subprocess.run", self.fake_run):
            redo = measure(self.root, self.binary, {"pp256": 1}, 2, [2, 4], 60, None)
        self.assertEqual(redo["threads"], 2)
        self.assertEqual(sorted(redo["models"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
