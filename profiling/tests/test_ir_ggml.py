"""The IR->ggml apparatus, checked against its own inputs rather than a campaign.

Three things have to hold for a clone to reproduce anything: the corpus pack must
be the corpus, the instantiator must turn a policy into the case text its manifest
describes, and the frozen prices must be the ones the provenance record names.  Each
of those is answered here from files in the repository -- no runner, no GPU, no
measurement, so this suite passes on a machine that has never built ggml.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from profiling.ir_ggml import corpus_es
from profiling.ir_ggml import instantiate as inst
from profiling.ir_ggml import prices_provenance

PACK = corpus_es.PACK
HERE = PACK.parent


class CorpusPackTests(unittest.TestCase):
    def test_the_pack_is_eight_hundred_declared_policies(self):
        blob = corpus_es.load(PACK)
        self.assertEqual(blob["count"], 800)
        self.assertEqual(len(blob["cases"]), 800)

    def test_every_case_still_hashes_to_what_was_packed(self):
        blob = corpus_es.load(PACK)
        bad = [name for name, entry in blob["cases"].items()
               if corpus_es._sha(corpus_es._canonical(corpus_es.nodes_of(entry)))
               != entry["sha256"]]
        self.assertEqual(bad, [])

    def test_node_order_survives_the_round_trip(self):
        # The order is part of the case: instantiate numbers nodes in file order, so a
        # re-sorted pack would emit the same graph under different indices.
        blob = corpus_es.load(PACK)
        name = "dense-compute-000-spare"
        nodes = corpus_es.nodes_of(blob["cases"][name])
        self.assertEqual(next(iter(nodes)), "g:request0:request_arrival")

    def test_unpack_writes_one_directory_per_case(self):
        blob = corpus_es.load(PACK)
        with tempfile.TemporaryDirectory(prefix="ir-corpus-") as folder:
            corpus_es.unpack(PACK, Path(folder))
            written = {p.name for p in Path(folder).iterdir() if (p / "nodes.json").exists()}
        self.assertEqual(written, set(blob["cases"]))


class InstantiateTests(unittest.TestCase):
    def build(self, names, model="qwen3-0.6b"):
        blob = corpus_es.load(PACK)
        folder = tempfile.TemporaryDirectory(prefix="ir-inst-")
        self.addCleanup(folder.cleanup)
        cases = Path(folder.name) / "in"
        for name in names:
            target = cases / name
            target.mkdir(parents=True)
            (target / "nodes.json").write_text(
                corpus_es._canonical(corpus_es.nodes_of(blob["cases"][name])),
                encoding="utf-8", newline="\n")
        out = Path(folder.name) / "out"
        code = inst.main(["--model", model, "--tokens", "64",
                          "--cases", str(cases), "--out", str(out)])
        self.assertEqual(code, 0)
        return out / model

    def test_a_dense_policy_becomes_a_case_with_the_models_geometry(self):
        out = self.build(["dense-compute-000-spare"])
        header = {}
        for line in (out / "dense-compute-000-spare.txt").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            header[key] = value
        self.assertEqual(header["model"], "qwen3-0.6b")
        self.assertEqual(header["cpu_mask"], "2,4")
        self.assertEqual(header["geometry_hidden"], "1024")
        # The toy substrate's 36 ns arrival is replaced by the model's own geometry.
        self.assertNotEqual(header["tokens"], "0")

    def test_a_moe_policy_for_a_model_with_a_shared_expert_declares_it(self):
        # The frozen `moe_ffn` price covers the shared expert, so the executor has to be
        # told the model has one; a model without one must not grow the line.
        out = self.build(["moe-compute-000-demand"], model="qwen1.5-moe-a2.7b")
        header = {}
        for line in (out / "moe-compute-000-demand.txt").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            header[key] = value
        self.assertEqual(header["shared"], "5632")
        self.assertEqual(header["dtype_down_shexp"], "q6_k")
        other = self.build(["moe-compute-000-demand"], model="granite-3.0-3b-a800m")
        lines = (other / "moe-compute-000-demand.txt").read_text(encoding="utf-8").splitlines()
        self.assertFalse([line for line in lines if line.startswith("shared ")])

    def test_the_manifest_names_every_case_it_wrote(self):
        out = self.build(["dense-compute-000-spare", "attention-compute-000-demand"])
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["cases"]),
                         {"dense-compute-000-spare", "attention-compute-000-demand"})
        self.assertEqual(manifest["cpu_mask"], "2,4")
        written = {p.stem for p in out.glob("*.txt")}
        self.assertEqual(written, set(manifest["cases"]))

    def test_a_mixed_corpus_keeps_its_moe_cases_for_a_dense_model(self):
        # A dense model cannot execute router/expert/combine, but the instantiated case
        # set is not the place to drop them: `run.py --families` keeps the half the model
        # can run and counts the rest as not applicable.
        out = self.build(["dense-compute-000-spare", "moe-compute-000-demand"])
        self.assertTrue((out / "moe-compute-000-demand.txt").exists())

    def test_an_all_moe_corpus_against_a_dense_model_is_refused(self):
        with self.assertRaises(SystemExit):
            self.build(["moe-compute-000-demand"], model="qwen3-0.6b")


def write_gguf(path: Path, tensors: dict[str, tuple[list[int], int]]) -> None:
    """The smallest file `read_gguf` and `instantiate.gguf_dtypes` accept.

    Only the metadata a case generator reads is written: names, shapes, types.  The
    tensor data blob is empty, which is enough because nothing in these tests loads
    bytes -- `verify_weights` is tested against the real files, not a stub.
    """
    import struct

    def text(value: str) -> bytes:
        raw = value.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for name, (dims, kind) in sorted(tensors.items()):
        body += text(name) + struct.pack("<I", len(dims))
        body += b"".join(struct.pack("<Q", d) for d in dims)
        body += struct.pack("<IQ", kind, 0)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1)
    header += text("general.architecture") + struct.pack("<I", 8) + text("qwen3")
    blob = header + body
    path.write_bytes(blob + b"\0" * (-len(blob) % 32))


class PerLayerWeightTests(unittest.TestCase):
    """A case that names its layers, and the file that decides their dtypes."""

    def instantiate(self, flags, spec="qwen3-0.6b"):
        blob = corpus_es.load(PACK)
        folder = tempfile.TemporaryDirectory(prefix="ir-layers-")
        self.addCleanup(folder.cleanup)
        cases = Path(folder.name) / "in"
        for name in ("dense-compute-000-spare", "dense-compute-001-demand"):
            target = cases / name
            target.mkdir(parents=True)
            (target / "nodes.json").write_text(
                corpus_es._canonical(corpus_es.nodes_of(blob["cases"][name])),
                encoding="utf-8", newline="\n")
        out = Path(folder.name) / "out"
        code = inst.main(["--model", spec, "--tokens", "64", "--cases", str(cases),
                          "--out", str(out)] + flags)
        self.assertEqual(code, 0)
        return Path(folder.name), out / spec

    def test_the_flag_names_every_compute_node_and_no_arrival(self):
        _, written = self.instantiate(["--per-layer-weights"])
        text = (written / "dense-compute-000-spare.txt").read_text(encoding="utf-8")
        nodes = [line.split() for line in text.splitlines() if line.startswith("node ")]
        arrivals = [w for w in nodes if w[2] == "arrival"]
        compute = [w for w in nodes if w[2] != "arrival"]
        self.assertTrue(arrivals and compute)
        # params are a comma-joined list of k=v terms after the dependency column
        def has_layer(words):
            return any(term.startswith("layer=")
                       for term in " ".join(words[5:]).replace(",", " ").split())

        self.assertFalse(any(has_layer(w) for w in arrivals))
        self.assertTrue(all(has_layer(w) for w in compute))

    def test_without_the_flag_the_case_is_the_one_the_tables_were_measured_on(self):
        _, written = self.instantiate([])
        text = (written / "dense-compute-000-spare.txt").read_text(encoding="utf-8")
        self.assertNotIn("layer=", text)
        self.assertNotIn("_l0 ", text)

    def test_the_files_dtypes_win_over_the_table_declaration(self):
        folder, _ = self.instantiate([])
        spec = inst.load_model("qwen3-0.6b")
        tensors = {}
        for layer in range(spec["layers"]):
            for role, stem in inst.ROLE_FILES.items():
                if role.endswith("_shexp"):
                    continue
                tensors[f"blk.{layer}.{stem}.weight"] = ([256, 8], 12)
        # one layer stores a different width, which is what a real Q4_K_M file does
        tensors["blk.3.attn_v.weight"] = ([256, 8], 14)
        path = folder / "tiny.gguf"
        write_gguf(path, tensors)
        header, extra = inst.gguf_dtypes(path, spec)
        self.assertEqual(header["v"], "q4_k")
        self.assertIn("dtype_v_l3 q6_k", extra)
        self.assertNotIn("dtype_v_l4 q6_k", extra)
        self.assertEqual(len(extra), 1)

    def test_a_file_missing_a_layer_is_named_rather_than_silently_short(self):
        folder, _ = self.instantiate([])
        spec = inst.load_model("qwen3-0.6b")
        tensors = {f"blk.{layer}.{stem}.weight": ([256, 8], 12)
                   for layer in range(spec["layers"] - 1)
                   for role, stem in inst.ROLE_FILES.items() if not role.endswith("_shexp")}
        path = folder / "short.gguf"
        write_gguf(path, tensors)
        with self.assertRaises(SystemExit) as caught:
            inst.gguf_dtypes(path, spec)
        self.assertIn("has no blk.27.attn_q.weight", str(caught.exception))


class ModelColumnTests(unittest.TestCase):
    """The whole-model column's own arithmetic: placements, cases, and the claim.

    Nothing here runs the executor.  What can be wrong before a GPU is ever touched is the
    case set (does every placement exist, does a decode round exist) and the claim (is a
    miss a miss); both are answered from the corpus generator.
    """

    def test_every_placement_from_nothing_on_the_accelerator_to_everything(self):
        from profiling.ir_ggml import model_column as mc
        self.assertEqual(mc.placements(28), [0, 7, 14, 21, 28])
        self.assertEqual(mc.placements(16), [0, 4, 8, 12, 16])
        # a shallow model must not emit the same placement five times
        self.assertEqual(mc.placements(2), [0, 1, 2])
        self.assertEqual(mc.placements(1), [0, 1])

    def test_a_case_carries_the_whole_stack_and_the_declared_decode_rounds(self):
        from profiling.ir_ggml import model_column as mc
        spec = inst.load_model("qwen3-0.6b")
        with tempfile.TemporaryDirectory(prefix="model-column-") as folder:
            written = mc.build_cases(spec, rounds=2, out=Path(folder))
            self.assertEqual(set(written), {mc.case_name(28, g) for g in (0, 7, 14, 21, 28)})
            text = (Path(folder) / mc.case_name(28, 14) / "nodes.json").read_text(encoding="utf-8")
        self.assertIn("layer27", text)
        self.assertIn("decode0", text)
        self.assertIn("decode1", text)
        self.assertNotIn("decode2", text)

    def test_the_claim_turns_a_miss_into_a_miss(self):
        from profiling.ir_ggml import model_column as mc
        summary = {"units_by_phase": {"attention:prefill": {"n": 10, "median": 0.95},
                                      "attention:decode": {"n": 10, "median": 1.15}},
                   "makespan_resolved": {"n": 1, "median": 0.97},
                   "state": {"verdict": "in_band"}}
        verdict = mc.claim(summary)
        self.assertTrue(verdict["cells"]["attention:prefill"]["within_band"])
        self.assertFalse(verdict["cells"]["attention:decode"]["within_band"])
        self.assertEqual(verdict["verdict"], "off_band")
        summary["units_by_phase"]["attention:decode"]["median"] = 1.05
        self.assertEqual(mc.claim(summary)["verdict"], "in_band")

    def test_an_off_band_tranche_is_not_a_reading(self):
        from profiling.ir_ggml import model_column as mc
        summary = {"units_by_phase": {"attention:prefill": {"n": 10, "median": 1.00}},
                   "state": {"verdict": "off_band:tranche_void"}}
        self.assertEqual(mc.claim(summary)["verdict"], "off_band")

    def test_units_are_split_by_how_many_nodes_they_cost(self):
        from profiling.ir_ggml import model_column as mc
        rows = [{"units": {
            "a": {"unit": "attention", "phase": "decode", "nodes": [1, 2, 3, 4], "ratio": 1.2},
            "b": {"unit": "attention", "phase": "decode", "nodes": [1, 2, 3, 4], "ratio": 1.0},
            "c": {"unit": "projection", "phase": "decode", "nodes": [1], "ratio": 0.9}}}]
        table = mc.by_unit_nodes(rows)
        self.assertEqual(table["attention:decode:4n"]["median"], 1.1)
        self.assertEqual(table["projection:decode:1n"]["median"], 0.9)

    def test_a_history_the_table_does_not_price_is_refused(self):
        from profiling.ir_ggml import model_column as mc
        with tempfile.TemporaryDirectory(prefix="model-column-") as folder:
            with self.assertRaises(SystemExit):
                mc.main(["--model", "qwen3-0.6b", "--out", folder,
                         "--decode-history", "65", "--analyse-only"])


class PriceProvenanceTests(unittest.TestCase):
    def test_every_model_has_a_frozen_table(self):
        for model in prices_provenance.ORIGIN:
            self.assertTrue((HERE / "prices" / model / "costs.json").is_file(), model)

    def test_the_tables_match_their_recorded_hashes(self):
        self.assertEqual(prices_provenance.main(["--check"]), 0)


class RunnerBuildTests(unittest.TestCase):
    def test_print_still_validates_the_checkout(self):
        script = HERE.parent / "native" / "build_ir_runner.py"
        llama = Path(tempfile.gettempdir()) / "not-a-llama-checkout"
        done = subprocess.run([sys.executable, str(script), "--print", "--llama", str(llama)],
                              capture_output=True, text=True)
        # --print still validates the checkout, so this is the refusal path, not a build.
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("does not look like a llama.cpp checkout", done.stderr)

    def test_fetch_and_llama_are_mutually_exclusive(self):
        script = HERE.parent / "native" / "build_ir_runner.py"
        done = subprocess.run([sys.executable, str(script), "--print", "--fetch",
                               "--llama", "."], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("not both", done.stderr)

    def test_fetch_clones_the_pinned_tag_and_checks_its_commit(self):
        # `--print --fetch` must not touch the network or the disk: it is the only way to
        # check the bootstrap path on a machine that already has its own ggml build.
        native = HERE.parent / "native"
        source = (native / "build_ir_runner.py").read_text(encoding="utf-8")
        self.assertIn('PINNED_TAG = "b8705"', source)
        self.assertRegex(source, r'PINNED_COMMIT = "[0-9a-f]{40}"')
        done = subprocess.run([sys.executable, str(native / "build_ir_runner.py"),
                               "--print", "--fetch"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("--branch b8705", done.stdout)
        self.assertIn("-DGGML_SOURCE_DIR=", done.stdout)
        self.assertIn("--target ggml-ir-runner", done.stdout)


if __name__ == "__main__":
    unittest.main()
