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


class PriceProvenanceTests(unittest.TestCase):
    def test_every_model_has_a_frozen_table(self):
        for model in prices_provenance.ORIGIN:
            self.assertTrue((HERE / "prices" / model / "costs.json").is_file(), model)

    def test_the_tables_match_their_recorded_hashes(self):
        self.assertEqual(prices_provenance.main(["--check"]), 0)


class RunnerBuildTests(unittest.TestCase):
    def test_print_shows_the_two_commands_and_runs_nothing(self):
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
