"""Finite serving extension gate: explicit semantics, not timing accuracy."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.serving_acceptance.run import MANIFEST_SHA256,PRIOR_MANIFEST_SHA256,COUNTS,source_snapshot

ROOT=Path(__file__).resolve().parents[1]


class ServingAcceptanceTests(unittest.TestCase):
    def test_frozen_requirements_keep_prior_baseline(self):
        manifest=ROOT/"experiments/serving_acceptance/manifest.json"
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(),MANIFEST_SHA256)
        prior=ROOT/"experiments/ir_acceptance/manifest.json"
        self.assertEqual(hashlib.sha256(prior.read_bytes()).hexdigest(),PRIOR_MANIFEST_SHA256)
        document=json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual({g:len(document[g]) for g in COUNTS},COUNTS)

    def test_portable_profile_free_single_cpu_gpu_serving_cases(self):
        # A clean checkout has no ignored native profiling/results directory.
        # Test the portable semantic adapters directly. The separate research
        # runner still strictly requires its local calibration evidence freeze.
        from experiments.serving_acceptance.cases import CASES
        from experiments.strategy_fidelity import write_json
        manifest=json.loads((ROOT/"experiments/serving_acceptance/manifest.json").read_text(encoding="utf-8"))
        identifiers={row["id"] for group in COUNTS for row in manifest[group]}
        self.assertEqual(set(CASES),identifiers)
        before=source_snapshot()
        with tempfile.TemporaryDirectory() as temporary:
            for group,count in COUNTS.items():
                self.assertEqual(len(manifest[group]),count)
                for requirement in manifest[group]:
                    identifier=requirement["id"]
                    with self.subTest(case=identifier):
                        folder=Path(temporary)/identifier;folder.mkdir()
                        write_json(folder/"requirement.json",requirement)
                        evidence=CASES[identifier](folder)
                        self.assertEqual(evidence.get("_status","pass"),"pass",evidence.get("reason",""))
        self.assertEqual(source_snapshot(),before,"source/reference bytes changed during portable semantic tests")


if __name__=="__main__":unittest.main()
