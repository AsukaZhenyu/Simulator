"""Offline context-role tests; the already-tested base verifier is mocked."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import analysis as a
import verify_context as v


def fixture(directory, family="ffn"):
    (directory / "raw").mkdir()
    (directory / "source").mkdir()
    source = directory / "compute_context.cpp"
    source.write_text("// fake context source\n", encoding="utf-8", newline="\n")
    snapshot = directory / "source" / source.name
    snapshot.write_bytes(source.read_bytes())
    flags = {"NVIDIA_TF32_OVERRIDE": "0", "GGML_CALIBRATION_GPU_DISPATCH_CPU_ID": "2"}
    manifest = {"observed_child_flags": flags,
                "inventory_before": {str(source): a.sha(source)},
                "source_snapshots": {str(source): {"path": "source/compute_context.cpp", "sha256": a.sha(snapshot)}}}
    a.dump(directory / "manifest.json", manifest)
    a.dump(directory / "suite.json", {"protocol": {"cpu_id": 0}})
    a.dump(directory / "measurements.json", [{"id": "row", "case": {"family": family}, "status": "measured", "split": "profile", "raw_sha256": "base-verifier-checks-this"}])
    payload = {"nvidia_tf32_override": "0",
               "profile_gpu_shard_thread": "dispatch_worker_same_as_parallel_strategy" if family == "ffn" else "not_applicable_attention_has_no_shard",
               "profile_gpu_shard_affinity_mask": 4 if family == "ffn" else None,
               "profile_gpu_shard_affinity_group": 0 if family == "ffn" else None}
    a.dump(directory / "raw" / "row.json", {"stdout": json.dumps(payload), "observed_child_flags": flags})
    return payload


class ContextContracts(unittest.TestCase):
    def test_ffn_same_worker_and_attention_not_applicable(self):
        for family in ("ffn", "attention"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp).resolve()
                fixture(directory, family)
                with patch.object(v, "verify_dual", return_value={"verified": True}):
                    self.assertTrue(v.verify(directory)["same_profile_execution_context_verified"])

    def test_mismatched_role_mask_group_rejected(self):
        for key, value in (("profile_gpu_shard_thread", "main_thread"), ("profile_gpu_shard_affinity_mask", 1), ("profile_gpu_shard_affinity_group", 1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp).resolve()
                payload = fixture(directory)
                payload[key] = value
                rawpath = directory / "raw" / "row.json"
                raw = v.r.document(rawpath)
                raw["stdout"] = json.dumps(payload)
                a.dump(rawpath, raw)
                with patch.object(v, "verify_dual", return_value={"verified": True}), self.assertRaisesRegex(ValueError, "strategy worker"):
                    v.verify(directory)

    def test_native_tf32_must_match_observed_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp).resolve()
            payload = fixture(directory)
            payload["nvidia_tf32_override"] = "1"
            rawpath = directory / "raw" / "row.json"
            raw = v.r.document(rawpath)
            raw["stdout"] = json.dumps(payload)
            a.dump(rawpath, raw)
            with patch.object(v, "verify_dual", return_value={"verified": True}), self.assertRaisesRegex(ValueError, "precision environment"):
                v.verify(directory)

    def test_manifest_tf32_override_must_be_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp).resolve()
            fixture(directory)
            manifest = v.r.document(directory / "manifest.json")
            manifest["observed_child_flags"].pop("NVIDIA_TF32_OVERRIDE")
            a.dump(directory / "manifest.json", manifest)
            with patch.object(v, "verify_dual", return_value={"verified": True}), self.assertRaisesRegex(ValueError, "TF32_OVERRIDE"):
                v.verify(directory)

    def test_context_source_snapshot_identity_required(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp).resolve()
            fixture(directory)
            (directory / "compute_context.cpp").write_text("changed\n", encoding="utf-8", newline="\n")
            with patch.object(v, "verify_dual", return_value={"verified": True}), self.assertRaisesRegex(ValueError, "context native source"):
                v.verify(directory)


if __name__ == "__main__":
    unittest.main()
