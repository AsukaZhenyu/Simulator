"""Protocol artifact tests only: no subprocesses or hardware are invoked."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent)]
import analysis as a
import run_native as r
import verify_native as v
from test_run_native import payload, suite


def fake_experiment(path):
    path.mkdir()
    (path / "raw").mkdir()
    (path / "source").mkdir()
    doc = suite()
    doc["cases"] = [doc["cases"][0]]
    doc["protocol"]["samples"] = 3
    doc["bootstrap"]["iterations"] = 100
    a.dump(path / "suite.json", doc)
    rows = []
    frozen = None
    model = None
    case = doc["cases"][0]
    declared = {case["id"]: r.calibration_recipes(case)}
    aliases = {case["id"]: r.observed_aliases(case)}
    flags = {"GGML_CUDA_DISABLE_GRAPHS": "1"}
    for split in ("profile", "heldout"):
        for index, planned in enumerate(r.jobs(doc, split)):
            row = dict(planned)
            row["command"] = r.command("probe.exe", case, doc["protocol"], row["seed"])
            row["raw_path"] = str(path / "raw" / (row["id"] + ".json"))
            native = payload(case, doc["protocol"], row["seed"])
            raw = {"command": row["command"], "observed_child_flags": flags,
                   "started_utc": "2026-10-17T10:00:00+00:00", "child_pid": 100 + index,
                   "returncode": 0, "stdout": json.dumps(native), "stderr": ""}
            row.update(status="measured", metrics=native["wall_ns"], checks=native["checks"],
                       workspaces=native["workspaces"], ports=native["ports"], boundary=native["boundary"], child_pid=raw["child_pid"])
            if split == "heldout":
                row["frozen_model_sha256_at_start"] = frozen
                raw["frozen_model_sha256_at_start"] = frozen
            a.dump(row["raw_path"], raw)
            row["raw_sha256"] = a.sha(row["raw_path"])
            rows.append(row)
        if split == "profile":
            model = a.freeze_model(rows, declared, aliases)
            a.dump(path / "prediction-model.json", model)
            frozen = a.sha(path / "prediction-model.json")
    a.dump(path / "measurements.json", rows)
    comparisons = {case["id"]: {strategy: ("weights_resident_inputs_resident_output_dram" if strategy.split("@", 1)[0].endswith("/resident") else "three_gpu_shard_weights_reloaded_then_output_dram")
                                            + (";dispatch_model" if "@" in strategy else ";ideal_model")
                               for strategy in r.calibration_recipes(case)}}
    report = a.evaluate(rows, model, doc["quality"], doc["bootstrap"],
                        scope="真实 F32 FFN 中间维 CPU/GPU 分片并行与 CPU 汇合，以及 post-RoPE Attention QK/softmax/PV 异构串行链。",
                        frozen_before_heldout=True, comparison_groups=comparisons)
    a.write_outputs(path / "analysis", report, model, rows)
    before, snapshots = {}, {}
    for source in (Path(a.__file__).resolve(), Path(r.__file__).resolve()):
        target = path / "source" / source.name
        target.write_bytes(source.read_bytes())
        before[str(source)] = a.sha(source)
        snapshots[str(source)] = {"path": "source/" + source.name, "sha256": a.sha(target)}
    manifest = {"inventory_before": before, "inventory_after": before, "assets_unchanged": True,
                "source_snapshots": snapshots, "model_frozen_before_heldout_sha256": frozen,
                "model_frozen_utc": "2026-10-17T09:00:00+00:00", "observed_child_flags": flags,
                "report_sha256": a.sha(path / "analysis" / "report.json")}
    a.dump(path / "manifest.json", manifest)
    return rows


class VerifyTests(unittest.TestCase):
    def test_reconstruct_full_fake_protocol_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "experiment"
            fake_experiment(path)
            result = v.verify(path)
            self.assertTrue(result["verified"])
            self.assertEqual(result["measurement_rows"], 4)
            self.assertTrue(result["profile_only_model_reconstructed"])
            self.assertEqual(result["accepted_predictions"], 0)

    def test_modified_raw_cannot_reuse_saved_report(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "experiment"
            rows = fake_experiment(path)
            raw = Path(rows[0]["raw_path"])
            raw.write_bytes(raw.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "hash"):
                v.verify(path)

    def test_model_change_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "experiment"
            fake_experiment(path)
            model_path = path / "prediction-model.json"
            model = r.document(model_path)
            model["cases"][0]["prices"]["cpu/sum"]["median_ns"] = 1
            a.dump(model_path, model)
            with self.assertRaisesRegex(ValueError, "model hash"):
                v.verify(path)

    def test_heldout_started_before_freeze_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "experiment"
            fake_experiment(path)
            manifest = r.document(path / "manifest.json")
            manifest["model_frozen_utc"] = "2026-10-17T11:00:00+00:00"
            a.dump(path / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "before the model"):
                v.verify(path)


if __name__ == "__main__":
    unittest.main()
