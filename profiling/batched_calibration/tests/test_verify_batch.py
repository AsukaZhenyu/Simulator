"""Tamper regressions over synthetic protocol archives; no hardware is run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from test_batch_contract import fake_payload, suite

SOURCE = Path(__file__).resolve().parents[1] / "verify_batch.py"
SPEC = importlib.util.spec_from_file_location("batch_archive_verifier_test", SOURCE)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


def archive(root):
    out = root / "experiment"
    out.mkdir()
    (out / "raw").mkdir()
    (out / "source").mkdir()
    native = root / "native-build"
    native.mkdir()
    binary = native / "synthetic-probe.exe"
    binary.write_bytes(b"Synthetic unit-test metadata, never executed.\n")
    generated = native / "compute_batch.cpp"
    generated.write_bytes(b"// Synthetic unit-test source, never compiled.\n")
    semantic_sources = {
        "generator_sha256": v.PROFILING / "native/execution_transforms_batch/generate_probe.py",
        "context_sha256": v.PROFILING / "native/execution_transforms_dual/compute_context.cpp",
        "helper_sha256": v.PROFILING / "native/matched_mechanisms/compute.cpp"}
    generated_manifest = {key: v.a.sha(path) for key, path in semantic_sources.items()}
    generated_manifest["generated_sha256"] = v.a.sha(generated)
    source_manifest = native / "source-manifest.json"
    v.a.dump(source_manifest, generated_manifest)
    build = native / "compute-batch-build.json"
    v.a.dump(build, dict(binary_sha256=v.a.sha(binary), source_manifest=generated_manifest))
    doc = suite()
    v.a.dump(out / "suite.json", doc)
    flags = dict(GGML_CALIBRATION_BATCH_REPETITIONS="16", GGML_CALIBRATION_GPU_DISPATCH_CPU_ID="2",
                 GGML_CUDA_DISABLE_GRAPHS="1", NVIDIA_TF32_OVERRIDE="0")
    rows, frozen = [], None
    case = doc["cases"][0]
    for split in ("profile", "heldout"):
        for index, planned in enumerate(v.base.jobs(doc, split)):
            row = dict(planned)
            _, payload = fake_payload(doc)
            payload["seed"] = row["seed"]
            normalized = v.batch.normalized_payload(payload, row, doc, 2)
            row["command"] = v.base.command(str(binary), case, doc["protocol"], row["seed"])
            row["raw_path"] = str(out / "raw" / (row["id"] + ".json"))
            record = dict(command=row["command"], observed_child_flags=flags,
                started_utc="2026-10-17T10:00:00+00:00", child_pid=100 + index,
                returncode=0, stdout=json.dumps(payload), stderr="")
            row.update(status="measured", metrics=normalized, raw_batch_metrics=payload["batch_wall_ns"],
                raw_batch_statistics={key: v.a.stats(values) for key, values in payload["batch_wall_ns"].items()},
                checks=payload["checks"], workspaces=payload["workspaces"], ports=payload["ports"],
                boundary=payload["boundary"], batch_repetitions=16, child_pid=record["child_pid"])
            if split == "heldout":
                row["frozen_model_sha256_at_start"] = frozen
                record["frozen_model_sha256_at_start"] = frozen
            v.a.dump(row["raw_path"], record)
            row["raw_sha256"] = v.a.sha(row["raw_path"])
            rows.append(row)
        if split == "profile":
            model = v.a.freeze_model(rows, {case["id"]: v.batch.recipes(case)},
                                    {case["id"]: v.base.observed_aliases(case)})
            v.a.dump(out / "prediction-model.json", model)
            frozen = v.a.sha(out / "prediction-model.json")
    v.a.dump(out / "measurements.json", rows)
    groups = {case["id"]: {name: ("resident" if name.split("@", 1)[0].endswith("resident") else "preload")
        + (";dispatch_model" if "@" in name else ";ideal_model") for name in v.batch.recipes(case)}}
    report = v.a.evaluate(rows, model, doc["quality"], doc["bootstrap"],
        scope="F32 compact FFN and post-RoPE Attention; repeated completed-work batch means, explicit CPU/GPU worker contexts.",
        frozen_before_heldout=True, comparison_groups=groups)
    report.update(batch_repetitions=16, raw_timing_unit="completed_work_batch_total",
                  older_isolated_protocol_replaced=False, gpu_capacity_proved=False)
    (out / "analysis").mkdir()
    v.a.dump(out / "analysis/report.json", report)
    inventory, snapshots = {}, {}
    sources = [Path(v.a.__file__).resolve(), Path(v.base.__file__).resolve(), Path(v.batch.__file__).resolve(),
               *semantic_sources.values(), generated, source_manifest, build]
    for index, source in enumerate(sources):
        original = str(source.resolve())
        archived = out / "source" / (str(index) + "-" + source.name)
        archived.write_bytes(source.read_bytes())
        inventory[original] = v.a.sha(source)
        snapshots[original] = dict(path=str(archived.relative_to(out)), sha256=v.a.sha(archived))
    inventory[str(binary)] = v.a.sha(binary)
    manifest = dict(schema="execution-transform-batch-manifest-0.1", assets_unchanged=True,
        inventory_before=inventory, inventory_after=inventory, source_snapshots=snapshots,
        timing_unit="completed_work_batch_total", derived_metric_unit="rounded_mean_completed_invocation_within_batch_ns",
        normalization="round(batch_total_ns / batch_repetitions); raw totals retained", sample_filtering="none",
        gpu_capacity_proved=False, observed_child_flags=flags,
        model_frozen_before_heldout_sha256=frozen, model_frozen_utc="2026-10-17T09:00:00+00:00",
        report_sha256=v.a.sha(out / "analysis/report.json"))
    v.a.dump(out / "manifest.json", manifest)
    return out, rows


class BatchArchiveTests(unittest.TestCase):
    def test_reconstructs_synthetic_batch_archive_without_executing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = archive(Path(directory).resolve())
            result = v.verify(path)
            self.assertTrue(result["verified"])
            self.assertEqual(result["measurement_rows"], 4)
            self.assertTrue(result["raw_batch_totals_and_normalization_verified"])
            self.assertFalse(result["physical_capacity_validated"])

    def test_raw_hash_change_and_changed_binary_rejected(self):
        for kind in ("raw", "binary"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                path, rows = archive(root)
                changed = Path(rows[0]["raw_path"]) if kind == "raw" else root / "native-build/synthetic-probe.exe"
                changed.write_bytes(changed.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "hash|asset missing or changed"):
                    v.verify(path)

    def test_changed_normalization_even_with_same_raw_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = archive(Path(directory).resolve())
            rows[0]["metrics"]["cpu/sum"][0] += 1
            v.a.dump(path / "measurements.json", rows)
            with self.assertRaisesRegex(ValueError, "normalization changed"):
                v.verify(path)

    def test_heldout_cannot_precede_frozen_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = archive(Path(directory).resolve())
            manifest = v.base.document(path / "manifest.json")
            manifest["model_frozen_utc"] = "2026-10-17T11:00:00+00:00"
            v.a.dump(path / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "previously frozen"):
                v.verify(path)

    def test_unknown_workspace_cannot_be_rewritten_and_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, rows = archive(Path(directory).resolve())
            raw = Path(rows[0]["raw_path"])
            record = v.base.document(raw)
            payload = json.loads(record["stdout"])
            payload["workspaces"]["gpu/shard/gate"]["workspace_bytes"] = 0
            record["stdout"] = json.dumps(payload)
            v.a.dump(raw, record)
            rows[0]["raw_sha256"] = v.a.sha(raw)
            v.a.dump(path / "measurements.json", rows)
            with self.assertRaisesRegex(ValueError, "unknown"):
                v.verify(path)


if __name__ == "__main__":
    unittest.main()
