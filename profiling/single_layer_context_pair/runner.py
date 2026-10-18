"""Counterbalanced same-PID diagnostics; never fit or accept latency parameters.

Prepare binds existing profile programs and a separately built paired executor.
Run executes only declared jobs under the shared measurement lock. Summary
rehashes raw records and reports every declared success/failure descriptively.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (ROOT, ROOT / "modeling", ROOT / "mapping", ROOT / "src", ROOT / "profiling"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from highlevel_accuracy.bridge import RUNTIME_FLAGS, verify_native
from single_layer_calibration import run as driver
from single_layer_calibration.analysis import _execution_scope, _validate_interval, _validate_native, cv, median
from state_calibration.telemetry import DeviceTelemetrySampler

KINDS = ("context_vs_isolated", "instrumentation")
MODES = {"context_vs_isolated": {"A": "isolated-components", "B": "context-trace"},
         "instrumentation": {"A": "bare-replay", "B": "context-trace"}}
SCHEMA = "single-layer-context-pair-campaign-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def local_sources():
    return sorted(path.resolve() for path in Path(__file__).parent.glob("*.py"))


def own_inventory(executable):
    executable = Path(executable).resolve()
    path = executable.parent / "build-manifest.json"
    build = driver.read(path)
    require(build.get("format_version") == "single-layer-context-pair-native-build-v2" and
            build.get("returncode") == 0 and build.get("diagnostic_only") is True and
            build.get("usable_for_parameter_fit") is False and build.get("usable_for_heldout_acceptance") is False,
            "successful diagnostic-only paired build required")
    require(build["binary_sha256"] == driver.sha(executable), "paired executable differs from build")
    bound = {**build["bound_files"], str(path): driver.sha(path), str(executable): driver.sha(executable),
             **{str(path): driver.sha(path) for path in local_sources()}}
    for source, digest in bound.items():
        require(driver.sha(source) == digest, "paired build/source input bytes changed: " + source)
    sources = {source: digest for source, digest in bound.items()
               if Path(source).suffix.lower() in {".py", ".cpp", ".h", ".hpp", ".inc", ".txt", ".json"}}
    return {"bound_sha256": bound, "source_sha256": sources, "build": build,
            "build_record_path": str(path), "executable": str(executable),
            "local_source_inventory": [str(path) for path in local_sources()],
            "file_stats": {source: [Path(source).stat().st_size, Path(source).stat().st_mtime_ns] for source in bound}}


def verify_sources(manifest, *, full=False):
    base_path = Path(manifest["base_manifest_path"])
    require(driver.sha(base_path) == manifest["base_manifest_sha256"], "base campaign manifest changed")
    base = driver.load_manifest(base_path.parent)
    driver.verify_current(base, full=full)
    own = manifest["paired_software"]
    require([str(path) for path in local_sources()] == own["local_source_inventory"], "paired Python inventory changed")
    stats = {path: [Path(path).stat().st_size, Path(path).stat().st_mtime_ns] for path in own["bound_sha256"]}
    require(stats == own["file_stats"], "paired source/runtime metadata changed")
    paths = own["bound_sha256"] if full else {**own["source_sha256"], own["executable"]: own["bound_sha256"][own["executable"]]}
    for path, digest in paths.items():
        require(driver.sha(path) == digest, "paired source/binary/runtime bytes changed: " + path)
    for program in manifest["programs"].values():
        require(driver.sha(program["path"]) == program["sha256"], "declared profile program changed")
    return base


def prepare(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    campaign = args.campaign.resolve()
    base = driver.load_manifest(campaign)
    driver.verify_current(base, full=True)
    require(base["protocol"]["independent_processes"] >= 3, "source campaign needs >=3 profile programs per case")
    own = own_inventory(args.executable)
    programs, jobs = {}, []
    for case_index, case in enumerate(base["cases"]):
        for repeat in range(3):
            key = case["program_keys"][f"profile-{repeat}"]
            binding = deepcopy(base["programs"][key])
            require(driver.sha(binding["path"]) == binding["sha256"], "existing profile program changed")
            program = driver.read(binding["path"])
            targets = [index for index, action in enumerate(program["actions"])
                       if action["kind"] == "COMPUTE" and action.get("device") == "gpu" and
                       any(item["operation_id"] == action["target_id"] and item["kind"] == "attention_pv" for item in program["bindings"])]
            require(len(targets) == 1, "exactly one GPU attention_pv target is required")
            programs[key] = binding
            for poll_index, poll in enumerate(("default", "0")):
                for kind_index, kind in enumerate(KINDS):
                    order = "ABBA" if (repeat + case_index + poll_index + kind_index) % 2 == 0 else "BAAB"
                    run_id = f"{case['id']}--repeat-{repeat}--poll-{poll}--{kind}"
                    jobs.append({"run_id": run_id, "case_id": case["id"], "repeat": repeat,
                                 "program_key": key, "seed": binding["seed"], "poll": poll,
                                 "pair_kind": kind, "order": order, "target_action_index": targets[0]})
    # A fixed alternation reduces mode/order/time confounding without consulting observations.
    jobs.sort(key=lambda job: (job["repeat"], job["case_id"] if job["repeat"] % 2 == 0 else job["case_id"][::-1], job["poll"], job["pair_kind"]))
    manifest = {"schema": SCHEMA, "created_at_utc": driver.utc(),
                "base_manifest_path": str(campaign / "manifest.json"),
                "base_manifest_sha256": driver.sha(campaign / "manifest.json"),
                "paired_software": own, "runtime_dirs": base["runtime_dirs"],
                "programs": programs, "jobs": jobs,
                "protocol": {"warmup_per_block": args.warmup, "samples_per_block": args.samples,
                             "blocks_per_process": 4, "telemetry_interval_ms": 50,
                             "native_timeout_seconds": 300, "inherited_affinity_unchanged": True,
                             "orders_chosen_before_measurements": True, "all_observations_retained": True},
                "diagnostic_only": True, "usable_for_parameter_fit": False,
                "usable_for_heldout_acceptance": False,
                "scope": "same_PID_paired_context_and_instrumentation; existing_profile_inputs_only; no_fitting_or_accuracy_claim"}
    manifest["manifest_sha256"] = driver.content_sha(manifest)
    verify_sources(manifest, full=True)
    driver.save(out / "manifest.json", manifest)
    return {"status": "prepared", "jobs": len(jobs), "out": str(out), "hardware_executed": False}


def load_manifest(out):
    manifest = driver.read(Path(out) / "manifest.json")
    unsigned = deepcopy(manifest)
    digest = unsigned.pop("manifest_sha256", None)
    require(manifest.get("schema") == SCHEMA and digest == driver.content_sha(unsigned), "paired manifest changed")
    require(manifest.get("diagnostic_only") is True and manifest.get("usable_for_parameter_fit") is False and
            manifest.get("usable_for_heldout_acceptance") is False, "paired diagnostics cannot become calibration data")
    require(len({job["run_id"] for job in manifest["jobs"]}) == len(manifest["jobs"]), "duplicate declared job")
    return manifest


def validate_native(program, raw, job, protocol, *, process_id):
    require(raw.get("format_version") == "single-layer-context-pair-native-v2", "paired native v2 schema required")
    require(raw.get("diagnostic_only") is True and raw.get("usable_for_parameter_fit") is False and
            raw.get("usable_for_heldout_acceptance") is False, "native diagnostic boundary differs")
    require(raw["process_id"] == process_id and raw["program_sha256"] == program["program_sha256"] and
            raw["seed"] == program["seed"] == job["seed"], "paired PID/program/seed differs")
    require(raw["pair_kind"] == job["pair_kind"] and raw["block_order"] == job["order"] and
            raw["target_action_index"] == job["target_action_index"], "paired context/order/target differs")
    require(raw["A_label"] == MODES[job["pair_kind"]]["A"] and raw["B_label"] == MODES[job["pair_kind"]]["B"],
            "paired A/B semantic labels differ")
    target = job["target_action_index"]
    require(raw["target_action"] == program["actions"][target], "paired target action differs")
    require(raw["same_PID_Executor_program_and_pool_for_all_blocks"] is True, "same-PID execution contract missing")
    require(raw["samples_per_block"] == protocol["samples_per_block"] and raw["warmup_per_block"] == protocol["warmup_per_block"], "block sample/warmup protocol differs")
    require(raw["threadpool"]["requested_poll"] == job["poll"], "paired polling contract differs")
    blocks = raw["blocks"]
    require(len(blocks) == 4 and job["order"] in {"ABBA", "BAAB"}, "exactly four counterbalanced blocks required")
    frequency = raw["native_clock"]["frequency_hz"]
    require(type(frequency) is int and frequency > 0, "native clock frequency invalid")
    require(raw["begin_tick"] <= raw["end_tick"], "paired root clock reversed")
    previous, scope, checked, descriptions = raw["begin_tick"], None, [], []
    for index, block in enumerate(blocks):
        label, native = job["order"][index], block["raw"]
        require(block["block_index"] == index and block["label"] == label, "block index/label order differs")
        require(previous <= block["block_begin_tick"] <= block["block_end_tick"] <= raw["end_tick"], "paired block clocks overlap/outside process")
        _validate_interval({"begin_tick": block["block_begin_tick"], "end_tick": block["block_end_tick"], "elapsed_ns": block["block_wall_ns"]}, frequency)
        previous = block["block_end_tick"]
        require(native["mode"] == MODES[job["pair_kind"]][label], "block label/native mode differs")
        require(native["process_id"] == process_id and native["program_sha256"] == program["program_sha256"], "block PID/program differs")
        require(native["native_clock"] == raw["native_clock"] and native["threadpool"] == raw["threadpool"], "block clock/pool differs")
        require(native["samples"] == protocol["samples_per_block"] and native["warmup"] == protocol["warmup_per_block"], "block sample inventory differs")
        require(native["actions"] == program["actions"], "block real action inventory differs")
        require(native.get("diagnostic_only") is True and native.get("usable_for_parameter_fit") is False and native.get("usable_for_heldout_acceptance") is False, "block cannot be used for fitting")
        current_scope = _execution_scope(native)
        require(scope is None or current_scope == scope, "block effective execution contract differs")
        scope = current_scope
        if native["mode"] == "isolated-components":
            require(native.get("component_subset") is True and len(native["components"]) == 1 and
                    native["components"][0]["action_index"] == target and
                    native["components"][0]["action"] == program["actions"][target] and
                    native["iterations"] == [] and native["total_samples_ns"] == [], "isolated target subset differs")
            component = native["components"][0]
            values = component["samples_ns"]
            require(len(values) == len(component["intervals"]) == protocol["samples_per_block"], "isolated sample count differs")
            intervals = component["intervals"]
            previous_sample_end = block["block_begin_tick"]
            for number, interval in enumerate(intervals):
                require(interval["sample"] == number and interval["elapsed_ns"] == values[number] and
                        type(values[number]) is int and values[number] > 0, "isolated interval/sample binding differs")
                _validate_interval(interval, frequency)
                require(interval["begin_tick"] >= previous_sample_end, "isolated sample clocks overlap")
                previous_sample_end = interval["end_tick"]
        else:
            _validate_native(program, native)
            intervals = native["iterations"]
            values = (native["total_samples_ns"] if job["pair_kind"] == "instrumentation" else
                      [iteration["action_intervals"][target]["elapsed_ns"] for iteration in native["iterations"]])
        require(all(block["block_begin_tick"] <= interval["begin_tick"] <= interval["end_tick"] <= block["block_end_tick"] for interval in intervals), "sampling interval outside its paired block")
        numerical = verify_native(program, native["verification_native"])
        checked.append(numerical)
        descriptions.append({"block_index": index, "label": label, "mode": native["mode"],
            "metric": "whole_invocation" if job["pair_kind"] == "instrumentation" else "selected_completed_GPU_attention_pv",
            "samples_ns": values, "median_ns": exact(median(values)), "cv": cv(values),
            "cpu_diagnostics": deepcopy(native.get("cpu_diagnostics")), "numerical_verification": numerical})
    label_medians = {label: median([Fraction(row["median_ns"]["numerator"], row["median_ns"]["denominator"]) for row in descriptions if row["label"] == label]) for label in ("A", "B")}
    require(label_medians["A"] > 0, "nonpositive paired reference")
    return {"blocks": descriptions, "label_median_ns": {label: exact(value) for label, value in label_medians.items()},
            "B_to_A_ratio": float(label_medians["B"] / label_medians["A"]),
            "execution_scope": scope, "independently_verified_blocks": len(checked),
            "cause": "unresolved; same_PID_counterbalancing_does_not_control_all_time_varying_state",
            "diagnostic_only": True, "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False}


def launch(out, manifest, job):
    binding = manifest["programs"][job["program_key"]]
    program = driver.read(binding["path"])
    directory = out / "jobs" / job["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    native_path = directory / "native.json"
    p = manifest["protocol"]
    command = [manifest["paired_software"]["executable"], binding["path"], job["pair_kind"],
               str(job["target_action_index"]), str(p["warmup_per_block"]), str(p["samples_per_block"]), job["poll"], job["order"], str(native_path)]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(manifest["runtime_dirs"] + [env.get("PATH", "")])
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(manifest["runtime_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    env.update({key: value for key, value in RUNTIME_FLAGS.items() if key != "matmul_precision"})
    record = {**job, "started_at_utc": driver.utc(), "command": command, "status": "failed",
              "diagnostic_only": True, "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False}
    process, sampler = None, None
    try:
        sampler = DeviceTelemetrySampler(interval_ms=p["telemetry_interval_ms"]).start()
        with (directory / "stdout.txt").open("x", encoding="utf-8") as stdout, (directory / "stderr.txt").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            record.update(process_id=process.pid, affinity=driver.affinity_readback(process))
            record["returncode"] = process.wait(timeout=p["native_timeout_seconds"])
        require(record["returncode"] == 0, "paired native process failed")
        raw = driver.read(native_path)
        record["analysis"] = validate_native(program, raw, job, p, process_id=process.pid)
        record["source_path"], record["source_sha256"] = str(native_path), driver.sha(native_path)
        record["program_path"], record["program_sha256"] = binding["path"], binding["sha256"]
        record["status"] = "measured"
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
            record["owned_child_terminated_after_failure"] = True
        if sampler is not None:
            sampler.stop()
            sampler.write_jsonl(directory / "telemetry.jsonl")
            driver.save(directory / "telemetry-metadata.json", sampler.metadata())
        record["completed_at_utc"] = driver.utc()
        record["raw_file_sha256"] = {path.name: driver.sha(path) for path in directory.iterdir() if path.is_file()}
        driver.save(directory / "record.json", record)
    print(json.dumps({"run_id": job["run_id"], "status": record["status"], "error": record.get("error")}), flush=True)
    return record


def collect(args):
    out = args.out.resolve()
    manifest = load_manifest(out)
    verify_sources(manifest, full=True)
    require(not (out / "phase.json").exists(), "paired campaign already completed; use a new output")
    records = []
    with driver.hardware_lock(out):
        for job in manifest["jobs"]:
            verify_sources(manifest)
            records.append(launch(out, manifest, job))
        verify_sources(manifest, full=True)
    phase = {"schema": "single-layer-context-pair-phase-v1", "manifest_sha256": manifest["manifest_sha256"],
             "records": records, "completed_at_utc": driver.utc(),
             "successful": sum(row["status"] == "measured" for row in records),
             "failed": sum(row["status"] != "measured" for row in records),
             "sources_verified_at_phase_end": True, "diagnostic_only": True,
             "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False}
    driver.save(out / "phase.json", phase)
    return {"status": "diagnostic_collection_completed", "successful": phase["successful"], "failed": phase["failed"]}


def summarize(args):
    out = args.out.resolve()
    manifest = load_manifest(out)
    verify_sources(manifest, full=True)
    phase = driver.read(out / "phase.json")
    require(phase["manifest_sha256"] == manifest["manifest_sha256"] and phase["sources_verified_at_phase_end"] is True, "paired phase source binding differs")
    records = phase["records"]
    require(len({row["run_id"] for row in records}) == len(records) and
            {row["run_id"] for row in records} == {job["run_id"] for job in manifest["jobs"]}, "paired phase omits or duplicates jobs")
    require(phase["successful"] == sum(row["status"] == "measured" for row in records) and
            phase["failed"] == sum(row["status"] != "measured" for row in records), "paired success/failure counts differ")
    groups = {}
    for row in records:
        key = (row["case_id"], row["poll"], row["pair_kind"])
        group = groups.setdefault(key, {"case_id": key[0], "poll": key[1], "pair_kind": key[2], "runs": [], "failed_runs": []})
        job = next(job for job in manifest["jobs"] if job["run_id"] == row["run_id"])
        require(all(row[name] == job[name] for name in job), "paired run declaration differs")
        directory = out / "jobs" / job["run_id"]
        require(driver.read(directory / "record.json") == row, "phase differs from original launched-process record")
        for name, digest in row["raw_file_sha256"].items():
            require(Path(name).name == name and driver.sha(directory / name) == digest,
                    "paired retained raw artifact changed")
        if row["status"] != "measured":
            group["failed_runs"].append({"run_id": row["run_id"], "error": row.get("error")})
            continue
        require(Path(row["source_path"]).resolve() == (directory / "native.json").resolve() and
                driver.sha(row["source_path"]) == row["source_sha256"], "paired raw source path or bytes changed")
        binding = manifest["programs"][job["program_key"]]
        raw = driver.read(row["source_path"])
        analyzed = validate_native(driver.read(binding["path"]), raw, job, manifest["protocol"], process_id=row["process_id"])
        require(analyzed == row["analysis"], "paired recorded analysis differs from raw evidence")
        group["runs"].append({"run_id": row["run_id"], "process_id": row["process_id"], "seed": row["seed"],
                              "order": row["order"], "source_path": row["source_path"], "source_sha256": row["source_sha256"], **analyzed})
    for group in groups.values():
        ratios = [row["B_to_A_ratio"] for row in group["runs"]]
        group["process_ratios_B_to_A"] = ratios
        group["diagnostic_median_process_ratio"] = statistics_median(ratios) if ratios else None
        group["all_predeclared_processes_measured"] = not group["failed_runs"] and len(group["runs"]) == 3
    report = {"schema": "single-layer-context-pair-summary-v1", "manifest_sha256": manifest["manifest_sha256"],
              "phase_sha256": driver.sha(out / "phase.json"), "groups": list(groups.values()),
              "declared_processes": len(manifest["jobs"]), "measured_processes": sum(len(g["runs"]) for g in groups.values()),
              "all_samples_retained": True, "diagnostic_only": True,
              "usable_for_parameter_fit": False, "usable_for_heldout_acceptance": False,
              "cause": "unresolved; same_PID_counterbalancing_retains_time_varying_state_and_block_boundary_effects"}
    driver.save(args.output.resolve() if args.output else out / "summary.json", report)
    return {"status": "diagnostic_summary_completed", "groups": len(groups), "declared_processes": len(manifest["jobs"])}


def statistics_median(values):
    # A descriptive ratio statistic, never an execution-time prediction.
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--campaign", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--executable", type=Path, required=True)
    prep.add_argument("--warmup", type=int, default=30)
    prep.add_argument("--samples", type=int, default=101)
    sub.add_parser("run").add_argument("--out", type=Path, required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        require(0 <= args.warmup <= 10000 and 3 <= args.samples <= 10000, "bounded explicit warmup and >=3 samples required")
        result = prepare(args)
    elif args.command == "run":
        result = collect(args)
    else:
        result = summarize(args)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
