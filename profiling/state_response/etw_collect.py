"""Launch an explicitly authorized, bounded elevated scheduler diagnosis.

This command can show UAC. It never changes a foreign recorder or kills the
broker, whose own finite deadline must always clean up its named session.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid

from . import collect as common
from profiling.state_calibration import run

ASSETS = Path(__file__).resolve().with_name("etw")


def prepare_config(path):
    instance = "LLMSimThread_" + uuid.uuid4().hex
    output = ASSETS / "results" / instance
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [ASSETS / name for name in ("launch-uac.ps1", "capture-owned-threads.ps1", "OwnedThreads.wprp")]
    for tool in ("wpr.exe", "xperf.exe"):
        value = shutil.which(tool)
        common._require(value is not None, "required existing Windows tracing tool unavailable: " + tool)
        files.append(Path(value))
    config = dict(schema="llm-sim-etw-controller-config-1", instance_name=instance, out_dir=str(output),
        maximum_trace_seconds=30, bound_files={str(p.resolve()): run.sha(p) for p in files})
    run.save(path, config)
    return config


def collect(out):
    out = Path(out).resolve()
    manifest = common.load_manifest(out)  # All source hashing precedes recorder startup.
    config = manifest["protocol"]["scheduler_trace"]
    common._require(len(manifest["jobs"]) == 1, "one bounded scheduler diagnosis job required")
    host = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    common._require(host is not None, "PowerShell host unavailable")
    command = [host, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
        str(ASSETS / "launch-uac.ps1"), "-Broker", str(ASSETS / "capture-owned-threads.ps1"),
        "-OutDir", config["out_dir"], "-InstanceName", config["instance_name"],
        "-Receipt", str(out / "uac-launch.json")]
    with run.measurement_lock():
        launched = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=180, check=False)
        (out / "uac-launch-stdout.txt").write_bytes(launched.stdout)
        (out / "uac-launch-stderr.txt").write_bytes(launched.stderr)
        common._require(launched.returncode == 0, "UAC launch failed; receipts retained")
        ready_path = Path(config["out_dir"]) / "capture-ready.json"
        deadline = time.monotonic() + 20
        while not ready_path.is_file():
            final = Path(config["out_dir"]) / "broker-final.json"
            common._require(not final.is_file(), "scheduler broker refused or failed before recording readiness")
            common._require(time.monotonic() < deadline, "scheduler recorder READY deadline exceeded")
            time.sleep(.02)
        row = common.execute_job(out, manifest, manifest["jobs"][0])
        run.save(out / "diagnostic-rows.json", [row])
    return dict(status="collected", process_id=row["process_id"], instance_name=config["instance_name"],
                raw_trace=config["out_dir"], exact_scheduler_intervals_validated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("config")
    setup.add_argument("--out", type=Path, required=True)
    recording = sub.add_parser("collect")
    recording.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_config(args.out) if args.command == "config" else collect(args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
