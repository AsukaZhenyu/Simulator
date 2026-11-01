"""Pinned external-source/native-component receipts for E1.

The 48-case expression suite is not ported to either external tool.  These are
bounded native API checks, deliberately not whole-system timing or unsupported-
feature claims.  Public repositories and private dependencies are prepared by
the operator; this script never edits external source or installs packages.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "LLMCompass": "2e015fd2ee750e6cad8d7152df52551e5b41ef20",
    "LLMServingSim": "a4053bc1161872420e1e0607cb3409ef659b828e",
}
REPOSITORIES = {
    "LLMCompass": "https://github.com/PrincetonUniversity/LLMCompass",
    "LLMServingSim": "https://github.com/casys-kaist/LLMServingSim",
}
FILES = {
    "LLMCompass": ["environment.yml", "docs/run.md", "software_model/utils.py",
                   "software_model/operators.py", "software_model/matmul.py",
                   "software_model/transformer.py", "hardware_model/device.py",
                   "hardware_model/compute_module.py", "hardware_model/io_module.py"],
    "LLMServingSim": ["serving/core/block_pool.py", "serving/core/config_builder.py",
                      "serving/core/pim_model.py", "serving/core/utils.py",
                      "serving/core/logger.py", "scripts/docker-sim.sh", ".gitmodules"],
}


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def run(argv, cwd, env=None, timeout=60):
    start = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        p = subprocess.run([str(x) for x in argv], cwd=cwd, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return dict(command=[str(x) for x in argv], cwd=str(cwd), started_utc=start,
                    returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
                    timeout=False)
    except subprocess.TimeoutExpired as e:
        return dict(command=[str(x) for x in argv], cwd=str(cwd), started_utc=start,
                    returncode=None, stdout=str(e.stdout or ""), stderr=str(e.stderr or ""),
                    timeout=True)


def result_from_stdout(receipt):
    for line in reversed(receipt["stdout"].splitlines()):
        if line.startswith("E1_NATIVE_JSON="):
            return json.loads(line.split("=", 1)[1])
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-native", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.run_native:
        lock = ROOT / "profiling/.measurement.lock"
        token = json.dumps({"pid":os.getpid(), "scope":"E1 external native components",
                            "utc":dt.datetime.now(dt.timezone.utc).isoformat()})
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as error:
            raise RuntimeError("Native external checks must wait for the measurement lock owner.") from error
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(token)
        def release():
            if lock.exists() and lock.read_text(encoding="utf-8") == token:
                lock.unlink()
        atexit.register(release)
    attempts = out / "receipts"
    number = 1
    while (attempts / f"attempt-{number:03d}").exists():
        number += 1
    receipt_dir = attempts / f"attempt-{number:03d}"
    receipt_dir.mkdir(parents=True)
    code_dir = receipt_dir / "code"
    code_dir.mkdir()
    (code_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    private_deps = out / "deps"
    if private_deps.exists():
        env["PYTHONPATH"] = str(private_deps) + os.pathsep + env.get("PYTHONPATH", "")
    inventory_code = (
        "import importlib.util,importlib.metadata as m,json,sys\n"
        "names=['numpy','pandas','torch','scalesim','networkx','rich','yaml','scipy','absl','tqdm']\n"
        "versions={}\n"
        "for name in names:\n"
        " try: versions[name]=m.version({'yaml':'PyYAML','absl':'absl-py'}.get(name,name))\n"
        " except m.PackageNotFoundError: versions[name]=None\n"
        "print(json.dumps({'python':sys.version,'packages':{n:bool(importlib.util.find_spec(n)) for n in names},'versions':versions}))\n")
    inventory = run([args.python, "-c", inventory_code],
                    ROOT, env)
    dump(receipt_dir / "dependency-inventory.json", inventory)
    tools = []
    for name in PINS:
        source = out / name
        head = run(["git", "rev-parse", "HEAD"], source)
        status = run(["git", "status", "--porcelain", "--untracked-files=no"], source)
        source_hashes = {p: hashlib.sha256((source / p).read_bytes()).hexdigest()
                         for p in FILES[name] if (source / p).is_file()}
        item = dict(name=name, repository=REPOSITORIES[name], commit=PINS[name],
                    source_url=REPOSITORIES[name] + "/tree/" + PINS[name],
                    head_receipt=head, status_receipt=status, source_sha256=source_hashes,
                    native_component_execution=False, whole_workload_comparison=False,
                    suite_48_ported=False, status="source_checked_native_not_run")
        if head["returncode"] != 0 or head["stdout"].strip() != PINS[name]:
            item["status"] = "setup_failed_wrong_source_version"
        elif status["returncode"] != 0 or status["stdout"].strip():
            item["status"] = "setup_failed_external_source_modified"
        elif args.run_native:
            script = ROOT / "experiments" / ("dac_validation_e1_external_" +
                                               ("compass" if name == "LLMCompass" else "servingsim") + ".py")
            item["native_probe_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
            (code_dir / script.name).write_bytes(script.read_bytes())
            native = run([args.python, script], source, env, timeout=90)
            payload = result_from_stdout(native)
            item.update(native_receipt=native, native_payload=payload,
                        native_component_execution=native["returncode"] == 0,
                        status="native_component_passed" if native["returncode"] == 0
                               else "setup_or_native_check_failed")
            if name == "LLMServingSim":
                built_in = run([args.python, "-m", "serving.core.block_pool"], source, env, timeout=60)
                item["native_builtin_selftest_receipt"] = built_in
                if built_in["returncode"] != 0:
                    item["status"] = "native_builtin_selftest_failed"
            item["post_execution_source_sha256"] = {
                p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in source_hashes}
            item["external_core_source_unchanged"] = item["post_execution_source_sha256"] == source_hashes
            if not item["external_core_source_unchanged"]:
                item["status"] = "failed_external_source_changed"
        dump(receipt_dir / (name + "-receipt.json"), item)
        tools.append(item)
    summary = dict(schema="e1-external-native-components-v2", tools=tools,
                   native_component_tools_passed=sum(t["status"] == "native_component_passed" for t in tools),
                   whole_workload_comparisons=0, suite_48_ported_tools=0,
                   receipt_directory=str(receipt_dir),
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   interpretation="Actual pinned native components only; unported cases remain untested, not unsupported. No whole-system accuracy, latency, adapter-effort, or superiority claim.",
                   external_source_changes=0)
    dump(receipt_dir / "summary.json", summary)
    dump(out / "summary.json", summary)
    print(json.dumps({"statuses": {t["name"]:t["status"] for t in tools},
                      "whole_workload_comparisons": 0}))


if __name__ == "__main__":
    main()
