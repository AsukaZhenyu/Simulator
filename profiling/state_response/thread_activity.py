"""Bounded snapshots of caller-owned native threads; not a latency predictor.

The optional .NET helper records refreshed OS thread-state snapshots and raw
QueryThreadCycleTime counters. It neither suspends threads nor starts ETW. State
samples cannot establish exact running/ready/waiting durations, and cycles are
never converted to time or treated as useful instructions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from profiling.state_calibration.telemetry import NativeClock
from .worker_activity import OwnedActor, actor_cpu_snapshot

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "profiling/native/state_response/thread_sampler.cs"
LAYOUT_SOURCE = SOURCE.with_name("thread_layout_guard.cpp")
CONFIG_SCHEMA = "owned-thread-snapshot-config-1"
OUTPUT_SCHEMA = "owned-thread-snapshots-1"
NT_PROVIDER = "nt_system_process_information_x64"
LEGACY_PROVIDER = "dotnet_process_threads"
REFERENCES = (
    "https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.process.threads",
    "https://learn.microsoft.com/en-us/dotnet/api/system.diagnostics.threadstate",
    "https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime",
    "https://learn.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntquerysysteminformation",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low=1, high=2**63 - 1):
    _require(type(value) is int and low <= value <= high, name + " must be an integer in the explicit range")
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compile_helper(out, *, framework=None, native_compiler=None, sdk_include=None, runner=subprocess.run):
    """Compile the explicit C# source; this never launches the sampled program."""
    out = Path(out).resolve()
    _require(not out.exists(), "helper build output must be a new directory")
    framework = Path(framework or Path(os.environ.get("WINDIR", "C:/Windows")) /
                     "Microsoft.NET/Framework64/v4.0.30319").resolve()
    compiler = framework / "csc.exe"
    dependencies = [compiler, SOURCE] + [framework / name for name in (
        "System.Web.Extensions.dll", "System.dll", "System.Core.dll", "mscorlib.dll")]
    _require(all(path.is_file() for path in dependencies), "explicit .NET Framework compiler/reference inputs unavailable")
    sources = {str(path): sha(path) for path in dependencies}
    out.mkdir(parents=True, exist_ok=False)
    layout_guard = None
    if native_compiler is not None or sdk_include is not None:
        _require(native_compiler is not None and sdk_include is not None,
                 "native compiler and SDK include directory must both be explicit")
        compiler_native, sdk = Path(native_compiler).resolve(), Path(sdk_include).resolve()
        vc_include = compiler_native.parents[3] / "include"
        _require(compiler_native.is_file() and (sdk / "um/winternl.h").is_file() and vc_include.is_dir(),
                 "explicit native compiler/SDK layout inputs unavailable")
        dependency_path = out / "layout-source-dependencies.json"
        layout_command = [str(compiler_native), "/nologo", "/Zs", "/EHsc", "/std:c++17",
            "/sourceDependencies", str(dependency_path), *["/I" + str(path) for path in (
                vc_include, sdk / "um", sdk / "shared", sdk / "ucrt")], str(LAYOUT_SOURCE)]
        result = runner(layout_command, capture_output=True, check=False)
        (out / "layout-stdout.txt").write_bytes(result.stdout)
        (out / "layout-stderr.txt").write_bytes(result.stderr)
        layout_guard = dict(command=layout_command, returncode=result.returncode, verified=result.returncode == 0,
            sdk_include=str(sdk), program_launched=False)
        save_new(out / "layout-guard.json", layout_guard)
        _require(result.returncode == 0 and dependency_path.is_file(), "SDK layout guard failed; logs retained")
        headers = read(dependency_path)["Data"]["Includes"]
        for path in (compiler_native, LAYOUT_SOURCE, *map(Path, headers)):
            sources[str(path)] = sha(path)
        sources[str(dependency_path)] = sha(dependency_path)
        sources[str(out / "layout-guard.json")] = sha(out / "layout-guard.json")
    binary = out / "thread-sampler.exe"
    command = [str(compiler), "/nologo", "/target:exe", "/optimize+", "/platform:x64",
               "/reference:" + str(framework / "System.Web.Extensions.dll"),
               "/out:" + str(binary), str(SOURCE)]
    completed = runner(command, capture_output=True, check=False)
    (out / "compiler-stdout.txt").write_bytes(completed.stdout)
    (out / "compiler-stderr.txt").write_bytes(completed.stderr)
    receipt = dict(schema="owned-thread-helper-build-1", command=command,
        returncode=completed.returncode, bound_files=sources,
        binary=str(binary), binary_sha256=sha(binary) if binary.is_file() else None,
        layout_guard=layout_guard,
        supported_state_providers=[LEGACY_PROVIDER] + ([NT_PROVIDER] if layout_guard is not None else []),
        hardware_program_launched=False)
    save_new(out / "build-manifest.json", receipt)
    _require(completed.returncode == 0 and binary.is_file(), "thread helper compilation failed; compiler logs retained")
    return receipt


def helper_sources(binary):
    binary = Path(binary).resolve()
    manifest_path = binary.parent / "build-manifest.json"
    receipt = read(manifest_path)
    _require(receipt.get("schema") == "owned-thread-helper-build-1" and receipt.get("returncode") == 0,
             "successful source-bound thread helper build required")
    _require(receipt.get("binary") == str(binary) and receipt.get("binary_sha256") == sha(binary),
             "thread helper binary differs from its build receipt")
    for name, expected in receipt["bound_files"].items():
        _require(sha(name) == expected, "thread helper build input changed: " + name)
    return dict(receipt["bound_files"], **{str(binary): sha(binary), str(manifest_path): sha(manifest_path)})


def make_config(actors, *, interval_ms=2, affinity_mask=64, max_duration_ms=15000, state_provider=NT_PROVIDER,
                clock=None, snapshot_reader=actor_cpu_snapshot):
    """Bind owned handles to explicit PID and process creation time before launch."""
    actors = tuple(actors)
    _require(1 <= len(actors) <= 8 and all(isinstance(actor, OwnedActor) for actor in actors),
             "one to eight caller-owned actors required")
    _require(len({actor.actor_id for actor in actors}) == len(actors) and
             len({actor.process_id for actor in actors}) == len(actors), "duplicate owned actor")
    _integer(interval_ms, "interval_ms", high=1000)
    _integer(max_duration_ms, "max_duration_ms", high=15000)
    _integer(affinity_mask, "affinity_mask")
    _require(state_provider in (NT_PROVIDER, LEGACY_PROVIDER), "explicit supported thread state provider required")
    rows, receipts = [], []
    for actor in actors:
        receipt = snapshot_reader(actor, clock=clock)
        _require(receipt["process_identity"]["status"] == "verified", "owned actor handle/PID identity unavailable")
        created = receipt["cpu_counter_snapshot"]["process"]["values"].get("creation_time_100ns")
        _integer(created, "owned process creation time")
        rows.append(dict(actor_id=actor.actor_id, process_id=actor.process_id, creation_time_100ns=created))
        receipts.append(receipt)
    return dict(schema=CONFIG_SCHEMA, actors=rows, interval_ms=interval_ms,
                affinity_mask=affinity_mask, max_duration_ms=max_duration_ms, state_provider=state_provider), receipts


def validate_output(output, config, *, process_id, clock):
    _require(output.get("schema") == OUTPUT_SCHEMA, "unknown thread snapshot output")
    _require(output.get("process_id") == process_id and output.get("clock") ==
             {key: clock.metadata()[key] for key in ("clock_id", "frequency_hz")}, "helper identity/clock differs")
    _require(output.get("requested_affinity_mask") == config["affinity_mask"] == output.get("observed_affinity_mask"),
             "helper affinity differs")
    _require(output.get("state_provider") == config["state_provider"], "thread state provider differs")
    _require(output.get("sampling_end_reason") == "owned_stop" and output.get("error") is None,
             "thread sampling failed or exceeded its maximum duration")
    records = output.get("records")
    _require(isinstance(records, list) and 1 <= len(records) == output.get("sample_count") and len(records) <= 20000,
             "thread snapshot inventory differs or exceeds bounds")
    expected = {actor["actor_id"]: actor for actor in config["actors"]}
    previous = None
    for index, row in enumerate(records):
        _require(row["sample_index"] == index and row["query_begin_tick"] <= row["query_end_tick"], "malformed snapshot span")
        _require(previous is None or row["query_begin_tick"] >= previous, "thread snapshots move backwards")
        previous = row["query_end_tick"]
        observed = {actor["actor_id"]: actor for actor in row["actors"]}
        _require(len(observed) == len(row["actors"]) and set(observed) == set(expected), "owned actor inventory differs")
        for name, actor in observed.items():
            _require(all(actor[key] == expected[name][key] for key in ("process_id", "creation_time_100ns")),
                     "thread actor PID/lifetime binding differs")
            _require(len(actor["threads"]) <= 128, "owned thread bound exceeded")
            _require(row["query_begin_tick"] <= actor["query_begin_tick"] <= actor["query_end_tick"] <= row["query_end_tick"],
                     "actor query escaped outer snapshot span")
            for thread in actor["threads"]:
                _require(thread.get("handle_owner_pid") in (None, actor["process_id"]), "thread handle belongs to another PID")
    _require(all(output.get(key) is False for key in ("exact_thread_state_durations_claimed", "cycles_converted_to_time",
        "useful_work_distinguished_from_runtime_spin", "used_as_predictor", "global_timer_settings_changed")),
        "thread observation scope was broadened")
    return output


class OwnedThreadSampler:
    """One bounded, owned helper. Only this helper is stopped or terminated."""
    def __init__(self, actors, directory, helper_binary, *, interval_ms=2, affinity_mask=64,
                 max_duration_ms=15000, state_provider=NT_PROVIDER, clock=None):
        self.actors, self.directory, self.binary = tuple(actors), Path(directory).resolve(), Path(helper_binary).resolve()
        self.interval_ms, self.affinity_mask, self.max_duration_ms = interval_ms, affinity_mask, max_duration_ms
        self.state_provider = state_provider
        self.clock = clock or NativeClock()
        self.process = None
        self.record = dict(schema="owned-thread-sampler-process-1", forced_termination=False)
        self.output = None
        self._streams = []

    def start(self):
        _require(self.process is None, "thread sampler already started")
        self.sources = helper_sources(self.binary)
        build = read(self.binary.parent / "build-manifest.json")
        _require(self.state_provider in build.get("supported_state_providers", [LEGACY_PROVIDER]),
                 "helper lacks the selected source-verified state provider")
        self.config, receipts = make_config(self.actors, interval_ms=self.interval_ms, affinity_mask=self.affinity_mask,
            max_duration_ms=self.max_duration_ms, state_provider=self.state_provider, clock=self.clock)
        self.directory.mkdir(parents=True, exist_ok=False)
        save_new(self.directory / "config.json", self.config)
        save_new(self.directory / "owned-identity.json", receipts)
        command = [str(self.binary), str(self.directory / "config.json"), str(self.directory / "ready.json"),
                   str(self.directory / "stop.json"), str(self.directory / "result.json")]
        stdout = (self.directory / "stdout.txt").open("xb")
        stderr = (self.directory / "stderr.txt").open("xb")
        self._streams = [stdout, stderr]
        self.record.update(command=command, launch_tick=self.clock.now(), helper_sources=self.sources)
        try:
            self.process = subprocess.Popen(command, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.record["process_id"] = self.process.pid
            deadline = time.monotonic() + 5
            while not (self.directory / "ready.json").exists():
                _require(self.process.poll() is None, "owned thread helper exited before READY")
                _require(time.monotonic() < deadline, "owned thread helper READY timeout")
                time.sleep(.01)
            ready = read(self.directory / "ready.json")
            _require(ready.get("schema") == "owned-thread-sampler-ready-1", "unknown thread helper READY schema")
            _require(ready["process_id"] == self.process.pid and ready["clock"] ==
                {key: self.clock.metadata()[key] for key in ("clock_id", "frequency_hz")}, "thread helper READY identity differs")
            _require(ready["observed_affinity_mask"] == self.affinity_mask, "thread helper READY affinity differs")
            _require(ready["state_provider"] == self.state_provider, "thread helper READY state provider differs")
            self.record.update(ready=ready, readiness_observed_tick=self.clock.now())
        except BaseException as error:
            self.record["startup_error"] = type(error).__name__ + ": " + str(error)
            if self.process is None:
                for stream in self._streams:
                    stream.close()
                self.record["no_child_was_created"] = True
                save_new(self.directory / "process.json", self.record)
            else:
                try:
                    self.stop(cleanup=True)
                except BaseException as cleanup_error:
                    self.record["startup_cleanup_error"] = type(cleanup_error).__name__ + ": " + str(cleanup_error)
            raise
        return self

    def stop(self, *, cleanup=False):
        if self.process is None or "returncode" in self.record:
            return self.output
        stop = self.directory / "stop.json"
        if not stop.exists():
            save_new(stop, dict(requested_tick=self.clock.now(), cleanup=cleanup))
        try:
            try:
                code = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.record["forced_termination"] = True
                try:
                    code = self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.record["killed_after_terminate"] = True
                    code = self.process.wait(timeout=5)
            self.record.update(returncode=code, process_exited_verified=self.process.poll() is not None,
                               exit_observed_tick=self.clock.now())
            if not cleanup:
                _require(code == 0 and not self.record["forced_termination"], "thread helper failed; diagnostic raw output retained")
                self.output = validate_output(read(self.directory / "result.json"), self.config,
                    process_id=self.process.pid, clock=self.clock)
        finally:
            for stream in self._streams:
                stream.close()
            if not (self.directory / "process.json").exists():
                save_new(self.directory / "process.json", self.record)
        return self.output

    @property
    def records(self):
        return [] if self.output is None else self.output["records"]

    def metadata(self):
        return dict(process=self.record, config=self.config, source_files=self.sources,
                    exact_thread_state_durations_claimed=False, used_as_predictor=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-out", required=True)
    parser.add_argument("--framework")
    parser.add_argument("--native-compiler")
    parser.add_argument("--sdk-include")
    args = parser.parse_args(argv)
    print(json.dumps(compile_helper(args.build_out, framework=args.framework,
        native_compiler=args.native_compiler, sdk_include=args.sdk_include), indent=2))


if __name__ == "__main__":
    main()
