"""Offline contracts for optional owned-thread snapshots, never hardware runs."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from profiling.state_response import thread_activity as t
from profiling.state_response.worker_activity import OwnedActor


class Clock:
    def __init__(self): self.tick = 100
    def now(self): self.tick += 1; return self.tick
    def metadata(self): return dict(clock_id="fixture-QPC", frequency_hz=1000, extra="retained only externally")


def owned_snapshot(actor, **kwargs):
    return dict(process_identity=dict(status="verified"),
        cpu_counter_snapshot=dict(process=dict(values=dict(creation_time_100ns=700))))


def config():
    return dict(schema=t.CONFIG_SCHEMA, actors=[dict(actor_id="main", process_id=12, creation_time_100ns=700)],
        interval_ms=2, affinity_mask=64, max_duration_ms=15000, state_provider=t.NT_PROVIDER)


def output():
    return dict(schema=t.OUTPUT_SCHEMA, process_id=44, clock=dict(clock_id="fixture-QPC", frequency_hz=1000),
        state_provider=t.NT_PROVIDER,
        requested_affinity_mask=64, observed_affinity_mask=64, sampling_end_reason="owned_stop", error=None,
        sample_count=1, records=[dict(sample_index=0, query_begin_tick=1, query_end_tick=9, actors=[
            dict(actor_id="main", process_id=12, creation_time_100ns=700, query_begin_tick=2, query_end_tick=8,
                threads=[dict(thread_id=5, handle_owner_pid=12, cycle_total=None, unknown=["query denied"])])])],
        exact_thread_state_durations_claimed=False, cycles_converted_to_time=False,
        useful_work_distinguished_from_runtime_spin=False, used_as_predictor=False, global_timer_settings_changed=False)


class OwnedThreadActivityTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "build").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "build", prefix="thread-sampler-test-")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_config_requires_borrowed_handle_identity_and_creation_time(self):
        actor = OwnedActor("main", 12, 123)
        actual, receipts = t.make_config([actor], clock=Clock(), snapshot_reader=owned_snapshot)
        self.assertEqual(actual, config())
        self.assertEqual(len(receipts), 1)
        def unknown(actor, **kwargs):
            return dict(process_identity=dict(status="unknown"))
        with self.assertRaisesRegex(ValueError, "identity unavailable"):
            t.make_config([actor], snapshot_reader=unknown)
        def missing(actor, **kwargs):
            return dict(process_identity=dict(status="verified"), cpu_counter_snapshot=dict(process=dict(values={})))
        with self.assertRaisesRegex(ValueError, "creation time"):
            t.make_config([actor], snapshot_reader=missing)

    def test_config_rejects_duplicate_pid_and_unbounded_duration_or_float(self):
        actor = OwnedActor("main", 12, 123)
        for kwargs in (dict(max_duration_ms=15001), dict(interval_ms=0), dict(interval_ms=1.0), dict(affinity_mask=True)):
            with self.assertRaises(ValueError): t.make_config([actor], snapshot_reader=owned_snapshot, **kwargs)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            t.make_config([actor, OwnedActor("other", 12, 124)], snapshot_reader=owned_snapshot)

    def test_output_retains_unknown_cycles_without_zero_or_duration_conversion(self):
        raw = output()
        self.assertIs(t.validate_output(raw, config(), process_id=44, clock=Clock()), raw)
        thread = raw["records"][0]["actors"][0]["threads"][0]
        self.assertIsNone(thread["cycle_total"])
        self.assertEqual(thread["unknown"], ["query denied"])

    def test_output_rejects_foreign_pid_lifetime_clock_or_affinity(self):
        variants = []
        for key, value in (("process_id",45), ("observed_affinity_mask",32), ("clock",dict(clock_id="other",frequency_hz=1000))):
            raw=output(); raw[key]=value; variants.append(raw)
        for key, value in (("creation_time_100ns",701), ("process_id",13)):
            raw=output(); raw["records"][0]["actors"][0][key]=value; variants.append(raw)
        raw=output();raw["records"][0]["actors"][0]["threads"][0]["handle_owner_pid"]=99;variants.append(raw)
        for raw in variants:
            with self.assertRaises(ValueError): t.validate_output(raw, config(), process_id=44, clock=Clock())

    def test_output_rejects_timeout_error_partial_inventory_and_claimed_predictor(self):
        for key,value in (("sampling_end_reason","maximum_duration"),("error","failure"),
                          ("sample_count",2),("used_as_predictor",True),("cycles_converted_to_time",True)):
            raw=output();raw[key]=value
            with self.assertRaises(ValueError): t.validate_output(raw, config(), process_id=44, clock=Clock())
        raw=output();raw["records"][0]["actors"] = []
        with self.assertRaises(ValueError): t.validate_output(raw, config(), process_id=44, clock=Clock())

    def test_query_spans_are_actual_nonoverlapping_outer_envelopes(self):
        raw=output();raw["records"][0]["actors"][0]["query_end_tick"]=10
        with self.assertRaisesRegex(ValueError,"escaped"):
            t.validate_output(raw,config(),process_id=44,clock=Clock())
        raw=output();second=deepcopy(raw["records"][0]);second["sample_index"]=1
        raw["records"].append(second);raw["sample_count"]=2
        with self.assertRaisesRegex(ValueError,"backwards"):
            t.validate_output(raw,config(),process_id=44,clock=Clock())

    def test_native_source_uses_supported_snapshots_and_private_timer_without_suspending(self):
        source=t.SOURCE.read_text(encoding="utf-8")
        for required in ("Process.Refresh()", "Process.Threads", "QueryThreadCycleTime", "GetProcessIdOfThread",
                         "CreateWaitableTimerEx", "GetThreadTimes", "MaxSamples = 20000", "MaxThreads = 128"):
            self.assertIn(required,source)
        for forbidden in ("SuspendThread", "timeBeginPeriod", "SetThreadPriority"):
            self.assertNotIn(forbidden,source)

    def fixture_framework(self):
        framework=self.folder/"framework";framework.mkdir()
        for name in ("csc.exe","System.Web.Extensions.dll","System.dll","System.Core.dll","mscorlib.dll"):
            (framework/name).write_bytes(name.encode())
        return framework

    def test_compiler_receipt_binds_source_compiler_references_and_never_executes_binary(self):
        framework=self.fixture_framework();calls=[]
        def runner(command, **kwargs):
            calls.append(command)
            binary=Path(next(arg[5:] for arg in command if arg.startswith("/out:")))
            binary.write_bytes(b"fixture executable, never launched")
            return SimpleNamespace(returncode=0,stdout=b"compiler fixture\n",stderr=b"")
        result=t.compile_helper(self.folder/"new-build",framework=framework,runner=runner)
        self.assertEqual(len(calls),1)
        self.assertFalse(result["hardware_program_launched"])
        files=t.helper_sources(result["binary"])
        self.assertIn(str(t.SOURCE),files)
        self.assertIn(str(framework/"csc.exe"),files)
        (framework/"System.dll").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"input changed"): t.helper_sources(result["binary"])

    def test_failed_compile_retains_receipt_and_cannot_overwrite_directory(self):
        framework=self.fixture_framework();out=self.folder/"failed-build"
        def runner(command,**kwargs): return SimpleNamespace(returncode=1,stdout=b"",stderr=b"compile error")
        with self.assertRaisesRegex(ValueError,"compilation failed"): t.compile_helper(out,framework=framework,runner=runner)
        self.assertEqual(t.read(out/"build-manifest.json")["returncode"],1)
        with self.assertRaisesRegex(ValueError,"new directory"): t.compile_helper(out,framework=framework,runner=runner)

    def test_save_new_preserves_lf_and_never_overwrites(self):
        target=self.folder/"receipt.json";t.save_new(target,{"value":1})
        self.assertNotIn(b"\r\n",target.read_bytes())
        with self.assertRaises(FileExistsError): t.save_new(target,{"value":2})

    def test_failed_helper_launch_closes_streams_and_retains_no_child_receipt(self):
        sampler=t.OwnedThreadSampler([OwnedActor("main",12,123)],self.folder/"launch",self.folder/"unused.exe",clock=Clock())
        t.save_new(self.folder/"build-manifest.json",dict(supported_state_providers=[t.NT_PROVIDER]))
        with mock.patch.object(t,"helper_sources",return_value={}), mock.patch.object(t,"make_config",return_value=(config(),[])), \
             mock.patch.object(t.subprocess,"Popen",side_effect=OSError("fixture denied")):
            with self.assertRaisesRegex(OSError,"fixture denied"): sampler.start()
        self.assertTrue(all(stream.closed for stream in sampler._streams))
        self.assertTrue(t.read(sampler.directory/"process.json")["no_child_was_created"])

    def test_empty_snapshot_is_not_successful_observation(self):
        raw=output();raw["sample_count"]=0;raw["records"]=[]
        with self.assertRaisesRegex(ValueError,"inventory"):
            t.validate_output(raw,config(),process_id=44,clock=Clock())


if __name__ == "__main__": unittest.main()
