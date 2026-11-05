"""Full Dense runner/export boundaries; all subprocess fixtures are synthetic."""
from __future__ import annotations
import contextlib
import copy
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_dense_layer_measurements as runner
import export_dense_layer as exporter
from llm_infer_model.m2.dense_contract import (CONTRACT, EXCLUDED, PROCESS_SCOPES, VARIANTS,
    build_plan, operations, ports, resident_accounting, stage_weight_roles, validate_case,
    validate_config, weight_shapes, DTYPES)
from llm_infer_model.m2.costs import MacroShape, parse_cost_table
from llm_infer_model.m2.spec import parse_layer_document
from llm_infer_model.tensor.spec import ggml_contiguous_layout, SpecError
from profile_contract import summarize_samples


def config():
    return validate_config({"hidden":16,"intermediate":32,"heads":4,"kv_heads":2,"head_dim":4,
        "attention_gate":"sigmoid","qk_norm":"rms","rope_mode":"normal","rope_dims":4,
        "rope_base":10000,"norm_epsilon":1e-6,"weight_dtypes":{role:"f32" for role in
            ("input_norm","q","k","v","gate_proj","q_norm","k_norm","output","post_attention_norm","gate","up","down")}})


def case(kind="attention", **changes):
    raw = {"id":kind,"kind":kind,"phase":"prefill","batch":1,"tokens":3,"visible_history":0,
           "config":config(),"threads":2}
    raw.update(changes)
    return validate_case(raw)


def suite(cases):
    return {"schema_version":"dense_layer_measurement_suite_v1","cases":cases}


def layer_document():
    c = config()
    shapes = weight_shapes(c)
    weights = []
    for role, shape in shapes.items():
        _, size = ggml_contiguous_layout(DTYPES[c["weight_dtypes"][role]], (*shape, *([1]*(4-len(shape)))))
        weights.append({"id":"W_"+role,"dtype":c["weight_dtypes"][role],"shape":list(shape),
                        "storage_bytes":size,"initial_location":"dram"})
    return {"format_version":"m2-layer-0.3","id":"synthetic-full-dense-fixture",
        "provenance":"synthetic_layer","hidden_size":16,"hidden_bytes_per_token":64,
        "kv_bytes_per_token":32,"attention":"full_causal","ffn":"dense","weights":weights,
        "physical":{"contract":CONTRACT,"scope":"complete_declared_decoder_block",
            "input_boundary":"hidden_activation","port_layout":"materialized_stage_ports",
            "kv_allocation":"logical_blocks","activation_dtype":"f32","kv_dtype":"f16",
            "config":c,"threads":2,"excluded_operations":[],
            "weight_groups":{kind:{role:"W_"+role for role in stage_weight_roles(c,kind)}
                             for kind in ("projection","attention","dense_ffn")}},
        "tasks":[{"id":"projection","kind":"projection","depends_on":[]},
                 {"id":"attention","kind":"attention","depends_on":["projection"]},
                 {"id":"dense_ffn","kind":"dense_ffn","depends_on":["attention"]}]}


def native(case, **changes):
    mem = resident_accounting(case["config"],case["kind"],case["tokens"],case["visible_history"])
    inputs, outputs = ports(case["config"], case["kind"], case["tokens"])
    port_accounting = {name:{"dtype":port["dtype"],"shape":list(port["ne"][:3]),"ne":list(port["ne"]),
        "bytes":__import__("math").prod(port["ne"])*4,"category":category}
        for category,declared in (("input",inputs),("output",outputs)) for name,port in declared.items()}
    if case["kind"] in ("attention","block"):
        length = case["visible_history"]+case["tokens"]
        head_dim, kv_heads = case["config"]["head_dim"], case["config"]["kv_heads"]
        for name,shape in (("key_cache",[head_dim,length,kv_heads]),
                           ("value_cache",[length,head_dim,kv_heads])):
            port_accounting[name] = {"dtype":"f16","shape":shape,
                "bytes":mem["kv_cache_bytes"]//2,"category":"kv_cache","ne":shape+[1]}
    timing = 3000 if case["kind"] == "block" else 1000
    result = {"schema":"ggml_dense_layer_probe/1","kind":"dense_layer","contract":CONTRACT,
        "stage":case["kind"],"backend":"CPU","batch":1,"config":case["config"],
        "q":case["tokens"],"history":case["visible_history"],"activation_dtype":"f32","kv_dtype":"f16",
        "implementation_variant":VARIANTS[case["kind"]],"included":operations(case["config"],case["kind"]),
        "excluded":list(EXCLUDED),"timing":"steady_clock_graph_compute_including_sync",
        "evict_bytes":0,"residency_regime":"resident_warm_cache_shared_across_samples",
        "scope":"resident_repeated_complete_declared_decoder_block","setup_compute_repetitions":1,
        "threads":2,"threads_requested":2,"threads_applied":2,"warmup":2,"samples":3,"seed":17,
        "upstream_setup_compute_repetitions":0 if case["kind"] == "block" else 1,
        "thread_management":"persistent_ggml_threadpool","input_lineage":"same_seed_complete_upstream_preparation_outside_timing",
        "position_dtype":"i32","port_accounting":port_accounting,
        "all_outputs_finite":True,"reference_check":{"passed":True},
        "kv_append_check":{"passed":True,"append_exact":True,"history_unchanged":True}
            if case["kind"] in ("attention","block") else None,
        "stage_port_digests":{port:"0123456789abcdef" for port in mem["output_ports_bytes"]},
        "port_digest_algorithm":"fnv1a64_native_tensor_bytes",
        "input_port_digests":{port:"0123456789abcdef" for port in mem["input_ports_bytes"]},
        "wall_ns":[timing,timing+1,timing-1],"workspace_bytes":128,
        "workspace_source":"ggml_gallocr_internal_buffer_plus_cpu_graph_plan",
        "cpu_cplan_work_size_bytes":32,
        "memory_accounting":{"resident_logical_bytes":mem["resident_logical_bytes"],
            "resident_allocated_bytes":mem["resident_logical_bytes"]+32,
            "internal_allocated_buffer_bytes":96,"cpu_cplan_work_size_bytes":32,"workspace_bytes":128,
            "workspace_bound":"conservative_allocated_capacity",
            "workspace_scope":"internal_reusable_allocator_capacity_including_alignment_plus_cpu_cplan_scratch_excludes_resident_tensors_host_reference_and_runtime_overheads"},**mem}
    result.update(changes)
    return result


class DenseRunnerTests(unittest.TestCase):
    def run_fake(self, payload, *, selected=None, exitcode=0, failure=None):
        selected = selected or case()
        with tempfile.TemporaryDirectory(prefix="dense-runner-test-") as folder:
            root = Path(folder)
            binary = root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
            binary.write_bytes(b"synthetic probe never executed")
            path = root / "suite.json"
            path.write_text(json.dumps(suite([selected])),encoding="utf-8",newline="\n")
            out = root / "output"
            text = payload if isinstance(payload,str) else json.dumps(payload)
            child = subprocess.CompletedProcess([],exitcode,text,"synthetic diagnostic")
            arguments = ["--suite",str(path),"--native-bin",str(root),"--output",str(out),
                         "--threads","2","--samples","3","--warmup","2"]
            with (mock.patch.object(runner,"ROOT",root),
                  mock.patch.object(runner,"environment",return_value={"mock":True}),
                  mock.patch.object(runner.subprocess,"run",return_value=child,side_effect=failure) as probe,
                  contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO())):
                code = runner.main(arguments)
            self.assertEqual(probe.call_count,1)
            self.assertFalse((root/".measurement.lock").exists())
            return {"code":code,"rows":json.loads((out/"measurements.json").read_text()),
                    "raw":json.loads((out/"raw/001.json").read_text()),
                    "files":{str(p.relative_to(out)):p.read_bytes() for p in out.rglob("*") if p.is_file()}}

    def test_complete_config_and_raw_samples_survive(self):
        for kind in ("projection","attention","dense_ffn","block"):
            with self.subTest(kind=kind):
                c = case(kind)
                result = self.run_fake(native(c),selected=c)
                self.assertEqual(result["code"],0)
                row = result["rows"][0]
                self.assertEqual(json.loads(result["raw"]["stdout"])["wall_ns"],row["native"]["wall_ns"])
                self.assertTrue(row["signature"]["native_binary_sha256"])
                self.assertTrue(row["eligibility"]["complete_declared_block_variant"])

    def test_materialized_port_shape_dtype_and_category_must_match(self):
        for key,value in (("dtype","f16"),("shape",[4,3,4]),("category","output"),("bytes",1)):
            c = case("attention")
            bad = copy.deepcopy(native(c)["port_accounting"])
            bad["hidden"][key] = value
            with self.subTest(key=key):
                self.assertEqual(self.run_fake(native(c,port_accounting=bad))["code"],1)

    def test_unknown_workspace_stays_unknown(self):
        result = self.run_fake(native(case(),workspace_bytes=None))
        self.assertEqual(result["code"],0)
        self.assertIsNone(result["rows"][0]["native"]["workspace_bytes"])
        self.assertFalse(result["rows"][0]["eligibility"]["workspace_known"])

    def test_native_binding_rejects_different_graph_or_protocol(self):
        for key,value in (("stage","block"),("history",2),("q",1),("batch",True),
                          ("contract","unknown"),("scope","partial"),("setup_compute_repetitions",0),
                          ("threads_applied",1),("samples",4),("included",[]),("all_outputs_finite",False)):
            with self.subTest(key=key):
                result = self.run_fake(native(case(),**{key:value}))
                self.assertEqual(result["code"],1)
                self.assertEqual(json.loads(result["raw"]["stdout"])[key],value)

    def test_native_rope_and_dtype_mismatch_rejected(self):
        for key,value in (("rope_base",20000),("norm_epsilon",1e-4),("qk_norm","none")):
            with self.subTest(key=key):
                changed = config(); changed[key] = value
                self.assertEqual(self.run_fake(native(case(),config=changed))["code"],1)

    def test_numeric_workspace_requires_allocator_evidence(self):
        self.assertEqual(self.run_fake(native(case(),workspace_bytes=0))["code"],1)
        self.assertEqual(self.run_fake(native(case(),workspace_source="unknown"))["code"],1)

    def test_missing_port_digest_and_kv_history_check_rejected(self):
        for changes in ({"stage_port_digests":{}},{"kv_append_check":{"passed":True,"append_exact":True,"history_unchanged":False}}):
            with self.subTest(changes=changes):
                self.assertEqual(self.run_fake(native(case(),**changes))["code"],1)

    def test_missing_or_malformed_port_digests_rejected(self):
        c = case("projection")
        for key in ("input_port_digests", "stage_port_digests"):
            for bad in ({}, {"undeclared":"0123456789abcdef"},
                        {port:"NOT_A_DIGEST" for port in native(c)[key]},
                        {port:"0123456789ABCDEF" for port in native(c)[key]}):
                with self.subTest(key=key,bad=bad):
                    self.assertEqual(self.run_fake(native(c,**{key:bad}),selected=c)["code"],1)

    def test_failures_and_timeouts_preserve_stdout(self):
        result = self.run_fake("partial output",exitcode=7)
        self.assertEqual(result["code"],1)
        self.assertEqual(result["raw"]["stdout"],"partial output")
        result = self.run_fake({},failure=subprocess.TimeoutExpired([],1,output=b"partial",stderr=b"diagnostic"))
        self.assertEqual(result["raw"]["stdout"],"partial")
        self.assertTrue(result["raw"]["timed_out"])

    def test_lf_files_and_no_output_overwrite(self):
        result = self.run_fake(native(case()))
        for contents in result["files"].values():
            self.assertNotIn(b"\r\n",contents)
            self.assertTrue(contents.endswith(b"\n"))

    def test_shared_lock_is_not_removed_by_rejected_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
            binary.write_bytes(b"mock")
            path = root / "suite.json"; path.write_text(json.dumps(suite([case()])))
            lock = root / ".measurement.lock"; lock.write_text("another process")
            with (mock.patch.object(runner,"ROOT",root),mock.patch.object(runner.subprocess,"run") as child,
                  contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit)):
                runner.main(["--suite",str(path),"--native-bin",str(root),"--threads","2"])
            child.assert_not_called()
            self.assertEqual(lock.read_text(),"another process")


class ThreadPlacementTests(unittest.TestCase):
    """Without an explicit mask the OS picks the workers' logical CPUs.

    On a 16-core/32-thread part that can be two SMT siblings of one core for
    one case and two cores on different CCDs for the next, which moves the
    median of a fixed shape by up to a factor of two between processes.
    """

    def test_no_mask_is_recorded_as_no_request(self):
        self.assertIsNone(runner.parse_cpu_ids(None, 2))

    def test_a_mask_needs_exactly_one_cpu_per_thread(self):
        self.assertEqual(runner.parse_cpu_ids("0,2", 2), [0, 2])
        with self.assertRaisesRegex(ValueError, "every worker"):
            runner.parse_cpu_ids("0", 2)
        with self.assertRaisesRegex(ValueError, "every worker"):
            runner.parse_cpu_ids("0,2,4", 2)

    def test_duplicate_and_malformed_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeat"):
            runner.parse_cpu_ids("0,0", 2)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            runner.parse_cpu_ids("0,x", 2)
        with self.assertRaisesRegex(ValueError, "processor group"):
            runner.parse_cpu_ids("0,64", 2)

    def test_the_mask_reaches_the_probe_sorted(self):
        command = runner.native_command(
            case(), Path("probe-dir"),
            SimpleNamespace(warmup=2, samples=3, seed=17,
                            cpu_ids=runner.parse_cpu_ids("2,0", 2)))
        self.assertEqual(command[command.index("--cpu-mask") + 1], "0,2")

    def test_an_absent_mask_leaves_the_os_scheduler_in_charge(self):
        command = runner.native_command(
            case(), Path("probe-dir"), SimpleNamespace(warmup=2, samples=3, seed=17))
        self.assertNotIn("--cpu-mask", command)


class DeclaredPlacementTests(unittest.TestCase):
    """A pinned mask is an identity, not a launch detail.

    The same two threads on the two SMT siblings of one physical core and on two
    cores of different CCDs price the same declared block differently (1.30x on
    the machine that produced these tables), so a price measured under one
    placement must never answer to a signature produced under another. The mask
    is emitted only when a run declared it, which is what keeps every table
    measured before the dimension existed byte-identical.
    """

    def test_an_undeclared_mask_keeps_the_historical_signature(self):
        args = SimpleNamespace(warmup=2, samples=3, seed=17)
        self.assertIsNone(runner.declared_cpu_ids(args))
        self.assertNotIn("cpu_mask",
                         runner.measurement_signature(case(), "00")["physical_signature"])

    def test_a_declared_mask_reaches_the_signature(self):
        args = SimpleNamespace(warmup=2, samples=3, seed=17, cpu_ids=[2, 4])
        self.assertEqual(runner.declared_cpu_ids(args), [2, 4])
        signature = runner.measurement_signature(
            case(), "00", cpu_ids=runner.declared_cpu_ids(args))["physical_signature"]
        self.assertEqual(signature["cpu_mask"], [2, 4])
        self.assertEqual(signature["thread_placement"], "explicit_strict_cpu_mask")

    def test_a_declared_mask_differs_from_the_undeclared_one_in_two_keys(self):
        free = runner.measurement_signature(case(), "00")["physical_signature"]
        pinned = runner.measurement_signature(case(), "00", cpu_ids=[2, 4])["physical_signature"]
        self.assertEqual(set(pinned) - set(free), {"cpu_mask", "thread_placement"})
        self.assertEqual({key: value for key, value in pinned.items()
                          if key not in ("cpu_mask", "thread_placement")}, free)

    def test_two_placements_differ_in_exactly_the_mask(self):
        first = runner.measurement_signature(case(), "00", cpu_ids=[2, 4])
        second = runner.measurement_signature(case(), "00", cpu_ids=[2, 6])
        self.assertNotEqual(first["physical_contract_sha256"], second["physical_contract_sha256"])
        self.assertEqual({key: value for key, value in second["physical_signature"].items()
                          if key != "cpu_mask"},
                         {key: value for key, value in first["physical_signature"].items()
                          if key != "cpu_mask"})
        self.assertNotEqual(first["physical_signature"]["cpu_mask"],
                            second["physical_signature"]["cpu_mask"])

    def test_a_mask_that_could_not_have_been_applied_is_refused(self):
        for mask in ([2], [2, 4, 6], [2, 2], [4, 2], [0, 64]):
            with self.subTest(mask=mask), self.assertRaises(SpecError):
                runner.measurement_signature(case(), "00", cpu_ids=mask)
        with self.assertRaises(SpecError):
            runner.measurement_signature(case(), "00", cpu_ids="2,4")


class DeclaredResidencyTests(unittest.TestCase):
    """The resident and cold protocols must never share a price or a signature.

    Consecutive samples under the resident protocol reuse whatever the previous
    sample left in cache; under the streaming protocol each sample starts with
    its operands evicted. The same geometry therefore has two different costs,
    and the only defence against mixing them is that neither the command nor the
    signature can be produced by the other regime's declaration.
    """

    def test_the_default_regime_keeps_the_historical_command_and_signature(self):
        args = SimpleNamespace(warmup=2, samples=3, seed=17)
        command = runner.native_command(case(), Path("probe-dir"), args)
        self.assertNotIn("--evict-bytes", command)
        self.assertNotIn("residency_regime",
                         runner.measurement_signature(case(), "00")["physical_signature"])

    def test_a_declared_eviction_reaches_the_probe_and_the_signature(self):
        args = SimpleNamespace(warmup=2, samples=3, seed=17, evict_bytes=1 << 28)
        command = runner.native_command(case(), Path("probe-dir"), args)
        self.assertEqual(command[command.index("--evict-bytes") + 1], str(1 << 28))
        signature = runner.measurement_signature(case(), "00",
                                                 residency=runner.declared_residency(args))
        self.assertEqual(signature["physical_signature"]["residency_regime"],
                         "streamed_cold_cache_before_each_sample")

    def test_the_two_regimes_differ_in_exactly_one_signature_key(self):
        warm = runner.measurement_signature(case(), "00")["physical_signature"]
        cold = runner.measurement_signature(case(), "00", residency="streaming")["physical_signature"]
        self.assertEqual(set(cold) - set(warm), {"residency_regime"})
        self.assertEqual({key: value for key, value in cold.items() if key != "residency_regime"},
                         warm)

    def test_a_native_record_that_contradicts_the_declared_regime_is_rejected(self):
        declared = SimpleNamespace(warmup=2, samples=3, seed=17, evict_bytes=1 << 28)
        c = case()
        cold = native(c, evict_bytes=1 << 28,
                      residency_regime="streamed_cold_cache_before_each_sample",
                      timing="steady_clock_graph_compute_including_sync_after_declared_cache_eviction")
        runner.validate_native_binding(c, cold, declared)
        for key, value in (("evict_bytes", 0),
                           ("residency_regime", "resident_warm_cache_shared_across_samples"),
                           ("timing", "steady_clock_graph_compute_including_sync")):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"native {key}"):
                    runner.validate_native_binding(c, {**cold, key: value}, declared)
        # A resident invocation must equally refuse a record that claims to have
        # been measured cold: the mislabelling can happen in either direction.
        with self.assertRaisesRegex(ValueError, "native evict_bytes"):
            runner.validate_native_binding(
                c, cold, SimpleNamespace(warmup=2, samples=3, seed=17))


class ProcessScopeTests(unittest.TestCase):
    """One process per case and one process per table are different protocols.

    Every entry of a price table inherits whatever machine state its process
    began with. Measuring the table inside one process keeps that state common
    to every entry, which is the only way the entry-to-entry ratios can be free
    of it -- and the only way to tell which protocol produced a price is the
    signature, so the scope must reach it in exactly one direction.
    """

    def test_the_default_scope_keeps_the_historical_signature(self):
        args = SimpleNamespace(warmup=2, samples=3, seed=17)
        self.assertIsNone(runner.declared_process_scope(args))
        self.assertNotIn("measurement_process_scope",
                         runner.measurement_signature(case(), "00")["physical_signature"])

    def test_the_two_scopes_differ_in_exactly_one_signature_key(self):
        one_each = runner.measurement_signature(case(), "00")["physical_signature"]
        one_table = runner.measurement_signature(case(), "00", process_scope="per_suite")["physical_signature"]
        self.assertEqual(set(one_table) - set(one_each), {"measurement_process_scope"})
        self.assertEqual({key: value for key, value in one_table.items()
                          if key != "measurement_process_scope"}, one_each)

    def test_the_manifest_declaration_maps_back_to_the_signature_argument(self):
        from llm_infer_model.m2.dense_contract import PROCESS_SCOPES
        self.assertIsNone(runner.dense_contract.declared_process_scope(None))
        self.assertIsNone(runner.dense_contract.declared_process_scope(PROCESS_SCOPES["per_case"]))
        self.assertEqual(runner.dense_contract.declared_process_scope(PROCESS_SCOPES["per_suite"]),
                         "per_suite")
        with self.assertRaises(SpecError):
            runner.dense_contract.declared_process_scope("one_probe_process_per_suite")

    class FakeProbe:
        """A stand-in for the probe's --cases-from-stdin loop, one line per case."""

        def __init__(self, cases, *, skip=None, die_after=None):
            self.cases = cases
            self.pending = []
            self.written = []
            self.stdin = self.stdout = self
            self.skip = skip
            self.die_after = die_after
            self.closed = False
            self.killed = False

        def write(self, text):
            self.written.append(text)
            if self.die_after is not None and len(self.written) > self.die_after:
                return
            args = text.rstrip("\n").split("\t")
            stage = args[args.index("--stage") + 1]
            matched = [c for c in self.cases if c["kind"] == stage]
            index = len(self.written) - 1
            if index == self.skip:
                self.pending.append(f"#case {index} 1\n")
            else:
                self.pending.append(json.dumps(native(matched[0])) + "\n")
                self.pending.append(f"#case {index} 0\n")

        def flush(self): pass

        def close(self): self.closed = True

        def wait(self): return 0

        def kill(self): self.killed = True

        def readline(self):
            return self.pending.pop(0) if self.pending else ""

    def run_scope(self, scope, cases, *, skip=None, die_after=None, extra=()):
        with tempfile.TemporaryDirectory(prefix="dense-scope-test-") as folder:
            root = Path(folder)
            binary = root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
            binary.write_bytes(b"synthetic probe never executed")
            path = root / "suite.json"
            path.write_text(json.dumps(suite(cases)), encoding="utf-8", newline="\n")
            out = root / "output"
            probe = self.FakeProbe(cases, skip=skip, die_after=die_after)
            arguments = ["--suite", str(path), "--native-bin", str(root), "--output", str(out),
                         "--threads", "2", "--samples", "3", "--warmup", "2",
                         "--process-scope", scope, "--seed", "17", *extra]
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(runner, "ROOT", root))
                stack.enter_context(mock.patch.object(runner, "environment", return_value={"mock": True}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                popen = None
                if scope == "per_case":
                    def fake_run(command, **kwargs):
                        stage = command[command.index("--stage") + 1]
                        chosen = [c for c in cases if c["kind"] == stage][0]
                        return subprocess.CompletedProcess(command, 0, json.dumps(native(chosen)), "")
                    stack.enter_context(mock.patch.object(runner.subprocess, "run", side_effect=fake_run))
                else:
                    popen = stack.enter_context(
                        mock.patch.object(runner.subprocess, "Popen", return_value=probe))
                code = runner.main(arguments)
            self.assertFalse((root / ".measurement.lock").exists())
            rows = json.loads((out / "measurements.json").read_text())
            manifest = json.loads((out / "manifest.json").read_text())
            raws = {p.name: json.loads(p.read_text()) for p in (out / "raw").glob("*.json")}
            return {"code": code, "rows": rows, "manifest": manifest, "raws": raws,
                    "probe": probe, "popen": popen, "out": out}

    def test_one_process_receives_every_case_verbatim(self):
        cases = [case("attention"), case("dense_ffn")]
        result = self.run_scope("per_suite", cases)
        self.assertEqual(result["code"], 0)
        self.assertEqual(result["popen"].call_count, 1)
        # Each line is exactly the command line a standalone invocation would
        # have used, minus argv[0]: the process is the only thing that is shared.
        standalone = [runner.native_command(c, result["out"], SimpleNamespace(
            warmup=2, samples=3, seed=17))[1:] for c in cases]
        self.assertEqual([line.rstrip("\n").split("\t") for line in result["probe"].written],
                         standalone)
        self.assertEqual([row["status"] for row in result["rows"]], ["measured", "measured"])
        self.assertEqual([row["case"]["kind"] for row in result["rows"]],
                         sorted(row["case"]["kind"] for row in result["rows"]))

    def test_every_case_keeps_its_own_record_and_signature(self):
        cases = [case("attention"), case("dense_ffn")]
        result = self.run_scope("per_suite", cases)
        for name, row in zip(("001.json", "002.json"), result["rows"]):
            raw = result["raws"][name]
            self.assertEqual(json.loads(raw["stdout"])["stage"], row["case"]["kind"])
            self.assertEqual(raw["process_group"], 1)
            self.assertEqual(row["raw_output"], f"raw/{name}")
            self.assertEqual(row["signature"]["physical_signature"]["measurement_process_scope"],
                             "one_probe_process_per_suite_in_declared_case_order")
            self.assertEqual(raw["command"], runner.native_command(
                row["case"], result["out"].parent, SimpleNamespace(warmup=2, samples=3, seed=17)))
        self.assertEqual(result["raws"]["001.json"]["stderr"], "")
        self.assertIn("suite-001.stderr.txt", result["raws"]["001.json"]["stderr_file"])

    def test_the_manifest_declares_the_group_and_the_whole_process_budget(self):
        cases = [case("attention"), case("dense_ffn")]
        protocol = self.run_scope("per_suite", cases)["manifest"]["protocol"]
        self.assertEqual(protocol["process_scope"],
                         "one_probe_process_per_suite_in_declared_case_order")
        self.assertEqual(len(protocol["process_groups"]), 1)
        group = protocol["process_groups"][0]
        self.assertEqual(sorted(group["case_ids"]), sorted(c["id"] for c in cases))
        self.assertEqual(protocol["per_process_timeout_s"], 180 * len(cases))

    def test_the_default_scope_runs_one_process_and_declares_no_group(self):
        cases = [case("attention"), case("dense_ffn")]
        result = self.run_scope("per_case", cases)
        self.assertIsNone(result["popen"])
        self.assertEqual(len(result["raws"]), len(cases))
        protocol = result["manifest"]["protocol"]
        self.assertEqual(protocol["process_scope"], "one_probe_process_per_case")
        self.assertEqual(protocol["process_groups"], [])
        for row in result["rows"]:
            self.assertNotIn("measurement_process_scope", row["signature"]["physical_signature"])

    def test_a_case_without_a_record_fails_and_never_borrows_the_next_one(self):
        cases = [case("attention"), case("dense_ffn")]
        result = self.run_scope("per_suite", cases, skip=1)
        self.assertEqual(result["code"], 1)
        self.assertEqual([row["status"] for row in result["rows"]], ["measured", "failed"])
        self.assertEqual(json.loads(result["raws"]["001.json"]["stdout"])["stage"], "attention")
        self.assertEqual(result["raws"]["002.json"]["stdout"], "")
        self.assertEqual(result["raws"]["002.json"]["returncode"], 1)

    def test_a_process_that_dies_marks_the_rest_as_not_run(self):
        cases = [case("attention"), case("dense_ffn")]
        result = self.run_scope("per_suite", cases, die_after=0, extra=("--keep-going",))
        self.assertEqual([row["status"] for row in result["rows"]], ["failed", "failed"])
        self.assertIn("ended early", result["rows"][0]["error"])
        self.assertIn("not run", result["rows"][1]["error"])


class DenseExportTests(unittest.TestCase):
    # A subclass that sets this runs every export test under the other residency
    # regime, so the cold path is covered by the same assertions as the resident one.
    EVICT_BYTES = 0
    # A subclass that sets this runs every export test with the table declared
    # as measured inside one process, so the scope reaches the same assertions.
    PROCESS_SCOPE = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dense-export-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); (self.root/"raw").mkdir()
        document = layer_document()
        if self.EVICT_BYTES:
            document["physical"]["residency"] = "streaming"
        if self.PROCESS_SCOPE:
            document["physical"]["process_scope"] = self.PROCESS_SCOPE
        self.layer = parse_layer_document(document)
        self.regime = "streaming" if self.EVICT_BYTES else None
        self.cases = [case(kind) for kind in ("projection","attention","dense_ffn","block")]
        self.binary = self.root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
        self.digest = hashlib.sha256(b"fake executable never run").hexdigest()
        self.manifest = {"schema_version":"dense_layer_measurement_profile_v1","contract":CONTRACT,
            "profile_id":"synthetic-evidence","suite":suite(self.cases),
            "protocol":{"backend":"CPU","batch":1,"threads":2,"warmup":2,"samples":3,"seed":17,
                "evict_bytes":self.EVICT_BYTES,
                "sequential_execution":True,"graphs_per_sample":1,"setup_compute_repetitions":1,"measured_graph_untimed_compute_repetitions":3,
                "upstream_setup_compute_repetitions_by_kind":{kind:0 if kind=="block" else 1 for kind in ("projection","attention","dense_ffn","block")},
                "total_untimed_compute_repetitions_by_kind":{kind:3 if kind=="block" else 4 for kind in ("projection","attention","dense_ffn","block")}},
            "software":{"runtime_sha256":{str(self.binary):self.digest}}}
        if self.EVICT_BYTES:
            self.manifest["protocol"]["residency_regime"] = "streamed_cold_cache_before_each_sample"
        if self.PROCESS_SCOPE:
            self.manifest["protocol"]["process_scope"] = PROCESS_SCOPES[self.PROCESS_SCOPE]
        self.args = SimpleNamespace(warmup=2,samples=3,seed=17,evict_bytes=self.EVICT_BYTES)
        self.rows = []
        for index,c in enumerate(self.cases):
            n = native(c)
            if self.EVICT_BYTES:
                n.update(evict_bytes=self.EVICT_BYTES,
                         residency_regime="streamed_cold_cache_before_each_sample",
                         timing="steady_clock_graph_compute_including_sync_after_declared_cache_eviction")
            summary = summarize_samples(n["wall_ns"])
            self.rows.append({"case_id":c["id"],"case":c,"status":"measured",
                "signature":runner.measurement_signature(c,self.digest,residency=self.regime,
                                                         process_scope=self.PROCESS_SCOPE),
                "native":{**n,"expected_samples":3,"summary":summary},
                "summary":summary,"quality_flags":[],"raw_output":f"raw/{index+1:03d}.json"})
            self.write_raw(index)

    def write_raw(self,index):
        row = self.rows[index]
        n = {key:value for key,value in row["native"].items() if key not in ("summary","expected_samples")}
        raw = {"case":row["case"],"command":runner.native_command(row["case"],self.binary.parent,self.args),
               "returncode":0,"stdout":json.dumps(n),"stderr":"synthetic not executed"}
        (self.root/row["raw_output"]).write_text(json.dumps(raw),encoding="utf-8",newline="\n")

    def pin(self,mask,*,placement="explicit_strict_cpu_mask",verified=True,declared=False):
        """Record a controlled thread placement exactly as the runner would.

        ``declared`` is what a run does now that the mask is part of the physical
        signature: the manifest says so and every record carries the mask. Left
        False it stays a command detail, which is how every campaign measured
        before the dimension existed recorded it.
        """
        self.args = SimpleNamespace(warmup=2,samples=3,seed=17,cpu_ids=list(mask),
                                    evict_bytes=self.EVICT_BYTES)
        self.manifest["protocol"]["cpu_mask"] = list(mask)
        self.manifest["protocol"]["thread_placement"] = placement
        if declared:
            self.manifest["protocol"]["cpu_mask_in_physical_signature"] = True
            for row in self.rows:
                row["signature"] = runner.measurement_signature(
                    row["case"],self.digest,residency=self.regime,
                    process_scope=self.PROCESS_SCOPE,cpu_ids=list(mask))
        for index,row in enumerate(self.rows):
            row["native"]["cpu_affinity"] = {"strict_cpu":True,"verified":verified,
                                             "verified_before_and_after_timing":verified,
                                             "binding_source":"GGML_threadpool_strict_cpu_mask"}
            self.write_raw(index)

    def declare_layer_mask(self,mask):
        """Give the layer document the same placement the table was measured at."""
        self.layer = dataclasses.replace(
            self.layer,physical=dataclasses.replace(self.layer.physical,
                                                    cpu_ids=tuple(mask) if mask else None))

    def modify(self,index,**changes):
        row = self.rows[index]; row["native"].update(changes)
        row["summary"] = row["native"]["summary"] = summarize_samples(row["native"]["wall_ns"])
        self.write_raw(index)

    def export(self,**kwargs):
        return exporter.export_cost_table(self.layer,self.rows,raw_directory=self.root,manifest=self.manifest,**kwargs)

    def test_exports_three_stages_joint_block_only_for_comparison(self):
        costs,report = self.export()
        self.assertEqual(len(costs["entries"]),3)
        self.assertNotIn("block",{entry["kind"] for entry in costs["entries"]})
        self.assertEqual(report["composition"][0]["absolute_relative_error"],0)
        self.assertIsNone(report["composition"][0]["within_explicit_tolerance"])
        self.assertFalse(report["calibration_claimed"])
        self.assertTrue(report["complete_declared_block_variant"])
        self.assertFalse(costs["validation_only"])

    def test_controlled_placement_is_reconstructed_and_recorded(self):
        self.pin([2,4])
        costs,report = self.export()
        self.assertFalse(costs["validation_only"])
        self.assertEqual(report["source_protocol"]["cpu_mask"],[2,4])
        self.assertEqual(report["source_protocol"]["thread_placement"],"explicit_strict_cpu_mask")

    def test_recorded_mask_must_appear_in_the_raw_command(self):
        self.pin([2,4])
        self.args = SimpleNamespace(warmup=2,samples=3,seed=17)
        for index in range(len(self.rows)): self.write_raw(index)
        with self.assertRaisesRegex(ValueError,"raw native command"):
            self.export()

    def test_a_mask_must_name_one_cpu_per_thread(self):
        for mask in ([2],[2,4,6],[2,2],[4,2]):
            with self.subTest(mask=mask):
                self.pin(mask)
                with self.assertRaisesRegex(ValueError,"cpu_mask"):
                    self.export()

    def test_placement_label_must_agree_with_the_mask(self):
        self.pin([2,4],placement="os_scheduler_choice_of_logical_cpus")
        with self.assertRaisesRegex(ValueError,"thread_placement disagrees"):
            self.export()

    def test_an_unverified_affinity_cannot_back_a_requested_mask(self):
        self.pin([2,4],verified=False)
        with self.assertRaisesRegex(ValueError,"not verified"):
            self.export()

    def test_a_mask_recorded_before_the_dimension_still_exports(self):
        # This is exactly what the archived campaigns recorded: a pinned command,
        # no declaration that the placement was part of the identity. Their
        # tables must stay reproducible from their own evidence.
        self.pin([2,4])
        costs,report = self.export()
        self.assertNotIn("cpu_mask_in_physical_signature",report["source_protocol"])
        for entry in costs["entries"]:
            self.assertNotIn("cpu_mask",entry["physical_signature"])

    def test_a_declared_mask_reaches_every_entry(self):
        self.declare_layer_mask([2,4])
        self.pin([2,4],declared=True)
        costs,report = self.export()
        self.assertTrue(report["source_protocol"]["cpu_mask_in_physical_signature"])
        for entry in costs["entries"]:
            self.assertEqual(entry["physical_signature"]["cpu_mask"],[2,4])
            self.assertEqual(entry["physical_signature"]["thread_placement"],
                             "explicit_strict_cpu_mask")

    def test_records_that_never_declared_the_mask_cannot_be_exported_as_declared(self):
        self.declare_layer_mask([2,4])
        self.pin([2,4],declared=True)
        for row in self.rows:
            row["signature"] = runner.measurement_signature(
                row["case"],self.digest,residency=self.regime,process_scope=self.PROCESS_SCOPE)
        with self.assertRaisesRegex(ValueError,"regenerated measurement signature"):
            self.export()

    def test_a_layer_declaring_another_placement_cannot_claim_the_table(self):
        self.pin([2,4],declared=True)
        self.declare_layer_mask([2,6])
        with self.assertRaisesRegex(ValueError,"model physical signature"):
            self.export()

    def test_a_layer_declaring_no_placement_cannot_claim_a_pinned_table(self):
        self.pin([2,4],declared=True)
        with self.assertRaisesRegex(ValueError,"model physical signature"):
            self.export()

    def test_a_manifest_that_claims_the_dimension_without_a_mask_is_refused(self):
        self.manifest["protocol"]["cpu_mask_in_physical_signature"] = True
        with self.assertRaisesRegex(ValueError,"records none"):
            self.export()

    def test_a_manifest_that_denies_the_dimension_is_refused(self):
        self.pin([2,4])
        self.manifest["protocol"]["cpu_mask_in_physical_signature"] = False
        with self.assertRaisesRegex(ValueError,"must be true"):
            self.export()

    def test_export_does_not_change_evidence(self):
        original = copy.deepcopy((self.rows,self.manifest))
        self.export()
        self.assertEqual(original,(self.rows,self.manifest))

    def test_missing_stage_or_unobserved_history_never_filled(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError,"complete exact shape group"):
            self.export()

    def test_stage_output_digest_must_equal_joint_port(self):
        self.modify(0,stage_port_digests={key:"fedcba9876543210" for key in self.rows[0]["native"]["stage_port_digests"]})
        with self.assertRaisesRegex(ValueError,"materialized port"):
            self.export()

    def test_stage_input_lineage_must_match_predecessor_materialized_ports(self):
        for index,port in ((0,"hidden"),(1,"q"),(1,"hidden"),(2,"attention_hidden")):
            with self.subTest(index=index,port=port):
                original = copy.deepcopy(self.rows)
                digests = dict(self.rows[index]["native"]["input_port_digests"])
                digests[port] = "fedcba9876543210"
                self.modify(index,input_port_digests=digests)
                with self.assertRaisesRegex(ValueError,"input"):
                    self.export()
                self.rows = original
                self.write_raw(index)

    def test_error_is_reported_and_only_explicit_tolerance_gates(self):
        self.modify(3,wall_ns=[1000,1001,999])
        costs,report = self.export()
        self.assertEqual(report["composition"][0]["absolute_relative_error"],2)
        self.assertFalse(costs["validation_only"])
        with self.assertRaisesRegex(ValueError,"exceeds explicit tolerance"):
            self.export(max_relative_error=0.2)
        costs,report = self.export(max_relative_error=0.2,allow_composition_error=True)
        self.assertTrue(costs["validation_only"])
        self.assertFalse(report["composition"][0]["within_explicit_tolerance"])

    def test_unstable_evidence_is_validation_only_if_explicitly_allowed(self):
        self.modify(0,wall_ns=[100,1000,10000])
        self.rows[0]["quality_flags"]=["high_observed_variability_cv_gt_0.1"]
        with self.assertRaisesRegex(ValueError,"unstable"):
            self.export()
        costs,report = self.export(allow_unstable=True)
        self.assertTrue(costs["validation_only"])
        self.assertTrue(report["validation_only"])

    def test_unknown_workspace_and_failed_records_cannot_be_waived(self):
        self.modify(0,workspace_bytes=None)
        with self.assertRaisesRegex(ValueError,"workspace is unknown"):
            self.export(allow_unstable=True)
        self.rows[0]["status"]="failed"
        with self.assertRaisesRegex(ValueError,"failed"):
            self.export(allow_unstable=True)

    def test_tampered_signature_raw_summary_and_runtime_hash_rejected(self):
        mutations = ("signature","raw","summary","runtime")
        for what in mutations:
            with self.subTest(what=what):
                original = copy.deepcopy((self.rows,self.manifest))
                if what == "signature": self.rows[0]["signature"]["physical_contract_sha256"]="0"*64
                elif what == "raw":
                    p = self.root/self.rows[0]["raw_output"]; raw=json.loads(p.read_text()); raw["stdout"]="{}";p.write_text(json.dumps(raw))
                elif what == "summary": self.rows[0]["summary"]["median_ns"]=1
                else: self.manifest["software"]["runtime_sha256"][str(self.binary)]="0"*64
                with self.assertRaises(ValueError): self.export()
                self.rows,self.manifest=original
                self.write_raw(0)

    def test_raw_path_escape_and_duplicate_shapes_rejected(self):
        self.rows[0]["raw_output"]="../other.json"
        with self.assertRaisesRegex(ValueError,"escapes"):
            self.export()
        self.rows[0]["raw_output"]="raw/001.json"
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.export()

    def test_a_scope_is_a_protocol_not_a_label(self):
        other = "per_case" if self.PROCESS_SCOPE else "per_suite"
        self.manifest["protocol"]["process_scope"] = PROCESS_SCOPES[other]
        with self.assertRaisesRegex(ValueError, "regenerated measurement signature"):
            self.export()

    def test_invalid_tolerance_cannot_be_waived(self):
        for value in (True,float("nan"),float("inf"),-1):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"tolerance"):
                self.export(max_relative_error=value,allow_composition_error=True)


class PerSuiteExportTests(DenseExportTests):
    """The same export assertions, run against a table priced in one process."""

    PROCESS_SCOPE = "per_suite"

    def test_a_per_suite_table_cannot_be_exported_against_a_per_case_layer(self):
        self.layer = parse_layer_document(layer_document())
        with self.assertRaisesRegex(ValueError, "model physical signature"):
            self.export()

    def test_a_per_suite_table_records_its_scope_in_every_entry(self):
        costs,_ = self.export()
        for entry in costs["entries"]:
            self.assertEqual(entry["physical_signature"]["measurement_process_scope"],
                             PROCESS_SCOPES["per_suite"])


class ColdRegimeExportTests(DenseExportTests):
    """The same export assertions, run under the declared cold-cache protocol."""

    EVICT_BYTES = 1 << 27

    def test_a_cold_table_cannot_be_exported_against_a_resident_layer(self):
        self.layer = parse_layer_document(layer_document())
        with self.assertRaisesRegex(ValueError, "model physical signature"):
            self.export()

    def test_a_cold_table_records_its_regime_in_every_entry(self):
        costs,_ = self.export()
        for entry in costs["entries"]:
            self.assertEqual(entry["physical_signature"]["residency_regime"],
                             "streamed_cold_cache_before_each_sample")


class HistoryInvariantPriceTests(unittest.TestCase):
    """Pricing one op three times must not become a depth trend.

    Decode costs 128, 512 and 1024 tokens of history; projection and the
    feed-forward read one token and none of it. Priced per shape, their three
    entries are replicates of one physical task. The declared pooled-median
    policy keeps every entry and its key, replaces the price with the median of
    the replicates, and says so in the entry's signature.
    """

    HISTORIES = (128, 512, 1024)
    # What the probe would read at each depth: the invariant stages only wobble,
    # attention climbs with history as it must.
    PRICES = {"projection": {128: 1000, 512: 1100, 1024: 1050},
              "dense_ffn": {128: 2000, 512: 2100, 1024: 2200},
              "attention": {128: 500, 512: 900, 1024: 1300},
              "block": {128: 3500, 512: 4100, 1024: 4550}}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dense-history-invariant-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); (self.root/"raw").mkdir()
        self.layer = parse_layer_document(layer_document())
        self.cases = [case(kind, phase="decode", tokens=1, visible_history=history,
                           id=f"{kind}-h{history}")
                      for kind in ("projection", "attention", "dense_ffn", "block")
                      for history in self.HISTORIES]
        self.binary = self.root / ("ggml-dense-layer-probe.exe" if os.name == "nt" else "ggml-dense-layer-probe")
        self.digest = hashlib.sha256(b"fake executable never run").hexdigest()
        kinds = tuple(sorted({c["kind"] for c in self.cases}))
        self.manifest = {"schema_version": "dense_layer_measurement_profile_v1", "contract": CONTRACT,
            "profile_id": "synthetic-history-invariant", "suite": suite(self.cases),
            "protocol": {"backend": "CPU", "batch": 1, "threads": 2, "warmup": 2, "samples": 3, "seed": 17,
                "evict_bytes": 0, "sequential_execution": True, "graphs_per_sample": 1,
                "setup_compute_repetitions": 1, "measured_graph_untimed_compute_repetitions": 3,
                "upstream_setup_compute_repetitions_by_kind": {k: 0 if k == "block" else 1 for k in kinds},
                "total_untimed_compute_repetitions_by_kind": {k: 3 if k == "block" else 4 for k in kinds}},
            "software": {"runtime_sha256": {str(self.binary): self.digest}}}
        self.args = SimpleNamespace(warmup=2, samples=3, seed=17, evict_bytes=0)
        self.rows = []
        for index, c in enumerate(self.cases):
            price = self.PRICES[c["kind"]][c["visible_history"]]
            n = native(c, wall_ns=[price, price+1, price-1])
            summary = summarize_samples(n["wall_ns"])
            self.rows.append({"case_id": c["id"], "case": c, "status": "measured",
                "signature": runner.measurement_signature(c, self.digest),
                "native": {**n, "expected_samples": 3, "summary": summary},
                "summary": summary, "quality_flags": [], "raw_output": f"raw/{index+1:03d}.json"})
            self.write_raw(index)

    def write_raw(self, index):
        row = self.rows[index]
        n = {key: value for key, value in row["native"].items() if key not in ("summary", "expected_samples")}
        raw = {"case": row["case"], "command": runner.native_command(row["case"], self.binary.parent, self.args),
               "returncode": 0, "stdout": json.dumps(n), "stderr": "synthetic not executed"}
        (self.root / row["raw_output"]).write_text(json.dumps(raw), encoding="utf-8", newline="\n")

    def export(self, **kwargs):
        return exporter.export_cost_table(self.layer, self.rows, raw_directory=self.root,
                                          manifest=self.manifest, **kwargs)

    def decode(self, costs, kind):
        return sorted((entry for entry in costs["entries"]
                       if entry["kind"] == kind and entry["phase"] == "decode"),
                      key=lambda entry: entry["visible_history"])

    def test_the_default_export_keeps_every_measured_price(self):
        costs, report = self.export()
        for kind in ("projection", "dense_ffn"):
            self.assertEqual([entry["duration_ns"] for entry in self.decode(costs, kind)],
                             [self.PRICES[kind][history] for history in self.HISTORIES])
            for entry in self.decode(costs, kind):
                self.assertNotIn("history_invariant_price_policy", entry["physical_signature"])
        self.assertIsNone(report["history_invariant_price_policy"])

    def test_pooling_replaces_only_the_history_invariant_prices(self):
        costs, report = self.export(price_policy="median_of_history_replicates")
        self.assertEqual([entry["duration_ns"] for entry in self.decode(costs, "projection")],
                         [1050, 1050, 1050])
        self.assertEqual([entry["duration_ns"] for entry in self.decode(costs, "dense_ffn")],
                         [2100, 2100, 2100])
        self.assertEqual([entry["duration_ns"] for entry in self.decode(costs, "attention")],
                         [500, 900, 1300])
        self.assertEqual(report["history_invariant_price_policy"],
                         "one_pooled_median_over_history_replicates")
        self.assertEqual(report["history_invariant_stages"], ["dense_ffn", "projection"])

    def test_a_pooled_price_declares_itself_in_its_signature(self):
        costs, _ = self.export(price_policy="median_of_history_replicates")
        for kind in ("projection", "dense_ffn"):
            for entry in self.decode(costs, kind):
                self.assertEqual(entry["physical_signature"]["history_invariant_price_policy"],
                                 "one_pooled_median_over_history_replicates")
        for entry in self.decode(costs, "attention"):
            self.assertNotIn("history_invariant_price_policy", entry["physical_signature"])

    def test_a_pooled_table_is_a_different_identity_from_the_measured_one(self):
        measured, _ = self.export()
        pooled, _ = self.export(price_policy="median_of_history_replicates")
        before = {entry["physical_signature"]["visible_history"]: entry["physical_signature"]
                  for entry in self.decode(measured, "projection")}
        after = {entry["physical_signature"]["visible_history"]: entry["physical_signature"]
                 for entry in self.decode(pooled, "projection")}
        for history in self.HISTORIES:
            self.assertEqual(set(after[history]) - set(before[history]),
                             {"history_invariant_price_policy"})
            self.assertEqual({k: v for k, v in after[history].items() if k != "history_invariant_price_policy"},
                             before[history])

    def test_the_evidence_keeps_every_reading_it_replaced(self):
        _, report = self.export(price_policy="median_of_history_replicates")
        evidence = {record["case_id"]: record for record in report["entries"]
                    if record["case_id"].startswith("projection")}
        for history in self.HISTORIES:
            record = evidence[f"projection-h{history}"]
            self.assertEqual(record["history_replicate_relative_spread"], 100/1050)
            self.assertEqual([item["duration_ns"] for item in record["history_replicates"]],
                             [1000, 1100, 1050])

    def test_the_composition_claim_is_made_against_the_exported_prices(self):
        _, report = self.export(price_policy="median_of_history_replicates")
        rows = {row["shape"]: row for row in report["composition"]}
        row = rows["decode/B1/q1/hist512"]
        # 1050 (projection) + 2100 (dense_ffn) + 900 (attention) against the
        # joint block's own median; the raw medians would have said 1100 + 2100.
        self.assertEqual(row["predicted_serial_stage_sum_ns"], 1050 + 2100 + 900)
        self.assertEqual(row["stage_prices_from"], "exported_table_durations")

    def test_replicates_spread_past_the_limit_are_not_pooled(self):
        index = [row["case_id"] for row in self.rows].index("dense_ffn-h1024")
        n = native(self.cases[index], wall_ns=[4000, 4001, 3999])
        self.rows[index]["native"].update(n)
        self.rows[index]["native"]["expected_samples"] = 3
        self.rows[index]["summary"] = self.rows[index]["native"]["summary"] = summarize_samples(n["wall_ns"])
        self.write_raw(index)
        with self.assertRaisesRegex(ValueError, "not replicate noise"):
            self.export(price_policy="median_of_history_replicates")

    def test_one_history_cannot_be_pooled_with_itself(self):
        self.rows = [row for row in self.rows if row["case"]["visible_history"] == 512]
        self.cases = [row["case"] for row in self.rows]
        self.manifest["suite"] = suite(self.cases)
        costs, report = self.export(price_policy="median_of_history_replicates")
        self.assertEqual(report["history_invariant_stages"], [])
        self.assertEqual([entry["duration_ns"] for entry in self.decode(costs, "projection")], [1100])
        self.assertEqual([entry["duration_ns"] for entry in self.decode(costs, "dense_ffn")], [2100])


if __name__ == "__main__":
    unittest.main()
