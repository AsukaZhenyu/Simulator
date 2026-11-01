"""Executable, scoped expression witnesses and explicit implementation accounting.

The external paths execute retained native Python interfaces plus a newly written,
counted serial coordinator. They do NOT execute complete external schedulers or
produce a whole-tool ease-of-use ranking. No native/GPU measurements occur.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

SIM = Path(__file__).resolve().parents[2]
ROOT = SIM.parent
HERE = Path(__file__).parent / "capability_witnesses"
OUT = SIM / "build/coordination-metrics-approved-20261030/expression"
EXT = SIM / "build/dac-validation-20261025/e1-external-v2"
for path in (SIM, SIM / "src", SIM / "mapping", SIM / "modeling", HERE,
             EXT / "LLMCompass", EXT / "deps", EXT / "LLMServingSim"):
    sys.path.insert(0, str(path))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location("witness_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stmt_nodes(node):
    """Every Python ast.stmt once, including imports/defs, excluding docstrings."""
    excluded = set()
    for scope in ast.walk(node):
        if isinstance(scope, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = scope.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                excluded.add(id(body[0]))
    return [n for n in ast.walk(node) if isinstance(n, ast.stmt) and id(n) not in excluded]


def leaf_count(obj):
    if isinstance(obj, dict):
        return sum(leaf_count(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(leaf_count(v) for v in obj)
    return 1


def accounting():
    rows = {}
    common_adapter = HERE / "serial_adapter.py"
    adapter_count = len(stmt_nodes(ast.parse(common_adapter.read_text(encoding="utf-8"))))
    for tool in ("ours", "compass", "serving"):
        path = HERE / (tool + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        policy = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "strategy")
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        policy_count = len(stmt_nodes(policy)) + len(imports)
        # The run wrapper contains a native model/interface probe. This probe
        # supplies/checks common mathematics only, never a policy decision.
        # Its entire source is retained and separately counted, not hidden.
        run_count = 0
        native_binding = 0
        if tool != "ours":
            run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
            run_count = 2  # public adapter function definition + coordinator call
            native_binding = len(stmt_nodes(run)) - run_count
        row = dict(tool=tool, policy_statements=policy_count,
                   added_coordination_statements=0 if tool == "ours" else adapter_count + run_count,
                   framework_change_statements=0, strategy_json_leaves=0,
                   excluded_native_common_model_binding_statements=native_binding,
                   raw_file_statements=len(stmt_nodes(tree)),
                   policy_source=str(path.relative_to(ROOT)), policy_sha256=digest(path),
                   shared_coordinator_source=None if tool == "ours" else str(common_adapter.relative_to(ROOT)),
                   shared_coordinator_sha256=None if tool == "ours" else digest(common_adapter),
                   count_unit="Python ast.stmt; defs/imports included; docstrings excluded")
        row["total_scoped_statements"] = row["policy_statements"] + row["added_coordination_statements"]
        rows[tool] = row
    return rows


def freeze():
    if (OUT / "PROTOCOL.json").exists():
        return json.loads((OUT / "PROTOCOL.json").read_text(encoding="utf-8"))
    inputs = []
    for split, hidden, q in (("development", 8, 1), ("holdout", 16, 3)):
        for case in ("dense_gpu", "cpu_attention", "expert_lru"):
            futures = (0, 2) if case == "expert_lru" else (0,)
            for future in futures:
                inputs.append(dict(id=f"{case}-{split}-f{future}", case=case, split=split,
                                   hidden=hidden, q=max(q, 3) if case == "expert_lru" else q,
                                   future=future))
    p = dict(schema="approved-expression-protocol-1", frozen_utc=datetime.now(timezone.utc).isoformat(),
             tasks=inputs, native_measurements=False, sources_modified=False,
             controlled_boundary="Same independently declared macro work, device assignments, operand readiness, terminal DRAM goals, and expert-LRU event sequence. Serial external coordination is not timing-equivalent to the full compiler.",
             no_whole_tool_claim=True, no_absolute_minimum_claim=True,
             input_independence="Common W/H and mathematical DAG from handwritten e1 algebra, not production graph/trace; private future DAG remains evaluator-owned, LRU decisions use current inputs and past resident history only.",
             count_rules=dict(python="each ast.stmt except docstrings; imports/defs count",
                              json="each primitive leaf; no strategy JSON needed here",
                              shared="whole coordinator counted once per tool for this entire three-policy cohort",
                              exclude="common W/H/profile construction, common math DAG, harness, native math probes and independent checks; every excluded native binding is separately counted and retained",
                              include="policy function, required imports, new adapter coordinator and public adapter wrapper",
                              result_boundary="Full original schedulers not executed. Reduction is reportable only for explicitly equivalent scoped obligations; whole-tool reduction is always null."),
             predeclared_checks=["same operation identities", "same device choice for every operation", "terminal DRAM goals", "native entrypoints actually called", "expert group miss/load and LRU victim sequences", "future suffix preserves prefill decisions"],
             native_repos={"LLMCompass": "2e015fd2ee750e6cad8d7152df52551e5b41ef20", "LLMServingSim": "a4053bc1161872420e1e0607cb3409ef659b828e"},
             source_counts_before_execution=accounting(),
             expansion_scope="FFN alias/lifetimes, prefetch and PD documented from previous versioned interface evidence but not included in new three-policy size denominator")
    write(OUT / "PROTOCOL.json", p)
    return p


def build_inputs(row):
    from experiments import dac_validation_e1 as e1
    from experiments.dac_validation_e1_oracle import declared_dag
    from llm_simulator import Device, Hardware, Model, Request, RequestFlow, Simulator
    case = row["case"]
    s = e1.spec("A06" if case == "expert_lru" else "A01", row["hidden"], row["q"], future=row["future"])
    s.update(layers=1, requests=1, decode_steps=3 if case == "expert_lru" else 0,
             attention_refined=False, ffn_refined=False, prefetch=False, handoff=False,
             physical_capacity_bytes=1048576, cache_budget_bytes=12 * row["hidden"] * 2 * row["hidden"])
    graph = declared_dag(s)
    params = dict(layers=1, hidden_size=s["hidden"], intermediate_size=s["intermediate"], heads=2, kv_heads=1)
    model = Model.moe(experts=3, top_k=1, **params) if case == "expert_lru" else Model.dense(**params)
    workload = RequestFlow(requests=(Request("request0", s["q"], s["decode_steps"], 0),),
                           routes=e1.routes_for(s), route_source="model-private declared synthetic trace" if case == "expert_lru" else None)
    hardware = Hardware(Device(1e9, 1e9), Device(2e9, 2e9), 1048576, 1048576, 1e9, 1e9, 1e9)
    costs = e1.WitnessCosts()
    object.__setattr__(costs, "workspace_bytes", 0)
    sim = Simulator(model=model, hardware=hardware, workload=workload, cost_model=costs)
    config = dict(hidden_size=s["hidden"], intermediate_size=s["intermediate"], num_attention_heads=2,
                  num_key_value_heads=1, num_hidden_layers=1, vocab_size=32, num_local_experts=3,
                  num_experts_per_tok=1, head_dim=s["hidden"] // 2, model_type="llama")
    path = OUT / "inputs" / row["id"]
    write(path / "model.json", config)
    write(path / "mathematical-workload.json", graph)
    write(path / "workload.json", s)
    external_hw = {device: SimpleNamespace(compute_module=SimpleNamespace(total_systolic_array_flops=rate,
                                    l2_bandwidth_per_cycle=1e9, clock_freq=1),
                                    io_module=SimpleNamespace(bandwidth=1e9))
                   for device, rate in (("cpu", 1e9), ("gpu", 2e9))}
    external_hw["vram_bytes"] = 1048576
    return s, graph, sim, external_hw, str((path / "model").resolve())


def operations_from_events(events, own):
    return [(e["target_id"], e["device"]) if own else (e["operation"], e["device"])
            for e in events if e["kind"] == "COMPUTE"]


def expert_events(events, own):
    result = []
    for e in events:
        kind = e["kind"]
        tid = (e.get("target_id") or "") if own else (e.get("tensor") or "")
        if ":E" not in tid:
            continue
        if kind == "COPY" and e.get("dst") == "vram":
            result.append(("load", tid))
        if kind == "EVICT":
            result.append(("evict", tid))
    return result


def execute(protocol):
    modules = {name: load(name) for name in ("ours", "compass", "serving")}
    records = []
    for row in protocol["tasks"]:
        s, graph, sim, hw, model_path = build_inputs(row)
        expert_bytes = 12 * s["hidden"] * s["intermediate"]
        parent = OUT / "runs" / row["id"]
        attempt = 1
        while (parent / ("attempt-" + str(attempt))).exists():
            attempt += 1
        run_dir = parent / ("attempt-" + str(attempt))
        with sim.compile(directory=run_dir / "ours-inputs", scheduling=modules["ours"].strategy(row["case"], expert_bytes)) as compiled:
            result = compiled.run()
            own_events = [asdict(e) for e in result.events]
            write(run_dir / "ours-events.json", own_events)
            write(run_dir / "ours-summary.json", result.summary())
            own_ops = operations_from_events(own_events, True)
            expected_ops = set(graph["operations"])
            if not result.success or {oid for oid, _ in own_ops} != expected_ops:
                raise AssertionError("our runtime failed exact independent operation contract")
        outcome = dict(**row, artifact_directory=str(run_dir.relative_to(OUT)), ours_completed=True, operations=len(expected_ops), comparisons={})
        for tool in ("compass", "serving"):
            policy = modules[tool].strategy(row["case"], expert_bytes)
            external = modules[tool].run(graph, policy, hw, model_path)
            write(run_dir / (tool + "-result.json"), external)
            ext_ops = operations_from_events(external["events"], False)
            exact_work_placement = sorted(ext_ops) == sorted(own_ops)
            xloads = expert_events(external["events"], False)
            oloads = expert_events(own_events, True)
            comparison = dict(exact_operation_devices=exact_work_placement,
                              goals_in_dram=len(external["goals_in_dram"]),
                              own_expert_events=oloads, external_expert_events=xloads,
                              exact_expert_sequence=oloads == xloads,
                              native_call_records=len(external["native_calls"]),
                              full_external_scheduler=False,
                              timing_equivalence_claim=False)
            comparison["scoped_obligations_match"] = exact_work_placement and (row["case"] != "expert_lru" or oloads == xloads)
            outcome["comparisons"][tool] = comparison
        write(run_dir / "comparison.json", outcome)
        records.append(outcome)
        print(row["id"], {k: v["scoped_obligations_match"] for k, v in outcome["comparisons"].items()}, flush=True)
    # The stock route result loses identities even for custom disclosed routes.
    route_controls = {str(e): modules["serving"].route_identity_control((e,), 3) for e in (0, 2)}
    write(OUT / "native-routing-identity-control.json", route_controls)
    counts = accounting()
    for tool in ("compass", "serving"):
        match = all(r["comparisons"][tool]["scoped_obligations_match"] for r in records)
        counts[tool]["scoped_equivalence_all_conditions"] = match
        counts[tool]["scoped_reduction_vs_ours_percent"] = (100 * (1 - counts["ours"]["total_scoped_statements"] / counts[tool]["total_scoped_statements"])) if match else None
        counts[tool]["whole_tool_reduction_percent"] = None
    capability = [
        dict(mechanism="Dense single-device", ours="native execution", compass="native operator/math model; explicit serial graph coordinator measured", serving="native sizes/placement/trace interfaces; explicit serial coordinator measured", status="executed scoped fixture; not whole original schedulers"),
        dict(mechanism="CPU Attention + GPU FFN", ours="native declaration", compass="new per-operation CPU/GPU coordination adapter", serving="new CPU compute placement/coordination adapter; native host memory is not CPU compute", status="executed scoped fixture"),
        dict(mechanism="causal expert identity + bounded LRU", ours="native expert_group LRU", compass="new identity/residency/LRU coordinator", serving="new identity/residency/LRU coordinator; stock route count loss directly executed", status="executed scoped fixture; history private to model until current work ready"),
        dict(mechanism="materialized FFN shards vs shared parent views", ours="previous full public-path evidence", compass="previous native shape witness + added allocation/view relations", serving="manual physical bytes accepted in previous exact parser witness; new allocation/lifetime rules needed for automatic derivation", status="previous evidence only; excluded from implementation-size cohort"),
        dict(mechanism="weight prefetch", ours="previous public-path evidence", compass="not executed in this comparison", serving="not executed in this comparison; do not infer absent", status="not evaluated here"),
        dict(mechanism="P/D handoff + prefetch", ours="components supported; this comparison not executed", compass="not evaluated here", serving="native P/D and KV movement already supported; this combination not evaluated", status="not evaluated here"),
    ]
    summary = dict(schema="approved-expression-size-1", status="completed_scoped_witnesses",
                   protocol_sha256=digest(OUT / "PROTOCOL.json"), tasks=len(records),
                   size_counting=counts, capability_table=capability, records=records,
                   original_route_control_same_result=route_controls["0"] == route_controls["2"],
                   claim="Implementation accounting for a defined macro-coordination adapter, not a complete simulator capability/performance/usability ranking.",
                   limitations=["Native math bindings are actual full-module imports, not AST substitutes.",
                                "External full scheduler/backends not invoked; adapter alone supplies execution coordination.",
                                "No concurrent prefetch/PD/alias lifetime implementation-size claims.",
                                "No numerical LLM or hardware timing validation; common math and profiles excluded.",
                                "Any failed equivalence withholds the cohort reduction rather than treating missing support as zero size."])
    write(OUT / "SUMMARY.json", summary)
    return summary


def report(summary):
    rows = []
    for tool, c in summary["size_counting"].items():
        rows.append(f"<tr><td>{tool}</td><td>{c['policy_statements']}</td><td>{c['added_coordination_statements']}</td><td>0</td><td>{c['total_scoped_statements']}</td><td>{c.get('scoped_reduction_vs_ours_percent')}</td></tr>")
    html = "<!doctype html><meta charset='utf-8'><title>Expression witnesses</title><style>body{font:16px sans-serif;max-width:1000px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style>"
    html += "<h1>可运行表达与实现规模：限定协调边界</h1><p>没有运行完整外部调度器，不能称作完整工具易用性排名。原始数学DAG、硬件、模型绑定、检查器与报告不计入策略规模；新增协调机制全部计入，各工具整组只计一次。</p>"
    html += "<table><tr><th>工具</th><th>策略声明</th><th>新增协调</th><th>框架修改</th><th>总计</th><th>同义务减幅%</th></tr>" + "".join(rows) + "</table>"
    html += "<p>未通过同义务检查的比例为null；原生完整任务的减幅始终为null。计数为Python ast.stmt（包含import与def，排除docstring）；共同JSON模型叶子不计，策略JSON为0。</p>"
    html += "<p><strong>不能将表中约90%的差值写成整个工具的易用性改进。</strong>外部路径运行了真实原生函数，但由本次新增的串行协调器负责动作；完整外部调度器没有执行，原生Dense最短程序规模也未测。因此该表是限定任务的已实现桥接代码规模，不是原生工具最小实现规模的排名。</p>"
    if "prefill_suffix_invariance" in summary:
        html += "<p>后缀扰动核验：三种实现分别在两种几何下，更换后续decode专家0→2，prefill执行/专家缓存决策保持一致。只证明这些已运行前缀的非预知性，不是任意程序的因果性证明。冻结后策略与适配器哈希均未改变。</p>"
    html += "<pre>" + escape(json.dumps(summary["capability_table"], ensure_ascii=False, indent=2)) + "</pre>"
    (OUT / "REPORT_zh.html").write_text(html, encoding="utf-8", newline="\n")


def audit_existing():
    """Complete predeclared checks from retained artifacts; no new simulation."""
    summary = json.loads((OUT / "SUMMARY.json").read_text(encoding="utf-8"))
    protocol = json.loads((OUT / "PROTOCOL.json").read_text(encoding="utf-8"))
    records = {r["id"]: r for r in summary["records"]}
    prefix_checks = []
    native_counts = []
    for split in ("development", "holdout"):
        for tool in ("ours", "compass", "serving"):
            prefixes = []
            for future in (0, 2):
                row = records[f"expert_lru-{split}-f{future}"]
                folder = OUT / row["artifact_directory"]
                raw = json.loads((folder / ("ours-events.json" if tool == "ours" else tool + "-result.json")).read_text(encoding="utf-8"))
                events = raw if tool == "ours" else raw["events"]
                prefix = []
                for event in events:
                    oid = event.get("target_id") if tool == "ours" else event.get("operation")
                    if event["kind"] == "COMPUTE" and "decode0" in (oid or ""):
                        break
                    if event["kind"] == "COMPUTE":
                        prefix.append(["compute", oid, event["device"]])
                    prefix.extend([list(x) for x in expert_events([event], tool == "ours")])
                prefixes.append(prefix)
            prefix_checks.append(dict(split=split, tool=tool, exact=prefixes[0] == prefixes[1],
                                      prefill_decisions=prefixes[0],
                                      observation_boundary="before first decode0 state-update compute; current routing only"))
    for row in records.values():
        folder = OUT / row["artifact_directory"]
        compass = json.loads((folder / "compass-result.json").read_text(encoding="utf-8"))
        serving = json.loads((folder / "serving-result.json").read_text(encoding="utf-8"))
        native_counts.append(dict(id=row["id"], compass_matmul_calls=sum(len(r["calls"]) for r in compass["native_calls"]),
                                  serving_calculate_sizes_calls=sum(r["native_sizes"] is not None for r in serving["native_calls"]),
                                  serving_indexed_cols_calls=len(serving["native_calls"])))
    current = accounting()
    source_unchanged = {tool: all(current[tool][key] == protocol["source_counts_before_execution"][tool][key]
                                for key in ("policy_sha256", "shared_coordinator_sha256", "total_scoped_statements"))
                        for tool in current}
    assert all(c["exact"] for c in prefix_checks)
    assert all(source_unchanged.values())
    assert len(stmt_nodes(ast.parse('"doc"\nimport ast\ndef f():\n    "doc"\n    return 1\n'))) == 3
    assert leaf_count({"x": [1, 2], "y": True}) == 3
    summary["prefill_suffix_invariance"] = prefix_checks
    summary["actual_native_entrypoint_call_counts"] = native_counts
    summary["frozen_policy_and_adapter_unchanged"] = source_unchanged
    summary["scope_caution"] = ("The 96/98 totals are the written native-interface-plus-coordinator implementations, not a proven minimal way to extend the original complete simulators. "
        "Dense is a common-support control, not evidence of absent native support. The shared coordinator dominates the reduction; do not advertise 90% lower whole-tool implementation effort.")
    summary["native_dense_control"] = dict(native_support_known=True, scoped_executed_cases=2,
        full_native_transformer_or_serving_scheduler_executed_here=False,
        stand_alone_native_program_size=None,
        limitation="This control checks native math/trace entrypoints and scoped device semantics, not the minimal native Dense program size.")
    version_receipts = {}
    for name, expected in protocol["native_repos"].items():
        repo = EXT / name
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(repo), "status", "--short"], text=True).strip()
        assert head == expected and not status
        version_receipts[name] = dict(commit=head, clean=True)
    summary["native_source_receipts"] = version_receipts
    summary["capability_source_evidence"] = [
        dict(mechanism="uniform block device", path=str((EXT / "LLMCompass/software_model/transformer.py").relative_to(ROOT)), lines="114-165", claim="native block roofline uses system.device for attention and FFN; operator-level manual alternatives remain possible"),
        dict(mechanism="expert identity reduction", path=str((EXT / "LLMServingSim/serving/core/gate_function.py").relative_to(ROOT)), lines="7-11,134-158", claim="route result retains per-rank expert counts; expert 0 and 2 give identical returned counts in the executed custom-router control"),
        dict(mechanism="attention device branch", path=str((EXT / "LLMServingSim/serving/core/trace_generator.py").relative_to(ROOT)), lines="1234-1247", claim="examined path selects PIM or NPU attention; our CPU compute branch is explicitly a new adapter, not a stock CPU scheduling field")]
    write(OUT / "FINAL_AUDIT.json", dict(status="completed", predeclared_prefix_checks=prefix_checks,
        source_unchanged=source_unchanged, native_counts=native_counts, counting_self_checks=True,
        harness_repairs=["ADVANCE may have null target_id: use empty string before expert filter",
                         "preserve occupied compile directories as numbered attempts",
                         "own COPY destination is dst, not location; correct observer filter"],
        repairs_change_policy_or_adapter=False, protocol_sha256=digest(OUT / "PROTOCOL.json"),
        harness_sha256=digest(__file__)))
    write(OUT / "SUMMARY.json", summary)
    report(summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()
    protocol = freeze()
    if args.audit_existing:
        audit_existing()
    elif not args.freeze_only:
        report(execute(protocol))
