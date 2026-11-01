"""Evidence-linked report for the three remaining accuracy studies.

Reports preserve measured failures and distinguish execution, coverage and
prediction acceptance. No report value is used to fit a performance model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def optional(path, evidence):
    path = Path(path)
    if not path.is_file():
        return None
    evidence.append(receipt(path))
    return read(path)


def moe_validation_evidence(collection, collection_manifest, evidence, *, heldout_summary=None):
    """Read sealed MoE coverage/calibration evidence, never a live final summary.

    Large native traces and the GGUF are referenced by their existing receipts;
    this report does not remeasure, refit or claim to re-audit their contents.
    A heldout corpus summary is read only when explicitly supplied after sealing.
    """
    collection = Path(collection).resolve()
    base = collection.parent
    frozen = base / "moe-coverage-freeze-v1"

    def required(path, expected_sha=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError("required MoE report evidence is missing: " + str(path))
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha is not None and digest != expected_sha:
            raise ValueError("MoE report evidence SHA256 changed: " + str(path))
        evidence.append(dict(path=str(path), sha256=digest, bytes=len(payload)))
        return json.loads(payload)

    barrier = required(frozen / "FREEZE_BARRIER.json")
    if barrier.get("schema") != "real-moe-coverage-freeze-barrier-1":
        raise ValueError("unknown MoE coverage freeze schema")
    if collection_manifest is None or barrier["source_manifest"]["sha256"] != receipt(collection / "manifest.json")["sha256"]:
        raise ValueError("MoE coverage freeze is not bound to the selected collection manifest")
    documents = {}
    for relative, digest in barrier["files"].items():
        path = frozen.joinpath(*relative.replace("\\", "/").split("/")).resolve()
        if not path.is_relative_to(frozen.resolve()):
            raise ValueError("MoE frozen evidence path escapes its directory")
        if path.suffix == ".json":
            documents[path.name] = required(path, digest)
        else:
            if not path.is_file() or receipt(path)["sha256"] != digest:
                raise ValueError("MoE frozen helper evidence changed: " + str(path))
            evidence.append(receipt(path))
    names = ("SUPPORTED_COMPONENT_INDEX.json", "UNSUPPORTED_PREDICTIONS_24.json",
             "WHOLE_MODEL_COVERAGE_GATE.json", "STRUCTURAL_AND_NUMERICAL_GATES.json")
    if any(name not in documents for name in names):
        raise ValueError("MoE freeze omits a required component/prediction/gate document")
    components, predictions, coverage, gates = (documents[name] for name in names)
    expected_schemas = ("real-moe-supported-component-index-1", "real-moe-frozen-unsupported-predictions-1",
                        "real-moe-whole-model-coverage-gate-1", "real-moe-separated-postcollection-gates-1")
    if tuple(row.get("schema") for row in (components, predictions, coverage, gates)) != expected_schemas:
        raise ValueError("unknown MoE component, prediction or gate schema")
    quantized_summary = required(components["source_summary"]["path"], components["source_summary"]["sha256"])
    for component in components["components"]:
        price = Path(component["price"]["path"]).resolve()
        if (not price.is_relative_to(frozen.resolve()) or not price.is_file()
                or receipt(price)["sha256"] != component["price"]["sha256"]):
            raise ValueError("MoE scoped component price evidence is absent or changed")
    calibration = required(base / "capacity-policy-v2-resident/CALIBRATION_REVIEW.json",
                           coverage["calibration_review"]["sha256"])
    trajectory = required(base / "capacity-policy-v2-resident/CALIBRATION_TRAJECTORY_DIAGNOSTIC.json",
                          coverage["trajectory_review"]["sha256"])
    if calibration.get("schema") != "real-moe-calibration-review-1" or trajectory.get("schema") != "real-moe-trajectory-diagnostic-1":
        raise ValueError("unknown MoE calibration or numerical trajectory schema")
    if coverage["model"]["sha256"] != collection_manifest["model"]["sha256"]:
        raise ValueError("MoE coverage gate/model binding differs")
    if not (components["declared_subgraph_profiles"] == components["completed_subgraph_profiles"] == 22
            and components["exported_scoped_prices"] == len(components["components"]) == 12
            and components["unstable_subgraph_prices_not_exported"] == 10):
        raise ValueError("MoE component-probe denominator differs from the frozen 22/12/10 evidence")
    entries = predictions["entries"]
    if (predictions["request_placement_count"] != 24 or len(entries) != 24
            or len({(row["request_id"], row["placement"]) for row in entries}) != 24
            or predictions["prediction_numeric_values_provided"] != 0
            or predictions["unsupported_values_are_null_not_zero"] is not True
            or any(row.get("prediction_status") != "unsupported"
                   or "predicted_duration_ns" not in row or row["predicted_duration_ns"] is not None
                   or row.get("accuracy_eligible") is not False for row in entries)):
        raise ValueError("MoE frozen unsupported/null prediction denominator or eligibility changed")
    if (coverage["coverage_passed"] is not False or coverage["prediction_status"] != "unsupported"
            or coverage["predicted_duration_ns"] is not None
            or coverage["missing_values_filled_with_zero"] is not False
            or components["whole_model_prediction_supported"] is not False):
        raise ValueError("MoE whole-model coverage claim differs; report integration requires explicit review")
    placements = calibration["placements_summary"]
    if set(placements) != {"cpu", "hybrid"} or calibration["calibration_unique_requests"] != 6:
        raise ValueError("MoE calibration request/placement denominator differs")
    strict = {row["placement"]: row for row in gates["rows"]}
    if set(strict) != set(placements) or any(strict[p]["strict_pair_status"] != placements[p]["pair_status"] for p in placements):
        raise ValueError("MoE strict calibration gate contradicts its review")
    if any(row.get("strict_failure_relaxed") is not False for row in strict.values()):
        raise ValueError("MoE strict pairing failure was relaxed")
    numerical_rows = []
    for row in trajectory["rows"]:
        cases = row["cases"]
        numerical_rows.append(dict(placement=row["placement"], mode=row["mode"], requests=len(cases),
            sampled_tokens=sum(case["outputs"] for case in cases),
            matching_samples=sum(case["matching_samples"] for case in cases),
            requests_with_sample_difference=sum(bool(case["mismatched_sample_indices"]) for case in cases),
            fixed_input_preserved=all(case["normal_input_stream_preserved"] for case in cases)))
    if {(row["placement"], row["mode"]) for row in numerical_rows} != {(p, m) for p in placements for m in ("replay", "trace")}:
        raise ValueError("MoE numerical diagnostic omits a replay/trace placement")
    corpus = None
    heldout_numerical_rows, heldout_route_rows = [], []
    if heldout_summary is not None:
        corpus = required(heldout_summary)
        if corpus.get("schema") != "real-moe-heldout-corpus-summary-1":
            raise ValueError("unknown sealed MoE heldout corpus summary schema")
        if (corpus["request_placement_cells"] != 24 or corpus["unsupported_request_placement_cells"] != 24
                or corpus["numeric_supported_request_placement_cells"] != 0
                or corpus["accuracy_evaluated"] is not False or corpus["whole_model_cost_coverage"] != "unsupported"):
            raise ValueError("MoE heldout collection cannot promote frozen unsupported predictions to accuracy passes")
        for kind, declared in (("normal_processes", 6), ("normal_request_observations", 72),
                               ("paired_request_observations", 48)):
            if corpus[kind + "_declared"] != declared or not 0 <= corpus[kind + "_completed"] <= declared:
                raise ValueError("MoE heldout declared/completed denominator differs: " + kind)
        if not corpus.get("files_index"):
            raise ValueError("sealed MoE heldout summary requires its evidence file index")
        route_coverage = corpus.get("route_coverage")
        if (not isinstance(route_coverage, list) or len(route_coverage) != len(placements)
                or {row.get("placement") for row in route_coverage} != set(placements)):
            raise ValueError("sealed MoE heldout summary requires both placement route/numerical gates")
        for row in route_coverage:
            if (row.get("strict_pair_status") not in ("passed", "failed")
                    or row.get("strict_failure_relaxed") is not False
                    or not isinstance(row.get("structural_inventory_status"), str)
                    or type(row.get("causal_fragments_exported")) is not bool
                    or not isinstance(row.get("exact_timing_coverage"), str)):
                raise ValueError("sealed MoE heldout strict gate/export fields are missing or relaxed")
            modes = row.get("modes")
            if not isinstance(modes, list) or len(modes) != 2 or {mode.get("mode") for mode in modes} != {"replay", "trace"}:
                raise ValueError("sealed MoE heldout summary requires replay and trace per placement")
            for mode in modes:
                fields = ("requests", "greedy_equal_requests", "greedy_sample_matches",
                          "greedy_sample_denominator", "fixed_output_stream_equal_requests")
                if (any(type(mode.get(field)) is not int or mode[field] < 0 for field in fields)
                        or mode["requests"] < 1
                        or mode["greedy_equal_requests"] > mode["requests"]
                        or mode["fixed_output_stream_equal_requests"] > mode["requests"]
                        or mode["greedy_sample_matches"] > mode["greedy_sample_denominator"]
                        or mode.get("acquisition_status") not in ("completed", "failed")):
                    raise ValueError("sealed MoE heldout numerical denominator/status is missing or invalid")
                heldout_numerical_rows.append(dict(placement=row["placement"], mode=mode["mode"],
                    acquisition_status=mode["acquisition_status"], strict_pair_status=row["strict_pair_status"],
                    requests=mode["requests"], greedy_equal_requests=mode["greedy_equal_requests"],
                    failed_requests=mode["requests"] - mode["greedy_equal_requests"],
                    greedy_sample_matches=mode["greedy_sample_matches"],
                    greedy_sample_denominator=mode["greedy_sample_denominator"],
                    fixed_output_stream_equal_requests=mode["fixed_output_stream_equal_requests"]))
            exported = row["causal_fragments_exported"]
            export = row.get("causal_export")
            if exported and (not isinstance(export, dict) or export.get("status") != "passed"
                             or type(export.get("decode_events")) is not int or export["decode_events"] < 0
                             or type(export.get("m2_decode_fragments")) is not int or export["m2_decode_fragments"] < 0
                             or export.get("whole_model_accuracy_certified") is not False):
                raise ValueError("sealed MoE heldout causal fragment export lacks its count or scope evidence")
            heldout_route_rows.append(dict(placement=row["placement"],
                structural_inventory_status=row["structural_inventory_status"],
                strict_pair_status=row["strict_pair_status"], causal_fragments_exported=exported,
                decode_events=export["decode_events"] if exported else None,
                m2_decode_fragments=export["m2_decode_fragments"] if exported else None,
                exact_timing_coverage=row["exact_timing_coverage"]))
    return dict(schema="remaining-validation-moe-evidence-1", freeze=barrier, components=components,
        unsupported_predictions=predictions, whole_model_coverage=coverage, separated_gates=gates,
        calibration=calibration, trajectory_diagnostic=trajectory, numerical_summary=numerical_rows,
        quantized_probe_summary=quantized_summary,
        heldout_corpus=corpus, heldout_numerical_summary=heldout_numerical_rows,
        heldout_route_summary=heldout_route_rows, complete_end_to_end_prediction_accuracy_certified=False)


def number(value, percent=False):
    if value is None:
        return "—"
    return f"{100 * value:.2f}%" if percent else f"{value:.3f}"


def table(headers, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def dense_rows(data):
    rows = []
    for case in data["cases"]:
        actual = [r["mean_ns"] for r in case["processes"] if r["status"] == "completed"]
        for name, provider in case["providers"].items():
            rows.append(dict(case=case["program_key"], layers=case["layers"], phase=case["phase"], strategy=case["strategy"],
                provider=name, predicted_ms=provider["predicted_ns"] / 1e6,
                measured_process_mean_ms=statistics.mean(actual) / 1e6 if actual else None,
                measured_process_min_ms=min(actual) / 1e6 if actual else None,
                measured_process_max_ms=max(actual) / 1e6 if actual else None,
                completed_processes=len(actual), mean_ape_max=provider["max_process_mean_ape"],
                block_ape_p90=provider["p90_block_ape"], passed=provider["accuracy_passed"],
                raw_cv_max=max((r["cv"] for r in case["processes"] if r["status"] == "completed"), default=None)))
    return rows


def dense_figures(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    colors = ("#16345b", "#407da3", "#b66546")
    names = ("CPU", "GPU", "GPU + CPU Attention")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    primary = [r for r in rows if r["provider"] == "history_conditional"]
    for ax, phase in zip(axes, ("prefill", "decode")):
        if not any(r["phase"] == phase for r in primary):
            ax.set_axis_off()
            ax.text(.5, .5, "Not included in this cohort", ha="center", va="center", transform=ax.transAxes)
            continue
        for strategy in range(3):
            selected = sorted((r for r in primary if r["phase"] == phase and r["strategy"] == strategy), key=lambda r: r["layers"])
            if not selected:
                continue
            xs = [r["layers"] for r in selected]
            observed = [r for r in selected if r["measured_process_mean_ms"] is not None]
            if observed:
                ax.errorbar([r["layers"] for r in observed], [r["measured_process_mean_ms"] for r in observed],
                    yerr=[[r["measured_process_mean_ms"]-r["measured_process_min_ms"] for r in observed],
                        [r["measured_process_max_ms"]-r["measured_process_mean_ms"] for r in observed]],
                    color=colors[strategy], marker="o", capsize=2, label=names[strategy] + " measured")
            ax.plot(xs, [r["predicted_ms"] for r in selected], color=colors[strategy], marker="x", linestyle="--", label=names[strategy] + " predicted")
        ax.set_title("Prefill (64 tokens)" if phase == "prefill" else "Decode (history 128)")
        ax.set_xlabel("Connected layers")
        ax.set_ylabel("Completed forward time (ms)")
        ax.set_xticks(sorted({r['layers'] for r in primary if r['phase']==phase}))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    legend = [Line2D([0], [0], color=color, lw=2, label=name) for color, name in zip(colors, ("CPU", "GPU", "Hybrid"))]
    legend += [Line2D([0], [0], color="#333333", marker="o", label="Measured"),
        Line2D([0], [0], color="#333333", marker="x", linestyle="--", label="Predicted")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.11), ncol=5, fontsize=9, frameon=False)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / ("dense_depth." + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate(out, dense, idle, external, moe, *, moe_heldout_summary=None, dynamic_report=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = []
    d = optional(Path(dense) / "summary.json", evidence)
    dm = optional(Path(dense) / "manifest.json", evidence)
    if d is not None and dm is None:
        raise ValueError("Dense results require their bound campaign manifest")
    i = optional(Path(idle) / "summary.json", evidence)
    im = optional(Path(idle) / "manifest.json", evidence)
    ic = optional(Path(idle) / "provider-comparison.json", evidence)
    controller_review = optional(Path(idle) / "ROOT_CONTROLLER_REVIEW.json", evidence)
    old_idle = optional(ROOT / "build/dynamic-validation-20261026/idle-v1/summary.json", evidence)
    optional(ROOT / "build/dynamic-validation-20261026/idle-v1/profile-observations.json", evidence)
    legacy = optional(ROOT / "build/h2d-history-20261026/campaign-v1/summary.json", evidence)
    x = optional(Path(external) / "evaluation.json", evidence)
    xf = optional(Path(external) / "coverage-failure.json", evidence)
    xr = optional(Path(external) / "RUNTIME_BOUNDARY_REVIEW.json", evidence)
    runtime_review_text = Path(external) / "RUNTIME_BOUNDARY_REVIEW_zh.md"
    if xr is not None:
        if not runtime_review_text.is_file():
            raise ValueError("runtime-boundary review text evidence missing")
        evidence.append(receipt(runtime_review_text))
    dynamic = None
    if dynamic_report is not None:
        dynamic_path = Path(dynamic_report).resolve()
        dynamic = optional(dynamic_path, evidence)
        if dynamic is None or dynamic.get("schema") != "dynamic-response-report-1":
            raise ValueError("complete dynamic figure report evidence required")
        if (dynamic.get("fitting_performed") is not False
                or dynamic.get("new_measurements_performed") is not False):
            raise ValueError("figure report must not remeasure or refit")
        if not (dynamic_path.parent / "REPORT_en.html").is_file():
            raise ValueError("dynamic figure report HTML missing")
        evidence.append(receipt(dynamic_path.parent / "REPORT_en.html"))
        if dynamic.get("external") is None:
            raise ValueError("dynamic figure report must bind external results")
        if x is not None:
            if dynamic["external"]["evaluation"]["sha256"] != receipt(Path(external) / "evaluation.json")["sha256"]:
                raise ValueError("dynamic figure report must bind the scored external cohort")
        elif (xf is None or dynamic["external"].get("status") != "coverage_failed"
              or dynamic["external"]["coverage_failure"]["sha256"] != receipt(Path(external) / "coverage-failure.json")["sha256"]):
            raise ValueError("dynamic figure report must bind the external support failure")
        selected_idle = [item for item in dynamic["idle"] if Path(item["campaign"]).resolve() == Path(idle).resolve()]
        if len(selected_idle) != 1 or selected_idle[0]["manifest"]["sha256"] != receipt(Path(idle) / "manifest.json")["sha256"]:
            raise ValueError("dynamic figure report must bind the selected idle cohort")
        artifact_groups = [dynamic["controllers"], *dynamic["idle"], *dynamic["external"]["figures"]]
        for group in artifact_groups:
            for artifact in group["artifacts"]:
                if receipt(artifact["path"])["sha256"] != artifact["sha256"]:
                    raise ValueError("dynamic figure artifact content changed")
                evidence.append(receipt(artifact["path"]))
    mp = optional(Path(moe) / "PREPARATION_STATUS.json", evidence)
    model_file = Path(moe) / "manifest.json"
    mm = optional(model_file, evidence)
    if mp is not None and mm is not None and any(mp['model'][k] != mm['model'][k] for k in ('sha256','bytes')):
        raise ValueError("MoE preparation/model manifest binding differs")
    if mp is None or mm is None:
        raise ValueError("MoE report requires preparation status and the bound collection manifest")
    me = moe_validation_evidence(moe, mm, evidence, heldout_summary=moe_heldout_summary)
    jobs = []
    for path in sorted((Path(moe) / "jobs").glob("*/execution.json")):
        job = optional(path, evidence)
        native = optional(job.get("native", {}).get("path", "__absent__"), evidence)
        if job['status'] == 'completed' and native is None:
            raise ValueError("completed MoE execution is missing native evidence")
        if native is not None and evidence[-1]['sha256'] != job['native']['sha256']:
            raise ValueError("MoE native evidence content changed")
        req = optional(path.parent / "request.json", evidence)
        if req is None:
            raise ValueError("MoE execution is missing request evidence; cannot omit it from denominator")
        if job.get('request') is not None and evidence[-1]['sha256'] != job['request']['sha256']:
            raise ValueError("MoE request evidence content changed")
        if req.get("mode") != "tokenize-only":
            jobs.append(dict(job_id=job.get("job_id", path.parent.name), status=job["status"],
                mode=req["mode"], placement=req["placement"], declared_requests=len(req["requests"]),
                completed_requests=len(native.get("results", [])) if native else 0,
                native=job.get("native"), error=job.get("error")))
    r = dense_rows(d) if d else []
    if r and dm:
        counters = {}
        for key in {v["case"] for v in r}:
            binding = dm["programs"][key]["program"]
            program = optional(binding["path"], evidence)
            if evidence[-1]["sha256"] != binding["sha256"]:
                raise ValueError("Dense program content binding differs")
            tensors = {t["id"]: t for t in program["graph"]["workload"]["tensors"]}
            actions = program["actions"]
            copies = [a for a in actions if a["kind"] == "COPY"]
            counters[key] = dict(compute_actions=sum(a["kind"] == "COMPUTE" for a in actions),
                copy_actions=len(copies),
                h2d_bytes=sum(tensors[a["target_id"]]["storage_bytes"] for a in copies if (a["src"], a["dst"]) == ("dram", "vram")),
                d2h_bytes=sum(tensors[a["target_id"]]["storage_bytes"] for a in copies if (a["src"], a["dst"]) == ("vram", "dram")))
        for row in r:
            row.update(counters[row["case"]])
    if r:
        with (out / "dense_cases.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(r[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(r)
        dense_figures(out, r)
    sections = []
    sections.append("<h2>如何读这份报告</h2><p>验收分为三层：程序是否正确执行、成本是否完整覆盖、独立数据的预测是否通过。真实执行成功不等于性能模型通过；CV单独描述观测波动，不用于替代预测误差。</p><p>预先门槛：每个独立进程均值绝对百分比误差≤10%，块级误差P90≤20%（higher）。参数只来自校准分项，冻结后才读取独立验收结果。失败与超时保留在分母中。</p>")
    overview = []
    if d:
        primary = [case['providers']['history_conditional'] for case in d['cases']]
        overview.append(['Dense连接层', f"{sum(row['accuracy_passed'] for row in primary)}/{len(primary)}情况通过；最差进程均值误差{number(max(row['max_process_mean_ape'] for row in primary), True)}", '缩小宽度F32、已知层数、固定历史Decode与串行回放，不代表真实整模型生成'])
    if me['heldout_corpus'] is not None:
        corpus = me['heldout_corpus']
        overview.append(['真实量化MoE', f"{corpus['normal_request_observations_completed']}/{corpus['normal_request_observations_declared']}条正常观测；24项整模型预测unsupported", '正常耗时与路由语料完成；成本覆盖和混合路由严格一致性仍有缺口，整模型精度未评估'])
    if i:
        overview.append(['H2D空闲/恢复', f"连续搬运{i['burst_mean_qualified_cases']}/8；首个搬运{i['first_response_accuracy_cases']}/8", '预测仅输入计划空闲；控制偏差保留，恢复准确性未全面通过'])
    if xf is not None:
        overview.append(['CPU/GPU后台负载', f"校准{xf['completed_profile_processes']}/{xf['declared_profile_processes']}；成本支持{xf['supported_response_cells']}/{xf['declared_response_cells']}", '支持门槛未通过，未拟合/冻结/采独立集，不报告虚构准确率'])
    elif x:
        provider = x['providers'][x['primary_provider']]
        overview.append(['CPU/GPU后台负载', f"{provider['passed']}/{provider['declared']}独立进程通过", '仅适用于已声明的同进程后台负载与两种连续Dense程序'])
    sections.append(table(['本轮验收线', '实际结果', '可以得出的结论'], overview))
    sections.append("<h2>1. Dense多层：真实连接的计算块</h2><p><b>预定正式矩阵：</b>同一单层结构连接4/8/16/32层；Prefill为64 token，Decode为1 token、固定历史128。H=512、FFN=1024、8个Attention头、2个KV头、F32。S0=CPU；S1=GPU；S2=GPU投影/FFN与CPU Attention。CPU线程数为1。实际完成范围以下表为准。</p>")
    if d:
        primary = [v for v in r if v["provider"] == "history_conditional"]
        pilot = dm is not None and dm.get("pilot") is True
        cohort = "Pilot/绘图预览（不是完整32层验收）" if pilot else "正式独立验收"
        coverage = ", ".join(f"{phase}:L={','.join(str(n) for n in sorted({v['layers'] for v in primary if v['phase']==phase}))}" for phase in sorted({v['phase'] for v in primary}))
        sections.append(f"<p class='callout'><b>{cohort}</b>：实际覆盖 {escape(coverage)}。独立进程完成 {d['completed_processes']}/{d['declared_processes']}；历史条件模型通过 {sum(v['passed'] for v in primary)}/{len(primary)} 种情况。CSV每种情况有两个提供者行，不能按CSV行数计算案例数。</p>")
        sections.append(table(["阶段", "层数", "策略", "实测均值/ms", "预测/ms", "最大进程均值误差", "块误差P90", "原始CV最大值", "验收"],
            [[v["phase"], v["layers"], f"S{v['strategy']}", number(v["measured_process_mean_ms"]), number(v["predicted_ms"]), number(v["mean_ape_max"], True), number(v["block_ape_p90"], True), number(v["raw_cv_max"], True), "通过" if v["passed"] else "未通过"] for v in primary]))
        sections.append("<figure><img src='dense_depth.png' alt='Measured and predicted connected Dense forward time'/><figcaption>Measured and predicted Dense forward time. Error bars span the completed independent process means. Hybrid uses GPU projection/FFN and CPU Attention. Same geometry, independent numeric inputs and fresh processes.</figcaption></figure>")
        sections.append(table(["对照提供者", "通过/总数"], [[name, f"{v['accuracy_passes']}/{v['cases']}"] for name, v in d["providers"].items()]))
    else:
        sections.append("<p>正式独立验收尚未形成summary；这里不以已通过的小规模pilot代替正式矩阵。</p>")
    sections.append("<p><b>证据范围：</b>这是缩小宽度的、真实GGML连接层计算块，包含独立权重、KV与层间依赖。校准包含所列深度的分项，因此验证的是已知工作负载下的分项组合预测，不是未知深度外推；固定历史Decode不是完整自回归生成。原生回放当前同步执行、预分配缓冲，不能证明真实异步重叠或物理驱逐。</p>")
    sections.append("<p>本实验包含当前编译器生成的逐块KV搬运开销；逐案例CSV同时列出COMPUTE/COPY次数与H2D/D2H字节。策略排名只适用于这套明确的执行与搬运实现，不能直接解释为所有真实LLM运行时的CPU/GPU硬件排名。</p>")
    sections.append("<p><b>工具准备开销：</b>32层GPU策略的动作导出明显较慢。独立软件cProfile定位到每动作重建全量公共观察和重复计算内存账本；固定单层结构后，这部分工作至少随动作数×张量数增长。它不计入推理计时，但会影响工具易用性和后续大规模搜索预算。本轮未为提速绕过策略、合法性检查或改变时序。</p>")
    sections.append("<h2>2. 真实MoE：实际权重、请求与路由</h2>")
    if mp:
        if mp["schema"] == "real-moe-preparation-status-3":
            if mm is None or mm["schema"] != "real-moe-collection-manifest-3":
                raise ValueError("v3 preparation requires its bound v3 collection manifest")
            contract = mm["model_contract"]
        elif mp["schema"] == "real-moe-preparation-status-2":
            contract = mp["model_contract"]
        else:
            raise ValueError("unknown MoE preparation schema")
        sections.append(f"<p>本机主实验采用OLMoE：{contract['layers']}层、{contract['experts']}专家、top-{contract['top_k']}，真实Q4_K_M GGUF约{mp['model']['bytes']/1e9:.2f} GB。完整SHA256与发布者一致。18个请求分为6个校准、12个留出，覆盖对话/代码/数学和短/长输入，提示模板与模型官方模板一致。</p>")
    sections.append("<p>原计划Qwen3.5-35B-A3B权重约20.5 GiB，而本机物理RAM约15.2 GiB；将其作为常驻内存实验会混入分页。因此明确修订主模型为OLMoE，原Qwen分词与准备证据保留。Q4_K_M内部按层混合精度，逐张量使用实际Q4_K/Q6_K/F32，不把F32成本冒充量化成本。</p>")
    component_index = me["components"]
    calibration = me["calibration"]
    placements = calibration["placements_summary"]
    strict = {row["placement"]: row for row in me["separated_gates"]["rows"]}
    sections.append("<p class='callout'><b>真实MoE尚未获得端到端预测准确性结论。</b>冻结的24个“留出请求×放置”单元全部为unsupported，预测值均为null，数值预测为0个；不能写成“误差通过0/24”，也不能以正常运行完成代替准确性验收。</p>")
    sections.append(table(["证据层", "已完成/已知", "允许解释"], [
        ["真实量化子图探针", f"{component_index['completed_subgraph_profiles']}/{component_index['declared_subgraph_profiles']}", "探针完成不代表价格稳定或整模型覆盖"],
        ["限定作用域的分项价格", component_index["exported_scoped_prices"], "仅精确匹配算子、后端、量化、形状、线程池、缓存及输入/路由条件时可查用"],
        ["拒绝导出的不稳定价格", component_index["unstable_subgraph_prices_not_exported"], "保留原始探针，不作为已知成本"],
        ["冻结的整模型留出预测", "0个数值；24个unsupported/null", "缺失项不置零、不插值、不用整模型乘数补齐"],
    ]))
    sections.append(table(["冻结时的组件覆盖", "状态", "限制"],
        [[row["component"], row["status"], row["reason"]] for row in me["whole_model_coverage"]["rows"]]))
    sections.append("<p>12个价格来自22组探针中的有限子集，并非12个真实请求或12个整模型预测。当前仍缺Attention及带padding的F16 KV、Embedding/Norm、完整LM-head边界、实际路由/激活/缓存状态匹配、Decode专家与尾块，以及异构backend衔接；现有分项不能直接相加冒充真实模型预测。</p>")
    sections.append(table(["6个校准请求的放置", "结构清单", "严格配对", "正常generate观测数", "平均Prefill/ms", "平均总时长/ms", "缺少精确价格的FFN q"], [
        [placement, strict[placement]["structural_inventory_status"], summary["pair_status"], len(summary["normal_requests"]),
         number(statistics.mean(row["prefill_ms"] for row in summary["normal_requests"])),
         number(statistics.mean(row["total_ms"] for row in summary["normal_requests"])),
         ", ".join(map(str, summary["q_shapes_without_prepared_profile"]))]
        for placement, summary in placements.items()
    ]))
    sections.append(table(["校准放置", "方式", "保留正常输入流", "贪心采样相同/比较位置", "发生采样差异的请求"], [
        [row["placement"], row["mode"], row["fixed_input_preserved"],
         f"{row['matching_samples']}/{row['sampled_tokens']}", f"{row['requests_with_sample_difference']}/{row['requests']}"]
        for row in me["numerical_summary"]
    ]))
    cpu_routes = placements["cpu"]["routing_analysis"]
    hybrid_trace = next(row for row in me["numerical_summary"] if row["placement"] == "hybrid" and row["mode"] == "trace")
    cross = calibration["cross_placement_same_generated_tokens"]
    sections.append(f"<p>CPU独立trace路由分析状态为{escape(cpu_routes['status'])}：{cpu_routes['decode_events']}个Decode事件展开为{cpu_routes['m2_fragments']}个逐层M2片段。CPU的replay与trace校准采样一致，但该结果仍不证明未插桩正常执行的隐含路由与trace完全相同。</p>")
    sections.append(f"<p><b>混合放置的严格校准配对失败，原判定保持不变。</b>其trace在{hybrid_trace['requests_with_sample_difference']}/{hybrid_trace['requests']}个请求、{hybrid_trace['sampled_tokens']-hybrid_trace['matching_samples']}/{hybrid_trace['sampled_tokens']}个采样位置出现差异；固定输入流保留与贪心采样一致性是两个独立检查。CPU和混合正常生成在{sum(not same for same in cross.values())}/{len(cross)}个校准请求中token序列不同，因此上表正常运行时间也不能直接证明相同token/路由负载下的策略优劣。</p>")
    if jobs:
        complete_normal = [v for v in jobs if v['status'] == 'completed' and v['mode'] == 'generate']
        complete_replay = [v for v in jobs if v['status'] == 'completed' and v['mode'] == 'replay']
        complete_trace = [v for v in jobs if v['status'] == 'completed' and v['mode'] == 'trace']
        sections.append(f"<p>文件索引中，正常generate完成{sum(v['completed_requests'] for v in complete_normal)}个请求观测；无回调的teacher-forced replay完成{sum(v['completed_requests'] for v in complete_replay)}个；独立trace完成{sum(v['completed_requests'] for v in complete_trace)}个。它们包含历史运行与重复观测，既不是不同测试提示数，也不是准确性验收分母。</p>")
        sections.append(table(["进程", "模式", "放置", "状态", "请求完成/声明"], [[v['job_id'], v['mode'], v['placement'], v['status'], f"{v['completed_requests']}/{v['declared_requests']}"] for v in jobs]))
    else:
        sections.append("<p>当前索引没有已完成的真实推理进程。模型/分词准备不能计入预测准确性通过数。</p>")
    sections.append("<p><b>验收边界：</b>正常generate/replay不安装路由回调；独立trace重放相同token，逐层逐token保存专家激活。trace耗时不能当正常推理真值。端到端预测还必须覆盖真实量化专家、Attention/KV、Embedding、LM-head与backend衔接；缺项明确拒绝，不回填整模型系数。</p>")
    sections.append("<p>实际执行配置显式使用常驻线程池与256-token microbatch。例如1029 tokens拆成256+256+256+256+5；末层FFN只处理需要输出的位置，不能假定16层都计算全部1029行。独立路由审计按每次forward、层与绝对位置检查，不能用q256价格线性外推所有尾块。</p>")
    if me["heldout_corpus"] is not None:
        corpus = me["heldout_corpus"]
        sections.append(table(["封存的留出语料采集", "完成/声明", "结论范围"], [
            ["正常generate进程", f"{corpus['normal_processes_completed']}/{corpus['normal_processes_declared']}", "真实运行语料"],
            ["正常请求观测", f"{corpus['normal_request_observations_completed']}/{corpus['normal_request_observations_declared']}", "重复进程观测，非72个不同提示"],
            ["replay/trace请求观测", f"{corpus['paired_request_observations_completed']}/{corpus['paired_request_observations_declared']}", "数值/路由证据，非正常计时替身"],
            ["端到端可评测预测", "0/24；24个unsupported", "accuracy_evaluated=false"],
        ]))
        sections.append(table(["留出放置", "方式", "采集状态", "放置整体严格配对", "贪心轨迹一致请求", "未通过请求（保留）", "贪心采样一致位置", "固定输出流一致请求"], [
            [row["placement"], row["mode"], row["acquisition_status"], row["strict_pair_status"],
             f"{row['greedy_equal_requests']}/{row['requests']}", row["failed_requests"],
             f"{row['greedy_sample_matches']}/{row['greedy_sample_denominator']}",
             f"{row['fixed_output_stream_equal_requests']}/{row['requests']}"]
            for row in me["heldout_numerical_summary"]
        ]))
        sections.append(f"<p><b>采集完成与严格轨迹检查分开计数。</b>留出replay/trace中共有{sum(row['failed_requests'] for row in me['heldout_numerical_summary'])}个请求观测未通过贪心轨迹一致性，全部保留，不放宽原门槛；固定输出流一致不等于未插桩正常执行的隐含路由已被证明一致。</p>")
        sections.append(table(["留出放置", "结构清单", "严格配对", "因果片段导出", "Decode事件", "逐层M2因果片段", "精确计时覆盖"], [
            [row["placement"], row["structural_inventory_status"], row["strict_pair_status"],
             row["causal_fragments_exported"], row["decode_events"] if row["decode_events"] is not None else "未导出",
             row["m2_decode_fragments"] if row["m2_decode_fragments"] is not None else "未导出", row["exact_timing_coverage"]]
            for row in me["heldout_route_summary"]
        ]))
        sections.append("<p>导出的CPU因果片段是实测路由输入证据；混合放置的严格配对失败不由结构清单通过抵消。片段导出不证明精确计时覆盖，也不将24个unsupported整模型预测变成准确性通过。</p>")
        sections.append("<p>该表只读取显式指定的封存留出摘要；采集完成仍不能补齐缺失成本，也没有将任何unsupported/null预测改成精度通过。</p>")
    else:
        sections.append("<p>本报告没有接入封存的留出语料摘要；上述校准和文件索引数据不替代独立留出分母。24个unsupported预测的冻结判定保持不变。</p>")
    sections.append("<h2>3. 设备波动：恢复与外部竞争分开检验</h2><p>输入为预先给出的外部工作计划，加上模型自己递推的执行历史。预测不读取目标未来的总利用率、时钟或实际完成边界。CPU/GPU共享状态演化框架；当前ON/OFF工作计划是受控负载条件，不能直接称为实测可用FLOP/s。</p>")
    if i:
        if im is None:
            raise ValueError("idle results require their bound controller manifest")
        if im['schema']=='dynamic-idle-campaign-v2':
            controller=im['protocol']['controller']['kind']
        elif im['schema']=='dynamic-idle-campaign-1':
            controller='plain Sleep controller (v1)'
        else:
            raise ValueError('unknown idle controller manifest schema')
        sections.append(f"<p class='callout'>实际cohort：<b>{escape(Path(idle).name)}</b>；清单：{escape(im['schema'])}；控制器：{escape(controller)}。2 MiB H2D独立进程完成 {i['completed_processes']}/{i['declared_processes']}；连续搬运整体验收 {i['burst_mean_qualified_cases']}/8；首个搬运均值验收 {i['first_response_accuracy_cases']}/8。</p>")
        sections.append(table(["主机内存", "计划空闲/ms", "连续搬运最大均值误差", "连续搬运最大P90误差", "首个搬运最大均值误差", "整体验收"], [[v['memory_mode'], v['requested_idle_ms'], number(v['maximum_process_burst_mean_ape'], True), number(v['maximum_process_burst_p90_ape'], True), number(v['maximum_first_copy_mean_ape'], True), "通过" if v['mean_response_qualified'] else "未通过"] for v in i['cases']]))
    else:
        sections.append("<p>新恢复试验尚未形成独立验收summary。</p>")
    sections.append("<p>第一轮普通Sleep控制存在明显偏差：计划1 ms实际约5–14 ms，不能将拟合常数解释为物理恢复时间。原始失败数据完整保留。另有独立高精度waitable timer版本，不忙等、不修改系统全局计时精度；控制精度与预测结论仅以下列独立采集文件为准，不能从实现方式推定通过。</p>")
    if old_idle:
        sections.append(f"<p>保留的第一轮结果：连续搬运整体验收 {old_idle['burst_mean_qualified_cases']}/8，首个搬运均值验收 {old_idle['first_response_accuracy_cases']}/8；不能仅凭首个动作改善宣称整段波动性已经解决。</p>")
    if ic:
        sections.append(table(["同一校准集的恢复对照", "连续搬运通过/8", "首个搬运均值通过/8"],
            [[name, f"{sum(v['mean_response_qualified'] for v in provider['cases'])}/8", f"{sum(v['first_response_accuracy_passed'] for v in provider['cases'])}/8"] for name, provider in ic['providers'].items()]))
    for split in ("profile", "heldout"):
        precision = optional(Path(idle) / (split + "-controller-precision.json"), evidence)
        if precision:
            sections.append(f"<p>高精度控制器（{split}）：{precision['qualified_processes']}/{precision['declared']}进程满足预先设定的控制精度；未删除慢等待。</p>")
    if x:
        sections.append(table(["外部竞争预测器", "独立进程通过/总数"], [[name, f"{v['passed']}/{v['declared']}"] for name, v in x['providers'].items()]))
        sections.append("<p>对照为同一校准数据的静态价格、仅当前外部状态的价格，以及可识别时的历史/恢复模型。2048次连续forward按16次报告一块，预测边界始终由自身递推，未对齐到实测块起点。</p>")
    elif xf is not None:
        if (xf.get("schema") != "dynamic-external-coverage-failure-1"
                or xf.get("model_fitted") is not False or xf.get("predictions_frozen") is not False
                or xf.get("accuracy_evaluated") is not False):
            raise ValueError("external coverage-failure evidence contradicts unfitted scope")
        sections.append(f"<p class='callout'>外部竞争校准完成 {xf['completed_profile_processes']}/{xf['declared_profile_processes']}；完整四态支持的动作语义 {xf['fully_supported_semantics']}/{xf['semantic_coordinates']}，已测响应格 {xf['supported_response_cells']}/{xf['declared_response_cells']}。GPU Attention 在 CPU/GPU 同时竞争状态缺少完整落在该状态内的调用。按预定支持门槛，本轮没有拟合或冻结该四态模型；独立采集 {xf['completed_heldout_processes']}/{xf['declared_heldout_processes']}，预测准确性未评估，不能记为数值通过或数值失败。</p>")
        sections.append("<p>三轮原始 PROFILE 都观察到 Attention 完成响应跨越共同竞争区间、在计划 CPU OFF 后结束。这里测到的是包含等待的完整调用，不能解释为 GPU 内核实际计算了数秒，也不能据此断定操作系统随机性或运行时死锁。混合程序的四态价格支持完整；它随整矩阵门槛一起停止，不能将其描述成同样缺价。所有原始观测保留，未用其他状态的价格填格，未追加改变窗口的测量以强行取得通过。</p>")
        if xr is not None:
            sections.append("<p><a href='" + escape(runtime_review_text.resolve().as_uri(), quote=True) + "'>只读运行时边界审查</a>：各负载使用独立GGML backend/pool，CUDA流与句柄也分开；没有找到足以解释秒级阻塞的显式共享锁或共享线程池。目标提交线程和CPU负载同绑CPU2；独立GGML对象不等于驱动上下文隔离，现有数据不能分离主机就绪/运行、驱动同步等待与设备服务。源码检查不代替这些物理阶段的测量，也没有排除运行时问题。</p>")
    else:
        sections.append("<p>外部竞争的24校准+24留出矩阵尚未形成evaluation；不以控制器实现或软件测试替代准确性结论。</p>")
    if dynamic is not None:
        def figure_link(name):
            return escape(os.path.relpath(dynamic_path.parent / name, out).replace("\\", "/"), quote=True)
        sections.append("<p><a href='" + figure_link("REPORT_en.html") + "'>设备波动的英文曲线、控制精度及矢量图</a>。模型沿自己的预测时间轴推进，实测曲线沿独立记录的时间轴展示，没有对齐到未来实测动作边界。</p>")
        sections.append("<figure><img src='" + figure_link(dynamic["controllers"]["figure"]) + "'><figcaption>Requested versus achieved idle; all independent processes retained.</figcaption></figure>")
        selected = next(item for item in dynamic["idle"] if Path(item["campaign"]).resolve() == Path(idle).resolve())
        sections.append("<figure><img src='" + figure_link(selected["figure"]) + "'><figcaption>Independent H2D burst responses and frozen forecasts.</figcaption></figure>")
        for plot in dynamic["external"]["figures"]:
            sections.append("<figure><img src='" + figure_link(plot["figure"]) + "'><figcaption>Completed-response observations under predeclared offered-work plans.</figcaption></figure>")
    old_result = legacy['providers']['history_conditional'] if legacy is not None else None
    preserved = f"旧报告的严格联合验收{old_result['fully_qualified_cases']}/{old_result['declared_cases']}保持原结果。" if old_result else "旧严格联合验收没有在本报告重新计算或重新标记。"
    sections.append("<h2>可以写入论文的边界</h2><p>可声明的结论以本报告表格为限。多层缩小块、真实MoE与受控波动是不同证据层，不能互相代替；" + preserved + "UMA/NPU仍是参数化分析，未增加真实硬件准确性声明。稳定性未通过不等于CV不可建模，也不意味着误差必然来自操作系统。</p>")
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>剩余准确性实验结果</title><style>body{font:17px/1.75 'Microsoft YaHei',sans-serif;color:#20344c;background:#f3f6fa;margin:0}main{max-width:1180px;margin:auto;background:white;padding:36px 48px}h1,h2{color:#14365d;line-height:1.35}h1{font-size:30px}h2{margin-top:38px;border-bottom:2px solid #e1e8f0;padding-bottom:9px;font-size:23px}table{border-collapse:collapse;min-width:650px;width:100%;font-size:14px}th,td{border:1px solid #d8e1eb;padding:9px;text-align:left}th{background:#e9f0f7}.scroll{overflow:auto;margin:18px 0}.callout{background:#eaf2fa;border-left:5px solid #3b6992;padding:13px 18px}figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{font:14px/1.5 'Times New Roman',serif;color:#53667d}code{word-break:break-all}a{color:#245e90}</style><main><h1>Dense深度、真实MoE与设备波动：实验结果</h1><p>单CPU + 单GPU · 2026-10-26–27 · 独立数据验收</p>" + "".join(sections) + "<h2>数据与复现索引</h2><p><a href='summary.json'>结构化结果</a> · <a href='manifest.json'>证据SHA256索引</a>" + (" · <a href='dense_cases.csv'>Dense逐情况CSV</a> · <a href='dense_depth.pdf'>可编辑/论文用矢量图PDF</a> · <a href='dense_depth.svg'>SVG</a>" if r else "") + "</p></main></html>"
    (out / "REPORT_zh.html").write_text(html, encoding="utf-8", newline="\n")
    summary = dict(schema="remaining-validation-report-1", generated_at_utc=datetime.now(timezone.utc).isoformat(),
        dense=d, idle=i, idle_controller_review=controller_review, previous_idle=old_idle, external=x, external_coverage_failure=xf, external_runtime_boundary_review=xr, dynamic_figure_report=dynamic, moe_preparation_snapshot=mp, moe_execution_index=jobs,
        moe_validation_evidence=me,
        preserved_legacy_joint_result=old_result,
        moe_observed_status=dict(normal_timing_collected=any(v['status']=='completed' and v['mode'] in ('generate','replay') for v in jobs),
            normal_generation_collected=any(v['status']=='completed' and v['mode']=='generate' for v in jobs),
            teacher_forced_replay_collected=any(v['status']=='completed' and v['mode']=='replay' for v in jobs),
            raw_routes_collected=any(v['status']=='completed' and v['mode']=='trace' for v in jobs),
            scoped_component_prices=me['components']['exported_scoped_prices'],
            unsupported_request_placement_cells=24, numeric_supported_request_placement_cells=0,
            calibration_strict_pairs={placement: row['pair_status'] for placement, row in me['calibration']['placements_summary'].items()},
            heldout_corpus_summary_explicitly_loaded=me['heldout_corpus'] is not None,
            complete_end_to_end_prediction_accuracy_certified=False),
        model_accuracy_is_distinct_from_execution=True, heldout_outcomes_used_for_fitting=False,
        raw_cv_is_prediction_gate=False)
    write(out / "summary.json", summary)
    write(out / "manifest.json", dict(schema="remaining-validation-report-evidence-1", evidence=evidence,
        generator=receipt(__file__), outputs=[receipt(p) for p in sorted(out.iterdir()) if p.is_file()]))
    return dict(report=str(out / "REPORT_zh.html"), evidence_files=len(evidence), dense_cases=len(d["cases"]) if d else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--dense", default=str(ROOT / "build/depth-validation-20261026/formal-v4"))
    parser.add_argument("--idle", default=str(ROOT / "build/dynamic-validation-20261026/idle-v2b"))
    parser.add_argument("--external", default=str(ROOT / "build/dynamic-validation-20261026/external-v1"))
    parser.add_argument("--moe", default=str(ROOT / "build/real-moe-validation-20261026/collection-v3"))
    parser.add_argument("--moe-heldout-summary", help="Explicit sealed HELDOUT_CORPUS_SUMMARY.json; never auto-read a live summary")
    parser.add_argument("--dynamic-report", help="Completed REPORT_DATA.json for independently scored dynamic figures")
    args = parser.parse_args()
    print(json.dumps(generate(args.out, args.dense, args.idle, args.external, args.moe,
                              moe_heldout_summary=args.moe_heldout_summary, dynamic_report=args.dynamic_report), ensure_ascii=False))


if __name__ == "__main__":
    main()
