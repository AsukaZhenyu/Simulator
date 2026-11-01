"""Render readable shared-storage evidence without inventing missing results.

Standard library only. This reads saved simulation/test/audit artifacts, does
not execute a policy, alter measurements, fit a model, or select winning inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LEGACY = 352
EXPECTED_SHARED = 104
COMPARE_FIELDS = (
    "success", "status", "termination_reason", "compilation_error", "error_code",
    "completed_makespan_ns", "makespan_ns", "peak_dram_bytes", "peak_vram_bytes",
    "h2d_bytes", "d2h_bytes", "ssd_read_bytes", "ssd_write_bytes", "decision_count",
    "event_counts", "cpu_event_span_ns", "gpu_event_span_ns", "copy_event_span_ns",
    "compute_event_span_ns", "cpu_gpu_event_overlap_ns", "copy_compute_event_overlap_ns",
    "any_engine_event_span_ns", "all_engine_event_uncovered_ns",
    "immutable_weight_reload_count", "immutable_weight_reload_bytes",
    "weight_eviction_count", "weight_released_allocation_bytes",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def read_optional(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def rows_or_unknown(path):
    value = read_optional(path)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"records must be a list of objects: {path}")
    keys = [(row.get("scenario_id"), row.get("candidate_id")) for row in value]
    if any(None in key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError(f"records contain missing or duplicate case identities: {path}")
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fmt(value):
    if value is None:
        return "未知／尚无产物"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def us(value):
    return "未知" if value is None else f"{value / 1000:.3f}"


def mib(value):
    return "未知" if value is None else f"{value / 1024**2:.4f}"


def index_rows(rows):
    return {(row["scenario_id"], row["candidate_id"]): row for row in rows}


def compare_legacy(baseline, current):
    before, after = index_rows(baseline), index_rows(current)
    common = sorted(before.keys() & after.keys())
    mismatches = []
    events_equal = 0
    outcomes_equal = 0
    for key in common:
        left, right = before[key], after[key]
        bad = [field for field in COMPARE_FIELDS
               if field not in left or field not in right or left[field] != right[field]]
        if bad:
            mismatches.append(dict(scenario_id=key[0], candidate_id=key[1], fields=bad))
        else:
            outcomes_equal += 1
        if (isinstance(left.get("event_sha256"), str) and
                left["event_sha256"] == right.get("event_sha256")):
            events_equal += 1
    return dict(expected_cases=EXPECTED_LEGACY, baseline_cases=len(baseline),
        current_cases=len(current), compared_cases=len(common),
        identical_outcomes=outcomes_equal, identical_event_hashes=events_equal,
        compared_fields=list(COMPARE_FIELDS), metric_mismatches=mismatches,
        missing_current_cases=[list(key) for key in sorted(before.keys()-after.keys())],
        unexpected_current_cases=[list(key) for key in sorted(after.keys()-before.keys())],
        passed=(len(before)==len(after)==len(common)==EXPECTED_LEGACY and
                outcomes_equal==events_equal==EXPECTED_LEGACY))


class Report:
    def __init__(self):
        self.markdown = []
        self.html = []

    def heading(self, title, level=2):
        self.markdown.append("#"*level + " " + title + "\n")
        self.html.append(f"<h{level}>{html.escape(title)}</h{level}>")

    def paragraph(self, text, *, callout=False):
        self.markdown.append(text + "\n")
        self.html.append(f'<p{chr(32)+"class=callout" if callout else ""}>{html.escape(text)}</p>')

    def table(self, headers, rows):
        self.markdown.append("| " + " | ".join(map(str, headers)) + " |\n| " +
                             " | ".join("---" for _ in headers) + " |\n" +
                             "\n".join("| " + " | ".join(fmt(value).replace("|", "\\|").replace("\n", " ")
                                for value in row) + " |" for row in rows) + "\n")
        self.html.append('<div class=table-wrap><table><thead><tr>' +
            "".join("<th>"+html.escape(str(value))+"</th>" for value in headers) +
            "</tr></thead><tbody>" + "".join("<tr>"+"".join("<td>"+html.escape(fmt(value))+"</td>"
                for value in row)+"</tr>" for row in rows)+"</tbody></table></div>")

    def code(self, text, language="python"):
        self.markdown.append(f"```{language}\n{text.rstrip()}\n```\n")
        self.html.append("<pre><code>"+html.escape(text.rstrip())+"</code></pre>")

    def evidence(self, label, value):
        self.markdown.append(f"完整证据：data/{label}\n")
        if value is None:
            self.paragraph(f"{label}：尚无产物，状态未知。")
        else:
            self.html.append('<details><summary>'+html.escape(label)+'（原始证据）</summary><pre>'+
                html.escape(json.dumps(value, ensure_ascii=False, indent=2))+'</pre></details>')

    def save(self, out, *, complete):
        title = "统一存储实现与回归实验报告"
        style = """
body{margin:0;background:#f3f5f8;color:#18283d;font:16px/1.75 'Microsoft YaHei',sans-serif}
main{max-width:1180px;margin:32px auto;background:white;padding:38px 48px;border-radius:12px}
h1{font-size:30px;color:#082c61}h2{font-size:23px;margin-top:36px;border-bottom:2px solid #e3eaf4;padding-bottom:8px}
h3{font-size:18px;margin-top:26px}p{max-width:1100px}.callout{background:#edf4ff;border-left:5px solid #175596;padding:14px 18px}
.banner{padding:12px 18px;border-radius:6px;background:#fff1d2;color:#704c00}.complete{background:#e8f5ed;color:#18552f}
.table-wrap{overflow:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.55}
th,td{border:1px solid #d6deea;text-align:left;vertical-align:top;padding:9px 11px}th{background:#edf2fa;white-space:nowrap}
tbody tr:nth-child(even){background:#fafbfd}pre{background:#f1f4f8;padding:14px;overflow:auto;white-space:pre-wrap;font-size:13px}
details{border:1px solid #d6deea;padding:10px 14px;margin:12px 0}summary{cursor:pointer;font-weight:bold}
a{color:#175596}@media(max-width:800px){main{margin:0;padding:20px}body{font-size:15px}}
"""
        banner = "证据齐全，按保存产物验收" if complete else "证据尚未齐全：缺失项目显示未知，不视为通过"
        body = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>'+title+'</title><style>'+style+'</style><main><div class="banner '+
            ('complete' if complete else '')+'">'+banner+'</div>'+"\n".join(self.html)+'</main></html>')
        (out/"REPORT_zh.html").write_text(body, encoding="utf-8", newline="\n")
        (out/"REPORT_zh.md").write_text("\n".join(self.markdown), encoding="utf-8", newline="\n")


def group_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_id"]].append(row)
    return dict(sorted(groups.items()))


def winning(rows):
    valid = [row for row in rows if row.get("success") is True
             and type(row.get("completed_makespan_ns")) is int]
    if not valid:
        return [], None
    best = min(row["completed_makespan_ns"] for row in valid)
    return sorted(row["candidate_id"] for row in valid if row["completed_makespan_ns"]==best), best


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (list,dict)) else value for key,value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"build/shared-storage-20261023")
    parser.add_argument("--legacy-dir", type=Path, default=Path("legacy-regression-v3"))
    parser.add_argument("--shared-dir", type=Path, default=Path("shared-acceptance-v3"))
    parser.add_argument("--baseline", type=Path, default=ROOT/"build/strategy-comparison-20261023/full-v1/records.json")
    parser.add_argument("--out", type=Path, default=ROOT.parent/"paper-write/第七教学周/experiment-reports/shared-storage-20261023")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    source, out = args.input.resolve(), args.out.resolve()
    legacy_dir=(source/args.legacy_dir).resolve()
    shared_dir=(source/args.shared_dir).resolve()
    baseline = rows_or_unknown(args.baseline.resolve())
    legacy = rows_or_unknown(legacy_dir/"records.json")
    shared = rows_or_unknown(shared_dir/"records.json")
    comparison = compare_legacy(baseline, legacy)
    supplied_comparison = read_optional(source/"regression-comparison.json")
    audit = read_optional(source/"shared-independent-audit.json")
    legacy_audit = read_optional(source/"legacy-independent-audit.json")
    tests = read_optional(source/"test-results.json")
    legacy_completion = read_optional(legacy_dir/"completion.json")
    shared_completion = read_optional(shared_dir/"completion.json")
    shared_protocol = read_optional(shared_dir/"protocol.json")
    selected_inputs = None
    if shared:
        relative = Path(shared[0].get("artifact_directory", "")) / "inputs/analytical-metadata.json"
        selected_path = (shared_dir/relative).resolve()
        if selected_path.is_relative_to(shared_dir):
            selected_inputs = read_optional(selected_path)
    formula_unchanged = supplied_comparison.get("formula_files_unchanged") if supplied_comparison else None
    accuracy_unchanged = supplied_comparison.get("frozen_accuracy_evidence_unchanged") if supplied_comparison else None
    shared_ok = sum(row.get("success") is True for row in shared)
    shared_identity_ok = bool(shared) and all(row.get("memory_mode")=="shared" and
        type(row.get("peak_shared_bytes")) is int and row.get("peak_shared_bytes",-1)>=0 and
        row.get("physical_memory_peaks_bytes")=={"shared":row.get("peak_shared_bytes")}
        for row in shared)
    shared_zero_payload = bool(shared) and all(row.get("h2d_bytes")==row.get("d2h_bytes")==0 for row in shared)
    completion_guards = dict(
        legacy_complete=legacy_completion.get("complete") is True if legacy_completion else None,
        shared_complete=shared_completion.get("passed") is True if shared_completion else None,
        legacy_records_hash=(legacy_completion.get("records_sha256")==sha(legacy_dir/"records.json"))
            if legacy_completion and (legacy_dir/"records.json").is_file() else None,
        shared_records_hash=(shared_completion.get("records_sha256")==stable_hash(shared))
            if shared_completion and (shared_dir/"records.json").is_file() else None,
    )
    completion_guards["test_log_hashes"] = bool(tests and tests.get("suites")) and all(
        (source/suite["log"]).is_file() and sha(source/suite["log"]) == suite.get("log_sha256")
        for suite in tests["suites"])
    complete = (comparison["passed"] and len(shared)==shared_ok==EXPECTED_SHARED and
        shared_identity_ok and shared_zero_payload and formula_unchanged is True and
        accuracy_unchanged is True and supplied_comparison and supplied_comparison.get("passed") is True and
        audit and audit.get("passed") is True and legacy_audit and legacy_audit.get("passed") is True and
        tests and tests.get("passed") is True and
        all(value is True for value in completion_guards.values()))
    complete = bool(complete)
    out.mkdir(parents=True, exist_ok=True)
    data = out/"data"
    data.mkdir(exist_ok=True)
    sources = {
        "baseline-records.json":args.baseline.resolve(),
        "legacy-records.json":legacy_dir/"records.json",
        "shared-records.json":shared_dir/"records.json",
        "regression-comparison.json":source/"regression-comparison.json",
        "shared-independent-audit.json":source/"shared-independent-audit.json",
        "legacy-independent-audit.json":source/"legacy-independent-audit.json",
        "test-results.json":source/"test-results.json",
        "legacy-completion.json":legacy_dir/"completion.json",
        "shared-completion.json":shared_dir/"completion.json",
        "shared-protocol.json":shared_dir/"protocol.json",
    }
    for suite in tests.get("suites", []) if tests else []:
        sources[suite["log"]] = source/suite["log"]
    sources["shared-library-example.log"] = source/"shared-library-example.log"
    for name, path in sources.items():
        if path.is_file():
            shutil.copy2(path, data/name)
    write_json(data/"report-regression-check.json", comparison)
    if selected_inputs is not None:
        write_json(data/"selected-shared-inputs.json", selected_inputs)
    write_csv(data/"legacy-records.csv", legacy)
    write_csv(data/"shared-records.csv", shared)
    receipt = {name:dict(path=str(path), exists=path.is_file(), sha256=sha(path) if path.is_file() else None)
               for name,path in sources.items()}
    write_json(data/"input-receipt.json", receipt)

    report = Report()
    report.heading("统一存储实现与回归实验报告", 1)
    report.paragraph("日期：2026-10-23。范围：单 CPU＋单 GPU；新增共享物理存储，保留独立 DRAM/VRAM。")
    report.paragraph("这次没有更换性能计算公式，也没有重新拟合 CPU/GPU 数据。改变的是“数据实际占几份内存、哪些设备可以访问、释放时能真正回收多少空间”。下述耗时来自未校准的解析模拟，不是某款真实统一内存设备的实测性能。", callout=True)
    report.heading("1．先看完成情况")
    suites = tests.get("suites", []) if isinstance(tests, dict) else []
    total_tests = sum(suite.get("tests",0) for suite in suites if type(suite.get("tests")) is int)
    report.table(["项目", "保存产物中的结果", "如何理解"], [
        ["原有独立存储组合", f"{len(legacy)}/{EXPECTED_LEGACY} 组有结果", "与修改前的同一组输入逐一对账"],
        ["旧结果完全相同", f"{comparison['identical_outcomes']}/{EXPECTED_LEGACY}", "包括耗时、内存峰值、搬运量、事件时段与动作计数"],
        ["旧事件内容完全相同", f"{comparison['identical_event_hashes']}/{EXPECTED_LEGACY}", "完整事件 JSON 的哈希一致，不只是均值接近"],
        ["新增共享存储组合", f"{shared_ok}/{EXPECTED_SHARED} 组到达目标；已有 {len(shared)} 组产物", "包含 Dense/MoE、Prefill/Decode、执行与存储策略组合"],
        ["专项与回归测试", f"累计 {total_tests} 次测试执行，整体 {fmt(tests.get('passed') if tests else None)}" if suites else None, "含专项重复覆盖；非法用例被正确拒绝也计为通过，跳过项另列"],
        ["独立审计", audit.get("passed") if audit else None, "独立检查事件成本、依赖、资源占用和共享容量"],
        ["旧模式独立审计", legacy_audit.get("passed") if legacy_audit else None, "另行重放原有模式的事件、容量与依赖，并核验保存产物的哈希"],
        ["计算公式文件未改变", formula_unchanged, "通过文件哈希检查"],
        ["冻结的准确性实验未改变", accuracy_unchanged, "本轮验证语义与回归，没有提高实机校准精度的声明"],
        ["完成标记与记录哈希", completion_guards, "只读本次指定的最终目录，旧的中断产物不参与验收"],
    ])
    if not complete:
        report.paragraph("本报告目前有缺失或未通过证据，不能当作完整验收结论。缺失项显示为未知，运行脚本时加 --require-complete 会以非零状态退出。", callout=True)

    report.heading("2．用几个例子理解本次改动")
    report.table(["情况", "原有独立存储", "共享存储的预期行为"], [
        ["CPU/GPU 使用同一份 10 GiB 权重", "CPU 与 GPU 各有独立副本时分别占空间", "同一分配可以共用，只计 10 GiB"],
        ["CPU 用 10 GiB 数据，GPU 用另一份 10 GiB 数据", "分别检查 DRAM/VRAM 容量", "合计 20 GiB；16 GiB 的共享池必须拒绝"],
        ["GPU 准备读取 CPU 已完成的数据", "真实 COPY：新缓冲区、搬运字节、搬运时间", "同一缓冲区使用 SHARE：零搬运字节，保留生产依赖"],
        ["显式要求复制数据", "产生独立目标分配", "仍产生独立分配并计费，不能假装成共享引用"],
        ["移除 GPU 引用，CPU 还持有同一数据", "移除独立 GPU 副本可以释放显存", "物理释放为 0 字节，不能虚报已经回收内存"],
        ["GPU 在运行，另一个视图想释放其输入", "已有读者锁保护", "锁覆盖同一物理分配，换个视图也不能绕过"],
        ["其他 APP 新占用内存", "按指定 DRAM/VRAM 池扣减", "使用 shared_external_bytes 对物理池扣减一次"],
    ])
    report.paragraph("这张表描述验收应满足的语义，不是厂商硬件测量。具体测试与独立审计状态见后文。共享存储不是把 H2D/D2H 带宽填成无穷大：它改变了副本身份和容量账本。")
    report.heading("3．哪些组件已适配")
    report.table(["组件", "现在怎样工作"], [
        ["公共 Python API", "Hardware.shared_memory 声明共享容量、对齐和运行时预留；旧 Hardware 默认继续使用 separate"],
        ["modeling", "记录物理分配与逻辑访问视图；workspace、真实副本、外部占用按物理池计入；计算公式保留"],
        ["IR 与 Compiler", "原有执行放置和数据准备接口继续使用；共享准备可编译为 SHARE；显式真实 COPY 保留成本要求"],
        ["缓存／预取／专家决策", "共同查询可访问性、增量占用、真实释放和独立备份；GPU 缓存预算保留为访问视图配额"],
        ["KV／多层／多轮／多请求／P/D", "共享 KV 不凭空多计一份；阶段交接和请求依赖仍受约束；支持状态保存与恢复"],
        ["服务资源／设备变化", "共享容量变化对同一个池生效；不自动推断新的共享带宽竞争参数"],
        ["profiling", "新增分配与计时边界元数据；未知同步成本保持未知，旧实测数据不被改写为统一内存数据"],
        ["GUI／产物", "共享模式只显示一条物理容量曲线；SHARE 是时间轴上的点；保存 peak_shared_bytes 与分配引用"],
    ])
    report.paragraph("计算仍是 max(计算工作量/有效算力, 访存字节/有效带宽) 加声明的启动开销；实际搬运仍按字节/链路带宽加固定延迟。SHARE 的零时间来自本轮明确声明的可见性抽象，不能解释成真实平台同步开销恰好为零。")
    report.code("""hardware = Hardware.shared_memory(
    cpu=Device(100e9, 20e9, launch_overhead_ns=500),
    gpu=Device(1e12, 100e9, launch_overhead_ns=2000),
    capacity_bytes=64 * 1024**2,
)
# 与原来相同的模型、工作负载、Scheduling 和 Simulator 流程
print(result.summary()["physical_memory_peaks_bytes"])
# 例如 {"shared": ...}；不要把逻辑 DRAM/VRAM 峰值再相加。""")

    report.heading("4．原有 352 组回归：独立存储是否被改变")
    report.paragraph("回归使用上一轮冻结的 4 个 Dense 场景和 12 个 MoE 场景，分别穷举 16／24 个既定候选，共 352 组。短／长 Prefill、Decode、多种专家路由均保留。两个仅 Prefill 的 phase_shift 输入与 hot 等价，仍作为原有控制组回归，不把它们当成独立发现。")
    before_idx = index_rows(baseline)
    comparison_rows = []
    for sid, entries in group_rows(legacy).items():
        identical = sum(all(field in before_idx.get((sid,row["candidate_id"]),{}) and
            field in row and before_idx[(sid,row["candidate_id"])][field]==row[field]
            for field in COMPARE_FIELDS) for row in entries)
        event_same = sum(before_idx.get((sid,row["candidate_id"]),{}).get("event_sha256") == row.get("event_sha256")
            and isinstance(row.get("event_sha256"), str) for row in entries)
        old_entries = [row for row in baseline if row["scenario_id"]==sid]
        old_best, old_time = winning(old_entries)
        new_best, new_time = winning(entries)
        comparison_rows.append([sid,len(entries),identical,event_same,us(old_time),us(new_time)])
    report.table(["场景", "结果数", "全部指标相同", "事件哈希相同", "修改前最短时间（µs）", "修改后最短时间（µs）"], comparison_rows)
    report.paragraph("这里的“完全相同”按整条候选记录检查；最佳值表只是方便阅读。路径、来源文件哈希、输入说明等元数据可以随实现变化，不能拿它们代替物理结果；实际对比字段已保存在 data/report-regression-check.json。")
    report.evidence("regression-comparison.json", supplied_comparison)
    if comparison["metric_mismatches"]:
        report.table(["不一致场景", "候选", "不一致字段"], [[row["scenario_id"],row["candidate_id"],row["fields"]] for row in comparison["metric_mismatches"]])

    report.heading("5．新增 104 组共享存储组合")
    report.paragraph("预设矩阵是 Dense 两个场景（Prefill=16/Decode=0；Prefill=4/Decode=1）各 16 个组合，加 MoE 三种路由（每种 Prefill=4/Decode=1）各 24 个组合，共 104 组。它们用于检查组合流程在共享存储下是否仍可执行；它们的输入长度与旧 352 组不同，不能直接把两张表的耗时相除当成统一内存加速比。")
    report.paragraph("e0=主要宏任务 GPU；e1=CPU；e2=FFN/专家 CPU；e3=Attention CPU；MoE 的 e4/e5 分别为 resident_first/current_cost 局部专家规则。s0/s1=LRU 无／有跨层预取；s2/s3=历史频率缓存无／有跨层预取。最优仅指本次有限候选，不是全局最优。")
    if selected_inputs:
        geometry=selected_inputs.get("model",{})
        hardware=selected_inputs.get("hardware",{})
        report.table(["公共实验设置", "声明值"], [
            ["层数／Hidden／FFN intermediate", f"{geometry.get('layers')} / {geometry.get('hidden_size')} / {geometry.get('intermediate_size')}"],
            ["Attention heads／KV heads", f"{geometry.get('heads')} / {geometry.get('kv_heads')}"],
            ["精度与 scratch", "F32；workspace=0 为显式简化假设，并非实测零分配"],
            ["CPU 参数", hardware.get("cpu")], ["GPU 参数", hardware.get("gpu")],
            ["共享物理容量", f"{mib(hardware.get('shared_capacity_bytes'))} MiB"],
            ["GPU 权重视图配额", shared_protocol.get("cache_budget_semantics") if shared_protocol else None],
            ["初始状态", "权重在 DRAM 逻辑视图，KV 从空状态开始；所有设备价格均为解析声明"],
        ])
    shared_rows = []
    for sid, entries in group_rows(shared).items():
        first=entries[0]
        best_ids, best_time=winning(entries)
        peaks=[row["peak_shared_bytes"] for row in entries if type(row.get("peak_shared_bytes")) is int]
        shared_rows.append([sid,first.get("prefill_tokens"),first.get("decode_steps"),
            first.get("route_pattern") or "Dense",sum(row.get("success") is True for row in entries),
            len(entries),", ".join(best_ids),us(best_time),
            f"{mib(min(peaks))}–{mib(max(peaks))}" if peaks else "未知",
            sum(row.get("shared_visibility_count",0) for row in entries),
            sum(row.get("h2d_bytes",0)+row.get("d2h_bytes",0) for row in entries)])
    report.table(["场景", "P Token", "D 步", "路由", "成功", "结果数", "最短候选（含并列）", "最短时间（µs）", "共享峰值范围（MiB）", "SHARE 总数", "H2D+D2H 总字节"], shared_rows)
    report.paragraph("共享模式里的 peak_dram_bytes 与 peak_vram_bytes 是同一个物理池的兼容视图，不是两个独立容量。报告使用 peak_shared_bytes；移除 GPU 访问视图仍可减少策略的缓存配额，但不一定减少物理占用。SHARE 次数是可见性交接次数，不能算作真实传输次数或硬件同步测量。")
    fixed_execution=defaultdict(list)
    for row in shared:
        fixed_execution[(row["scenario_id"],row["candidate_id"].split("-")[0])].append(row)
    comparable=[entries for entries in fixed_execution.values() if len(entries)==4 and
        all(row.get("success") is True and type(row.get("completed_makespan_ns")) is int for row in entries)]
    equal_storage=sum(len({row["completed_makespan_ns"] for row in entries})==1 for entries in comparable)
    if comparable:
        report.paragraph(f"一个容易理解的结果：固定执行策略后，{len(comparable)} 组四种存储策略对比中，{equal_storage} 组耗时完全相同。这里权重一开始就在共享池、物理容量充足，跨 CPU/GPU 的预载只是取得访问视图，不再产生原来的搬运等待。因此，原来针对 H2D 搬运设计的优化，在这批输入中不一定有性能收益。这不意味着缓存／预取在 SSD、紧容量或真实同步条件下都没有价值。", callout=True)
    report.paragraph("本轮仍保留 192 KiB 的 GPU 权重访问视图配额，以复用原有策略组合；它是策略约束，不是额外一块物理显存。局部专家规则也保留自己的选设备、可行性筛选顺序，不能把 current_cost 的名字理解为已经选出统一内存系统上的最优设备。")
    report.table(["共享矩阵一致性检查", "结果"], [
        ["所有记录声明 shared，并报告一致的共享物理峰值", shared_identity_ok if shared else None],
        ["共享准备未产生 H2D/D2H 真实流量", shared_zero_payload if shared else None],
        ["独立审计", audit.get("passed") if audit else None],
    ])

    report.heading("6．专项测试与独立审计")
    report.paragraph("专项测试检查“该能执行的能执行、该拒绝的会拒绝”。例如 16 GiB 放不下两份不同的 10 GiB 数据、读者仍在运行时不能释放其物理输入，正确报错就是通过；它不代表相应非法策略成功执行。")
    report.table(["测试组", "测试数", "失败", "错误", "跳过", "状态"],
        [[suite.get("name"),suite.get("tests"),suite.get("failures"),suite.get("errors"),suite.get("skipped"),suite.get("passed")] for suite in suites])
    report.paragraph("主要覆盖：共享／真实副本身份；容量、对齐、workspace；跨视图读者锁；回收释放字节；独立备份与 SSD；Prefill/Decode 和 KV 交接；多请求退休；因果路由；checkpoint 恢复；显式服务资源和外部内存变化。具体以保存的测试日志与审计检查为准。")
    if audit:
        totals=audit.get("totals",{})
        report.table(["独立审计读取的内容", "数量／结论"], [
            ["已审计组合", audit.get("case_count")],
            ["全部事件", totals.get("events")],
            ["计算事件", totals.get("compute_events")],
            ["零搬运 SHARE", totals.get("share_events")],
            ["实际释放为 0 字节的事件", totals.get("zero_release_events")],
            ["合计真实释放的物理字节", totals.get("physical_released_bytes")],
            ["全部组合通过", audit.get("passed")],
        ])
        report.paragraph("独立审计直接读取保存的图、事件和逐步内存快照，不通过性能器重新计算答案。104 组矩阵的审计针对固定设备状态；外部占用、服务资源和恢复行为另由相应专项测试覆盖。释放 0 字节的事件被如实保留，没有把一次删除引用伪记成释放一份权重。")
    report.evidence("test-results.json", tests)
    report.evidence("shared-independent-audit.json", audit)
    report.evidence("legacy-independent-audit.json", legacy_audit)

    report.heading("7．这次完成了什么，尚未证明什么")
    report.table(["可以得出的结论", "目前不能得出的结论"], [
        ["统一存储可以接入现有策略编译与事件推进框架", "已经准确预测 Apple、Jetson、Spark 或手机的真实性能"],
        ["共享引用和真实副本可被区分，物理容量不会因为两个视图而重复计算", "统一内存的一切访问／同步都没有开销"],
        ["使用相同计算价格，在共享拓扑下运行策略组合", "旧独显测量数据自动成为 UMA 实测成本"],
        ["旧 352 组是否逐项不变由回归产物确认", "已经实现新的最优搜索算法或证明论文级收益"],
        ["可注入外部可用容量／服务资源变化", "自动精确模拟 CPU/GPU 共享带宽、缓存一致性和页迁移"],
    ])
    report.paragraph("下一步若进行具体硬件分析，应先选择一台可测试设备，测量对应任务与交接边界，再声明是否需要共享带宽竞争参数。本轮没有增加实测设备，也没有更改冻结的准确性结论。优先用当前抽象比较容量、放置、KV 与执行协作的影响，同时将没有测量的同步／竞争因素明确标注为假设。", callout=True)

    report.heading("8．完整共享矩阵与复现入口")
    for sid, entries in group_rows(shared).items():
        report.heading(sid,3)
        report.table(["候选", "到达目标", "耗时（µs）", "共享峰值（字节）", "SHARE", "真实 H2D+D2H（字节）", "GPU／CPU 计算时段（ns）"],
            [[row["candidate_id"], row.get("success"),us(row.get("completed_makespan_ns")),
              row.get("peak_shared_bytes"),row.get("shared_visibility_count"),
              row.get("h2d_bytes",0)+row.get("d2h_bytes",0),
              f"{fmt(row.get('gpu_event_span_ns'))} / {fmt(row.get('cpu_event_span_ns'))}"]
             for row in sorted(entries,key=lambda row:tuple(int(part[1:]) for part in row["candidate_id"].split("-")))])
    report.paragraph("data/legacy-records.csv 是完整旧模式回归；data/shared-records.csv 是完整共享矩阵。data/ 保存原始 JSON、比较结果、测试与审计副本及输入 SHA-256。每条记录的 artifact_directory 指向原实验目录中的 IR、事件、策略审计和输入文件。")
    report.paragraph("如需重新执行模拟，而非重新排版报告，可在已安装本库的环境中使用以下命令。输出目录必须是新的空目录；与旧版本对比还需要保留的旧基线原始产物。")
    report.code("python experiments/strategy_comparison_campaign.py --out build/my-separate-regression --workers 4\n"
                "python experiments/shared_storage_campaign.py --out build/my-shared-acceptance --workers 4\n"
                "python experiments/shared_storage_audit.py --campaign build/my-shared-acceptance --out build/my-shared-audit.json", "bash")
    report.code("python experiments/shared_storage_report.py --require-complete", "bash")
    report.paragraph(f"原始实验目录：{source}")
    report.paragraph(f"本报告使用的回归目录：{legacy_dir}；共享验收目录：{shared_dir}")
    report.paragraph(f"原 352 组基线：{args.baseline.resolve()}")
    report.save(out, complete=complete)
    summary=dict(complete=complete, legacy_cases=len(legacy), shared_cases=len(shared),
        shared_successes=shared_ok, identical_legacy_outcomes=comparison["identical_outcomes"],
        identical_legacy_event_hashes=comparison["identical_event_hashes"],
        shared_physical_identity_check=shared_identity_ok, shared_zero_payload_check=shared_zero_payload,
        formula_files_unchanged=formula_unchanged, frozen_accuracy_evidence_unchanged=accuracy_unchanged,
        tests_passed=tests.get("passed") if tests else None,
        independent_audit_passed=audit.get("passed") if audit else None,
        legacy_independent_audit_passed=legacy_audit.get("passed") if legacy_audit else None,
        completion_guards=completion_guards,
        fixed_execution_storage_groups=len(comparable),
        equal_latency_storage_groups=equal_storage,
        limitations=dict(uncalibrated=True,numerical_llm_execution=False,
            new_hardware_measurements=False,automatic_shared_bandwidth_contention=False))
    write_json(out/"summary.json", summary)
    delivery={str(path.relative_to(out)).replace("\\","/"):sha(path)
              for path in sorted(out.rglob("*")) if path.is_file() and path.name!="delivery.json"}
    write_json(out/"delivery.json", dict(report_summary=summary,files=delivery,input_receipt=receipt))
    print(json.dumps(dict(summary,report=str(out/"REPORT_zh.html")),ensure_ascii=False))
    return 1 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
