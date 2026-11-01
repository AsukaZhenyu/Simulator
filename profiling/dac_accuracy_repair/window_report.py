"""Independent audit and append-only report for the 36-process window study.

This module neither launches native work nor fits parameters. REPORT_zh.html,
the 153-process study and its failures remain immutable. The new report is a
copy of that document with a separately identified actual-evidence appendix.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics

from .report import (DEFAULT_BUILD, DEFAULT_REPORT, csv_new, h, higher_quantile,
                     link, percent, read, receipt, sha, status, table,
                     transfer_value, us, write_new)

MODES = ("pageable", "registered")
TINY = (64, 1024, 2048, 16384)
FROZEN = ("action-profile-frozen.json", "transfer-pageable-frozen.json",
          "transfer-registered-frozen.json", "predictions.json")
LABEL = {"pageable": "普通可分页内存", "registered": "显式注册内存"}


def mean(values):
    return statistics.mean(values)


def cv(values):
    return statistics.stdev(values) / mean(values)


def recompute_metrics(processes, sizes, expected_count, calibration_ok):
    by_size = []
    for size in sizes:
        group = [row for row in processes if row["bytes"] == size]
        if len(group) != 3:
            continue
        measured = mean(row["mean_ns"] for row in group)
        prediction = group[0]["predicted_ns"]
        by_size.append(dict(bytes=size, observed_ns=measured, predicted_ns=prediction,
            ape=abs(prediction - measured) / measured,
            between_process_cv=cv([row["mean_ns"] for row in group])))
    size_ape = max((row["ape"] for row in by_size), default=None)
    process_ape = max((abs(row["predicted_ns"] - row["mean_ns"]) / row["mean_ns"] for row in processes), default=None)
    errors = [abs(row["predicted_ns"] - value) / value for row in processes for value in row["trial_ns"]]
    p90 = higher_quantile(errors) if errors else None
    cvs = sum(row["cv"] <= .1 for row in processes)
    observers = sum(abs(row["observer_effect"]) <= .05 for row in processes)
    original = len(processes) == expected_count and len(by_size) == len(sizes) and size_ape <= .1 and cvs == observers == expected_count
    strict = original and process_ape <= .1 and p90 <= .2
    return dict(expected_processes=expected_count, completed_processes=len(processes), sizes=by_size,
        maximum_size_mean_ape=size_ape, maximum_process_ape=process_ape, trial_p90_ape=p90,
        cv_passes=cvs, observer_passes=observers,
        recorded_original_size_mean_predicates=original, recorded_strict_process_mean_predicates=strict,
        qualification_allowed_by_calibration=calibration_ok,
        qualified_original_size_mean_gate=calibration_ok and original,
        qualified_strict_process_mean_gate=calibration_ok and strict)


def audit(window, summary):
    window = Path(window).resolve()
    manifest = read(window / "manifest.json")
    choice = read(window / "window-choice.json")
    barrier = read(window / "window-choice-barrier.json")
    source_summary_path = Path(manifest["source_summary"]["path"])
    source = source_summary_path.parent
    old = read(source_summary_path)
    checks, sources, raw_rows = [], {}, []

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def equal(name, actual, expected):
        numeric = type(actual) in (int, float) and type(expected) in (int, float)
        passed = math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-7) if numeric else actual == expected
        check(name, passed, actual=actual, expected=expected)

    def bound(item):
        actual = receipt(item["path"])
        sources[actual["path"]] = dict(actual, expected_sha256=item["sha256"])
        check("hash:" + actual["path"], actual["sha256"] == item["sha256"])

    def at(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    for item in (receipt(window / "manifest.json"), receipt(window / "summary.json"), manifest["source_campaign"], manifest["source_summary"], manifest["source_frozen_barrier"], manifest["binary"]):
        bound(item)
    for binding in manifest["sources"]:
        bound(binding["source"])
        bound(binding["archived"])
    for item in manifest["runtime"]:
        bound(item)
    for binding in manifest["programs"].values():
        bound(binding["program"])
        bound(binding["reference"])
    old_barrier = read(source / "frozen-barrier.json")
    for name in FROZEN:
        equal("unchanged_old_formula_or_prediction:" + name, sha(window / name), old_barrier["files"][name])
        equal("unchanged_source_formula_or_prediction:" + name, sha(source / name), old_barrier["files"][name])
    for name, digest in barrier["files"].items():
        bound(dict(path=str(window / name), sha256=digest))
    protocol = manifest["protocol"]
    check("one_fixed_candidate_no_refit", protocol["candidate_count"] == 1 and protocol["candidate_repeats"] == 4096 and protocol["short_repeats"] == 256 and protocol["parameter_refit"] is False and summary["parameter_refit"] is False)
    check("predeclared_calibration_and_holdout_inventory", protocol["calibration_bytes"] == 4096 and protocol["tiny_heldouts"] == list(TINY) and len(manifest["jobs"]) == 36 and protocol["heldout_runs_even_if_calibration_rejects"])
    check("original_failures_preserved", summary["original_results_modified"] is False and protocol["original_R256_failures_retained"] is True)
    check("choice_frozen_without_target", choice["heldout_read"] is False and choice["parameter_refit"] is False and barrier["heldout_read"] is False)
    executions = []
    for split in ("transfer-profile", "transfer-heldout"):
        file = window / f"{split}-executions.json"
        bound(receipt(file))
        executions.extend(read(file))
    planned = {job["id"]: job for job in manifest["jobs"]}
    check("frozen_full_job_bindings", len(executions) == 36 and all(record["job"] == planned.get(record["job"]["id"]) for record in executions))
    check("complete_unique_job_inventory", sorted(record["job"]["id"] for record in executions) == sorted(planned))
    equal("completed_count", summary["completed"], sum(record["status"] == "completed" for record in executions))
    equal("failed_count", summary["failed"], sum(record["status"] != "completed" for record in executions))
    check("all_owned_children_exited", all(record.get("process_exited_verified") for record in executions))
    check("calibration_before_choice", all(at(record["completed_at_utc"]) <= at(choice["frozen_at_utc"]) for record in executions if record["job"]["split"] == "transfer-profile"))
    check("heldouts_after_choice_barrier", all(at(record["started_at_utc"]) >= at(barrier["frozen_at_utc"]) for record in executions if record["job"]["split"] == "transfer-heldout"))
    check("choice_before_barrier", at(choice["frozen_at_utc"]) <= at(barrier["frozen_at_utc"]))
    native_ok = clock_ok = state_ok = pointer_ok = byte_ok = prep_ok = affinity_ok = True
    for record in executions:
        if record["status"] != "completed":
            continue
        bound(record["native"])
        native, job = read(record["native"]["path"]), record["job"]
        binding = manifest["programs"][job["program_key"]]
        program, oracle = read(binding["program"]["path"]), read(binding["reference"]["path"])
        req = job["request"]
        native_ok = native_ok and native["request"] == req and native["process_id"] == record["process_id"] and native["program_sha256"] == program["program_sha256"] and native["source_sha256"] == manifest["native_source_sha256"] and native["caller_source_sha256_match"]
        process, thread = native["process_affinity"], native["main_thread_affinity"]
        affinity_ok = affinity_ok and process["requested_mask"] == process["observed_mask"] == process["final_mask"] == manifest["process_affinity_mask"] and process["applied_before_backend_initialization"] and thread["requested_mask"] == thread["observed_mask"] == thread["final_mask"] == 4 and thread["restored"]
        expected_prep = dict(reset_before_batch=False, untimed_transfer_prime_passes=1,
            timed_replays_per_batch=job["repeats"], fixed_input_replay=True,
            warmup_iterations_min=req["sampling"]["warmup"], warmup_ms_min=req["sampling"]["warmup_min_ms"],
            equal_warmup_per_memory_mode_visit=True)
        prep_ok = prep_ok and native["preparation_contract"] == expected_prep and len(native["trials"]) == 32 and native["measurement_api"] == "ggml_tensor_copy_then_gpu_cpu_synchronize" and native["completion_semantic"] == "host_response_including_copy_api_and_gpu_cpu_completion"
        expected_kind = "registered_host" if job["memory_mode"] == "registered" else "unregistered_pageable"
        pointer_ok = pointer_ok and native["host_registration_restored"] and len(native["visits"]) == 1 and all(t["outside_trial_timers"] for t in native["host_registration_transitions"])
        proof = native["visits"][0]["full_byte_verification"]
        byte_ok = byte_ok and proof["full_byte_equality"] and proof["checked_bytes"] == oracle["bytes"] and proof["source_sha256"] == proof["output_sha256"] == oracle["sha256"]
        for row in native["trials"]:
            clock_ok = clock_ok and row["end_tick"] > row["begin_tick"] and row["repeats"] == job["repeats"] and abs((row["end_tick"] - row["begin_tick"]) * 1e9 / native["clock"]["frequency_hz"] - row["wall_ns"]) <= 1 and row["actual_stamped"] == (req["order"][row["block"]] == "B")
            state = row["preblock_state"]
            state_ok = state_ok and state["sample_tick"] <= row["begin_tick"] and state["derived_from_target_duration"] is False and state["outside_timed_interval"]
            pointer_ok = pointer_ok and row["host_pointer"].get(expected_kind) is True and row["memory_mode"] == job["memory_mode"]
        bare = [row["wall_ns"] / row["repeats"] for row in native["trials"] if not row["actual_stamped"]]
        stamped = [row["wall_ns"] / row["repeats"] for row in native["trials"] if row["actual_stamped"]]
        raw_rows.append(dict(job=job["id"], split=job["split"], bytes=binding["bytes"], memory_mode=job["memory_mode"], round=job["round"], repeats=job["repeats"], mean_ns=mean(bare), cv=cv(bare), trial_ns=bare, observer_effect=mean(stamped) / mean(bare) - 1, raw=record["native"]))
    for name, value in (("native_source_request_program_identity", native_ok), ("QPC_repeat_and_observer_order", clock_ok), ("causal_outside_timer_preblock_state", state_ok), ("actual_host_mode_registration_restore", pointer_ok), ("complete_independent_byte_oracle", byte_ok), ("explicit_sampling_preparation_contract", prep_ok), ("same_broad_process_and_pinned_main_affinity", affinity_ok)):
        check(name, value)
    calibration = [row for row in raw_rows if row["split"] == "transfer-profile"]
    long = [row for row in calibration if row["repeats"] == 4096]
    short = [row for row in calibration if row["repeats"] == 256]
    cvs = sum(row["cv"] <= .1 for row in long)
    obs = sum(abs(row["observer_effect"]) <= .05 for row in long)
    equal("calibration_long_cv_passes", choice["long_cv_passes"], cvs)
    equal("calibration_long_observer_passes", choice["long_observer_passes"], obs)
    shifts = []
    for mode in MODES:
        a = [row for row in short if row["memory_mode"] == mode]
        b = [row for row in long if row["memory_mode"] == mode]
        check("calibration_mode_inventory:" + mode, len(a) == len(b) == 3)
        shift = mean(row["mean_ns"] for row in b) / mean(row["mean_ns"] for row in a) - 1
        shifts.append(shift)
        equal("calibration_aggregate_shift:" + mode, next(row["aggregate_mean_shift"] for row in choice["modes"] if row["memory_mode"] == mode), shift)
        for round_id in range(3):
            first = next(row for row in a if row["round"] == round_id)
            second = next(row for row in b if row["round"] == round_id)
            expected = next(row for row in choice["individual_adjacent_pairs"] if row["memory_mode"] == mode and row["round"] == round_id)
            for key, value in (("short_mean_ns", first["mean_ns"]), ("long_mean_ns", second["mean_ns"]), ("individual_mean_shift", second["mean_ns"] / first["mean_ns"] - 1), ("short_cv", first["cv"]), ("long_cv", second["cv"]), ("short_observer_effect", first["observer_effect"]), ("long_observer_effect", second["observer_effect"])):
                equal(f"adjacent_pair:{mode}:{round_id}:{key}", expected[key], value)
    accepted = len(calibration) == 12 and len(long) == 6 and cvs == obs == 6 and all(abs(shift) <= .05 for shift in shifts)
    equal("frozen_calibration_acceptance", choice["accepted"], accepted)
    equal("summary_calibration_equals_frozen_choice", summary["calibration"], choice)
    predictions = read(window / "window-predictions.json")
    check("new_window_predictions_frozen_before_barrier", at(predictions["frozen_at_utc"]) <= at(barrier["frozen_at_utc"]) and predictions["heldout_read"] is False)
    lookup = {}
    for row in predictions["predictions"]:
        model = read(window / f"transfer-{row['memory_mode']}-frozen.json")
        prediction = transfer_value(model["parameters"], row["bytes"])
        equal(f"unchanged_formula_projection:{row['memory_mode']}:{row['bytes']}", row["predicted_ns"], prediction)
        check(f"explicit_cross_window_projection:{row['memory_mode']}:{row['bytes']}", row["formula_from_repeats"] == 256 and row["measurement_repeats"] == 4096 and row["parameter_refit"] is False)
        lookup[row["memory_mode"], row["bytes"]] = prediction
    check("eight_predeclared_window_predictions", set(lookup) == {(mode, size) for mode in MODES for size in TINY})
    new = [row for row in raw_rows if row["split"] == "transfer-heldout"]
    for row in new:
        row["predicted_ns"] = lookup[row["memory_mode"], row["bytes"]]
        row["ape"] = abs(row["predicted_ns"] - row["mean_ns"]) / row["mean_ns"]
    for mode in MODES:
        tiny = [row for row in new if row["memory_mode"] == mode]
        large = [dict(row, repeats=256) for row in old["D2H"][mode]["processes"] if row["bytes"] not in TINY]
        check("twelve_reused_large_rows:" + mode, len(large) == 12 and all(row["bytes"] >= 262144 for row in large))
        for row in large:
            bound(row["raw"])
            native = read(row["raw"]["path"])
            bare = [r["wall_ns"] / r["repeats"] for r in native["trials"] if not r["actual_stamped"]]
            stamped = [r["wall_ns"] / r["repeats"] for r in native["trials"] if r["actual_stamped"]]
            equal("reused_large_mean:" + row["raw"]["path"], row["mean_ns"], mean(bare))
            equal("reused_large_cv:" + row["raw"]["path"], row["cv"], cv(bare))
            equal("reused_large_observer:" + row["raw"]["path"], row["observer_effect"], mean(stamped) / mean(bare) - 1)
        for name, processes, sizes, count in (("new_R4096_tiny", tiny, TINY, 12), ("combined_policy", tiny + large, sorted(TINY + tuple(sorted({row["bytes"] for row in large}))), 24)):
            result = recompute_metrics(processes, sizes, count, accepted)
            expected = summary["D2H"][mode][name]
            for field, value in result.items():
                if field == "sizes":
                    equal(f"window_size_inventory:{mode}:{name}", [row["bytes"] for row in expected[field]], [row["bytes"] for row in value])
                    for a, b in zip(expected[field], value):
                        for key, number in b.items():
                            equal(f"window_size:{mode}:{name}:{b['bytes']}:{key}", a[key], number)
                else:
                    equal(f"window_gate:{mode}:{name}:{field}", expected[field], value)
        equal("original_R256_outcome_unchanged:" + mode, summary["D2H"][mode]["original_R256_qualification_unchanged"], old["D2H"][mode]["qualified_original_size_mean_gate"])
    return dict(schema="dac-accuracy-repair-window-independent-audit-1", created_at_utc=datetime.now(timezone.utc).isoformat(), passed=all(row["passed"] for row in checks), checks=checks, check_count=len(checks), failed_checks=[row for row in checks if not row["passed"]], sources=list(sources.values()), native_raw_processes_read=len(raw_rows), reused_original_large_raws=24, calibration_recomputed_from_raw=True, frozen_parameters_refit=False, native_launches=0), raw_rows


def audit_catalog(path):
    """Check all 16 finite-grid lookups and unsupported requests; no timing."""
    from llm_infer_model.performance.validated_transfer import ValidatedTransferCatalog

    path = Path(path).resolve()
    manifest = read(path)
    runtime = ValidatedTransferCatalog.from_manifest(manifest)
    checks, sources, lookups = [], [receipt(path)], []
    expected_grid = {str(size): 4096 for size in TINY}
    expected_grid.update({str(size): 256 for size in (262144, 2097152, 8388608, 33554432)})

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    def bound(item):
        actual = receipt(item["path"])
        sources.append(actual)
        check("hash:" + actual["path"], actual["sha256"] == item["sha256"])
        return read(item["path"])

    check("two_observed_memory_modes", set(manifest["entries"]) == set(MODES))
    check("no_refit_or_old_profile_promotion", manifest["coefficients_changed"] is False and manifest["parameter_refit"] is False and manifest["original_profile_qualification"] == dict(pageable=False, registered=False))
    rejected = 0
    for mode in MODES:
        entry = manifest["entries"][mode]
        qualification = bound(entry["qualification_receipt"])
        profile = bound(qualification["source_profile"])
        main = bound(qualification["main_summary"])
        window = bound(qualification["window_summary"])
        main_audit = bound(qualification["independent_audit"])
        window_audit = bound(qualification["window_independent_audit"])
        check("receipt_schemas_and_accepted_audits:" + mode,
              qualification["schema"] == "prepared-transfer-sampling-policy-qualification-1" and main_audit["passed"] is True and window_audit["passed"] is True and window_audit["schema"] == "dac-accuracy-repair-window-independent-audit-1")
        check("source_parameters_and_contract_unchanged:" + mode,
              entry["parameters"] == profile["parameters"] and entry["source_contract"] == profile["contract"] and entry["minimum_bytes"] == profile["minimum_bytes"] and entry["maximum_bytes"] == profile["maximum_bytes"] and profile["qualified"] is False)
        check("exact_eight_byte_replay_coordinates:" + mode,
              entry["validated_replays"] == qualification["validated_replays"] == expected_grid)
        check("qualified_only_new_recorded_sampling_policy:" + mode,
              entry["qualified"] is True and window["D2H"][mode]["combined_policy"]["qualified_strict_process_mean_gate"] is True and main["D2H"][mode]["qualified_strict_process_mean_gate"] is False and qualification["original_R256_qualification_changed"] is False and qualification["single_copy_latency_qualified"] is False and qualification["coefficients_changed"] is False and qualification["parameter_refit"] is False)
        for size, repeats in expected_grid.items():
            target = copy.deepcopy(entry["source_contract"])
            target["preparation_contract"]["timed_replays_per_batch"] = repeats
            result = runtime.prediction(int(size), contract=target)
            expected = transfer_value(profile["parameters"], int(size))
            passed = math.isclose(result["duration_ns"], expected, rel_tol=1e-13, abs_tol=1e-8) and result["qualified_for_requested_contract"] is True and result["requested_replays"] == repeats and result["sampling_equivalence_claimed"] is False and result["coefficients_changed"] is False
            check(f"finite_lookup:{mode}:{size}:R{repeats}", passed, duration_ns=result["duration_ns"], expected_ns=expected)
            lookups.append(dict(memory_mode=mode, bytes=int(size), repeats=repeats, duration_ns=result["duration_ns"], qualified=passed))
            wrong = copy.deepcopy(target)
            wrong["preparation_contract"]["timed_replays_per_batch"] = 1
            try:
                runtime.prediction(int(size), contract=wrong)
                refused = False
            except ValueError:
                refused = True
            check(f"single_copy_rejected:{mode}:{size}", refused)
            rejected += refused
        unseen = copy.deepcopy(entry["source_contract"])
        unseen["preparation_contract"]["timed_replays_per_batch"] = 4096
        try:
            runtime.prediction(128, contract=unseen)
            refused = False
        except ValueError:
            refused = True
        check("unseen_in_envelope_byte_rejected:" + mode, refused)
        rejected += refused
    sources.append(receipt(Path(__file__)))
    sources.append(receipt(Path(__file__).resolve().parents[2] / "modeling/llm_infer_model/performance/validated_transfer.py"))
    return dict(schema="dac-accuracy-repair-catalog-independent-audit-1", created_at_utc=datetime.now(timezone.utc).isoformat(), passed=all(row["passed"] for row in checks), checks=checks, check_count=len(checks), failed_checks=[row for row in checks if not row["passed"]], sources=sources, finite_lookups=lookups, lookups_checked=len(lookups), unsupported_requests_rejected=rejected, native_launches=0, benchmark_or_parameter_fit=False)


def build(window=DEFAULT_BUILD / "window-validation-v1", report_dir=DEFAULT_REPORT, catalogs=()):
    window, report_dir = Path(window).resolve(), Path(report_dir).resolve()
    if not (window / "summary.json").is_file():
        raise FileNotFoundError("最终 window summary 不存在；不生成占位追加报告")
    v1_path = report_dir / "REPORT_zh.html"
    v1_receipt = receipt(v1_path)
    summary, manifest = read(window / "summary.json"), read(window / "manifest.json")
    old_summary = read(manifest["source_summary"]["path"])
    catalogs = [Path(path).resolve() for path in catalogs]
    if any(not path.is_file() for path in catalogs):
        raise FileNotFoundError("catalog link requires an existing actual artifact")
    independent, rows = audit(window, summary)
    audit_path = report_dir / "independent-audit-window.json"
    if audit_path.exists():
        retained = read(audit_path)
        if any(retained[field] != independent[field] for field in ("passed", "checks", "sources")):
            raise ValueError("retained window audit differs from current read-only recomputation")
        independent = retained
    else:
        write_new(audit_path, independent)
    csv_rows = [dict(job=row["job"], split=row["split"], bytes=row["bytes"], memory_mode=row["memory_mode"], round=row["round"], repeats=row["repeats"], mean_ns=row["mean_ns"], trial_cv=row["cv"], observer_effect=row["observer_effect"], predicted_ns=row.get("predicted_ns"), process_mean_ape=row.get("ape"), raw_path=row["raw"]["path"], raw_sha256=row["raw"]["sha256"]) for row in rows]
    csv_new(report_dir / "window-processes-independent.csv", csv_rows)
    choice = summary["calibration"]
    parts = ['<section id="window-study"><h2>8. 独立追加：固定长采样窗口研究</h2><div class="notice">前面的 153 进程及六进程诊断全部原样保留。本节另外使用 36 个新进程，评估不同的 block-mean 采样协议。R256 未通过的旧结果不改变；R4096 也不代表单次 COPY 延迟分布已合格。</div>']
    parts.append(table(["新研究检查", "实际结果"], [["新进程完成", h(f"{summary['completed']}/{summary['declared']}；失败 {summary['failed']}")], ["唯一候选窗口", "R4096，预先固定，不继续增大到结果通过"], ["4 KiB 校准决定", status(choice["accepted"])], ["长窗校准 CV / 观察器", h(f"{choice['long_cv_passes']}/6；{choice['long_observer_passes']}/6")], ["独立 raw 复算", status(independent["passed"]) + h(f"；{independent['check_count']} 条检查")]]))
    parts.append('<h3>8.1 怎么决定窗口：先校准，再冻结，再留出</h3><p>在 4 KiB 上，每种内存各做 3 组相邻的新进程 R256/R4096 对照，总计 12 进程；短/长进程顺序 AB/BA 交错，进程内 bare/stamped 仍为 ABBA/BAAB。它是相邻的不同进程，不能称作同进程因果对照。</p><p>接受候选须同时满足：全部 6 个长窗校准进程 CV≤10%、观察器差≤5%；各内存类别的三进程聚合长/短均值变化绝对值≤5%。随后冻结选择与原公式的预测；无论校准接受与否，都采集原四个小尺寸（64 B、1 KiB、2 KiB、16 KiB）×两内存×3 个新进程的 R4096 留出数据。没有读取留出结果后重新选窗口或重拟合参数。</p>')
    parts.append(table(["内存", "短窗进程", "长窗进程", "聚合均值变化", "≤5%"], [[h(LABEL[row["memory_mode"]]), h(row["short_completed"]), h(row["long_completed"]), percent(row["aggregate_mean_shift"]), status(row["mean_shift_pass"])] for row in choice["modes"]]))
    parts.append(table(["内存/轮次", "短窗 µs", "长窗 µs", "个别进程对均值变化", "短/长 CV", "短/长观察器差"], [[h(f"{LABEL[row['memory_mode']]} / {row['round']}"), us(row["short_mean_ns"]), us(row["long_mean_ns"]), percent(row["individual_mean_shift"]), h(percent(row["short_cv"]) + " / " + percent(row["long_cv"])), h(percent(row["short_observer_effect"]) + " / " + percent(row["long_observer_effect"]))] for row in choice["individual_adjacent_pairs"]]))
    parts.append('<p>冻结的均值迁移门按每种内存的聚合值判断；个别相邻进程对的变化仍全部列出。聚合门通过不表示每一对变化≤5%，也不证明差异仅来自方差缩小；进程状态与时间漂移仍可能影响均值。</p>')
    parts.append('<h3>8.2 原公式不改，新的采样契约单独验收</h3><p>本轮直接使用前述冻结公式的参数，对 R4096 的 block-mean 每 COPY 响应作显式跨窗口投影。参数文件的 R256 准备契约和 qualified=False 均未修改，也没有静默声称 R256 契约与 R4096 完全相等。新留出结果仅评估这个已声明的 R4096 投影。</p>')
    overview = []
    for mode in MODES:
        for field, label in (("new_R4096_tiny", "4 小尺寸：新 R4096"), ("combined_policy", "8 尺寸：混合采样协议")):
            data = summary["D2H"][mode][field]
            overview.append([h(LABEL[mode]), h(label), h(f"{data['completed_processes']}/{data['expected_processes']}"), percent(data["maximum_size_mean_ape"]), percent(data["maximum_process_ape"]), percent(data["trial_p90_ape"]), h(f"{data['cv_passes']}/{data['expected_processes']}"), h(f"{data['observer_passes']}/{data['expected_processes']}"), status(data["qualified_strict_process_mean_gate"])])
    parts.append(table(["内存", "协议", "进程证据", "最差尺寸均值误差", "最差进程均值误差", "试次 P90", "CV 通过", "观察器通过", "严格门"], overview))
    details = []
    for mode in MODES:
        for row in summary["D2H"][mode]["new_R4096_tiny"]["sizes"]:
            group = [r for r in rows if r["split"] == "transfer-heldout" and r["memory_mode"] == mode and r["bytes"] == row["bytes"]]
            details.append([h(LABEL[mode]), h(row["bytes"]), us(row["observed_ns"]), us(row["predicted_ns"]), percent(row["ape"]), percent(max(r["ape"] for r in group)), h(f"{sum(r['cv'] <= .1 for r in group)}/3"), h(f"{sum(abs(r['observer_effect']) <= .05 for r in group)}/3")])
    parts.append(table(["内存", "字节数", "新实测 µs", "冻结预测 µs", "尺寸均值误差", "最大进程误差", "CV 门", "观察器门"], details))
    dense_count = len(old_summary['programs'])
    parts.append('<h3>8.3 “混合采样协议”不等于旧实验改判</h3><ul><li>每种内存的 combined policy 有 24 个进程证据：12 个新 R4096 小尺寸进程＋12 个原 R256 大尺寸进程。原四个大尺寸（256 KiB、2 MiB、8 MiB、32 MiB）没有重新采集。</li><li>这是按字节数选择测量窗口的新协议；它的准备契约必须包含窗口规则及这 8 个离散留出字节坐标。中间未测尺寸不据此自动合格。不是把八种尺寸都称为 R4096，也不是原 153 个进程重新验收通过。</li><li>CV 与 P90 对象是多次 COPY 后的平均响应块；窗口变长后的低 CV 不证明逐 COPY 的延迟抖动被建模或消除。完整 Dense 仍保留前述主 hot ' + h(f"{old_summary['counts']['hot_qualified_programs']}/{dense_count}") + '、新 prep ' + h(f"{old_summary['counts']['preparation_matched_programs']}/{dense_count}") + ' 的原结果。</li><li>原 R256 模型和数据的资格标志不翻转。新协议若在运行时启用，需要独立的 scope/资格收据；不得用于未验证的单次 COPY、Dense 前驱上下文或真实硬件带宽缩放。</li></ul>')
    parts.append(table(["保留的原 R256 资格", "结果"], [[h(LABEL[mode]), status(summary["D2H"][mode]["original_R256_qualification_unchanged"])] for mode in MODES]))
    source_items = [(window / "manifest.json", "36 进程预设矩阵与固定候选"), (window / "window-choice.json", "只由校准决定的冻结选择"), (window / "window-choice-barrier.json", "选择/预测/原参数 SHA 屏障"), (window / "window-predictions.json", "留出前冻结的原公式投影"), (window / "summary.json", "窗口研究实际结果"), (audit_path, "本节独立原始数据复算"), (report_dir / "window-processes-independent.csv", "独立逐进程指标 CSV"), (v1_path, "原始 v1 报告（未覆盖）")]
    source_items.extend((path, "运行时契约、资格或软件验收依据：" + path.name) for path in catalogs)
    parts.append('<h3>8.4 证据链接</h3><ul>' + ''.join('<li>' + link(path, label, report_dir) + '</li>' for path, label in source_items) + '</ul></section>')
    original = v1_path.read_text(encoding="utf-8")
    if '</body>' not in original:
        raise ValueError("v1 report lacks body boundary")
    addition = ''.join(parts)
    document = original.replace('</body>', addition + '</body>')
    latest_status = '；'.join(LABEL[mode] + '新混合采样协议严格门：' + status(summary['D2H'][mode]['combined_policy']['qualified_strict_process_mean_gate']) for mode in MODES)
    document = document.replace('<body>', '<body><div class="notice">这是追加版报告。' + latest_status + '。原 v1 的 R256 失败与 Dense 结果完整保留；新增证据见 <a href="#window-study">第 8 节：固定长窗口研究</a>。</div>', 1)
    destination = report_dir / "REPORT_zh_v2.html"
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document)
    if sha(v1_path) != v1_receipt["sha256"]:
        raise ValueError("v1 report unexpectedly changed")
    write_new(report_dir / "window-report-sources.json", dict(schema="dac-accuracy-repair-window-report-sources-1", sources=[v1_receipt, receipt(window / "summary.json"), receipt(window / "manifest.json"), receipt(window / "window-choice-barrier.json"), receipt(Path(__file__)), receipt(audit_path), *[receipt(path) for path in catalogs]], original_report_sha256_unchanged=True, parameter_refit=False, native_launches=0))
    return dict(report=str(destination), audit_passed=independent["passed"], checks=independent["check_count"], original_v1_unchanged=True, declared=summary["declared"], completed=summary["completed"], failed=summary["failed"], native_launches=0, parameter_refit=False)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, default=DEFAULT_BUILD / "window-validation-v1")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--catalog", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(build(args.window, args.report_dir, args.catalog), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
