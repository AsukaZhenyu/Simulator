"""Publish a reviewed report revision without changing measured evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from profiling.dac_validation.e2a import read, write, receipt, require, sha


def finalize(out, destination, software):
    out, destination, software = Path(out).resolve(), Path(destination).resolve(), Path(software).resolve()
    audit = read(out/'independent-audit.json')
    summary = read(out/'summary.json')
    require(audit['passed'] and audit['summary_sha256']==sha(out/'summary.json'), 'independent audit does not bind results')
    checks = read(software)
    require(checks['exit_code']==0, 'software verification failed')
    rows = summary['providers']['history_conditional']['cases']
    mean_pass = sum(r['gates']['each_primary_process_mean_ape'] for r in rows)
    p90_pass = sum(r['gates']['pooled_primary_block_p90_ape'] for r in rows)
    worst_mean = max(r['maximum_primary_process_mean_ape'] for r in rows)
    worst_p90 = max(r['pooled_primary_block_ape_p90'] for r in rows)
    primary_pass = summary['providers']['history_conditional']['fully_qualified_cases']
    static_pass = summary['providers']['context_static']['fully_qualified_cases']
    old_pass = summary['providers']['old_hot']['fully_qualified_cases']
    static_accuracy = sum(r['gates']['each_primary_process_mean_ape'] for r in summary['providers']['context_static']['cases'])
    idle = read(destination/'evidence/idle-summary.json')
    clock = read(destination/'evidence/clock-summary.json')
    pairs = idle['paired_contrasts']
    require(len(pairs)==4 and all(p['status']=='complete' and p['complete_activity_coverage_both'] for p in pairs),
            'reviewed idle statement requires four complete physical contrasts')
    idle_deltas = [p['target_action_changes']['host_wall_ns']['delta']/1000 for p in pairs]
    device_deltas = [p['target_action_changes']['device_memcpy_union_ns']['relative'] for p in pairs]
    total_processes=audit['completed_native_processes']+clock['completed_processes']+idle['completed']
    source = destination/'REPORT_zh.html'
    original = receipt(source)
    html = source.read_text(encoding='utf-8')
    additions = (
        '<div class="callout"><b>准确性与稳定性分开看：</b>'
        f'六个案例的进程均值误差门槛通过 {mean_pass}/6，最差进程均值误差 {worst_mean:.2%}；'
        f'区块误差 P90 门槛通过 {p90_pass}/6，最差 {worst_p90:.2%}。'
        f'同时满足准确性、CV 与观测一致性的完整验收仍为 {primary_pass}/6。'
        '这里的误差针对八次重放的区块平均，不代表单个算子的尾延迟。</div>'
        f'<p>这轮对照也限定了收益来源：在同一批独立验收数据上，旧热测价格的完整验收为 {old_pass}/6，'
        f'同上下文固定价格为 {static_pass}/6，历史条件价格为 {primary_pass}/6。历史条件化的均值准确性通过 '
        f'{mean_pass}/6，静态对照为 {static_accuracy}/6。不能据此声称逻辑复用状态已经充分解释设备波动。</p>'
        f'<p><b>空闲对照的新证据：</b>请求空闲 5 ms 后，目标 H2D 本身增加约 {min(idle_deltas):.0f}–{max(idle_deltas):.0f} μs；'
        '这些增加的时间不包含 Sleep，主要落在设备搬运开始前。四对诊断中的设备搬运时长变化约为 '
        f'{min(device_deltas):+.2%} 到 {max(device_deltas):+.2%}。注册内存下也有该现象，因而不能把它全归到普通内存暂存。'
        '这支持继续研究空闲后的提交/就绪恢复；尚不能区分 CPU 调度、驱动准备与设备唤醒，'
        '也不能把带 CUPTI 的跨进程对照直接认作无观测器的真实恢复参数。</p>'
        f'<p>主实验 {audit["completed_native_processes"]} 个进程的证据、校准参数、冻结顺序与验收指标已由独立脚本重新计算；'
        f'软件回归 {checks["passed"]} 项通过。'
        f'另有 {clock["completed_processes"]} 个频率诊断与 {idle["completed"]} 个空闲诊断均成功完成，'
        f'合计 {total_processes} 个测量进程。旧报告与旧测量保持原样。'
        '<a href="evidence/independent-audit.json">独立复核</a>；'
        '<a href="evidence/software-tests.json">软件验证</a>。</p>'
    )
    anchor = '<h2>1.'
    require(anchor in html, 'report section anchor absent')
    html = html.replace(anchor, additions+'\n'+anchor, 1)
    labels = {
        'each_primary_process_mean_ape': '每个主测进程的均值误差≤10%',
        'pooled_primary_block_p90_ape': '合并区块误差P90≤20%',
        'each_primary_process_cv': '每个主测进程CV≤10%',
        'each_observer_process_cv': '每个观测进程CV≤10%（辅助）',
        'each_paired_observer_relative_mean_difference': '每对主测/观测均值差≤5%',
    }
    for key, label in labels.items(): html=html.replace(key,label)
    html=html.replace('evidence/report-receipt.json','evidence/report-receipt-v2.json')
    for path in (out/'independent-audit.json',software):
        shutil.copy2(path,destination/'evidence'/path.name)
    path = destination/'REPORT_zh_v2.html'
    with path.open('x',encoding='utf-8',newline='\n') as stream: stream.write(html)
    record=dict(schema='h2d-history-reviewed-report-2', original_report=original,
        original_report_receipt=receipt(destination/'evidence/report-receipt.json'),
        final_report=receipt(path), generator=receipt(Path(__file__)),
        audit=receipt(destination/'evidence/independent-audit.json'),
        software=receipt(destination/'evidence/software-tests.json'),
        mean_accuracy_passes=mean_pass,p90_accuracy_passes=p90_pass,fully_qualified_cases=primary_pass,
        worst_process_mean_ape=worst_mean,worst_block_p90_ape=worst_p90,
        measurements_or_predictions_changed=False,physical_cause_fully_identified=False)
    write(destination/'evidence/report-receipt-v2.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--software',required=True);a=p.parse_args()
    import json
    print(json.dumps(finalize(a.out,a.destination,a.software),ensure_ascii=False))


if __name__=='__main__':main()
