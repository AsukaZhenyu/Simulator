"""Post-acquisition provenance and descriptive context checks, never refitting."""
from pathlib import Path
import json
import statistics
import shutil
import subprocess
import os
import time

from profiling.dac_validation import e2a
from .report import csv_write


def close(out):
    out=Path(out).resolve();root=Path(__file__).resolve().parents[2]
    manifest=e2a.read(out/"manifest.json");summary=e2a.read(out/"summary.json");groups={}
    for ex in e2a.read(out/"observer-executions.json"):
        if ex["status"]!="completed":continue
        native=e2a.read(ex["native"]["path"]);key=ex["job"]["program_key"]
        means={name:statistics.mean(t["wall_ns"]/t["repeats"] for t in native["trials"] if t["actual_stamped"]==stamped) for name,stamped in (("bare",False),("stamped",True))}
        groups.setdefault(key,[]).append(dict(**means,raw=ex["native"]))
    rows=[]
    for program in summary["programs"]:
        key=program["program_key"];obs=groups[key];bare=statistics.mean(p["bare"] for p in obs);stamped=statistics.mean(p["stamped"] for p in obs)
        rows.append(dict(program_key=key,layers=program["layers"],strategy=program["strategy"],heldout_mean_ns=program["measured_mean_ns"],
            observer_bare_mean_ns=bare,observer_stamped_mean_ns=stamped,observer_bare_vs_heldout_relative_shift=bare/program["measured_mean_ns"]-1,
            observer_stamped_vs_bare_relative_effect=stamped/bare-1))
    csv_write(out/"observer_context_comparison.csv",rows)
    e2a.write(out/"observer-context-comparison.json",dict(schema="dac-refinement-descriptive-context-check-1",generated_at_utc=e2a.utc(),rows=rows,
        native_sources=groups,post_hoc_analysis=True,new_model_fitted=False,new_timing_collected=False,
        primary_heldout_context="steady::run_full resets before each trial and directly times8 fixed-input replays",
        observer_context="each ABBA/BAAB block resets, executes one additional untimed full pass, then times8 replays",
        interpretation="Within-observer bare/stamped agreement does not make observer action prices unbiased for the primary heldout context. Between-phase/fresh-process state remains confounded with the explicit preconditioning difference; this is descriptive, not causal A/B."))
    scope=e2a.read(out/"stage-comparison-scope.json")
    scope.update(different_conditioning_from_primary_heldout=True,
        no_direct_heldout_stage_attribution=True,
        scope="same exported program and new binary, separate observer phase with extra immediate preconditioning; inclusive QPC diagnostic responses only",
        context_check=e2a.receipt(out/"observer-context-comparison.json"))
    e2a.write(out/"stage-comparison-scope-v2.json",scope)
    base=(out/"REPORT_zh.html").read_text(encoding="utf-8")
    note='<p class="note"><strong>观测口径边界：</strong>observer的每个A/B块在reset后执行额外一次未计时完整前向；主heldout直接从reset进入计时重放。两阶段为不同新进程与时段。observer内A/B接近，不能证明其动作分项等于主heldout分项，也不能把跨阶段变化唯一归因于额外前向。下表保留这种差异，不用它重新拟合或按目标挑模型。</p>'
    note+='<div class="scroll"><table><tr><th>程序</th><th>主heldout ms</th><th>observer bare ms</th><th>observer stamped ms</th><th>bare相对主heldout变化</th></tr>'
    for r in rows:note+=f'<tr><td>{r["program_key"]}</td><td>{r["heldout_mean_ns"]/1e6:.3f}</td><td>{r["observer_bare_mean_ns"]/1e6:.3f}</td><td>{r["observer_stamped_mean_ns"]/1e6:.3f}</td><td>{100*r["observer_bare_vs_heldout_relative_shift"]:+.2f}%</td></tr>'
    note+='</table></div>'
    base=base.replace('<h2>4 相同程序的分项诊断</h2>','<h2>4 相同程序的分项诊断</h2>'+note)
    (out/"REPORT_zh_v2.html").write_text(base,encoding="utf-8",newline="\n")
    # A software-only test receipt; no native unit tests or GPU work is enabled.
    command=[os.sys.executable,"-m","unittest","profiling.tests.test_context_refinement","-v"]
    begin=time.perf_counter();test=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(root/x) for x in ("modeling","mapping","profiling"))),capture_output=True,text=True)
    (out/"software-tests-stdout.txt").write_text(test.stdout,encoding="utf-8",newline="\n")
    (out/"software-tests-stderr.txt").write_text(test.stderr,encoding="utf-8",newline="\n")
    e2a.write(out/"tests-receipt.json",dict(command=command,returncode=test.returncode,elapsed_seconds=time.perf_counter()-begin,
        passed=test.returncode==0,native_measurement_executed=False,source=e2a.receipt(root/"profiling/tests/test_context_refinement.py"),
        stdout=e2a.receipt(out/"software-tests-stdout.txt"),stderr=e2a.receipt(out/"software-tests-stderr.txt")))
    snapshots=[]
    for source in manifest["sources"]:
        original=Path(source["path"]);e2a.require(e2a.sha(original)==source["sha256"],"executed source changed before archival")
        target=out/"measurement_sources"/original.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,target);snapshots.append(dict(original=source,archived=e2a.receipt(target)))
    e2a.write(out/"source-snapshots.json",dict(schema="dac-refinement-source-archive-1",sources=snapshots,
        runtime_libraries_hashed_not_duplicated=True,postprocessors=[e2a.receipt(p) for p in (Path(__file__),Path(__file__).with_name("report.py"),Path(__file__).with_name("metadata.py"))]))
    allrows=[ex for split in ("profile","heldout","observer","transfer-profile","transfer-heldout") for ex in e2a.read(out/(split+"-executions.json"))]
    e2a.require(len(allrows)==111 and all(ex["status"]=="completed" and ex["process_exited_verified"] for ex in allrows),"declared process receipt incomplete")
    for ex in allrows:e2a.require(e2a.sha(ex["native"]["path"])==ex["native"]["sha256"],"raw native evidence changed")
    for f,k in (("response-profile-frozen.json","response-profile-frozen.json"),("predictions.json","predictions.json")):
        e2a.require(e2a.sha(out/f)==e2a.read(out/"frozen-barrier.json")[k],"response freeze changed")
    e2a.require(e2a.sha(out/"transfer-frozen.json")==e2a.read(out/"transfer-barrier.json")["sha256"],"D2H freeze changed")
    e2a.write(out/"final-evidence.json",dict(schema="dac-refinement-final-evidence-1",completed_at_utc=e2a.utc(),
        native_processes_declared=111,native_processes_completed=111,all_raw_sha256_rechecked=True,
        all_measurement_sources_archived=True,all_owned_children_exited=True,new_parameters_fitted_after_heldout=False,
        source_archive=e2a.receipt(out/"source-snapshots.json"),tests=e2a.receipt(out/"tests-receipt.json"),
        primary_report=e2a.receipt(out/"REPORT_zh_v2.html"),
        files=[e2a.receipt(p) for p in out.rglob("*") if p.is_file() and p.name!="final-evidence.json"]))
    return dict(report=str(out/"REPORT_zh_v2.html"),summary=str(out/"summary.json"),tests_passed=test.returncode==0,native_processes=111,context_rows=rows)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",required=True,type=Path)
    print(close(parser.parse_args().out),flush=True)
