"""Post hoc reset-plus-prime response check using already frozen predictions.

No new acquisition, model selection, parameter update or replacement of the
primary reset-only evaluation occurs. All six programs and four independent
observer processes per program remain in the denominator.
"""
from pathlib import Path
from fractions import Fraction
import statistics
import html

from profiling.dac_validation import e2a
from .report import csv_write


def run(out):
    out=Path(out).resolve();root=Path(__file__).resolve().parents[2]
    barrier=e2a.read(out/"frozen-barrier.json");predictions=e2a.read(out/"predictions.json")
    e2a.require(all(e2a.sha(out/f)==barrier[f] for f in ("predictions.json","response-profile-frozen.json")),"frozen predictions changed")
    primary=e2a.read(out/"summary.json");observations=e2a.read(out/"observer-executions.json")
    programs=[];process_rows=[];allerrors={m:[] for m in ("hot","matched")};sources=[]
    for p in predictions:
        key=p["program_key"];processes=[];trials=[]
        for ex in observations:
            if ex["job"]["program_key"]!=key:continue
            e2a.require(ex["status"]=="completed" and ex["started_at_utc"]>=barrier["frozen_at_utc"],"observer missing or predates freeze")
            e2a.require(e2a.sha(ex["native"]["path"])==ex["native"]["sha256"],"observer raw changed")
            n=e2a.read(ex["native"]["path"]);a=[t["wall_ns"]/t["repeats"] for t in n["trials"] if not t["actual_stamped"]]
            b=[t["wall_ns"]/t["repeats"] for t in n["trials"] if t["actual_stamped"]]
            e2a.require(len(a)==16 and len(b)==16,"secondary balanced-block coverage differs")
            mean=statistics.mean(a);cv=statistics.stdev(a)/mean;effect=statistics.mean(b)/mean-1;trials+=a
            process=dict(job=ex["job"]["id"],round=ex["job"]["round"],process_id=ex["process_id"],mean_ns=mean,cv=cv,
                observer_effect=effect,cv_gate=cv<=.1,observer_gate=abs(effect)<=.05,bare_trial_averages=a,raw=ex["native"])
            processes.append(process);sources.append(ex["native"])
        e2a.require(len(processes)==4,"all four observer processes required")
        modes={}
        for mode in ("hot","matched"):
            price=float(Fraction(p["predictions"][mode]["duration_ns"]));apes=[abs(price-t["mean_ns"])/t["mean_ns"] for t in processes]
            p90=e2a.quantile([abs(price-v)/v for v in trials],.9)
            quality=all(a<=.10 for a in apes) and all(t["cv_gate"] and t["observer_gate"] for t in processes) and p90<=.20
            modes[mode]=dict(predicted_ns=price,median_process_ape=statistics.median(apes),maximum_process_ape=max(apes),
                mean_process_ape=statistics.mean(apes),mean_gate_passes=sum(a<=.1 for a in apes),trial_p90_ape=p90,
                secondary_all_thresholds_pass=quality)
            allerrors[mode]+=apes
            for t,ape in zip(processes,apes):process_rows.append(dict(program_key=key,layers=p["layers"],strategy=p["strategy"],mode=mode,
                job=t["job"],process_id=t["process_id"],prediction_ns=price,bare_mean_ns=t["mean_ns"],ape=ape,cv=t["cv"],
                mean_gate=ape<=.1,cv_gate=t["cv_gate"],observer_effect=t["observer_effect"],observer_gate=t["observer_gate"]))
        first=next(t for t in primary["programs"] if t["program_key"]==key)
        programs.append(dict(program_key=key,layers=p["layers"],strategy=p["strategy"],primary_reset_only_mean_ns=first["measured_mean_ns"],
            secondary_reset_plus_prime_mean_ns=statistics.mean(t["mean_ns"] for t in processes),
            secondary_vs_primary_shift=statistics.mean(t["mean_ns"] for t in processes)/first["measured_mean_ns"]-1,
            primary_modes=first["modes"],secondary_modes=modes,secondary_cv_passes=sum(t["cv_gate"] for t in processes),processes=processes))
    source_lines=[]
    for path,needles in ((root/"profiling/dac_refinement/native/refinement_probe.cpp",("executor.reset();full();auto &f","f.begin=steady::tick();")),
                         (root/"profiling/native/steady_costs/probe.cpp",("if(full)executor.reset();","frame.begin=tick();","for(int repeat=0;repeat<request.repeats;++repeat)"))):
        lines=path.read_text(encoding="utf-8").splitlines();matches=[]
        for i,line in enumerate(lines,1):
            if any(needle in line for needle in needles):matches.append(dict(line=i,text=line.strip()))
        source_lines.append(dict(source=e2a.receipt(path),matches=matches))
    summary=dict(schema="dac-refinement-reset-prime-secondary-1",generated_at_utc=e2a.utc(),
        analysis_type="post hoc secondary diagnostic; all programs retained; frozen predictions used unchanged",
        new_acquisition=False,new_parameter_fitting=False,primary_evaluation_replaced=False,
        observer_bare_independent_processes=24,programs=programs,
        totals={mode:dict(median_process_ape=statistics.median(errors),maximum_process_ape=max(errors),
                mean_gate_passes=sum(a<=.1 for a in errors),secondary_qualified_programs=sum(p["secondary_modes"][mode]["secondary_all_thresholds_pass"] for p in programs)) for mode,errors in allerrors.items()},
        secondary_cv_passes=sum(p["secondary_cv_passes"] for p in programs),
        exact_contexts=dict(primary="reset then directly timed8 fixed-input cyclic replays",secondary="reset, additional untimed complete pass, then timed8 fixed-input cyclic replays;16 bare blocks per process"),
        interpretation="The native code establishes a preparation-contract mismatch. Between-phase/fresh-process state remains confounded with priming; response changes cannot be uniquely assigned to a particular physical cause. This diagnostic cannot select a model per case or rewrite primary qualification.",
        native_code_evidence=source_lines,frozen_barrier=e2a.receipt(out/"frozen-barrier.json"),sources=sources)
    e2a.write(out/"reset-prime-secondary.json",summary);csv_write(out/"reset_prime_process_metrics.csv",process_rows)
    comparisons=[]
    for p in programs:
        for mode in ("hot","matched"):
            second=p["secondary_modes"][mode];first=p["primary_modes"][mode]
            comparisons.append(dict(program_key=p["program_key"],mode=mode,primary_mean_ns=p["primary_reset_only_mean_ns"],
                secondary_mean_ns=p["secondary_reset_plus_prime_mean_ns"],secondary_vs_primary_shift=p["secondary_vs_primary_shift"],
                primary_maximum_process_ape=first["maximum_process_ape"],secondary_maximum_process_ape=second["maximum_process_ape"],
                primary_mean_gate_passes=first["process_mean_gate_passes"],secondary_mean_gate_passes=second["mean_gate_passes"],
                secondary_cv_passes=p["secondary_cv_passes"],primary_qualified=first["fully_qualified"],secondary_thresholds_pass=second["secondary_all_thresholds_pass"]))
    csv_write(out/"reset_prime_program_comparison.csv",comparisons)
    body='<h2>附录 已冻结预测下的reset加prime次要分析</h2><p class="note">以下为post hoc secondary诊断，不替代主验收，不新增采样或参数，不按结果逐例选模型。全部6程序、每程序4独立observer进程均保留。原生代码明确observer每块在reset后额外完整prime一次；主heldout不执行这一步。跨进程/时段差异仍与prime混杂，不能把量化变化全部当成prime的纯因果效应。</p>'
    body+='<div class="scroll"><table><tr><th>程序/模型</th><th>主reset-only最大APE</th><th>次要reset+prime最大APE</th><th>次要均值门槛</th><th>次要CV</th><th>次要全部门槛</th></tr>'
    for p in comparisons:
        body+=f'<tr><td>{html.escape(p["program_key"]+" / "+p["mode"])}</td><td>{100*p["primary_maximum_process_ape"]:.2f}%</td><td>{100*p["secondary_maximum_process_ape"]:.2f}%</td><td>{p["secondary_mean_gate_passes"]}/4</td><td>{p["secondary_cv_passes"]}/4</td><td>{p["secondary_thresholds_pass"]}</td></tr>'
    body+='</table></div><p><a href="reset-prime-secondary.json">完整数据与原生代码路径行号</a>；<a href="reset_prime_process_metrics.csv">逐进程CSV</a>；<a href="reset_prime_program_comparison.csv">并排比较CSV</a>。前序方案对某口径更接近时，应限定准备契约，不能推广为统一默认。</p>'
    report=(out/"REPORT_zh_v2.html").read_text(encoding="utf-8").replace('</main></html>',body+'</main></html>')
    (out/"REPORT_zh_v3.html").write_text(report,encoding="utf-8",newline="\n")
    e2a.write(out/"final-evidence-v2.json",dict(schema="dac-refinement-final-evidence-2",generated_at_utc=e2a.utc(),
        prior_evidence=e2a.receipt(out/"final-evidence.json"),secondary=e2a.receipt(out/"reset-prime-secondary.json"),
        secondary_source=e2a.receipt(__file__),primary_report=e2a.receipt(out/"REPORT_zh_v3.html"),
        files=[e2a.receipt(p) for p in out.rglob('*') if p.is_file() and p.name!='final-evidence-v2.json']))
    return dict(totals=summary["totals"],secondary_cv_passes=summary["secondary_cv_passes"],report=str(out/"REPORT_zh_v3.html"),
        hybrid=[dict(program=p['program_key'],modes=p['secondary_modes']) for p in programs if p['strategy']==2])


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",required=True,type=Path)
    print(run(parser.parse_args().out),flush=True)
