"""Pre-target historical reference and auditable row-level refinement reports."""
from pathlib import Path
from fractions import Fraction
import csv
import json
import statistics
import html

from llm_infer_model.performance.reusable_action_costs import ReusableResponseTable, semantic_action
from profiling.dac_validation import e2a


def csv_write(path,rows):
    with Path(path).open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ["status"],lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def html_report(out,summary,comparisons,modes,stage_rows):
    esc=lambda v:html.escape(str(v))
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>上下文性能修正实验</title><style>body{font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#172638;background:#eef3f7;margin:0}main{max-width:1150px;margin:30px auto;padding:36px;background:white;border-radius:12px}h1,h2{color:#143654}h1{font-size:30px}h2{margin-top:34px}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border:1px solid #cbd5df;padding:10px;text-align:left}th{background:#e8eff5}tr:nth-child(even){background:#f7f9fc}.scroll{overflow:auto}.note{border-left:5px solid #a45c20;background:#fff4e7;padding:16px}.meta{color:#567}code{background:#edf1f5;padding:2px 5px}a{color:#1d568d}</style><main><h1>上下文性能修正实验</h1>']
    parts.append('<p class="meta">完成时间 '+esc(summary['evaluated_at_utc'])+' UTC；本轮矩阵在计时前固定，旧结果保留。</p>')
    parts.append('<p class="note"><strong>结论：</strong>前序条件化方案改善两层混合方案的均值误差，但在其他方案上变差，不能作为统一默认模型。独立的新热重复价格与前序匹配价格均在完整目标开始前冻结，未使用整图残差拟合。</p>')
    parts.append('<h2>1 范围与通过条件</h2><p>仅测试 H512/P64、单层/两层、全CPU S0/全GPU S1/GPU投影与FFN加CPU Attention S2，F32、单CPU线程、单CUDA GPU、关闭CUDA Graphs、固定输入warm串行重放。不是自回归、多请求并发、MoE缓存或真实UMA验证。</p>')
    parts.append('<p>111个预设进程：15独立动作校准、24完整目标、24同进程ABBA/BAAB观测、24 D2H训练、24 D2H独立尺寸验收。完成进程 '+str(summary['counts']['completed_processes'])+'/111。程序资格要求每个进程均值APE≤10%、CV≤10%、合并trial-average P90 APE≤20%，且4个观测对照均≤5%。旧0/6是另一矩阵的组级完整资格，不能与新程序级通过数合并。</p>')
    parts.append('<h2>2 模型与测量</h2><p>基础hot方案重复同一准备好的动作；可选前序方案先执行未计时的完整working-set sweep和prefix，再只计时目标动作。modeling负责前序语义坐标与串行响应求和，profiling负责采样、求独立动作均值、冻结和验收。两者使用同一CPU/GPU框架；前序轴是待验证的经验假设，不是对所有缓存/频率/运行时机制的因果识别。</p><p>坐标不含策略名、program/layer ID、seed或完整目标耗时。首动作循环前序只适用于已热身固定输入的重复执行。<a href="conditioning-metadata.json">prefix长度与逻辑catalogue bytes</a>仅作诊断，不等于CPU cache工作集、实际物理峰值或包含CUDA私有workspace的完整内存。</p>')
    parts.append('<h2>3 独立完整目标与消融</h2>')
    parts.append(table(['目标','实测ms','旧冻结ms','fresh-hot ms','前序匹配ms','hot完整资格','前序完整资格'],[[r['program_key'],f"{r['measured_mean_ns']/1e6:.3f}",f"{r['historical_predicted_ns']/1e6:.3f}",f"{r['hot_predicted_ns']/1e6:.3f}",f"{r['matched_predicted_ns']/1e6:.3f}",r['hot_qualified'],r['matched_qualified']] for r in comparisons]))
    parts.append(table(['模型','进程数','均值APE中位数','最大均值APE','进程均值门槛通过'],[[m,r['processes'],f"{100*r['median_process_ape']:.2f}%",f"{100*r['maximum_process_ape']:.2f}%",r['process_mean_gate_passes']] for m,r in modes.items()]))
    parts.append('<p>fresh-hot完整程序资格 '+str(summary['counts']['hot_qualified_programs'])+'/6；前序匹配 '+str(summary['counts']['matched_qualified_programs'])+'/6。旧冻结表是保留原binary/采样上下文的次要参考，不是新binary上的严格同上下文对照。主消融是新binary下的fresh-hot与matched。</p>')
    failures=[(p['program_key'],t['job'],t['cv']) for p in summary['programs'] for t in p['processes'] if t.get('status')=='completed' and t['cv']>.1]
    parts.append('<p>目标CV未通过：'+esc('; '.join(f'{k} / {j}: {100*v:.2f}%' for k,j,v in failures) or '无')+'。观测对照通过 '+str(sum(o['passed'] for p in summary['programs'] for o in p['observer']))+'/24。前序方案不能按已看到的目标结果与hot逐个择优拼接。</p>')
    parts.append('<h2>4 相同程序的分项诊断</h2><p>下表使用新同程序observer中的stamped动作响应。观测值是包含派发与完成等待的QPC响应，非无偏GPU活动时间；用于核对是否存在分项误差抵消，不用来拟合新价格或修正已冻结预测。</p>')
    chosen=[r for r in stage_rows if r['strategy']==2 and r['category'] in ('COPY:dram->vram','COPY:vram->dram','COMPUTE:cpu:attention')]
    parts.append(table(['目标','分项','旧价格ms','hot价格ms','matched价格ms','stamped响应ms'],[[r['program_key'],r['category'],f"{r['old_frozen_ns']/1e6:.3f}",f"{r['fresh_hot_ns']/1e6:.3f}",f"{r['matched_ns']/1e6:.3f}",f"{r['observed_stamped_mean_ns']/1e6:.3f}"] for r in chosen]))
    d=summary['D2H'];parts.append('<h2>5 D2H独立尺寸与物理阶段</h2>')
    parts.append('<p>最大独立尺寸均值APE '+f"{100*d['maximum_size_mean_ape']:.2f}%"+'；CV通过 '+str(d['cv_passes'])+'/24；bare/stamped差异通过 '+str(d['observer_passes'])+'/24。预设新口径完整资格：'+esc(d['qualified'])+'。物理阶段仅拆GGML copy API调用、后续GPU同步与CPU同步；copy API本身可能包含staging与内部同步，不应叫作PCIe活动时间。</p>')
    parts.append(table(['字节','预测μs','实测尺寸均值μs','APE'],[[r['bytes'],f"{r['predicted_ns']/1000:.3f}",f"{r['observed_ns']/1000:.3f}",f"{100*r['ape']:.2f}%"] for r in d['heldout_sizes']]))
    rows=[]
    for size in sorted({r['bytes'] for r in d['processes']}):
        rs=[r for r in d['processes'] if r['bytes']==size]
        rows.append([size,*[f"{statistics.mean(r['phases_ns'][k] for r in rs)/1000:.3f}" for k in ('copy_call_ns','gpu_sync_ns','cpu_sync_ns')],'; '.join(sorted({json.dumps(r['host_pointer'],sort_keys=True) for r in rs}))])
    parts.append(table(['字节','copy API μs','GPU后同步μs','CPU后同步μs','实际host pointer查询'],rows))
    parts.append('<h2>6 结果边界与复现</h2><p>本轮不重写旧0/6、旧D2H失败或旧多层13.09%低估。新数据仍揭示前序标签与conditioning方法的适用限制；既不按层号选模型，也不把整图残差塞入参数。新provider保留为显式可选；基础模型保持不变。D2H尺寸响应资格即使通过，也不辨识纯PCIe带宽轴。真实存储驱逐、MoE缓存性能、UMA、多GPU和动态资源轨迹不属于本轮验证。</p>')
    parts.append('<ul>'+''.join('<li><a href="'+name+'">'+name+'</a></li>' for name in ('summary.json','comparison-summary.json','manifest.json','response-profile-frozen.json','frozen-barrier.json','transfer-frozen.json','transfer-barrier.json','process_metrics.csv','program_comparison.csv','action_prices.csv','category_prices.csv','observed_action_response.csv','stage_process_metrics.csv','D2H_metrics.csv','conditioning-metadata.json'))+'</ul></main></html>')
    (Path(out)/'REPORT_zh.html').write_text(''.join(parts),encoding='utf-8',newline='\n')


def baseline(out):
    out=Path(out).resolve();root=Path(__file__).resolve().parents[2]
    e2a.require(not(out/"heldout-started.json").exists(),"historical reference must be saved before new targets")
    oldfile=root/"build/dac-validation-20261025/e2a-v2/profile-frozen.json"
    old=e2a.read(oldfile);manifest=e2a.read(out/"manifest.json");predictions=e2a.read(out/"predictions.json")
    table=ReusableResponseTable(tuple(old["entries"]));rows=[];prices=[];categories=[]
    for prediction in predictions:
        key=prediction["program_key"];binding=manifest["programs"][key];program=e2a.read(binding["program"]["path"])
        historical=table.evaluate(program,old["context"]);grouped={}
        for i,(action,value) in enumerate(zip(program["actions"],historical["action_ns"])):
            semantic=semantic_action(program,i)
            kind=action["kind"]
            category=kind+(":"+action["device"]+":"+semantic["semantic_op"] if kind=="COMPUTE" else ":"+action["src"]+"->"+action["dst"] if kind=="COPY" else "")
            row=dict(program_key=key,layers=binding["layers"],strategy=binding["strategy"],action_index=i,category=category,
                old_frozen_ns=float(value),fresh_hot_ns=float(Fraction(prediction["predictions"]["hot"]["action_ns"][i])),
                matched_ns=float(Fraction(prediction["predictions"]["matched"]["action_ns"][i])))
            prices.append(row)
            sums=grouped.setdefault(category,dict(old_frozen_ns=0.,fresh_hot_ns=0.,matched_ns=0.,action_count=0))
            for column in ("old_frozen_ns","fresh_hot_ns","matched_ns"):sums[column]+=row[column]
            sums["action_count"]+=1
        for category,values in grouped.items():categories.append(dict(program_key=key,layers=binding["layers"],strategy=binding["strategy"],category=category,**values))
        rows.append(dict(program_key=key,old_predicted_ns=float(historical["duration_ns"]),new_hot_ns=float(Fraction(prediction["predictions"]["hot"]["duration_ns"])),
                         matched_ns=float(Fraction(prediction["predictions"]["matched"]["duration_ns"]))))
    e2a.write(out/"historical-baseline-predictions.json",dict(schema="dac-refinement-historical-reference-1",
        frozen_at_utc=e2a.utc(),source_profile=e2a.receipt(oldfile),new_response_profile=e2a.receipt(out/"response-profile-frozen.json"),
        new_predictions=e2a.receipt(out/"predictions.json"),predictions=rows,categories=categories,target_times_read=False,
        boundary="The historical table retains its original binary and sampling context. This secondary reference does not isolate parameter drift from binary/harness/context differences. The prespecified primary ablation is fresh-hot versus matched using the new binary."))
    csv_write(out/"action_prices.csv",prices);csv_write(out/"category_prices.csv",categories)
    return dict(pre_target_reference=True,programs=len(rows),action_rows=len(prices))


def finish(out):
    out=Path(out).resolve();summary=e2a.read(out/"summary.json");baseline=e2a.read(out/"historical-baseline-predictions.json")
    histories={row["program_key"]:row for row in baseline["predictions"]};processes=[];observer=[];comparisons=[]
    for program in summary["programs"]:
        key=program["program_key"];prices={"historical":histories[key]["old_predicted_ns"],
            **{mode:program["modes"][mode]["predicted_ns"] for mode in ("hot","matched")}}
        for p in program["processes"]:
            if p["status"]!="completed":
                processes.append(dict(program_key=key,layers=program["layers"],strategy=program["strategy"],mode="unknown",job=p["job"],predicted_ns=None,measured_ns=None,ape=None,cv=None,mean_gate=False,cv_gate=False,process_status=p["status"]))
                continue
            for mode,price in prices.items():
                error=abs(price-p["mean_ns"])/p["mean_ns"]
                processes.append(dict(program_key=key,layers=program["layers"],strategy=program["strategy"],mode=mode,job=p["job"],predicted_ns=price,measured_ns=p["mean_ns"],ape=error,cv=p["cv"],mean_gate=error<=.1,cv_gate=p["cv"]<=.1,process_status=p["status"]))
        for i,p in enumerate(program["observer"]):observer.append(dict(program_key=key,process_index=i,status=p["status"],effect=p.get("effect"),passed=p["passed"]))
        comparisons.append(dict(program_key=key,layers=program["layers"],strategy=program["strategy"],measured_mean_ns=program["measured_mean_ns"],
            **{mode+"_predicted_ns":price for mode,price in prices.items()},
            **{mode+"_aggregate_ape":abs(price-program["measured_mean_ns"])/program["measured_mean_ns"] for mode,price in prices.items()},
            hot_qualified=program["modes"]["hot"]["fully_qualified"],matched_qualified=program["modes"]["matched"]["fully_qualified"]))
    transfers=[]
    for row in summary["D2H"]["processes"]:
        size=next(s for s in summary["D2H"]["heldout_sizes"] if s["bytes"]==row["bytes"])
        transfers.append(dict(bytes=row["bytes"],mean_ns=row["mean_ns"],cv=row["cv"],observer_effect=row["observer_effect"],
            size_mean_ape=size["ape"],**row["phases_ns"],host_pointer=json.dumps(row["host_pointer"],sort_keys=True),raw_path=row["raw"]["path"],raw_sha256=row["raw"]["sha256"]))
    csv_write(out/"process_metrics.csv",processes);csv_write(out/"observer_metrics.csv",observer)
    csv_write(out/"program_comparison.csv",comparisons);csv_write(out/"D2H_metrics.csv",transfers)
    fresh=[p for p in processes if p["mode"]=="hot" and p["process_status"]=="completed"]
    matched=[p for p in processes if p["mode"]=="matched" and p["process_status"]=="completed"]
    historical=[p for p in processes if p["mode"]=="historical" and p["process_status"]=="completed"]
    modes={mode:dict(processes=len(rows),median_process_ape=statistics.median(r["ape"] for r in rows),maximum_process_ape=max(r["ape"] for r in rows),process_mean_gate_passes=sum(r["mean_gate"] for r in rows)) for mode,rows in (("historical",historical),("hot",fresh),("matched",matched))}
    paired=[dict(job=a["job"],hot_ape=a["ape"],matched_ape=b["ape"],matched_minus_hot_ape=b["ape"]-a["ape"]) for a,b in zip(fresh,matched)]
    e2a.write(out/"comparison-summary.json",dict(schema="dac-refinement-ablation-summary-1",modes=modes,
        program_comparisons=comparisons,paired_process_errors=paired,
        matched_lower_process_error_count=sum(r["matched_minus_hot_ape"]<0 for r in paired),
        original_complete_qualification_unchanged=True,historical_reference_context_not_new_binary=True,
        common_new_binary_primary_ablation="hot versus matched; no target-fit parameters"))
    stage_rows=[];stage_processes=[]
    prices={}
    with (out/"action_prices.csv").open(encoding="utf-8",newline="") as stream:
        for row in csv.DictReader(stream):prices[(row["program_key"],int(row["action_index"]))]=row
    observed={}
    for ex in e2a.read(out/"observer-executions.json"):
        if ex["status"]!="completed":continue
        key=ex["job"]["program_key"];native=e2a.read(ex["native"]["path"])
        frames=[r for r in native["trials"] if r["actual_stamped"]];per_action={}
        for frame in frames:
            for action in frame["actions"]:per_action.setdefault(action["action_index"],[]).append(action["wall_ns"]/frame["repeats"])
        per_category={}
        for index,values in per_action.items():
            price=prices[(key,index)];category=price["category"]
            per_category[category]=per_category.get(category,0.)+statistics.mean(values)
        for category,value in per_category.items():
            observed.setdefault((key,category),[]).append(value)
            stage_processes.append(dict(program_key=key,job=ex["job"]["id"],category=category,
                stamped_response_ns=value,process_id=ex["process_id"],raw_path=ex["native"]["path"],raw_sha256=ex["native"]["sha256"]))
    for row in baseline["categories"]:
        key=(row["program_key"],row["category"]);values=observed.get(key,[])
        actual=statistics.mean(values) if values else None
        stage_rows.append(dict(**row,observed_stamped_mean_ns=actual,observed_processes=len(values),
            **{m+"_minus_observed_ns":row[m+"_ns"]-actual if actual is not None else None for m in ("old_frozen","fresh_hot","matched")}))
    csv_write(out/"stage_process_metrics.csv",stage_processes)
    csv_write(out/"observed_action_response.csv",stage_rows)
    e2a.write(out/"stage-comparison-scope.json",dict(schema="dac-refinement-stage-diagnostic-1",
        scope="same-program fresh observer processes; QPC inclusive completed-action responses, not unbiased GPU active time",
        four_processes_per_program=True,target_stamps_used_for_parameters=False,
        response_profile_unchanged=e2a.sha(out/"response-profile-frozen.json")==e2a.read(out/"frozen-barrier.json")["response-profile-frozen.json"],
        why_needed="Whole-program error improvements can conceal compensating category errors; this ledger compares both independently frozen alternatives with existing stamped actions."))
    html_report(out,summary,comparisons,modes,stage_rows)
    files=[e2a.receipt(p) for p in out.rglob("*") if p.is_file() and p.name!="artifacts.json"]
    e2a.write(out/"artifacts.json",dict(schema="dac-refinement-artifacts-1",files=files))
    return modes


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("baseline","finish"));p.add_argument("--out",required=True,type=Path)
    a=p.parse_args();print(baseline(a.out) if a.command=="baseline" else finish(a.out),flush=True)
