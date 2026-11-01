"""New, separately frozen long-window transfer experiment prompted by v1 noise.

No v1 fitted term or heldout value enters this version. Only source-bound probe
inputs and binaries are reused. The same sizes, model candidates and thresholds
are retained; the changed repeat window is explicit. Longer bursts can change
state, so results do not replace historical single-action/strategy prices.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from . import task1


def prepare(out, previous):
    out=Path(out).resolve();previous=Path(previous).resolve()
    out.mkdir(parents=True,exist_ok=False)
    original=task1.read(previous/"manifest.json")
    jobs=[]
    for row in original["jobs"]:
        if row["split"]=="observer":continue
        request=dict(row["request"],repeats=256)
        jobs.append(dict(row,request=request))
    keys={j["program_key"] for j in jobs}
    manifest=dict(original,schema="dac-followup-task1-long-window-plan-1",prepared_at_utc=task1.utc(),
                  jobs=jobs,programs={key:original["programs"][key] for key in keys},
                  sources=[*task1.current_sources(),task1.receipt(__file__)],
                  profile_context=task1.fresh_profile_context(repeats=256))
    manifest.pop("pre_measurement_refinement",None)
    manifest["reused_program_manifest"]=task1.receipt(previous/"manifest.json")
    manifest["protocol"]=dict(original["protocol"],observer_processes=0,sham_processes=0,
        repeats=256,previous_repeats=8,reason="v1 calibration shows large short-window dispersion; longer batch is a new falsifiable sampling contract",
        version1_outcomes_retained=True,version1_parameters_or_heldout_used_for_fit=False,
        same_model_candidates_sizes_and_thresholds=True,
        potential_mean_or_state_shift="longer bursts may change service context; independent old/new mean shift reported after freeze; no old qualification replacement")
    task1.write(out/"manifest.json",manifest)
    task1.write(out/"observer-executions.json",[])
    return dict(prepared=str(out),jobs=len(jobs),repeats=256,native_timing_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--previous",type=Path,required=True)
    args=parser.parse_args();print(prepare(args.out,args.previous))

if __name__=="__main__":main()
