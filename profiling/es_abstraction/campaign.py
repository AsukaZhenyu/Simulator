"""Execute the frozen E/S native plan, including its sign-independent precision rule."""
from pathlib import Path
import argparse
import os
import shutil
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from profiling.moe_policy_validation import campaign as base
from profiling.es_abstraction.run_native import run
from experiments.es_abstraction.analyze import run as analyze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True)
    a=ap.parse_args();folder=a.campaign.resolve()
    sources={str(p.resolve()):base.digest(p) for p in (
        Path(__file__),Path(__file__).with_name('run_native.py'),Path(__file__).with_name('calibration.py'),
        base.ROOT/'experiments/es_abstraction/analyze.py')}
    receipt=folder/'ACQUISITION_SOURCE_SNAPSHOT.json'
    if not receipt.exists():
        base.save(receipt,dict(created=base.utc(),sources=sources))
        for p in sources:
            target=folder/'executed_sources'/Path(p).relative_to(base.ROOT)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    elif base.read(receipt)['sources']!=sources:raise ValueError('acquisition source drift')
    if not (folder/'pilot/PILOT.json').exists():run(folder,pilot=True)
    run(folder,rounds=5)
    initial=folder/'ANALYSIS_5.json'
    first=base.read(initial) if initial.exists() else analyze(folder,initial,5)
    extras=first['extra_round_workloads']
    ep=folder/'EXTRA_ROUNDS_DECISION.json'
    if not ep.exists():base.save(ep,dict(created=base.utc(),rule='95% gain CI width>10pp; all unique policies in each affected workload',
        initial_analysis_sha256=base.digest(initial),workloads=extras))
    if extras:run(folder,start_round=5,rounds=10,workload_ids=extras)
    final=folder/'ANALYSIS_FINAL.json'
    if not final.exists():analyze(folder,final,0)
    after={p:base.digest(p) for p in sources}
    if after!=sources:raise ValueError('acquisition code changed')
    p=folder/'ACQUISITION_COMPLETE.json'
    if not p.exists():base.save(p,dict(created=base.utc(),source_unchanged=True,manifest_sha256=base.digest(folder/'FROZEN_PREDICTIONS.json'),
        final_analysis_sha256=base.digest(final),extra_round_workloads=len(extras)))

if __name__=='__main__':main()
