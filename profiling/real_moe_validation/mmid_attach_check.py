"""Bounded tool-attachment diagnostic; never replace original failed jobs."""
from pathlib import Path
import argparse
import json
import subprocess
import time
import psutil
from .dataset import read, write
from .tap_v4 import receipt, sha
from profiling.state_calibration.run import measurement_lock

def run(out, prepared):
    out,prepared=Path(out).resolve(),Path(prepared).resolve()
    if out.exists():raise ValueError('exclusive attachment diagnostic output required')
    m=read(prepared/'PREPARED.json');b=read(prepared/'BUILD_COMPLETE.json')
    tool=m['compute_sanitizer']['file'];binary=b['binary']
    for r in (tool,binary):
        if sha(r['path'])!=r['sha256']:raise ValueError('frozen tool/binary changed')
    out.mkdir(parents=True)
    jobs=[]
    for variant in ('original','syncwarp'):
        suite=prepared/('racecheck-'+variant+'.json')
        command=[tool['path'],'--tool','racecheck','--error-exitcode','97',
                 '--target-processes','application-only','--launch-timeout','10',
                 '--kill','--log-file',str(out/(variant+'.log')),
                 binary['path'],'--run',str(suite),str(out/(variant+'-result.json'))]
        jobs.append(dict(variant=variant,suite=receipt(suite),command=command))
    write(out/'PLAN.json',dict(jobs=jobs,tool=tool,binary=binary,source=receipt(__file__),
        purpose='qualify debugger attachment outside app-container after preserved timeout; not new performance data',
        timeout_seconds=35,registry_or_driver_changes=False))
    records=[]
    with measurement_lock():
        for j in jobs:
            begin=time.monotonic();p=None
            with (out/(j['variant']+'.stdout.txt')).open('x',encoding='utf-8',newline='\n') as a,(out/(j['variant']+'.stderr.txt')).open('x',encoding='utf-8',newline='\n') as z:
                p=subprocess.Popen(j['command'],stdout=a,stderr=z)
                row=dict(variant=j['variant'],pid=p.pid,timeout=False)
                try:p.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    row['timeout']=True
                    owner=psutil.Process(p.pid)
                    for child in owner.children(recursive=True):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    p.kill();p.wait(timeout=10)
                row.update(returncode=p.returncode,elapsed_seconds=time.monotonic()-begin)
            records.append(row);write(out/(j['variant']+'-execution.json'),row)
            print(json.dumps(row),flush=True)
    write(out/'COMPLETE.json',dict(jobs=records,old_failures_preserved=True,
                                  numerical_root_cause_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--prepared',required=True)
    a=p.parse_args();run(a.out,a.prepared)
