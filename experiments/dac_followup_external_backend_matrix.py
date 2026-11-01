"""Run a bounded native ServingSim/ASTRA fixture matrix in a private WSL prefix.

The first L2/P16 job remains an independent, retained earlier run. This runner
adds the other three cells. It changes only local fixture configs, not source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def run(base):
    base = Path(base).resolve()
    repo = base / 'ServingSim'
    binary = base / 'astra-build/bin/AstraSim_Analytical_Congestion_Unaware'
    env = dict(os.environ, PYTHONPATH=str(base / 'site'),
               LD_LIBRARY_PATH=str(base / 'prefix/usr/lib/x86_64-linux-gnu'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONIOENCODING='utf-8')
    checks = []
    for layers, prompt in ((2, 64), (4, 16), (4, 64)):
        label = f'l{layers}-p{prompt}'
        run_id = f'whole-{label}-matrix-v1'
        attempt = f'whole-cli-{label}-matrix-v1'
        out = base / attempt
        out.mkdir(exist_ok=False)
        model = dict(model_type='llama', hidden_size=128, intermediate_size=512,
                     num_hidden_layers=layers, num_attention_heads=2,
                     num_key_value_heads=2, head_dim=64, vocab_size=1024,
                     torch_dtype='float16', max_position_embeddings=4096)
        workload = dict(input_toks=prompt, output_toks=1, arrival_time_ns=0)
        dump(repo / 'configs/model/followup/tiny-dense.json', model)
        (repo / 'workloads/one.jsonl').write_text(json.dumps(workload) + '\n', encoding='utf-8', newline='\n')
        dump(out / 'model-at-attempt.json', model)
        dump(out / 'workload-at-attempt.json', workload)
        dump(out / 'cluster-at-attempt.json', json.loads((repo / 'configs/cluster/followup.json').read_text()))
        argv = [sys.executable, '-m', 'serving', '--cluster-config',
                'configs/cluster/followup.json', '--dataset', 'workloads/one.jsonl',
                '--dtype', 'float16', '--max-num-seqs', '1', '--max-num-batched-tokens',
                '64', '--block-size', '16', '--no-enable-prefix-caching',
                '--no-enable-chunked-prefill', '--save-trace-text', '--keep-inputs',
                '--run-id', run_id, '--output', str(out / 'sim.csv'), '--log-level', 'INFO']
        start = time.perf_counter()
        proc = subprocess.Popen(argv, cwd=repo, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=10)
        (out / 'stdout.txt').write_text(stdout, encoding='utf-8', newline='\n')
        (out / 'stderr.txt').write_text(stderr, encoding='utf-8', newline='\n')
        clean = re.sub(r'\x1b\[[0-9;]*[mK]', '', stdout + stderr)
        clocks = re.findall(r'Total clocks \(ns\)\s*:?\s*([0-9,]+)', clean)
        summary = dict(schema='actual-servingsim-whole-cli-1', argv=argv,
                       cwd=str(repo), pid=proc.pid, returncode=proc.returncode,
                       timed_out=timed_out, wall_seconds=time.perf_counter() - start,
                       reported_clocks=[int(x.replace(',', '')) for x in clocks],
                       native_cpp_binary=str(binary),
                       native_cpp_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                       cost_profiles_are_declared_synthetic_fixtures=True,
                       real_hardware_accuracy_claimed=False, source_core_modified=False,
                       layers=layers, prompt=prompt, run_id=run_id,
                       status='completed' if proc.returncode == 0 and clocks else 'failed_actual_cli_attempt')
        dump(out / 'summary.json', summary)
        checks.append(summary)
        print(json.dumps(summary), flush=True)
        if summary['status'] != 'completed':
            raise RuntimeError('Actual CLI failure retained at ' + str(out))
        adapter = Path(__file__).with_name('dac_followup_external_backend.py')
        check = subprocess.run([sys.executable, str(adapter), '--build', str(base),
                                '--run-id', run_id, '--attempt', attempt],
                               text=True, capture_output=True, timeout=30)
        (out / 'adapter.stdout.txt').write_text(check.stdout, encoding='utf-8', newline='\n')
        (out / 'adapter.stderr.txt').write_text(check.stderr, encoding='utf-8', newline='\n')
        if check.returncode:
            raise RuntimeError('Comparison adapter failure retained at ' + str(out))
        print(check.stdout, flush=True)
    dump(base / 'backend-matrix-execution.json', checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    run(parser.parse_args().build)
