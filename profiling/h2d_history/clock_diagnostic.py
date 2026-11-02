"""Declared eight-process passive clock/link diagnosis; no prediction fitting.

Read-only NVML samples include the target workload and may perturb execution.
Their timestamps share absolute QPC with the native probe. They are diagnostic
observations, never future external-resource input or per-action service rates.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import threading

from profiling.dac_validation import e2a
from profiling.state_calibration import run, telemetry

ROOT = Path(__file__).resolve().parents[2]
read, write, receipt, sha, require = e2a.read, e2a.write, e2a.receipt, e2a.sha, e2a.require


class LinkProvider(telemetry.NVMLProvider):
    """Extend the existing read-only provider without changing its implementation."""

    LINK_FIELDS = {
        'current_pcie_generation': 'nvmlDeviceGetCurrPcieLinkGeneration',
        'current_pcie_width': 'nvmlDeviceGetCurrPcieLinkWidth',
    }

    def read(self):
        result = super().read()
        values, unknown = result['values'], result['unknown']
        for field, symbol in self.LINK_FIELDS.items():
            try:
                if self._error or self._closed:
                    raise OSError(self._error or 'NVML provider closed')
                value = ctypes.c_uint()
                function = self._function((symbol,), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)])
                self._check(function(self._handle, ctypes.byref(value)), symbol)
                values[field] = value.value
            except Exception as error:
                unknown[field] = f'{type(error).__name__}: {error}'
        result['status'] = 'partial' if values and unknown else 'ok' if values else 'unavailable'
        return result


def clock_ns(tick, frequency):
    return tick * 1_000_000_000 // frequency


def library_receipt(provider):
    try:
        require(provider._lib is not None, 'NVML library unavailable')
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        query = kernel.GetModuleFileNameW
        query.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
        query.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32768)
        count = query(provider._lib._handle, buffer, len(buffer))
        require(0 < count < len(buffer), 'cannot resolve loaded NVML module path')
        return dict(status='observed', **receipt(buffer.value))
    except Exception as error:
        return dict(status='unknown', error=f'{type(error).__name__}: {error}')


class Sampler:
    def __init__(self, device_index, period_ms):
        self.clock = telemetry.NativeClock()
        require(self.clock.clock_id == 'windows_qpc', 'same Windows QPC clock required')
        self.provider = LinkProvider(device_index)
        self.period_ms = period_ms
        self.stop = threading.Event()
        self.rows = []
        self.errors = []
        self.started_tick = None
        self.ended_tick = None
        self.waiter = telemetry._make_waiter(self.stop)
        self.thread = threading.Thread(target=self._loop, name='passive-nvml-clock-diagnostic', daemon=True)
        self.module = library_receipt(self.provider)

    def _loop(self):
        try:
            self.started_tick = self.clock.now()
            period = max(1, round(self.clock.frequency_hz * self.period_ms / 1000))
            deadline = self.started_tick
            while not self.stop.is_set():
                first = self.clock.now()
                observation = self.provider.read()
                last = self.clock.now()
                self.rows.append(dict(sample_index=len(self.rows), begin_tick=first, end_tick=last,
                    begin_ns=clock_ns(first, self.clock.frequency_hz), end_ns=clock_ns(last, self.clock.frequency_hz),
                    query_wall_ns=clock_ns(last-first, self.clock.frequency_hz), observation=observation))
                if len(self.rows) >= 100000:
                    raise RuntimeError('declared telemetry sample envelope exceeded')
                deadline += period
                now = self.clock.now()
                # Skip missed deadlines. No catch-up burst or busy polling.
                if deadline <= now:
                    deadline += ((now-deadline)//period+1)*period
                if self.waiter.wait(max(0, (deadline-self.clock.now())/self.clock.frequency_hz)):
                    break
        except BaseException as error:
            self.errors.append(f'{type(error).__name__}: {error}')
        finally:
            self.ended_tick = self.clock.now()

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.waiter.interrupt()
        self.thread.join(timeout=10)
        stopped = not self.thread.is_alive()
        if not stopped:
            self.errors.append('NVML polling thread did not exit within ten seconds; no safe forced thread termination')
        # A stuck read must not race nvmlShutdown; preserve explicit failure.
        if stopped:
            self.provider.close()
            self.waiter.close()
        periods = [b['begin_ns']-a['begin_ns'] for a, b in zip(self.rows, self.rows[1:])]
        return dict(schema='h2d-history-passive-telemetry-1', enabled=True,
            clock=self.clock.metadata(), timestamp_conversion='floor(absolute QPC ticks * 1e9 / frequency)',
            requested_period_ms=self.period_ms, sample_count=len(self.rows),
            actual_begin_period_ns=dict(min=min(periods) if periods else None,
                median=statistics.median(periods) if periods else None,
                mean=statistics.mean(periods) if periods else None,
                max=max(periods) if periods else None),
            started_tick=self.started_tick, ended_tick=self.ended_tick,
            provider=self.provider.metadata(), loaded_nvml=self.module,
            waiter=self.waiter.metadata(), polling_thread_exited_verified=stopped,
            errors=self.errors, samples=self.rows,
            persisted_after_child_exit=True, diagnostic_only=True,
            target_utilization_used_as_future_prediction_input=False,
            available_service_inferred=False, external_only_state=False,
            semantic_limits=[
                'Queries include the target workload and other applications; no external-only attribution.',
                'Utilization is a driver-defined rolling window, not instantaneous service.',
                'Device-memory utilization is not host DRAM bandwidth or saturation.',
                'Clock and PCIe link states are observations, not calibrated action throughput.',
                'NVML readings need not represent the complete preceding query bracket.',
                'Sampling spans initialization, warmup and execution; native QPC brackets identify timed batches.',
                'Passive queries and the companion thread can change execution and require an observer comparison.'])


def prepare(out, campaign, period_ms=10):
    out, campaign = Path(out).resolve(), Path(campaign).resolve()
    require(not out.exists(), 'new diagnostic output directory required')
    require(1 <= period_ms <= 1000, 'requested period must be within 1..1000 ms')
    parent = read(campaign/'manifest.json')
    require(parent['schema'] == 'h2d-history-campaign-1', 'existing H2D history manifest required')
    for item in [parent['binary'], *parent['runtime']]:
        require(sha(item['path']) == item['sha256'], 'parent binary/runtime changed')
    source_receipt = next(s['source'] for s in parent['sources'] if s['source']['path'].replace('\\', '/').endswith('/h2d_history/native/history_probe.cpp'))
    require(sha(source_receipt['path']) == source_receipt['sha256'], 'native source changed')
    out.mkdir(parents=True)
    binary = out/'executed_binary'/Path(parent['binary']['path']).name
    binary.parent.mkdir(); shutil.copy2(parent['binary']['path'], binary)
    sources = []
    for path in (Path(__file__), Path(telemetry.__file__), Path(run.__file__), Path(e2a.__file__)):
        path = path.resolve(); target = out/'executed_sources'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        sources.append(dict(source=receipt(path), archived=receipt(target)))
    jobs = []
    # Rotated adjacent on/off order across both modes and both full contexts.
    layouts = [(1, 1, 'target-hot', (False, True)), (1, 1, 'full', (True, False)),
               (2, 2, 'full', (False, True)), (2, 2, 'target-hot', (True, False))]
    for layers, strategy, mode, observer_order in layouts:
        key = f'l{layers}-h512-p64-s{strategy}-profile'
        program = read(parent['programs'][key]['program']['path'])
        target = next(i for i, a in enumerate(program['actions']) if a['kind']=='COPY'
            and a['src']=='dram' and a['dst']=='vram' and a['target_id'].endswith(':W_up'))
        tensor = next(t for t in program['graph']['workload']['tensors']
            if t['id']==program['actions'][target]['target_id'])
        require(tensor['storage_bytes'] == 2097152, 'declared target must be actual two-MiB W_up')
        for enabled in observer_order:
            sampling = e2a.request('full' if mode=='full' else 'action', trials=16)
            sampling['repeats'] = 8
            sampling['target_action_index'] = target if mode=='target-hot' else None
            request = dict(schema='h2d-history-request-1', source_sha256=source_receipt['sha256'],
                sampling=sampling, expected_process_mask=4294967295, mode=mode,
                instrumentation='bare', memory_mode='pageable', history_intervention='none',
                target_action_index=target if mode=='target-hot' else None)
            jobs.append(dict(id=f'clock-{key}-{mode}-telemetry-{int(enabled)}',
                program_key=key, mode=mode, telemetry_enabled=enabled, request=request,
                device_index=program['physical'].get('cuda_device', 0), target_action_index=target,
                target_storage_bytes=tensor['storage_bytes']))
    require(len(jobs)==8, 'exactly eight processes required')
    manifest = dict(schema='h2d-history-clock-diagnostic-campaign-1', prepared_at_utc=e2a.utc(),
        parent_manifest=receipt(campaign/'manifest.json'), native_source=source_receipt,
        binary=receipt(binary), original_binary=parent['binary'], sources=sources,
        programs={k:parent['programs'][k] for k in {j['program_key'] for j in jobs}},
        runtime=parent['runtime'], runtime_dirs=parent['runtime_dirs'], jobs=jobs,
        protocol=dict(total_processes=8, period_ms=period_ms, timeout_seconds=300,
            trials=16, repeats=8, warmup_min_ms=500, retries=0,
            prediction_parameters_fitted=False, independent_validation_claimed=False,
            whole_target_duration_used_as_parameter=False, telemetry_used_as_future_input=False,
            observer_comparison='adjacent independently initialized processes; one process per cell; includes between-process variation, not pure causal overhead',
            persistence='native and parent telemetry serialized after timed target completes',
            resource_settings_modified=False))
    write(out/'manifest.json', manifest)
    return dict(out=str(out), **manifest['protocol'])


def check(manifest):
    for value in [manifest['parent_manifest'], manifest['native_source'], manifest['binary'], *manifest['runtime']]:
        require(sha(value['path']) == value['sha256'], 'bound manifest/source/binary/runtime changed')
    for binding in manifest['programs'].values():
        for field in ('program', 'reference'):
            value = binding[field]; require(sha(value['path'])==value['sha256'], 'bound mathematical input changed')
    for value in manifest['sources']:
        for side in ('source', 'archived'):
            require(sha(value[side]['path']) == value[side]['sha256'], 'bound diagnostic source changed')


def verify_native(native, job, binding, process_id):
    import numpy as np
    require(native['request']==job['request'], 'native request mismatch')
    require(native['process_id']==process_id, 'native PID mismatch')
    require(native['source_sha256']==job['request']['source_sha256'], 'native source mismatch')
    require(native['program_sha256']==read(binding['program']['path'])['program_sha256'], 'program hash mismatch')
    require(native['native_repeat_consistency']['passed'], 'repeat consistency failed')
    proof, oracle = native['verification_native'], read(binding['reference']['path'])
    require(proof['copy_values_equal'], 'copy byte verification failed')
    require(set(proof['ports'])==set(oracle), 'independent port inventory mismatch')
    for tensor, values in oracle.items():
        actual, reference = np.asarray(proof['ports'][tensor]), np.asarray(values)
        require(actual.shape==reference.shape and np.all(np.isfinite(actual))
            and np.allclose(actual,reference,rtol=5e-5,atol=5e-5), 'independent numeric mismatch '+tensor)
    require(len(native['trials'])==16, 'native trial count mismatch')
    require(all(row['repeats']==8 and not row['actual_stamped'] for row in native['trials']), 'native timing contract changed')


def collect(out):
    out = Path(out).resolve(); manifest = read(out/'manifest.json'); check(manifest)
    executions = []
    with run.measurement_lock():
        for job in manifest['jobs']:
            check(manifest)
            folder = out/'jobs'/job['id']
            if (folder/'execution.json').exists():
                executions.append(read(folder/'execution.json')); continue
            folder.mkdir(parents=True, exist_ok=False)
            request_path, result_path = folder/'request.json', folder/'result.json'
            write(request_path, job['request'])
            binding = manifest['programs'][job['program_key']]
            command = [manifest['binary']['path'], binding['program']['path'], str(request_path), str(result_path)]
            execution = dict(job=job, command=command, started_at_utc=e2a.utc(), status='failed')
            process, sampler, handles, telemetry_result = None, None, [], None
            clock = telemetry.NativeClock()
            try:
                if job['telemetry_enabled']:
                    sampler = Sampler(job['device_index'], manifest['protocol']['period_ms']); sampler.start()
                execution['parent_launch_tick'] = clock.now()
                environment = run.child_environment(manifest['runtime_dirs']); environment['OPENBLAS_NUM_THREADS']='1'
                process, stdout, stderr, affinity = run.launch(command, environment,
                    folder/'stdout.txt', folder/'stderr.txt', 4294967295)
                handles = [stdout, stderr]
                execution.update(process_id=process.pid, process_affinity=affinity)
                code = process.wait(timeout=manifest['protocol']['timeout_seconds'])
                execution.update(returncode=code, process_exited_verified=True, forced_termination=False,
                    parent_exit_observed_tick=clock.now())
                require(code==0, 'native process failed; retained without retry')
                native = read(result_path); verify_native(native, job, binding, process.pid)
                require(native['clock']==clock.metadata() or (
                    native['clock']['clock_id']==clock.clock_id and native['clock']['frequency_hz']==clock.frequency_hz),
                    'native and passive QPC clocks differ')
                execution.update(status='completed', native=receipt(result_path), independent_oracle_passed=True)
            except BaseException as error:
                execution['error'] = repr(error)
                if process is not None and process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)
                    execution['forced_termination'] = True
                if process is not None:
                    execution.update(returncode=process.returncode, process_exited_verified=process.poll() is not None,
                        parent_exit_observed_tick=clock.now())
            finally:
                if sampler is not None:
                    telemetry_result = sampler.finish()
                else:
                    telemetry_result = dict(schema='h2d-history-passive-telemetry-1', enabled=False,
                        samples=[], sample_count=0, nvml_initialized=False,
                        diagnostic_only=True, persisted_after_child_exit=True)
                # Nothing is written by the polling thread during native timing.
                write(folder/'telemetry.json', telemetry_result)
                execution['telemetry'] = receipt(folder/'telemetry.json')
                execution['telemetry_thread_exited_verified'] = telemetry_result.get('polling_thread_exited_verified', True)
                execution['telemetry_observation_available'] = (
                    telemetry_result.get('provider', {}).get('available', False)
                    if job['telemetry_enabled'] else None)
                if not execution['telemetry_thread_exited_verified']:
                    execution['status']='failed'; execution['telemetry_error']='polling thread still active'
                for handle in handles: handle.close()
                execution['completed_at_utc']=e2a.utc(); write(folder/'execution.json', execution)
            executions.append(execution)
            print(json.dumps(dict(job=job['id'],status=execution['status'],error=execution.get('error'))),flush=True)
            require(execution['telemetry_thread_exited_verified'], 'stop campaign: own telemetry thread remains active')
    if not (out/'executions.json').exists(): write(out/'executions.json', executions)
    return dict(declared=8, attempted=len(executions), completed=sum(e['status']=='completed' for e in executions))


def summarize(out):
    out = Path(out).resolve(); manifest = read(out/'manifest.json'); check(manifest)
    rows, groups = [], {}
    for job in manifest['jobs']:
        folder = out/'jobs'/job['id']; execution = read(folder/'execution.json')
        row = dict(job=job['id'], program_key=job['program_key'], mode=job['mode'],
            telemetry_enabled=job['telemetry_enabled'], status=execution['status'],
            telemetry_observation_available=execution.get('telemetry_observation_available'),
            execution=receipt(folder/'execution.json'))
        if execution['status']=='completed':
            native, observed = read(folder/'result.json'), read(folder/'telemetry.json')
            times = [t['wall_ns']/t['repeats'] for t in native['trials']]
            row.update(mean_ns=statistics.mean(times), early_mean_ns=statistics.mean(times[:8]),
                late_mean_ns=statistics.mean(times[8:]), cv=statistics.stdev(times)/statistics.mean(times),
                native=receipt(folder/'result.json'), telemetry=receipt(folder/'telemetry.json'))
            row['late_over_early_minus_one']=row['late_mean_ns']/row['early_mean_ns']-1
            windows = []
            for trial in native['trials']:
                samples = [s for s in observed['samples'] if trial['begin_ns']<=s['begin_ns'] and s['end_ns']<=trial['end_ns']]
                values = {}
                for field in (*telemetry.NVMLProvider.FIELDS, *LinkProvider.LINK_FIELDS):
                    known = [s['observation']['values'][field] for s in samples if field in s['observation']['values']]
                    values[field] = dict(n=len(known), min=min(known) if known else None,
                        mean=statistics.mean(known) if known else None, max=max(known) if known else None)
                windows.append(dict(trial=trial['trial'], complete_query_brackets=len(samples), values=values,
                    per_action_state_identified=False))
            row['native_batch_observations']=windows
            groups.setdefault((job['program_key'],job['mode']),{})[job['telemetry_enabled']]=row
        rows.append(row)
    contrasts=[]
    for (key,mode), pair in groups.items():
        contrasts.append(dict(program_key=key,mode=mode,
            enabled_over_disabled_mean_minus_one=pair[True]['mean_ns']/pair[False]['mean_ns']-1 if set(pair)=={False,True} else None,
            interpretation='one fresh process per condition; diagnostic contrast combines observer effect and between-process variation'))
    summary=dict(schema='h2d-history-clock-diagnostic-summary-1', rows=rows, contrasts=contrasts,
        declared_processes=8, completed_processes=sum(r['status']=='completed' for r in rows),
        causal_mechanism_identified=False, calibrated_dynamic_model=False, hardware_accuracy_qualified=False,
        future_target_telemetry_used_for_prediction=False)
    require(not (out/'summary.json').exists(), 'summary already exists')
    write(out/'summary.json',summary)
    return dict(summary=str(out/'summary.json'), completed_processes=summary['completed_processes'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','collect','summarize'))
    parser.add_argument('--out',required=True)
    parser.add_argument('--campaign')
    parser.add_argument('--period-ms',type=float,default=10)
    args=parser.parse_args()
    if args.command=='prepare':
        require(args.campaign is not None, '--campaign required for prepare')
        result=prepare(args.out,args.campaign,args.period_ms)
    else: result=collect(args.out) if args.command=='collect' else summarize(args.out)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
