"""Bounded real physical-stage collection with disjoint reference sensors.

This collector measures hardware, but does not claim that reference-kernel
throughput is neutral universal availability. All timing/model conclusions
remain separately qualified. It never fits a duration during collection.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT/'src', ROOT/'modeling', ROOT/'mapping', ROOT/'profiling'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from profiling.state_calibration import run
from profiling.state_calibration.telemetry import NativeClock
from profiling.resource_calibration.collect import OwnedChild
from profiling.state_response.worker_activity import OwnedActor
from profiling.state_response.thread_affinity import pin_owned_main_thread


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def ns(clock, tick):
    return tick * 1_000_000_000 // clock.frequency_hz


def prepare(args):
    from profiling.highlevel_accuracy.cases import build_case
    from profiling.highlevel_accuracy.bridge import export_plan, reference_ports
    from profiling.highlevel_accuracy.run import case_specs
    from profiling.state_response.collect import hardware_identity, python_sources

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    specification = deepcopy(next(row['kwargs'] for row in case_specs('representative')
        if row['id'] == 'h128-q16-history0-attention-ffn-cut128'))
    specification['threads'] = 1
    compiled = build_case(out/'inputs', **specification).compile()
    programs = {}
    for split, offset in (('profile', 0), ('heldout', 100)):
        for repeat in range(3):
            key = f'{split}-{repeat}'
            program = export_plan(compiled, seed=args.seed + offset + repeat)
            path, oracle = out/'programs'/f'{key}.json', out/'references'/f'{key}.json'
            run.save(path, program)
            run.save(oracle, reference_ports(program))
            programs[key] = dict(path=str(path), sha256=run.sha(path), oracle=str(oracle),
                oracle_sha256=run.sha(oracle), seed=program['seed'], program_sha256=program['program_sha256'])
    states = [('idle', 0, 0), ('cpu', 1, 0), ('gpu', 0, 1), ('both', 1, 1)]
    jobs = [dict(id='smoke-cupti', split='smoke', program_key='profile-0', condition='idle',
                 cpu_background=0, gpu_background=0, instrumentation='cupti', references=False, samples=2)]
    for split in ('profile', 'heldout'):
        for repeat in range(3):
            order = states[repeat:] + states[:repeat]
            if split == 'heldout':
                order = list(reversed(order))
            for condition, cpu, gpu in order:
                jobs.append(dict(id=f'{split}-{repeat}-{condition}', split=split,
                    program_key=f'{split}-{repeat}', condition=condition, cpu_background=cpu,
                    gpu_background=gpu, instrumentation='cupti', references=True, samples=32))
    for repeat in range(3):
        modes = [('bare', False), ('cupti', False), ('cupti', True)]
        modes = modes[repeat:] + modes[:repeat]
        for instrument, sensors in modes:
            jobs.append(dict(id=f'observer-{repeat}-{instrument}-{int(sensors)}', split='observer',
                program_key=f'profile-{repeat}', condition='idle', cpu_background=0,
                gpu_background=0, instrumentation=instrument, references=sensors, samples=32))
    executable = args.binary.resolve()
    background_binary = args.background_binary.resolve()
    require(executable.is_file() and background_binary.is_file(), 'explicit existing binaries required')
    files = {str(executable): run.sha(executable), str(background_binary): run.sha(background_binary)}
    for source in getattr(getattr(compiled, '_compiled', None), 'sources', ()):
        path = Path(source['path'])
        require(run.sha(path) == source['sha256'], 'compiled source drift: '+str(path))
        require(path.is_file(), 'compiled input missing: '+str(path))
        files[str(path.resolve())] = run.sha(path)
    for binary in (executable, background_binary):
        build_path = binary.parent/'build-manifest.json'
        build = run.read(build_path)
        require(build.get('returncode') == 0 and build.get('binary_sha256') == run.sha(binary),
                'successful source-bound binary required')
        files[str(build_path)] = run.sha(build_path)
        for filename, expected in build['bound_files'].items():
            require(run.sha(filename) == expected, 'native build input changed: '+filename)
            files[filename] = expected
    for folder in (ROOT/'modeling/llm_infer_model', ROOT/'profiling/available_service',
                   ROOT/'profiling/native/physical_stages', ROOT/'profiling/state_calibration',
                   ROOT/'profiling/resource_calibration', ROOT/'profiling/state_response',
                   ROOT/'profiling/highlevel_accuracy', ROOT/'mapping/tensor_mapping', ROOT/'src/llm_simulator'):
        for path in folder.rglob('*'):
            if path.name in {'physical_records.py', 'test_physical_records.py'}:
                continue
            if path.is_file() and path.suffix in ('.py', '.cpp', '.hpp', '.txt') and not any(
                    part.startswith('build') or part in {'results', '__pycache__'} for part in path.relative_to(folder).parts[:-1]):
                files[str(path.resolve())] = run.sha(path)
    for folder in args.runtime_dir:
        for path in folder.glob('*.dll'):
            files[str(path.resolve())] = run.sha(path)
    for binding in programs.values():
        for field in ('path', 'oracle'):
            files[binding[field]] = run.sha(binding[field])
    jobs_by_role = dict(reference_cpu=39, reference_gpu=52, cpu_background=0, gpu_background=14)
    manifest = dict(schema='physical-service-campaign-1', prepared_at_utc=run.utc(),
        binary=str(executable), background_binary=str(background_binary),
        runtime_dirs=[str(p.resolve()) for p in args.runtime_dir], hardware=hardware_identity(),
        programs=programs, jobs=jobs, sources=files, source_inventory_sha256=digest(files),
        protocol=dict(warmup=20, reference_cpu_samples=4096, reference_gpu_samples=32,
            main_process_mask=4, main_thread_mask=4, cpu_reference_mask=4, gpu_reference_mask=48,
            cpu_background_mask=4, gpu_background_mask=48, cpu_threads=1,
            main_samples=32, go_lead_ms=20, job_timeout_seconds=90, background_timeout_seconds=110,
            background_settle_ms=300, reference_order=['gpu', 'cpu'], references_after_target=True,
            max_reference_hold_ns=10_000_000_000, targets=jobs_by_role,
            primary_claim='physical-stage collection and reference-conditioned diagnostic calibration',
            universal_available_resource_measurement_claimed=False,
            target_timing_used_to_derive_supply=False, model_fit_during_collection=False,
            parameter_freeze_required_before_heldout=True, observer_relative_error_max=0.05,
            relative_error_max=0.20, p90_relative_error_max=0.20, minimum_independent_runs=3,
            tracer='direct_CUPTI_common_QPC_ns', cuda_graphs='disabled', numeric_dtype='F32'))
    run.save(out/'manifest.json', manifest)
    snapshot = out/'recorded-sources'
    entries = []
    for filename, expected in files.items():
        path = Path(filename)
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix not in {'.py', '.cpp', '.hpp', '.txt'} or any(part.startswith('build') for part in relative.parts):
            continue
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected, 'snapshot source changed: '+filename)
        with target.open('xb') as stream:
            stream.write(content)
        require(run.sha(target) == expected, 'snapshot copy differs: '+str(target))
        entries.append(dict(original_path=filename, archive_path=str(target), sha256=expected))
    run.save(snapshot/'archive.json', dict(schema='physical-stage-source-snapshot-1',
        manifest_sha256=run.sha(out/'manifest.json'), entries=entries))
    return dict(status='prepared', jobs=len(jobs), native_timed_work_started=False)


def verify(manifest):
    require(manifest['source_inventory_sha256'] == digest(manifest['sources']), 'source inventory changed')
    for filename, expected in manifest['sources'].items():
        require(run.sha(filename) == expected, 'bound source changed: '+filename)


def wait_complete(child, clock, timeout):
    path = child.directory/'sampling-complete.json'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        require(child.process.poll() is None, 'owned actor exited before sampling-complete: '+child.name)
        if path.exists():
            try:
                value = run.read(path)
            except json.JSONDecodeError:
                time.sleep(.005)
                continue
            require(value['process_id'] == child.process.pid, 'sampling identity differs')
            return value
        time.sleep(.005)
    raise TimeoutError('sampling deadline: '+child.name)


def execute(out, manifest, job):
    from profiling.highlevel_accuracy.bridge import verify_native
    directory = out/'jobs'/job['id']
    directory.mkdir(parents=True, exist_ok=False)
    clock = NativeClock()
    program_binding = manifest['programs'][job['program_key']]
    program = run.read(program_binding['path'])
    protocol = manifest['protocol']
    env = run.child_environment(manifest['runtime_dirs'])
    for key in ('SIMULATOR_CUDA_SCHEDULE', 'STATE_PROBE_FIXED_ITERATIONS', 'STATE_PROBE_READY_FILE', 'STATE_PROBE_STOP_FILE'):
        env.pop(key, None)
    children, pins, completions = [], [], {}
    row = dict(schema='physical-service-execution-1', job=job, started_at_utc=run.utc(),
               program=program_binding, source_inventory_sha256=manifest['source_inventory_sha256'])

    def native(name, mode, instrument, target, count, mask):
        folder = directory/name
        command = [manifest['binary'], program_binding['path'], mode, instrument,
            'none' if target is None else str(target), str(protocol['warmup']), str(count), '0',
            str(folder/'ready.json'), str(folder/'go.json'), str(folder/'stop.json'), str(folder/'result.json')]
        child = OwnedChild(name, folder, command, env, clock, affinity_mask=mask, ready_timeout=30, stop_timeout=30)
        children.append(child)
        actor = OwnedActor.from_popen(name, child.process)
        pin = pin_owned_main_thread(actor, child.record['ready']['main_thread_id'], mask & -mask, clock=clock)
        pins.append(pin)
        run.save(folder/'thread-affinity.json', pin.receipt)
        return child

    def start_native(child):
        start = clock.now() + protocol['go_lead_ms'] * clock.frequency_hz // 1000
        run.save(child.directory/'go.json', dict(schema='physical-stages-go-1', clock_id='windows_qpc',
            frequency_hz=clock.frequency_hz, start_tick=start,
            end_tick=start+protocol['job_timeout_seconds']*clock.frequency_hz))
        completions[child.name] = wait_complete(child, clock, protocol['job_timeout_seconds'])

    def background(device):
        folder = directory/(device+'-background')
        role = device+'-worker'
        target = protocol['targets'][device+'_background']
        command = [manifest['background_binary'], program_binding['path'], role, 'bare', str(target),
            str(protocol['warmup']), str(folder/'ready.json'), str(folder/'go.json'), str(folder/'stop.json'), str(folder/'result.json')]
        child = OwnedChild(role, folder, command, env, clock,
            affinity_mask=protocol[device+'_background_mask'], ready_timeout=30, stop_timeout=30)
        children.append(child)
        return child

    try:
        backgrounds = []
        for device in ('cpu', 'gpu'):
            if job[device+'_background']:
                backgrounds.append(background(device))
        main = native('main', 'full', job['instrumentation'], None, job['samples'], protocol['main_process_mask'])
        refs = []
        if job['references']:
            for device in ('gpu', 'cpu'):
                refs.append(native('before-'+device, 'target', 'cupti', protocol['targets']['reference_'+device],
                    protocol['reference_'+device+'_samples'], protocol[device+'_reference_mask']))
        bg_start = clock.now()+clock.frequency_hz//50
        for child in backgrounds:
            end = bg_start+protocol['background_timeout_seconds']*clock.frequency_hz
            run.save(child.directory/'go.json', dict(schema='state-response-go-v1',clock=clock.metadata(),
                start_tick=bg_start,end_tick=end,phases=[dict(phase_index=0,begin_tick=bg_start,end_tick=end,
                    cpu_replay=job['cpu_background'],gpu_replay=job['gpu_background'])]))
        if backgrounds:
            time.sleep(protocol['background_settle_ms']/1000)
        for child in refs:
            start_native(child)
        start_native(main)
        if job['references']:
            for device in ('gpu', 'cpu'):
                child = native('after-'+device, 'target', 'cupti', protocol['targets']['reference_'+device],
                    protocol['reference_'+device+'_samples'], protocol[device+'_reference_mask'])
                start_native(child)
        row['status'] = 'sampled'
    except BaseException as error:
        row.update(status='failed', error=repr(error))
    finally:
        for pin in pins:
            try:
                pin.observe()
            except Exception as error:
                row.setdefault('affinity_observation_errors', []).append(repr(error))
            try:
                pin.close()
            except Exception as error:
                row.setdefault('affinity_close_errors', []).append(repr(error))
            row.setdefault('thread_affinity', []).append(pin.receipt)
        processes = []
        for child in reversed(children):
            try:
                processes.append(child.stop(cleanup=row.get('status') != 'sampled'))
            except Exception as error:
                row.setdefault('cleanup_errors', []).append(dict(name=child.name,error=repr(error)))
        row['processes'] = processes
        row['cleanup_verified'] = len(processes) == len(children) and all(p.get('process_exited_verified') and not p.get('forced_termination') for p in processes)
        row['completed'] = completions
        row['ended_at_utc'] = run.utc()
        row['numerical'] = {}
        if row.get('status') == 'sampled':
            try:
                for child in children:
                    path = child.directory/'result.json'
                    result = run.read(path)
                    if child.name.endswith('-worker'):
                        verification = result['verification_native']
                    else:
                        require(result['schema'] == 'physical-stage-native-1', 'native schema differs')
                        verification = result['verification_native']
                        row.setdefault('capture_status', {})[child.name] = dict(
                            instrumentation=result['instrumentation'],
                            all_requested_records_retained=result['cupti'].get('all_requested_records_retained'),
                            dropped_records=result['cupti'].get('cupti_dropped_records'),
                            callback_errors=result['cupti'].get('callback_errors'),
                            record_count=len(result['cupti'].get('records', [])))
                    receipt = verify_native(program, verification)
                    row['numerical'][child.name] = dict(passed=True, **receipt)
                    row.setdefault('raw_files', {})[child.name] = dict(path=str(path),sha256=run.sha(path))
                require(not row.get('affinity_observation_errors') and not row.get('affinity_close_errors'), 'thread affinity observation/cleanup failed')
                require(row['cleanup_verified'] and all(p['returncode']==0 for p in processes), 'owned cleanup/native exit failed')
                row['status'] = 'completed'
            except Exception as error:
                row.update(status='failed', error=repr(error))
        run.save(directory/'execution.json', row)
    require(row['status'] == 'completed', 'physical collection failed; retained '+str(directory/'execution.json'))
    return row


def collect(args):
    out = args.out.resolve()
    manifest = run.read(out/'manifest.json')
    verify(manifest)
    if args.split == 'heldout':
        require(args.barrier is not None, 'a saved frozen barrier is required before heldout launch')
        barrier = run.read(args.barrier)
        require(barrier.get('frozen_at_utc') and barrier.get('manifest_sha256') == run.sha(out/'manifest.json'),
                'barrier differs from the prepared campaign')
        frozen_path = Path(barrier['frozen_file'])
        require(run.sha(frozen_path) == barrier['frozen_file_sha256'], 'frozen file changed after barrier')
    run.save(out/(args.split+'-collection-started.json'), dict(started_at_utc=run.utc(),split=args.split,
        barrier=None if args.barrier is None else dict(path=str(args.barrier.resolve()),sha256=run.sha(args.barrier))))
    rows = []
    with run.measurement_lock():
        for job in manifest['jobs']:
            if job['split'] != args.split:
                continue
            verify(manifest)
            if args.split == 'heldout':
                require(run.sha(frozen_path) == barrier['frozen_file_sha256'], 'frozen model changed during heldout collection')
            row = execute(out, manifest, job)
            rows.append(row)
            print(json.dumps(dict(job=job['id'],status=row['status'],main_pid=next(
                p['pid'] for p in row['processes'] if p['name']=='main'),cleanup_verified=row['cleanup_verified'])),flush=True)
    run.save(out/(args.split+'-executions.json'), rows)
    return dict(status='collected',split=args.split,jobs=len(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    command = sub.add_parser('prepare')
    command.add_argument('--out',type=Path,required=True)
    command.add_argument('--binary',type=Path,required=True)
    command.add_argument('--background-binary',type=Path,required=True)
    command.add_argument('--runtime-dir',type=Path,action='append',required=True)
    command.add_argument('--seed',type=int,required=True)
    command = sub.add_parser('collect')
    command.add_argument('--out',type=Path,required=True)
    command.add_argument('--split',choices=('smoke','observer','profile','heldout'),required=True)
    command.add_argument('--barrier',type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args) if args.command=='prepare' else collect(args)))


if __name__ == '__main__':
    main()
