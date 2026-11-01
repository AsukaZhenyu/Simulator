"""Audit the bounded actual ServingSim backend campaign and bind its evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def git_receipt(path, windows_checkout=False):
    result = {}
    for key, args in [('head', ['rev-parse', 'HEAD']),
                      ('tracked_status', ['status', '--porcelain', '--untracked-files=no'])]:
        options = ['-c', 'core.autocrlf=true'] if windows_checkout else []
        proc = subprocess.run(['git', *options, '-C', str(path), *args], text=True,
                              capture_output=True, check=True)
        result[key] = proc.stdout.strip()
    assert not result['tracked_status'], (str(path), result)
    result['checkout_eol_contract'] = 'original Windows checkout core.autocrlf=true' if windows_checkout else 'native LF checkout'
    return result


def audit(base):
    base = Path(base).resolve()
    serving = base / 'ServingSim'
    original = ROOT / 'build/dac-validation-20261025/e1-external-v2/LLMServingSim'
    source = git_receipt(original, windows_checkout=True)
    assert source['head'] == 'a4053bc1161872420e1e0607cb3409ef659b828e'
    raw_status = subprocess.run(['git','-C',str(original),'status','--porcelain','--untracked-files=no'],
                                capture_output=True,text=True,check=True)
    eol = subprocess.run(['git','-C',str(original),'ls-files','--eol',
                          'serving/__main__.py','serving/core/router.py'],
                         capture_output=True,text=True,check=True)
    dump(base/'windows-checkout-eol-witness.json',dict(
        schema='readonly-cross-platform-git-eol-witness-1',
        raw_git_status_stdout=raw_status.stdout,
        raw_status_line_count=len(raw_status.stdout.splitlines()),
        raw_git_status_stderr=raw_status.stderr,
        git_ls_files_eol_stdout=eol.stdout,
        normalized_read_argv=['git','-c','core.autocrlf=true','-C',str(original),
                              'status','--porcelain','--untracked-files=no'],
        normalized_read_stdout=source['tracked_status'],
        configuration_written=False,source_content_modified=False,
        interpretation='Windows checkout CRLF is unchanged. WSL default Git read treats CRLF as content changes; command-local core.autocrlf=true verifies canonical tracked content without rewriting files/config.'))
    astra = base / 'astra-sim'
    nested = [astra, astra / 'extern/graph_frontend/chakra',
              astra / 'extern/helper/fmt', astra / 'extern/helper/spdlog',
              astra / 'extern/memory_backend/analytical',
              astra / 'extern/network_backend/analytical',
              astra / 'extern/network_backend/analytical/extern/yaml-cpp']
    # The network submodule's yaml-cpp checkout is under extern/ or dependencies/
    # depending on this pinned upstream layout. Resolve from observed .git files.
    if not (nested[-1] / '.git').exists():
        hits = list((astra / 'extern/network_backend/analytical').rglob('yaml-cpp/.git'))
        assert len(hits) == 1, hits
        nested[-1] = hits[0].parent
    pins = {str(path.relative_to(base)): git_receipt(path) for path in nested}
    assert pins['astra-sim']['head'] == 'd3469945c50410cac2e727c9ab627670c965db2a'
    copies = []
    for path in (original / 'serving').rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            target = serving / 'serving' / path.relative_to(original / 'serving')
            assert target.is_file() and sha(target) == sha(path), str(target)
            copies.append(dict(relative=str(path.relative_to(original)), sha256=sha(path)))
    chakra = astra / 'extern/graph_frontend/chakra'
    chakra_copies = []
    for path in (chakra / 'src').rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        target = base / 'site/chakra/src' / path.relative_to(chakra / 'src')
        assert target.is_file() and sha(target) == sha(path), str(target)
        chakra_copies.append(dict(relative=str(path.relative_to(chakra)), sha256=sha(path)))
    cases = [(2, 16, 'whole-l2-p16-v2', 'whole-cli-002')]
    cases += [(l, p, f'whole-l{l}-p{p}-matrix-v1', f'whole-cli-l{l}-p{p}-matrix-v1')
              for l, p in ((2, 64), (4, 16), (4, 64))]
    rows = []
    artifacts = []
    for layers, prompt, run_id, attempt in cases:
        out = base / attempt
        cli = json.loads((out / 'summary.json').read_text())
        compare = json.loads((out / 'backend-comparison.json').read_text())
        decoded = json.loads((out / 'decoded-ets.json').read_text())
        target = next(row for row in decoded if 'instance0_batch0' in row['path'])
        assert all(sha(Path(row['path'])) == row['sha256'] for row in decoded)
        graph = json.loads((out / 'imported-event-graph.json').read_text())
        whole_graph = json.loads((out / 'imported-whole-protocol-graph.json').read_text())
        costs = json.loads((out / 'node-costs.json').read_text())
        assert cli['returncode'] == 0 and cli['status'] == 'completed'
        assert compare['status'] == 'pass' and compare['difference_ns'] == 0
        assert compare['whole_protocol_difference_ns'] == 0
        assert compare['bootstrap_node']['name'] == 'COMP_NODE_event_1ns_0'
        assert compare['bootstrap_et_duration_ns'] == 1
        assert compare['node_types'] == dict(MEM_LOAD_NODE=1, COMP_NODE=9*layers+4, MEM_STORE_NODE=1)
        tasks = {task['id']: task for task in graph['tasks']}
        assert len(tasks) == len(target['nodes'])
        whole_tasks={task['id']:task for task in whole_graph['tasks']}
        assert len(whole_tasks)==len(tasks)+1
        assert whole_tasks['native-first-arrival-event']['depends_on']==[]
        assert whole_tasks['node-0']['depends_on']==['native-first-arrival-event']
        for node in target['nodes']:
            key = 'node-' + str(node['id'])
            expected = {'node-' + str(x) for x in node['data_deps'] + node['ctrl_deps']}
            assert set(tasks[key]['depends_on']) == expected
            # These four native ETs are serial. Verify that rather than assuming
            # arbitrary DAG sums are a valid independent event-simulator oracle.
            assert expected == (set() if node['id'] == 0 else {'node-' + str(node['id'] - 1)})
            if node['id'] != 0:
                assert whole_tasks[key]['depends_on']==tasks[key]['depends_on']
        inputs = serving / 'astra-sim/inputs/runs' / run_id
        trace = next(path for path in (inputs / 'trace').rglob('instance0_batch0.txt'))
        lines = trace.read_text().splitlines()
        native_rows = [line.split() for line in lines[3:] if line.strip()]
        assert len(native_rows) == 9*layers+4 == int(lines[1])
        independent_compute = sum(int(row[1]) for row in native_rows)
        assert independent_compute == compare['native_compute_ns']
        assert int(native_rows[0][3]) == 4*prompt
        assert int(native_rows[-1][7]) == 4*prompt
        independent_io = 2*(100+(4*prompt)//16)
        assert sum(row['duration_ns'] for row in costs) == independent_compute + independent_io
        assert compare['native_target_elapsed_ns'] == independent_compute + independent_io
        assert compare['native_protocol_total_ns'] == independent_compute + independent_io + 1
        rows.append(dict(layers=layers, hidden=128, prompt=prompt, output_tokens=1,
                         native_compute_nodes=9*layers+4, native_memory_nodes=2,
                         input_bytes=4*prompt, native_sampler_output_bytes=4*prompt,
                         native_compute_ns=independent_compute, memory_ns=independent_io,
                         target_interval_ns=compare['native_target_elapsed_ns'],
                         native_total_clock_ns=compare['native_protocol_total_ns'],
                         ours_modeling_event_dag_ns=compare['ours_modeling_event_dag_ns'],
                         difference_ns=compare['difference_ns'],
                         whole_protocol_difference_ns=compare['whole_protocol_difference_ns'],
                         cli_wall_seconds=cli['wall_seconds'], run_id=run_id, attempt=attempt,
                         target_et_sha256=compare['target_et_sha256']))
        artifacts.extend(path for path in out.rglob('*') if path.is_file())
        artifacts.extend(path for path in inputs.rglob('*') if path.is_file())
    binary = base / 'astra-build/bin/AstraSim_Analytical_Congestion_Unaware'
    steps = json.loads((base / 'native-build-steps.json').read_text())
    assert all(step['returncode'] == 0 for step in steps)
    failed = json.loads((base / 'whole-cli-001/summary.json').read_text())
    assert failed['status'] == 'failed_actual_cli_attempt'
    anchors = []
    for path, first, last, reason in [
        (serving/'serving/core/router.py',227,232,'First arrival clamps to max(1, arrival_time_ns).'),
        (serving/'serving/__main__.py',611,619,'Frontend builds explicit first-arrival event workload.'),
        (serving/'serving/core/trace_generator.py',1855,1874,'Event runtime is explicitly emitted into trace.'),
        (base/'site/chakra/src/converter/llm_converter.py',210,213,'Compute price is copied to duration_micros field.'),
        (astra/'astra-sim/workload/Workload.cc',178,184,'Pinned fixed-price branch consumes duration as ns without x1000.'),
        (astra/'extern/memory_backend/analytical/AnalyticalMemory.cc',269,275,'Analytical remote memory latency plus truncated byte/bandwidth term.'),
    ]:
        snippet = '\n'.join(path.read_text().splitlines()[first-1:last])
        anchors.append(dict(path=str(path.relative_to(base)), first_line=first,
                            last_line=last, sha256=sha(path), reason=reason, snippet=snippet))
        artifacts.append(path)
    receipts = [path for path in base.glob('*.json')
                if path.name not in ('backend-summary.json','backend-audit.json','backend-manifest.json')]
    artifacts += receipts + [binary, base/'run_whole_cli_002.py']
    artifacts += [Path(__file__), Path(__file__).with_name('dac_followup_external_backend.py'),
                  Path(__file__).with_name('dac_followup_external_backend_matrix.py')]
    artifacts += list((base/'whole-cli-001').rglob('*'))
    artifacts += list((serving/'profiler/perf/declared-fixture/followup/tiny-dense').rglob('*.csv'))
    unique = sorted({path.resolve() for path in artifacts if path.is_file()}, key=str)
    manifest = [dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path))
                for path in unique]
    dump(base/'backend-manifest.json', dict(schema='dac-followup-backend-manifest-1', artifacts=manifest))
    audit = dict(schema='dac-followup-backend-audit-1', status='pass',
                 actual_jobs_checked=4, native_et_edges_match_imported_graph=True,
                 independent_native_text_trace_sums_match=True,
                 independent_io_bytes_and_formula_match=True, explicit_bootstrap_et_matches=True,
                 source_servingsim=source, astra_and_required_submodule_pins=pins,
                 unmodified_serving_core_files=len(copies), unmodified_chakra_python_files=len(chakra_copies),
                 copied_core_hashes=copies, copied_chakra_hashes=chakra_copies,
                 source_anchors=anchors, artifact_count=len(manifest),
                 manifest_sha256=sha(base/'backend-manifest.json'))
    dump(base/'backend-audit.json',audit)
    summary = dict(schema='dac-followup-native-serving-backend-summary-1', status='pass',
                   actual_complete_native_cli_jobs=4, actual_native_backend_target_timings=4,
                   actual_native_backend_binary_sha256=sha(binary), actual_backend='ASTRA analytical congestion_unaware',
                   fixture_profiles=True, independent_real_hardware_accuracy=False,
                   common_cost_semantics_comparison=True, cost_refit_from_target=False,
                   source_core_modified=False, native_request_output_numerics_verified=False,
                   ours_path='llm_infer_model.performance.Graph -> evaluate (event DAG)',
                   ours_strategy_ir_compiler_for_these_serving_ets=False,
                   highlevel_dense_frontend_coverage=False,
                   native_physical_allocator_accuracy_claimed=False,
                   layers=[2,4], prompts=[16,64], hidden=128, dtype='float16', cases=rows,
                   max_target_interval_difference_ns=max(abs(r['difference_ns']) for r in rows),
                   max_full_protocol_difference_ns=max(abs(r['whole_protocol_difference_ns']) for r in rows),
                   bootstrap='Explicit first-arrival event: router max(1, arrival), encoded 1 ns for these arrival=0 fixtures; not a fitted overhead.',
                   native_sampler_output='Native generator emits prompt_count*4 bytes for these fixtures; preserved exactly, not a verified LLM sampler algorithm.',
                   unsuccessful_cli_attempts_retained=['whole-cli-001'],
                   historical_frontend_only_campaign='external-v1/attempt-002: 24 traces, zero backend timings; remains unchanged.',
                   no_global_package_install=True,
                   workspace_deb_bytes=sum(p.stat().st_size for p in (base/'debs').glob('*.deb')),
                   workspace_wheel_bytes=sum(p.stat().st_size for p in (base/'wheels').glob('*.whl')),
                   configure_seconds=next(s['wall_seconds'] for s in steps if '-S' in s['argv']),
                   compile_seconds=next(s['wall_seconds'] for s in steps if '--build' in s['argv']),
                   audit_file='backend-audit.json', audit_sha256=sha(base/'backend-audit.json'),
                   manifest_file='backend-manifest.json', manifest_sha256=sha(base/'backend-manifest.json'))
    dump(base/'backend-summary.json',summary)
    report = [
        '# Actual native LLMServingSim backend comparison', '',
        'Four complete native CLI jobs executed scheduler -> trace -> Chakra -> ASTRA C++ -> request completion. '
        'Both sides consume the same declared synthetic component prices and analytical memory contract. '
        'This verifies matched event composition, not independent hardware accuracy or performance superiority.', '',
        '| Layers | Prompt | Compute (ns) | Memory (ns) | Native target (ns) | Ours event DAG (ns) | Difference (ns) |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        report.append(f"| {r['layers']} | {r['prompt']} | {r['native_compute_ns']} | {r['memory_ns']} | {r['target_interval_ns']} | {r['ours_modeling_event_dag_ns']} | {r['difference_ns']} |")
    report += ['',
        'The native total clock includes a separate 1 ns first-arrival event. This is explicitly present in the generated ET '
        'and follows router max(1, arrival); no fitted startup constant is added. Reconstructing the full protocol '
        'from that event and the target DAG also matches all four native total clocks exactly.', '',
        'The native generator emits 4*prompt bytes for both the request input and sampler output in these fixtures '
        '(64 B or 256 B). We preserve that generator boundary. This is not a numerical validation of a real LLM sampler.', '',
        'Our imported path is modeling.performance.Graph/evaluate. These ServingSim ETs do not pass through '
        'StrategyIR/compiler/tensor_engine or the high-level Model.dense frontend. The separate earlier Compass '
        'campaign covers 32 real StrategyIR compiler/tensor-engine placement checks.', '',
        'The native C++ backend was built from pinned ASTRA with only necessary pinned submodules. All copied '
        'ServingSim/Chakra core files match their source hashes. Dependencies were extracted into this workspace; '
        'no global package installation or external core patch was performed. The failed initial CLI attempt '
        '(missing explicit fixture model field) and package/download failures are retained.', '',
        'Evidence: backend-summary.json, backend-audit.json, backend-manifest.json, native-build-steps.json; '
        'per-case raw stdout/stderr, native ETs/text traces, imported graph, node prices and comparison JSON.', '',
        'Scope: H=128, two/four layers, prefill 16/64 with one generated output token, TP1, one request, '
        'resident weights and declared synthetic FP16 profiles. No decode loop, MoE, multi-request pipeline, '
        'P/D handoff, complete physical allocation accuracy or real-device speed claim is established by this matrix.', '',
    ]
    (base/'BACKEND_REPORT.md').write_text('\n'.join(report),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='pass', actual_jobs=4, max_difference_ns=0,
                          artifact_count=len(manifest), summary=str(base/'backend-summary.json'))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',required=True)
    audit(parser.parse_args().build)
