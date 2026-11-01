"""Freeze the external-interface coordination audit before executing cases."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Simulator/build/coordination-metrics-20261030/external'
TASKS = [
    ('dense_placement', 'Dense', 'CPU/GPU placement'),
    ('dense_prepare', 'Dense', 'demand/preparation'),
    ('attention_placement', 'refined_attention', 'CPU/GPU placement'),
    ('attention_refine', 'refined_attention', 'opaque/refined Attention'),
    ('ffn_partition', 'FFN_partition', 'partition fraction'),
    ('ffn_representation', 'FFN_partition', 'materialized/shared-parent view'),
    ('pipeline_placement', 'request_pipeline', 'CPU/GPU placement'),
    ('pipeline_prepare', 'request_pipeline', 'demand/preparation'),
    ('pd_placement', 'PD', 'phase placement'),
    ('pd_handoff', 'PD', 'handoff policy'),
    ('moe_execution', 'causal_MoE', 'execution rule'),
    ('moe_cache', 'causal_MoE', 'cache rule'),
]
CATEGORIES = {
    'allocation_alias': 'Physical allocation identities, alias/view-parent relations beyond declared policy.',
    'copy_payload': 'Actual copy byte amounts beyond declared shapes, routes and representation.',
    'reader_lifetime': 'Reader lifetime / release safety conditions beyond retention policy.',
    'preparation_dependency': 'Copy/readiness/execution dependencies beyond intended preparation/placement.',
}
SOURCE_ROOTS = {
    'LLMCompass': ROOT/'Simulator/build/dac-validation-20261025/e1-external-v2/LLMCompass',
    'LLMServingSim': ROOT/'Simulator/build/dac-validation-20261025/e1-external-v2/LLMServingSim',
    'Chakra': ROOT/'Simulator/build/dac-followup-20261025/external-wsl-build-v1/astra-sim/extern/graph_frontend/chakra',
}
SOURCE_FILES = {
    'LLMCompass': ['utils.py', 'software_model/utils.py', 'software_model/matmul.py',
                   'software_model/operators.py', 'software_model/softmax.py'],
    'LLMServingSim': ['serving/core/memory_model.py', 'serving/core/config_builder.py',
                      'serving/core/trace_generator.py', 'serving/core/graph_generator.py'],
    'Chakra': ['src/converter/llm_converter.py'],
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'PROTOCOL.json'
    if target.exists():
        raise SystemExit('Refusing to overwrite the frozen external protocol')
    doc = {
        'schema': 'coordination-external-protocol-1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_before_adapter_execution',
        'question': 'What additional coordination categories must a user declare after a paired high-level policy edit?',
        'tasks': [dict(id=i, structure=s, edit=e) for i,s,e in TASKS],
        'categories': CATEGORIES,
        'unit': 'At most one unit per independent category per paired edit; score is 0..4 only when all required obligations are actually evaluated.',
        'count_rule': {
            'count': 'Category of an additional, explicit, caller-supplied coordination value required by the executed path, beyond common workload/hardware/implementation and intended policy inputs.',
            'do_not_count': ['High-level policy selection itself', 'Repeated tensors/layers/events',
                'Adapter-derived facts', 'Information already supplied equally in workload/hardware/implementation contracts'],
            'partial': 'A partial interface witness receives evaluated-category evidence but no whole-task score. Unsupported or unimplemented paths receive null, never 4 or failure.',
            'zero': 'A frozen generic adapter that derives all required facts without extra user declarations receives zero, even if it matches our automation.',
            'interpretation': 'Observed declaration burden on the executed interface/adapter path, not a theoretical lower bound or human-effort measure.',
        },
        'fairness': [
            'Existing native automation and a once-built generic parameterized adapter are permitted without an artificial code-size restriction.',
            'The adapter may not call our mapping/modeling compiler or consume its emitted action list as input.',
            'Original tool definitions are executed unmodified with fixture dependencies where full imports are unnecessary; extracted source symbols and hashes are retained.',
            'A manual physical trace is accepted as a supported representation when the original parser accepts it.',
            'No missing field is interpreted as a fundamental inability; no deliberately broken foreign input is used to manufacture an error.',
            'Source/schema support, partial contract coverage and complete execution validation are separate statuses.',
            'No GPU/native performance test, whole-tool speed/accuracy comparison or physical OOM assertion.',
            'Each semantic family is equally weighted; trace expansion cannot increase automation credit.',
        ],
        'known_executable_scope_before_results': {
            'LLMCompass': 'Exact logical Tensor, Matmul and selected operator definitions. Its mapper and hardware modeling are acknowledged. A generic cross-operation ownership/lifetime adapter is not yet claimed implemented.',
            'LLMServingSim': 'Exact size/configuration functions, trace schema and original Chakra Layer parser. Existing batching/routing/offloading/KV automation is acknowledged. Full serving-engine/ASTRA execution is not part of this audit.',
            'common_warning': 'Interface acceptance does not certify all high-level coordination obligations. Complete scores require comparison to independent obligation artifacts.',
        },
        'development_and_holdout': {
            'inputs': 'Use the shared root task manifest for the 12 paired edits; bind its immutable hash before running held-out cases.',
            'split': 'Public development dimensions configure the generic adapter; held-out sizes/budgets are checked after source freeze.',
            'freeze_receipt': 'ADAPTER_FREEZE.json records source hashes and exact input manifest hashes before result evaluation.',
            'post_freeze_changes': 'Any bug fix gets a new version and explicit receipt; do not overwrite failed artifacts or silently refit to held-out results.',
        },
        'source_hashes': {str((SOURCE_ROOTS[k]/f).relative_to(ROOT)): sha(SOURCE_ROOTS[k]/f)
                          for k, files in SOURCE_FILES.items() for f in files if (SOURCE_ROOTS[k]/f).exists()},
        'no_cases_executed_by_this_script': True,
    }
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'protocol': str(target), 'sha256': sha(target), 'tasks': len(TASKS), 'cases_executed': 0}))

if __name__ == '__main__':
    freeze()
