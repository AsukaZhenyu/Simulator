from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from profiling.depth_validation.programs import build_one, file_sha, materialize, matrix, reseed_case, validate_compact
from profiling.highlevel_accuracy.bridge import canonical, export_plan, reference_ports, sha
from profiling.highlevel_accuracy.multilayer import build_multilayer_case


class DepthProgramsTests(unittest.TestCase):
    def setUp(self):
        parent = Path(__file__).resolve().parents[2]/'build/test-depth-programs'
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(parent.resolve()))

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, name='case', strategy=2, layers=2):
        result = build_one(self.folder/name,layers=layers,strategy=strategy,
            hidden=8,intermediate=16,heads=2,kv_heads=1,head_dim=4,tokens=3,
            asset_pool=self.folder/'assets',seed=123)
        program = json.loads(Path(result['program']['path']).read_text(encoding='utf-8'))
        return result,program

    def test_declared_matrix_is_18_distinct_cases(self):
        rows=matrix()
        self.assertEqual(len(rows),18)
        self.assertEqual(len({r['id'] for r in rows}),18)
        self.assertEqual({r['layers'] for r in rows if r['phase']=='prefill'},{4,8,16,32})
        self.assertEqual(sum(r['phase']=='decode' for r in rows),6)

    def test_old_json_canonical_encoding_unchanged(self):
        value=dict(z=[1,1.25,-2.4],a=dict(x=True))
        self.assertEqual(canonical(value),json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False))

    def test_array_identity_binds_dtype_and_numeric_bits(self):
        first=np.asarray([1.0,2.0],dtype=np.float32)
        self.assertEqual(sha(first),sha(first.copy()))
        self.assertNotEqual(sha(first),sha(np.asarray([1,2],dtype=np.int32)))
        changed=first.copy();changed[1]=2.5
        self.assertNotEqual(sha(first),sha(changed))

    def test_compact_roundtrip_preserves_all_reference_ports(self):
        result,program=self.build()
        self.assertEqual(program['initial_values'],{})
        self.assertFalse(result['native_timing_executed'])
        restored=materialize(program)
        self.assertEqual(restored['program_sha256'],program['materialized_program_sha256'])
        self.assertTrue(all(isinstance(v,np.ndarray) for v in restored['initial_values'].values()))
        expected=json.loads(Path(result['reference']['path']).read_text(encoding='utf-8'))
        observed=reference_ports(restored)
        self.assertEqual(expected,observed)
        self.assertEqual(len(program['invocation_scopes']),2)
        self.assertEqual(program['invocation_scopes'][1]['incoming_hidden'],
                         program['invocation_scopes'][0]['hidden_outputs'])

    def test_strategy_variants_share_identical_numeric_asset(self):
        first,_=self.build('cpu',0)
        second,_=self.build('gpu',1)
        self.assertEqual(first['initial_assets'],second['initial_assets'])
        self.assertEqual(len(list((self.folder/'assets').glob('*.bin'))),1)

    def test_modified_binary_asset_is_rejected(self):
        result,program=self.build()
        path=Path(result['initial_assets']['path'])
        with path.open('r+b') as stream:
            first=stream.read(1);stream.seek(0);stream.write(bytes([first[0]^1]))
        with self.assertRaisesRegex(ValueError,'file bytes/hash'):
            materialize(program)

    def test_offset_tampering_cannot_be_rehashed_into_valid_layout(self):
        _,program=self.build()
        changed=deepcopy(program)
        first=next(iter(changed['initial_assets']['tensors'].values()))
        first['offset_bytes']=4
        changed['identity']['initial_assets']=sha(changed['initial_assets'])
        changed.pop('program_sha256');changed['program_sha256']=sha(changed)
        with self.assertRaisesRegex(ValueError,'offset/overlap'):
            validate_compact(changed)

    def test_layer_limit_rejects_33_before_artifacts(self):
        target=self.folder/'bad'
        with self.assertRaisesRegex(ValueError,'1..32'):
            build_one(target,layers=33,strategy=0)
        self.assertFalse(target.exists())

    def test_reseed_same_seed_preserves_exact_materialized_program(self):
        result,program=self.build()
        regenerated=reseed_case(result['program']['path'],self.folder/'same-seed',seed=123,
                                asset_pool=self.folder/'assets')
        updated=json.loads(Path(regenerated['program']['path']).read_text(encoding='utf-8'))
        self.assertEqual(updated['initial_assets'],program['initial_assets'])
        self.assertEqual(updated['materialized_program_sha256'],program['materialized_program_sha256'])
        self.assertEqual(Path(result['reference']['path']).read_bytes(),
                         Path(regenerated['reference']['path']).read_bytes())
        self.assertFalse(regenerated['compiler_or_simulator_rerun'])

    def test_new_seed_changes_data_but_preserves_graph_without_recompile(self):
        result,program=self.build()
        with (patch('profiling.depth_validation.programs.build_multilayer_case',side_effect=AssertionError('recompile')),
              patch('profiling.depth_validation.programs.export_plan',side_effect=AssertionError('reexport'))):
            regenerated=reseed_case(result['program']['path'],self.folder/'new-seed',seed=124,
                                    asset_pool=self.folder/'assets')
        updated=json.loads(Path(regenerated['program']['path']).read_text(encoding='utf-8'))
        for field in ('graph','actions','bindings','invocation_scopes','frontend_ir','physical'):
            self.assertEqual(updated[field],program[field])
        self.assertNotEqual(updated['initial_assets']['file']['sha256'],program['initial_assets']['file']['sha256'])
        for tid,entry in program['initial_assets']['tensors'].items():
            newer=updated['initial_assets']['tensors'][tid]
            if entry['dtype']=='GGML_TYPE_I32':self.assertEqual(entry['sha256'],newer['sha256'])
            else:self.assertNotEqual(entry['sha256'],newer['sha256'])
        self.assertEqual(reference_ports(materialize(updated)),
                         json.loads(Path(regenerated['reference']['path']).read_text(encoding='utf-8')))

    def test_reseed_invalid_seed_rejected_before_output(self):
        target=self.folder/'bad-seed'
        with self.assertRaisesRegex(ValueError,'uint32'):
            reseed_case({},target,seed=-1)
        self.assertFalse(target.exists())

    def test_lightweight_driver_preserves_public_trace_duration_and_numerics(self):
        recipes=(('cpu','cpu','cpu'),('gpu','gpu','gpu'),('gpu','cpu','gpu'))
        for layers in (1,2):
            for strategy,devices in enumerate(recipes):
                with self.subTest(layers=layers,strategy=strategy):
                    plan=build_multilayer_case(self.folder/f'equivalence-l{layers}-s{strategy}',
                        layers=layers,hidden=8,intermediate=16,heads=2,kv_heads=1,
                        head_dim=4,tokens=3,history=0,threads=1,devices=devices).compile()
                    full=export_plan(plan,seed=321,numeric_arrays=True)
                    with (patch.object(type(plan),'run',side_effect=AssertionError('full debug driver called')),
                          patch('tensor_mapping.unified_strategy.observation_hash',side_effect=AssertionError('full observation hash called'))):
                        light=export_plan(plan,seed=321,numeric_arrays=True,lightweight_actions=True)
                    self.assertEqual(light['actions'],full['actions'])
                    self.assertEqual(light['original_simulation'],full['original_simulation'])
                    self.assertEqual(light['graph'],full['graph'])
                    self.assertEqual(light['bindings'],full['bindings'])
                    self.assertEqual(reference_ports(light),reference_ports(full))


if __name__=='__main__': unittest.main()
