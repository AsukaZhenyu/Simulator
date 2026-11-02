"""Independent post-collection audit; frozen collector/native files stay intact.

Run: python -m profiling.workload_accuracy.tests.test_v2_evidence DIRECTORY
No native executable is launched. Both metadata and payload births are checked.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from profiling.workload_accuracy.contract import integer, same, sha, validate_native
from profiling.workload_accuracy.run import verify


def audit_expert_allocation_order(case,payload):
    """Check every metadata/payload allocation against an independent barrier map."""
    births={event['id']:event for event in payload['memory']['events'] if event['event']=='allocate'}
    deaths={event['id']:event for event in payload['memory']['events'] if event['event']=='release'}
    count=metadata=0
    for li,layer in enumerate(case['layers']):
        if layer['kind']!='moe':
            continue
        events=payload['residency'][li]['events']
        known={event['round']:event['sequence'] for event in events if event['event']=='route_known'}
        begin={event['round']:event['sequence'] for event in events if event['event']=='expert_ffn_begin'}
        for birth in births.values():
            if birth['layer']!=li or not birth['component'].startswith('resident_expert.'):
                continue
            count+=1;metadata+=birth['component'].endswith('.metadata')
            death=deaths.get(birth['id'])
            if death is None:
                raise ValueError('expert allocation has no release')
            if layer['residency']=='all':
                same(birth['round'],-1,'route-independent all-resident initialization')
                same(death['round'],len(case['rounds']),'all-resident lifetime through final round')
                continue
            ri=birth['round'];sequence=integer(birth.get('sequence'),'expert metadata/payload birth sequence')
            if ri not in known or not known[ri]<sequence<begin[ri]:
                raise ValueError('expert metadata/payload allocated before route_known or after expert FFN began')
            if death['round']<len(case['rounds']):
                ri=death['round'];sequence=integer(death.get('sequence'),'expert metadata/payload release sequence')
                if ri not in known or not known[ri]<sequence<begin[ri]:
                    raise ValueError('expert metadata/payload evicted outside current route-known barrier')
    return dict(all_expert_allocation_births_and_releases_after_route_known=True,
                expert_allocations_checked=count,expert_metadata_allocations_checked=metadata,
                expert_payload_allocations_checked=count-metadata)


def audit_directory(directory):
    root=Path(directory).resolve()
    report=verify(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    rows=[];mutations=[]
    for record in manifest['records']:
        path=root/record['raw_output'];raw=json.loads(path.read_text(encoding='utf-8'))
        native=json.loads(raw['stdout'])
        row=audit_expert_allocation_order(raw['case'],native)
        rows.append(dict(raw_output=path.name,raw_sha256=sha(path),**row))
        active=[li for li,layer in enumerate(raw['case']['layers']) if layer['kind']=='moe' and layer['residency']=='active']
        if not active:
            continue
        li=active[0]
        for kind in ('metadata','payload'):
            changed=copy.deepcopy(native)
            birth=next(event for event in changed['memory']['events'] if event['event']=='allocate'
                       and event['layer']==li and event['round']==0 and event['component'].startswith('resident_expert.')
                       and event['component'].endswith('.metadata')==(kind=='metadata'))
            barrier=next(event['sequence'] for event in changed['residency'][li]['events'] if event['event']=='route_known' and event['round']==0)
            birth['sequence']=barrier-1
            try:
                audit_expert_allocation_order(raw['case'],changed)
            except ValueError as error:
                mutations.append(dict(raw_output=path.name,mutation='early_expert_'+kind+'_birth',rejected=True,reason=str(error)))
            else:
                raise ValueError('independent audit accepted early '+kind+' allocation')
    return dict(schema='workload_independent_all_expert_lifetime_audit/1',
                collector_native_model_sources_changed=False,
                manifest_sha256=sha(root/'manifest.json'),frozen_model_sha256=sha(root/'frozen-model.json'),
                independent_auditor_sha256=sha(__file__),all_records_verified=True,
                metadata_and_payload_causality_pass=True,corruption_tests_pass=True,
                records=rows,corruption_tests=mutations,
                timing_acceptance=report['accepted'],physical_capacity_verified=False)


class IndependentV2EvidenceTests(unittest.TestCase):
    def test_native_metadata_and_payload_early_allocation_corruptions(self):
        for backend in ('cpu','cuda'):
            path=ROOT/f'profiling/workload_accuracy/formal-v2-{backend}-20261018/heldout-1-0.raw.json'
            if not path.exists():
                self.skipTest('optional previously collected native evidence absent')
            raw=json.loads(path.read_text(encoding='utf-8'));native=json.loads(raw['stdout'])
            result=audit_expert_allocation_order(raw['case'],native)
            self.assertGreater(result['expert_metadata_allocations_checked'],0)
            self.assertGreater(result['expert_payload_allocations_checked'],0)
            for kind in ('metadata','payload'):
                changed=copy.deepcopy(native)
                birth=next(event for event in changed['memory']['events'] if event['event']=='allocate' and event['layer']==0 and event['round']==0 and event['component'].startswith('resident_expert.') and event['component'].endswith('.metadata')==(kind=='metadata'))
                known=next(event['sequence'] for event in changed['residency'][0]['events'] if event['event']=='route_known' and event['round']==0)
                birth['sequence']=known-1
                with self.assertRaisesRegex(ValueError,'metadata/payload allocated before route_known'):
                    audit_expert_allocation_order(raw['case'],changed)


if __name__=='__main__':
    if len(sys.argv)==2 and not sys.argv[1].startswith('-'):
        folder=Path(sys.argv[1]);result=audit_directory(folder)
        (folder/'independent-expert-lifetime-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({key:value for key,value in result.items() if key not in ('records','corruption_tests')},indent=2))
    else:
        unittest.main()
