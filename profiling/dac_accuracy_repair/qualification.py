"""Attach independent qualification without modifying frozen measurements."""
from __future__ import annotations

import copy
from pathlib import Path

from profiling.dac_validation import e2a
from llm_infer_model.performance.prepared_transfer import PreparedTransferModel


def promote(out, *, audit_receipt):
    """Qualify only exact, independently audited prepared transfer contracts.

    The supplied independent audit must bind this exact summary. Its schema is
    intentionally checked by the caller; this helper independently rechecks the
    immutable prediction barrier and all declared strict transfer predicates.
    """
    out=Path(out).resolve()
    e2a.require(e2a.sha(audit_receipt['path'])==audit_receipt['sha256'], 'independent audit changed')
    audit=e2a.read(audit_receipt['path'])
    e2a.require(audit.get('schema')=='dac-accuracy-repair-independent-audit-1' and audit.get('passed') is True,
                'a passing independent evidence audit is required')
    summary=e2a.read(out/'summary.json');barrier=e2a.read(out/'frozen-barrier.json')
    summary_path=(out/'summary.json').resolve();summary_hash=e2a.sha(summary_path)
    e2a.require(any(Path(row['path']).resolve()==summary_path and row['sha256']==summary_hash
                    for row in audit.get('sources',[])), 'audit does not bind the current summary')
    for name, value in barrier['files'].items():e2a.require(e2a.sha(out/name)==value, 'frozen price or prediction changed')
    published=[]
    for mode in ('pageable','registered'):
        result=summary['D2H'][mode]
        if not result['qualified_strict_process_mean_gate']:continue
        rows=result['processes']
        e2a.require(len(rows)==24 and all(r['ape']<=.1 and r['cv']<=.1 and abs(r['observer_effect'])<=.05 for r in rows)
                    and result['trial_p90_ape']<=.2 and result['maximum_size_mean_ape']<=.1,
                    'strict transfer qualification predicates differ')
        frozen_path=out/f'transfer-{mode}-frozen.json';frozen=e2a.read(frozen_path)
        evidence=out/f'transfer-{mode}-qualification.json'
        e2a.write(evidence,dict(schema='prepared-transfer-qualification-1',qualified_at_utc=e2a.utc(),
            contract=frozen['contract'],frozen_profile=e2a.receipt(frozen_path),summary=e2a.receipt(out/'summary.json'),
            independent_audit=audit_receipt,calibration_changed=False,price_refitted=False,
            predicates=dict(size_mean_ape_max=.1,each_process_mean_ape_max=.1,pooled_trial_p90_ape_max=.2,
                            each_process_trial_cv_max=.1,each_observer_effect_max=.05),
            evidence_scope='warm fixed-input repeated inclusive host-completed transfer; exact measured memory and preparation only',
            not_single_cold_copy_or_physical_payload_bandwidth_qualification=True))
        promoted=copy.deepcopy(frozen);promoted['qualified']=True;promoted['qualification_receipt']=e2a.receipt(evidence)
        path=out/f'transfer-{mode}-qualified.json';e2a.write(path,promoted)
        e2a.require(promoted['parameters']==frozen['parameters'] and promoted['contract']==frozen['contract'], 'qualification changed prices or scope')
        model=PreparedTransferModel.from_profile(promoted)
        model.prediction(262144,contract=promoted['contract'])
        published.append(e2a.receipt(path))
    return published


def promote_window_catalog(out, window, *, audit_receipt, window_audit_receipt):
    """Publish only the finite size/replay grid with independent evidence.

    This is a different sampling contract from the original short-window
    campaign. Its coefficients remain the original pre-heldout freeze.
    """
    from llm_infer_model.performance.validated_transfer import ValidatedTransferCatalog
    out,window=Path(out).resolve(),Path(window).resolve()
    summary=e2a.read(out/'summary.json');extension=e2a.read(window/'summary.json')
    for item,schema,summary_path in ((audit_receipt,'dac-accuracy-repair-independent-audit-1',out/'summary.json'),
                                    (window_audit_receipt,'dac-accuracy-repair-window-independent-audit-1',window/'summary.json')):
        e2a.require(e2a.sha(item['path'])==item['sha256'],'independent qualification audit changed')
        audit=e2a.read(item['path'])
        e2a.require(audit.get('schema')==schema and audit.get('passed') is True,'independent evidence audit must pass')
        e2a.require(any(Path(row['path']).resolve()==summary_path and row['sha256']==e2a.sha(summary_path)
                        for row in audit['sources']),'independent audit does not bind the summary')
    e2a.require(extension['parameter_refit'] is False and extension['original_results_modified'] is False
                and extension['calibration']['accepted'],'window calibration/parameter-transfer rule did not pass')
    entries={}
    for mode in ('pageable','registered'):
        price=out/f'transfer-{mode}-frozen.json';copied=window/price.name
        e2a.require(e2a.sha(price)==e2a.sha(copied),'original coefficients or source preparation changed')
        result=extension['D2H'][mode]['combined_policy'];frozen=e2a.read(price)
        e2a.require(result['qualified_strict_process_mean_gate'] and result['completed_processes']==24
                    and result['cv_passes']==24 and result['observer_passes']==24
                    and result['maximum_process_ape']<=.1 and result['trial_p90_ape']<=.2,
                    'combined prepared-mean qualification did not pass')
        grid={str(row['bytes']):4096 if row['bytes']<=16384 else 256 for row in result['sizes']}
        e2a.require(set(map(int,grid))=={64,1024,2048,16384,262144,2097152,8388608,33554432},'validated grid differs')
        proof=window/f'transfer-{mode}-sampling-policy-qualification.json'
        e2a.write(proof,dict(schema='prepared-transfer-sampling-policy-qualification-1',qualified_at_utc=e2a.utc(),
            source_profile=e2a.receipt(price),main_summary=e2a.receipt(out/'summary.json'),
            window_summary=e2a.receipt(window/'summary.json'),independent_audit=audit_receipt,
            window_independent_audit=window_audit_receipt,validated_replays=grid,
            coefficients_changed=False,parameter_refit=False,original_R256_qualification_changed=False,
            single_copy_latency_qualified=False,
            qualification_scope='finite byte grid; warm completed-copy block means under the explicit size-dependent replay policy'))
        entries[mode]=dict(source_contract=frozen['contract'],parameters=frozen['parameters'],
            minimum_bytes=frozen['minimum_bytes'],maximum_bytes=frozen['maximum_bytes'],
            validated_replays=grid,qualification_receipt=e2a.receipt(proof),qualified=True)
    catalog=dict(schema='prepared-transfer-validated-catalog-1',entries=entries,
        evidence_scope='Finite byte grid on the recorded hardware/runtime. Small copies use R4096 block means; large copies use R256. No single-copy, original-short-window, unseen-byte or physical-bandwidth qualification.',
        original_profile_qualification={mode:summary['D2H'][mode]['qualified_original_size_mean_gate'] for mode in entries},
        coefficients_changed=False,parameter_refit=False)
    model=ValidatedTransferCatalog.from_manifest(catalog)
    for entry in entries.values():
        for size,repeats in entry['validated_replays'].items():
            contract=copy.deepcopy(entry['source_contract']);contract['preparation_contract']['timed_replays_per_batch']=repeats
            e2a.require(model.prediction(int(size),contract=contract)['qualified_for_requested_contract'],'qualified finite-grid lookup failed')
    destination=window/'validated-transfer-catalog.json';e2a.write(destination,catalog)
    return e2a.receipt(destination)
