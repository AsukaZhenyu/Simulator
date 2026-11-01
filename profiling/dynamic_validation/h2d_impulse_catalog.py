"""Export finite coordinate qualifications without modifying frozen v11 data.

Only eligibility metadata is added. Original prices, formula parameters and the
failed full-grid first-response flag are preserved. No observations are fitted,
collected, filtered or modified by this post-validation export.
"""
from __future__ import annotations
import argparse,copy,hashlib,itertools,json
from pathlib import Path

from llm_infer_model.performance.transfer_lifecycle_catalog import ScopedTransferLifecycle
from llm_infer_model.performance.parameters import UnknownParameter

ROOT=Path(__file__).resolve().parents[2]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def receipt(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def require(condition,message):
    if not condition:raise ValueError(message)

def export(out):
    out=Path(out).resolve()
    destination=out/'scoped-lifecycle-catalog.json';verification_path=out/'SCOPED_CATALOG_VERIFICATION.json'
    require(not destination.exists() and not verification_path.exists(),'new scoped export files required')
    manifest=read(out/'manifest.json');barrier=read(out/'frozen-barrier.json')
    require(barrier['manifest_sha256']==sha(out/'manifest.json') and barrier['fit_sha256']==sha(out/'fit.json') and barrier['predictions_sha256']==sha(out/'heldout-predictions.json'),'original freeze changed')
    original=read(out/'validated-lifecycle-catalog.json');evaluation=read(out/'INDEPENDENT_EVALUATION.json');fit=read(out/'fit.json');predictions=read(out/'heldout-predictions.json')['rows']
    sizes=tuple(manifest['protocol']['sizes_bytes']);idles=tuple(manifest['protocol']['idle_ms'])
    require(original['qualification_evidence']['sha256']==sha(out/'INDEPENDENT_EVALUATION.json'),'original qualification evidence changed')
    require(evaluation['parameters_refitted'] is False and evaluation['raw_samples_removed']==0,'target-blind complete independent evaluation required')
    models=[];checks=[];excluded=[]
    core=('schema','context_sha256','preparation_sha256','route','memory_mode','sustained','first_impulses','maximum_copies')
    for old in original['models']:
        memory=old['memory_mode'];rows=[r for r in evaluation['first_rows'] if r['memory_mode']==memory]
        require(len(rows)==len(sizes)*len(idles),'complete first-coordinate inventory required')
        require({(r['bytes'],r['idle_ms']) for r in rows}==set(itertools.product(sizes,idles)),'first coordinates differ')
        for r in rows:require(r['passed']==(r['APE']<=.10),'original first gate differs')
        source=next(p for p in fit['models'] if p['memory_mode']==memory)
        require(all(old[k]==source[k] for k in core),'validated catalog numerical parameters differ from frozen profile')
        require(old['first_qualified']==all(r['passed'] for r in rows),'original full-grid flag differs')
        profile=copy.deepcopy(old)
        profile['qualified_first_coordinates']=[[r['bytes'],r['idle_ms']*1000000] for r in rows if r['passed']]
        profile['first_coordinate_qualification_evidence']=[str(out/'INDEPENDENT_EVALUATION.json')+'#sha256='+sha(out/'INDEPENDENT_EVALUATION.json')]
        profile['original_full_grid_first_flag_preserved']=True
        profile['unqualified_first_coordinates']=[dict(bytes=r['bytes'],requested_idle_ns=r['idle_ms']*1000000,APE=r['APE']) for r in rows if not r['passed']]
        model=ScopedTransferLifecycle.from_profile(profile)
        kw=dict(context_sha256=old['context_sha256'],preparation_sha256=old['preparation_sha256'])
        for p in (p for p in predictions if p['memory_mode']==memory):
            first=model.first_response_ns(p['bytes'],p['idle_ms']*1000000,**kw,require_qualified=False)
            window=model.window_ns(p['bytes'],p['idle_ms']*1000000,p['copies'],**kw,require_qualified=False)
            require(first==p['first_prediction_ns'] and window==p['window_prediction_ns'],'new consumer changed a frozen numerical prediction')
        for r in rows:
            rejected=False
            try:model.first_response_ns(r['bytes'],r['idle_ms']*1000000,**kw)
            except UnknownParameter:rejected=True
            require(rejected==(not r['passed']),'coordinate first qualification differs from original independent gate')
            checks.append(dict(memory_mode=memory,bytes=r['bytes'],idle_ms=r['idle_ms'],expected_first_qualified=r['passed'],consumer_rejected=rejected))
            if not r['passed']:excluded.append(dict(r))
        windows=[r for r in evaluation['window_rows'] if r['memory_mode']==memory]
        require(len(windows)==len(sizes)*len(idles)*2 and all(r['passed'] for r in windows),'this export requires the independently qualified complete-window catalog')
        for r in windows:require(model.window_ns(r['bytes'],r['idle_ms']*1000000,r['copies'],**kw)==r['prediction_ns'],'qualified window price differs')
        for size,idle,n in ((8192,2000000,None),(65536,1000000,None),(65536,2000000,32),(16777216,2000000,None)):
            rejected=False
            try:
                if n is None:model.first_response_ns(size,idle,**kw)
                else:model.window_ns(size,idle,n,**kw)
            except UnknownParameter:rejected=True
            require(rejected,'unknown/unqualified coordinate silently substituted')
        require(model.first_qualified==old['first_qualified'] and model.window_qualified==old['window_qualified'],'original global qualification flags changed')
        models.append(profile)
    catalog=dict(schema='scoped-completed-transfer-lifecycles-1',models=models,qualification_evidence=receipt(out/'INDEPENDENT_EVALUATION.json'),
        original_catalog=receipt(out/'validated-lifecycle-catalog.json'),frozen_fit=receipt(out/'fit.json'),frozen_predictions=receipt(out/'heldout-predictions.json'),
        modeling_adapter='llm_infer_model.performance.transfer_lifecycle_catalog.ScopedTransferLifecycle',
        full_grid_first_qualified=evaluation['first_qualified'],full_grid_window_qualified=evaluation['window_qualified'],
        qualified_first_coordinates=sum(len(p['qualified_first_coordinates']) for p in models),excluded_first_coordinates=excluded,
        formula_and_numerical_parameters_unchanged=True,raw_data_changed=False,parameters_refitted=False,hardware_executed=False,
        first_MAPE=evaluation['first_MAPE'],first_worst_APE=evaluation['first_worst_APE'],window_MAPE=evaluation['window_MAPE'],window_worst_APE=evaluation['window_worst_APE'])
    frozen_model=next(s for s in manifest['sources'] if Path(s['source']['path']).name=='transfer_lifecycle.py')
    for key in ('source','archived'):require(sha(frozen_model[key]['path'])==frozen_model[key]['sha256'],'original calibrated modeling source changed')
    write(destination,catalog)
    verification=dict(schema='scoped-transfer-catalog-verification-1',status='passed',scoped_catalog=receipt(destination),
        consumer_source=receipt(ROOT/'modeling/llm_infer_model/performance/transfer_lifecycle_catalog.py'),
        original_calibrated_model_source=frozen_model,coordinate_checks=checks,
        unchanged_frozen_predictions_checked=len(predictions),qualified_window_coordinates_checked=len(evaluation['window_rows']),
        first_coordinates=len(checks),qualified_first_coordinates=sum(r['expected_first_qualified'] for r in checks),excluded_first_coordinates=len(excluded),
        original_full_grid_flags_preserved=True,unknown_coordinates_rejected=True,raw_data_changed=False,parameters_refitted=False,hardware_executed=False)
    write(verification_path,verification)
    return dict(catalog=receipt(destination),verification=receipt(verification_path),first_MAPE=evaluation['first_MAPE'],first_worst_APE=evaluation['first_worst_APE'],
        qualified_first_coordinates=verification['qualified_first_coordinates'],first_coordinates=verification['first_coordinates'],window_coordinates=len(evaluation['window_rows']),
        full_grid_first_qualified=evaluation['first_qualified'],full_grid_window_qualified=evaluation['window_qualified'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(export(a.out),ensure_ascii=False))

if __name__=='__main__':main()
