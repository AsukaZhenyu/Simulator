"""Deterministic actual activation traces, with revisiting changing routes."""
import argparse
import copy
import json
import math
from pathlib import Path


def case(kind='dense', backend='CPU', seed=7101, *, hidden=16, intermediate=24):
    layer={'kind':'dense'} if kind=='dense' else {'kind':'moe','experts':4,'top_k':2,'residency':'active'}
    value=dict(schema='workload_accuracy_case/1',id=kind+'-two-layer-changing-trace',backend=backend,
               cuda_device=0,threads=1,seed=seed,hidden=hidden,intermediate=intermediate,
               heads=4,kv_heads=2,head_dim=hidden//4,rope_dims=hidden//4,rope_base=10000.,
               norm_eps=1e-6,attention_gate='none',qk_norm='none',max_context=8,
               layers=[copy.deepcopy(layer),copy.deepcopy(layer)],rounds=[],tracked_budget_bytes=64*1024**2)
    for ri,n in enumerate((2,1,1,1)):
        x=[[math.sin((t+1)*(j+3)*.071+ri*.31+seed*.001)*.35 for j in range(hidden)] for t in range(n)]
        routes=[];scores=[]
        for li in range(2):
            if kind=='dense':
                routes.append(None);scores.append(None)
            else:
                # Round 0 experts 0/1, round 1 2/3, round 2 0/1, round 3 1/2.
                a=(0,2,0,1)[ri]
                routes.append([[(a+li)%4,(a+li+1)%4] for _ in range(n)])
                scores.append([[.4,.6] for _ in range(n)])
        value['rounds'].append(dict(hidden_input=x,routes=routes,route_scores=scores))
    return value


def suite(backend='CPU'):
    return dict(schema='workload_accuracy_suite/1',id='fresh-traced-workload-'+backend.lower(),
                samples=5,warmup=1,processes=3,
                profile_seeds=[7101,7102,7103],heldout_seeds=[8101,8102,8103],
                quality={'max_within_process_cv':.15,'max_between_process_cv':.20,'max_heldout_relative_error':.20},
                cases=[case('dense',backend),case('moe',backend)])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['CPU','CUDA'],default='CPU')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(suite(args.backend),indent=2)+'\n',encoding='utf-8')
