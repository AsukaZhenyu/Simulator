"""Record conditioning inventories without fitting a new cost dimension."""
from pathlib import Path
from llm_infer_model.performance.context_response import context_coordinate
from profiling.dac_validation import e2a


def record(out):
    out=Path(out).resolve();manifest=e2a.read(out/"manifest.json");rows=[]
    for key,indices in manifest["protocol"]["profile_groups"].items():
        binding=manifest["programs"][key];p=e2a.read(binding["program"]["path"])
        tensors=p["graph"]["workload"]["tensors"]
        catalogue=sum(t["storage_bytes"] for t in tensors)
        actions=p["actions"]
        for i in indices:
            rows.append(dict(program_key=key,program=binding["program"],action_index=i,
                full_sweep_actions=len(actions),full_sweep_tensor_catalogue_bytes=catalogue,
                full_sweep_tensor_count=len(tensors),prefix_actions=i,
                prefix_physical_services=sum(a["kind"] in {"COPY","COMPUTE"} for a in actions[:i]),
                response_coordinate=context_coordinate(p,i,manifest["context"])))
    e2a.write(out/"conditioning-metadata.json",dict(schema="dac-refinement-conditioning-inventory-1",
        generated_at_utc=e2a.utc(),manifest=e2a.receipt(out/"manifest.json"),source=e2a.receipt(__file__),
        rows=rows,new_axis_fitted=False,
        inventory_scope="logical tensor catalogue bytes; not measured CPU cache working set, physical peak, memory traffic, or private CUDA workspace",
        cyclic_scope="first physical service uses previous full pass's last physical service; warm fixed-input replay only, not cold start or autoregressive execution"))
    return dict(rows=len(rows),new_axis_fitted=False)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",required=True,type=Path)
    print(record(parser.parse_args().out))
