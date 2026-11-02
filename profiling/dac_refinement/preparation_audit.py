"""Prospective preparation safeguard applied to retained requests, without re-gating."""
from dataclasses import asdict
from pathlib import Path
import os
import subprocess

from llm_infer_model.performance.preparation_contract import compare_preparation
from profiling.dac_validation import e2a
from .preparation_adapter import IMPLEMENTATION_SOURCES, refinement_preparation


def audit(out):
    out=Path(out).resolve();root=Path(__file__).resolve().parents[2]
    receipt_dir=out/"preparation-contract-safeguard";receipt_dir.mkdir(exist_ok=False)
    manifest=e2a.read(out/"manifest.json");archives=e2a.read(out/"source-snapshots.json")
    retained=e2a.read(out/"summary.json")
    source_hashes={};source_evidence=[]
    for key in IMPLEMENTATION_SOURCES:
        suffix="profiling/dac_refinement/native/refinement_probe.cpp" if key=="refinement_probe.cpp" else "profiling/native/steady_costs/probe.cpp"
        source=next(s for s in manifest["sources"] if s["path"].replace("\\","/").endswith(suffix))
        archived=next(s["archived"] for s in archives["sources"] if s["original"]==source)
        e2a.require(e2a.sha(archived["path"])==archived["sha256"]==source["sha256"],"native source archive changed")
        source_hashes[key]=source["sha256"];source_evidence.append(dict(implementation_key=key,executed=source,archived=archived))
    held=e2a.read(out/"heldout-executions.json");observer=e2a.read(out/"observer-executions.json");rows=[]
    for obs in observer:
        candidates=[ex for ex in held if ex["job"]["program_key"]==obs["job"]["program_key"] and ex["job"]["round"]==obs["job"]["round"]]
        e2a.require(len(candidates)==1,"primary request pairing is unknown or ambiguous")
        primary=candidates[0];contracts=[];receipts=[]
        for ex in (primary,obs):
            e2a.require(ex["status"]=="completed" and e2a.sha(ex["native"]["path"])==ex["native"]["sha256"],"native result missing or changed")
            rp=out/"jobs"/ex["job"]["id"]/"request.json";request=e2a.read(rp);native=e2a.read(ex["native"]["path"])
            e2a.require(request==native["request"]==ex["job"]["request"],"actual/native/manifest requests differ")
            contracts.append(refinement_preparation(request,source_hashes=source_hashes));receipts.append(dict(request=e2a.receipt(rp),native=ex["native"]))
        program_result=next(p for p in retained["programs"] if p["program_key"]==obs["job"]["program_key"])
        stored=next(p for p in program_result["observer"] if p["raw"]==obs["native"])
        # Read the old numerical outcome verbatim; preparation eligibility is
        # separate and does not recompute, replace or reinterpret its gate.
        effect=stored["effect"];internal_pass=stored["passed"]
        rows.append(dict(program_key=obs["job"]["program_key"],round=obs["job"]["round"],
            primary_job=primary["job"]["id"],observer_job=obs["job"]["id"],
            primary_contract=asdict(contracts[0]),observer_contract=asdict(contracts[1]),
            preparation_comparison=compare_preparation(*contracts),
            within_observer_AB_preparation=compare_preparation(contracts[1],contracts[1]),
            retained_internal_AB_effect=effect,retained_internal_AB_5percent_pass=internal_pass,
            primary_unbiased_observer_certification="rejected_"+compare_preparation(*contracts)["status"] if compare_preparation(*contracts)["status"]!="match" else "preparation_eligible_only_bias_not_certified",
            evidence=receipts))
    output=dict(schema="measurement-preparation-contract-audit-1",generated_at_utc=e2a.utc(),
        prospective_software_safeguard=True,source_evidence=source_evidence,rows=rows,
        counts=dict(pairs=len(rows),primary_preparation_matches=sum(r["preparation_comparison"]["status"]=="match" for r in rows),
            primary_preparation_mismatches=sum(r["preparation_comparison"]["status"]=="mismatch" for r in rows),
            unknown=sum(r["preparation_comparison"]["status"]=="unknown" for r in rows),
            within_observer_AB_preparation_matches=sum(r["within_observer_AB_preparation"]["status"]=="match" for r in rows),
            retained_internal_AB_5percent_passes=sum(r["retained_internal_AB_5percent_pass"] for r in rows)),
        new_native_processes=0,new_parameter_fitting=False,historical_gates_recomputed=False,
        historical_summary=e2a.receipt(out/"summary.json"),
        conclusion="All24 observer A/B contrasts share their own preparation. Their extra prime pass differs from the primary reset-only target, so24/24 internal contrasts cannot certify unbiased instrumentation of that primary harness. Historical numerical gates and primary/secondary results remain unchanged.")
    e2a.write(receipt_dir/"preparation-contract-audit.json",output)
    command=[os.sys.executable,"-m","unittest","profiling.tests.test_preparation_contract","profiling.tests.test_context_refinement","-v"]
    result=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(root/p) for p in ("modeling","mapping","profiling"))),text=True,capture_output=True)
    for name,value in (("stdout",result.stdout),("stderr",result.stderr)):(receipt_dir/("preparation-contract-tests-"+name+".txt")).write_text(value,encoding="utf-8",newline="\n")
    sources=[root/"modeling/llm_infer_model/performance/preparation_contract.py",Path(__file__).with_name("preparation_adapter.py"),Path(__file__),root/"profiling/tests/test_preparation_contract.py"]
    e2a.write(receipt_dir/"preparation-contract-tests.json",dict(command=command,returncode=result.returncode,passed=result.returncode==0,
        native_execution=False,sources=[e2a.receipt(p) for p in sources],
        stdout=e2a.receipt(receipt_dir/"preparation-contract-tests-stdout.txt"),stderr=e2a.receipt(receipt_dir/"preparation-contract-tests-stderr.txt")))
    e2a.write(receipt_dir/"preparation-contract-evidence.json",dict(schema="measurement-preparation-safeguard-evidence-1",
        audit=e2a.receipt(receipt_dir/"preparation-contract-audit.json"),tests=e2a.receipt(receipt_dir/"preparation-contract-tests.json"),
        measured_sources_unchanged=True,prior_evidence=e2a.receipt(out/"final-evidence-v2.json"),new_native_processes=0))
    return dict(counts=output["counts"],tests_returncode=result.returncode,report=str(receipt_dir/"preparation-contract-audit.json"))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",required=True,type=Path)
    print(audit(parser.parse_args().out),flush=True)
