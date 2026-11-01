"""Inspectable external-version/setup receipts, never simulated comparisons."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess

PINNED=(
    dict(name="LLMCompass",repository="https://github.com/PrincetonUniversity/LLMCompass",
        commit="2e015fd2ee750e6cad8d7152df52551e5b41ef20",
        verified_commit_url="https://github.com/PrincetonUniversity/LLMCompass/commit/2e015fd2ee750e6cad8d7152df52551e5b41ef20",
        source_observation="Official pinned repository includes software_model, hardware_model and design_space_exploration. It must not be described as having no software or strategy modeling.",
        source_url="https://github.com/PrincetonUniversity/LLMCompass/tree/2e015fd2ee750e6cad8d7152df52551e5b41ef20"),
    dict(name="LLMServingSim",repository="https://github.com/casys-kaist/LLMServingSim",
        commit="a4053bc1161872420e1e0607cb3409ef659b828e",
        verified_commit_url="https://github.com/casys-kaist/LLMServingSim/commit/a4053bc1161872420e1e0607cb3409ef659b828e",
        source_observation="Official pinned README describes a Python serving frontend, heterogeneous/disaggregated tiers, MoE and parallelism. It must not be described as a simulator with no serving strategy support.",
        source_url="https://github.com/casys-kaist/LLMServingSim/tree/a4053bc1161872420e1e0607cb3409ef659b828e"),
)


def run_external(output):
    from experiments.strategy_fidelity import write_json
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    requirements=["whole macro placement","attention and FFN refinement","materialized partition/merge",
        "cache and prefetch","two request pipeline","causal expert scheduling","refinement+cache",
        "refinement+prefetch","pipeline+cross-layer-prefetch","PD KV handoff","expert migration/cache","shared physical parent views"]
    for item in PINNED:
        command=["git","ls-remote",item["repository"]+".git","HEAD"]
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=20)
            receipt=dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
            setup="remote_reachable_not_installed" if result.returncode==0 else "setup_failed_network"
        except Exception as error:
            receipt=dict(command=command,error=repr(error));setup="setup_failed"
        row=dict(**item,status=setup,created_utc=datetime.now(timezone.utc).isoformat(),
            commit_verification="Official GitHub commit link inspected via web tool on 2026-10-25; this is a frozen version, not a claim about the latest HEAD.",
            source_review_only=True,executable_comparison_run=False,setup_receipt=receipt,
            requirements=[dict(requirement=r,status="untested",adapter_lines=None,core_changes=None) for r in requirements],
            fair_interpretation="Setup failure or untested requirement is not evidence of unsupported expression; no performance or productivity superiority is claimed.")
        write_json(output/(item["name"]+".json"),row);rows.append(row)
    summary=dict(schema="e1-external-attempts-v1",tools=rows,declared_attempts=2,
        executed_comparison_count=0,complete_as_execution_comparison=False,
        claim="Version and setup/source-review receipts only. Novelty versus these capable tools remains an explicit limitation.")
    write_json(output/"summary.json",summary)
    return summary
