"""Offline immutable-program read cache; unchanged E2A price arithmetic.

This adapter avoids re-decoding a multi-million-element initializer for every
primitive repetition. It verifies all measured input hashes before fitting and
never caches native result documents or changes the original freeze function.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

from . import e2a


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve();manifest=e2a.read(out/"manifest.json")
    e2a.require(not(out/"heldout-started.json").exists(),"offline cache refinement must precede target collection")
    sources={};executions=e2a.read(out/"profile-executions.json")
    expected={j["id"]:j for j in manifest["jobs"] if j["split"]=="profile"}
    e2a.require(len(executions)==len(expected) and len({e["job"]["id"] for e in executions})==len(expected),"profile process coverage differs")
    identities=set();previous_end=None
    for execution in executions:
        e2a.require(execution["status"]=="completed","all independent profile jobs must complete")
        e2a.require(execution["job"]==expected[execution["job"]["id"]],"profile launch differs from frozen inventory")
        ep=out/"jobs"/execution["job"]["id"]/"execution.json"
        e2a.require(e2a.read(ep)==execution,"aggregate execution differs from original receipt")
        e2a.require(execution["returncode"]==0 and execution["forced_termination"] is False and execution["process_exited_verified"] is True and execution["numerical"]["passed"],"profile normal exit/numerical proof absent")
        identity=(execution["process_id"],execution["started_at_utc"])
        e2a.require(identity not in identities,"duplicate process launch identity")
        identities.add(identity)
        e2a.require(previous_end is None or execution["started_at_utc"]>=previous_end,"sequential owned process lifetimes overlap")
        previous_end=execution["completed_at_utc"]
        sources[str(ep.resolve())]=e2a.sha(ep)
        for source in (execution["native"],execution["program"]["program"],execution["program"]["reference"]):
            sources[source["path"]]=source["sha256"]
    for source in manifest["sources"]:sources[source["path"]]=source["sha256"]
    begin=time.perf_counter()
    for path,expected in sources.items():e2a.require(e2a.sha(path)==expected,"frozen source data changed: "+path)
    verification=dict(schema="dac-e2a-offline-freeze-verification-1",verified_at_utc=e2a.utc(),sources=sources,
        manifest=e2a.receipt(out/"manifest.json"),adapter=e2a.receipt(__file__),
        original_freezer=e2a.receipt(e2a.__file__),hash_verification_wall_seconds=time.perf_counter()-begin,
        purpose="immutable program parsing cache only; original price arithmetic and source observations unchanged",
        target_measurements_read=False)
    e2a.write(out/"freeze-source-verification.json",verification)
    original_read=e2a.read;cache={};hits=0
    def cached_program_read(path):
        nonlocal hits
        path=Path(path).resolve()
        if path.parent.name!="programs":return original_read(path)
        if path in cache:hits+=1;return cache[path]
        cache[path]=original_read(path);return cache[path]
    e2a.read=cached_program_read
    started=time.perf_counter()
    try:result=e2a.freeze(out)
    finally:e2a.read=original_read
    receipt=dict(schema="dac-e2a-offline-freeze-cache-1",verification=e2a.receipt(out/"freeze-source-verification.json"),
        source=e2a.receipt(__file__),original_freezer=e2a.receipt(e2a.__file__),
        unique_programs_decoded=len(cache),immutable_program_cache_hits=hits,
        freeze_wall_seconds=time.perf_counter()-started,parameters=e2a.receipt(out/"profile-frozen.json"),
        predictions=e2a.receipt(out/"predictions.json"),original_parameter_arithmetic_unchanged=True,
        model_or_target_adjustment_performed=False)
    e2a.write(out/"freeze-cache-receipt.json",receipt)
    print(json.dumps(dict(result=result,cache=receipt),indent=2,allow_nan=False))


if __name__=="__main__":main()
