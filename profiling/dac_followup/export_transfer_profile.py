"""Export explicit response parameters without promoting them to default costs."""
from __future__ import annotations

import argparse
from pathlib import Path
from . import task1


def export(out,hardware_receipt):
    out=Path(out).resolve();hardware_receipt=Path(hardware_receipt).resolve()
    manifest=task1.read(out/"manifest.json");summary=task1.read(out/"summary.json")
    frozen=task1.read(out/"transfer-model-frozen.json");barrier=task1.read(out/"frozen-barrier.json")
    task1.require(task1.sha(out/"transfer-model-frozen.json")==barrier["frozen_sha256"],"frozen model differs")
    jobs=[j for j in manifest["jobs"] if j["split"] in {"calibration","transfer-heldout"}]
    repeats={j["request"]["repeats"] for j in jobs};task1.require(len(repeats)==1,"sampling contexts mixed")
    post=task1.read(hardware_receipt)
    entries=[]
    for route,row in summary["transfer"].items():
        model=frozen["models"][route]["selected"]
        entries.append(dict(route=route,qualified=row["qualified"],qualification_scope="heldout size-response mean error and trial CV only",
            status="qualified_size_response" if row["qualified"] else "heldout_validation_failed",
            parameters={key:model[key] for key in ("fixed_ns","small_ns_per_byte","large_ns_per_byte","breakpoint_bytes")},
            minimum_bytes=min(task1.TRAIN+task1.HELDOUT),maximum_bytes=max(task1.TRAIN+task1.HELDOUT),
            independent_test=dict(maximum_size_mean_ape=row["maximum_heldout_size_mean_ape"],process_cv_passes=row["process_cv_passes"],processes=row["processes"],
                                  limits=dict(maximum_size_mean_ape=.10,maximum_process_trial_cv=.10)),
            exact_measured_anchor_preservation_required=True,physical_bandwidth_scaling_qualified=False,
            effective_byte_service_counterfactual_requires_explicit_assumption=True,
            cost_table_is_not_automatically_installed=True))
    document=dict(schema="explicit-transfer-response-profile-1",created_at_utc=task1.utc(),entries=entries,
        scope=dict(binary=manifest["binaries"]["transfer"],runtime_dll_hashes=manifest["runtime_sha256"],
                   host_memory_contract=task1.HOST,boundary=manifest["protocol"]["transfer_boundary"],
                   cpu_compute_threads=1,main_thread_affinity_mask=4,process_affinity_mask=4,
                   warmup_minimum_iterations=20,warmup_minimum_ms=500,trials=8,repeats=next(iter(repeats)),
                   cuda_graphs="disabled",numerical_precision="F32",source_buffer_values="fixed procedural bytes",
                   timing="inclusive per-COPY mean from fixed-input batches; initialization/allocation/checking excluded",
                   hardware_identity=post["hardware"],hardware_identity_origin="post-campaign static query, matching prior E2; not pre-timing telemetry",
                   hardware_post_matches_prior_e2=post["all_shared_identity_fields_equal"],pretiming_resource_trajectory_measured=False,
                   loaded_runtime_module_resolution_measured=False),
        reuse_policy=dict(require_matching_hardware_binary_runtime_host_memory_and_sampling_context=True,
                          extrapolation_outside_tested_envelope=False,unqualified_route_rejected=True,
                          historical_e2_exact_costs_replaced=False,complete_strategy_accuracy_qualified=False),
        evidence=[task1.receipt(out/name) for name in ("manifest.json","transfer-model-frozen.json","frozen-barrier.json","calibration-executions.json","transfer-heldout-executions.json","summary.json")]+[task1.receipt(hardware_receipt)],
        exporter=task1.receipt(__file__))
    task1.write(out/"transfer-response-profile.json",document)
    return dict(profile=str(out/"transfer-response-profile.json"),qualified_routes=[row["route"] for row in entries if row["qualified"]])

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True);parser.add_argument("--hardware-receipt",type=Path,required=True)
    args=parser.parse_args();print(export(args.out,args.hardware_receipt))
