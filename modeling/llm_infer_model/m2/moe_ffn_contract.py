"""Physical F32 MoE FFN trace replay contract; independent of synthetic M2 layers.

An external trace prescribes grouped work. Random probe router weights produce
selected probabilities and a separately reported top-k discovery. This contract
never claims that those random weights generated the external trace.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import statistics

CONTRACT = "ggml_cpu_resident_moe_ffn_f32_v1"
NATIVE_SCHEMA = "resident_moe_ffn_native_v1"
SUITE_SCHEMA = "moe_measurement_suite_v1"
MEASUREMENT_SCHEMA = "moe_measurements_v1"
TIMING_SCOPE = "steady_clock_completed_cpu_graph_with_persistent_threadpool"
STABILITY_CV_LIMIT = 0.1
INTERLEAVED_CONTRACT = "ggml_cpu_resident_moe_ffn_f32_v2"
INTERLEAVED_TIMING_SCOPE = "interleaved_materialized_stages_and_joint_on_shared_resident_graph"

def sampling_mode(case):
    return case.get("sampling_mode", "isolated_warm")

def contract_for(case):
    return INTERLEAVED_CONTRACT if sampling_mode(case)=="interleaved_joint_context" else CONTRACT

def timing_scope_for(case):
    return INTERLEAVED_TIMING_SCOPE if sampling_mode(case)=="interleaved_joint_context" else TIMING_SCOPE



def _int(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in the supported range")
    return value


def validate_case(case):
    if not isinstance(case, dict):
        raise ValueError("MoE measurement case must be an object")
    unknown = set(case) - {"id", "config", "routes", "threads", "backend", "phase", "work_items", "source", "comment", "sampling_mode"}
    if unknown:
        raise ValueError(f"unsupported MoE case fields: {sorted(unknown)}")
    out = copy.deepcopy(case)
    if sampling_mode(out) not in ("isolated_warm", "interleaved_joint_context"):
        raise ValueError("unsupported MoE sampling_mode")
    if not isinstance(out.get("id"), str) or not out["id"].strip():
        raise ValueError("case id must be a nonempty string")
    if out.get("backend") != "CPU":
        raise ValueError("MoE physical probe supports explicit backend CPU only; no fallback")
    _int(out.get("threads"), "threads", maximum=512)
    config = out.get("config")
    if not isinstance(config, dict) or set(config) != {"hidden", "intermediate", "experts", "top_k", "tokens"}:
        raise ValueError("config must explicitly declare hidden/intermediate/experts/top_k/tokens")
    for key, maximum in (("hidden",32768),("intermediate",65536),("experts",1024),("tokens",4096),("top_k",1024)):
        _int(config[key], key, maximum=maximum)
    if config["top_k"] > config["experts"]:
        raise ValueError("top_k must not exceed experts")
    routes = out.get("routes")
    if not isinstance(routes, list) or len(routes) != config["tokens"]:
        raise ValueError("routes must cover every flattened token exactly once")
    for route in routes:
        if not isinstance(route, list) or len(route) != config["top_k"]:
            raise ValueError("each route must contain exactly top_k experts")
        for expert in route:
            _int(expert, "expert id", 0, config["experts"]-1)
        if len(set(route)) != len(route):
            raise ValueError("an expert cannot be repeated in a token route")
    if "phase" in out and out["phase"] not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")
    if "work_items" in out:
        items = out["work_items"]
        if not isinstance(items, list) or len(items) != config["tokens"]:
            raise ValueError("work_items must map every flattened token")
        seen = set()
        for item in items:
            if not isinstance(item, dict) or set(item) != {"request_id", "absolute_position"}:
                raise ValueError("work_items require request_id and absolute_position")
            if not isinstance(item["request_id"], str) or not item["request_id"]:
                raise ValueError("work item request_id must be nonempty")
            _int(item["absolute_position"], "absolute_position", 0)
            key = (item["request_id"], item["absolute_position"])
            if key in seen:
                raise ValueError("duplicate token work item")
            seen.add(key)
    return out


def groups(case):
    case = validate_case(case)
    result = {}
    for token, route in enumerate(case["routes"]):
        for slot, expert in enumerate(route):
            result.setdefault(expert, []).append((token, slot))
    return dict(sorted(result.items()))


def stage_names(case):
    return ["router", *(f"expert.{expert}" for expert in groups(case)), "combine", "joint"]


def stage_ports(case, stage):
    case = validate_case(case)
    if stage not in stage_names(case):
        raise ValueError("stage is not active in this route trace")
    c = case["config"]
    h, i, e, k, q = (c[key] for key in ("hidden", "intermediate", "experts", "top_k", "tokens"))
    ports = {}
    def add(name, a, b, category, dtype="f32"):
        ports[name] = {"dtype": dtype, "ne": [a,b,1,1], "bytes": a*b*4, "category": category}
    add("hidden_input", h,q,"input")
    router, combine = stage in ("router","joint"), stage in ("combine","joint")
    selected = int(stage.split(".")[1]) if stage.startswith("expert.") else None
    if router:
        add("router_weight",h,e,"weight")
        add("route_ids",k,q,"input","i32")
        add("selected_scores",k,q,"output")
        add("discovered_topk",k,q,"output","i32")
    elif combine:
        add("selected_scores",k,q,"input")
    for expert, rows in groups(case).items():
        n, suffix = len(rows), f".{expert}"
        if stage == "joint" or selected == expert:
            for role in ("gate","up"):
                add(role+suffix,h,i,"weight")
            add("down"+suffix,i,h,"weight")
            add("rows"+suffix,n,1,"input","i32")
            add("expert_output"+suffix,h,n,"output")
        elif combine:
            add("expert_output"+suffix,h,n,"input")
        if combine:
            add("rows"+suffix,n,1,"input","i32")
            add("slots"+suffix,n,1,"input","i32")
    if combine:
        add("hidden_output",h,q,"output")
    return ports


def stage_memory(case, stage):
    ports = stage_ports(case, stage)
    values = {category+"_bytes": sum(p["bytes"] for p in ports.values() if p["category"] == category)
              for category in ("weight","input","output")}
    values["resident_logical_bytes"] = sum(values.values())
    return values


def contract_hash(signature):
    if not isinstance(signature, dict):
        raise ValueError("physical signature must be an object")
    return hashlib.sha256(json.dumps(signature,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()


def physical_signature(case, native_binary_sha256=None):
    case = validate_case(case)
    if native_binary_sha256 is not None and (not isinstance(native_binary_sha256,str) or len(native_binary_sha256)!=64 or any(c not in "0123456789abcdef" for c in native_binary_sha256)):
        raise ValueError("native binary hash must be SHA256")
    signature = {"contract":contract_for(case),"scope":"resident_moe_ffn_subgraph","backend":"CPU","dtype":"f32",
            "config":case["config"],"routes":case["routes"],"threads_requested":case["threads"],"threads_applied":case["threads"],
            "routing_mode":"prescribed_trace_replay","router_discovery_used_for_grouping":False,
            "random_weights_not_model_weights":True,"timing_scope":timing_scope_for(case),
            "router_operations":["linear","softmax","argsort_topk_discovery","gather_prescribed_probabilities","renormalize_selected_probabilities","materialized_copy"],
            "expert_operations":["gather_trace_token_rows","gate_projection","up_projection","swiglu","down_projection","materialized_copy"],
            "combine_operations":["gather_slot_scores","weight_expert_rows","scatter_add_to_tokens","sum_experts","residual_add","materialized_copy"],
            "excluded_operations":["attention","normalization","QKV_projection","KV_cache","weight_transfer","original_model_kernel"],
            "stage_ports":{stage:stage_ports(case,stage) for stage in stage_names(case)},
            "work_items":case.get("work_items"),"phase":case.get("phase"),"native_binary_sha256":native_binary_sha256}
    if "sampling_mode" in case:
        signature["sampling_mode"]=sampling_mode(case)
    if sampling_mode(case)=="interleaved_joint_context":
        signature["workspace_context"]="shared_joint_allocator_and_common_CPU_plan_arena"
    return signature


def build_plan(suite):
    if not isinstance(suite,dict) or suite.get("schema_version") != SUITE_SCHEMA or set(suite)-{"schema_version","cases","source","comment"}:
        raise ValueError(f"suite must use {SUITE_SCHEMA}")
    cases=suite.get("cases")
    if not isinstance(cases,list) or not cases:
        raise ValueError("suite cases must be nonempty")
    cases=[validate_case(case) for case in cases]
    if len({case["id"] for case in cases})!=len(cases):
        raise ValueError("case ids must be unique")
    if len({contract_for(case) for case in cases})!=1:
        raise ValueError("one suite cannot mix isolated-v1 and shared-context-v2 timing contracts")
    return cases


def statistics_ns(values):
    if not isinstance(values,list) or len(values)<3:
        raise ValueError("timing requires at least three samples")
    for value in values:
        _int(value,"sample_ns")
    mean=statistics.mean(values)
    return {"median_ns":statistics.median(values),"mean_ns":mean,"min_ns":min(values),"max_ns":max(values),
            "cv":statistics.pstdev(values)/mean,"count":len(values)}


def validate_native(native, case):
    case=validate_case(case)
    if not isinstance(native,dict):
        raise ValueError("native result must be an object")
    validate_case({"id":case["id"],"config":native.get("config"),"routes":native.get("routes"),"threads":native.get("threads_requested"),"backend":native.get("backend")})
    _int(native.get("threads_applied"),"threads_applied")
    for field, expected_bool in (("random_weights_not_model_weights",True),("router_discovery_used_for_grouping",False)):
        if native.get(field) is not expected_bool:
            raise ValueError("native boolean contract mismatch")
    expected={"schema_version":("resident_moe_ffn_native_v2" if sampling_mode(case)=="interleaved_joint_context" else NATIVE_SCHEMA),"contract":contract_for(case),"backend":"CPU","routing_mode":"prescribed_trace_replay",
              "scope":"resident_moe_ffn_subgraph","random_weights_not_model_weights":True,"router_discovery_used_for_grouping":False,
              "dtype":"f32","config":case["config"],"routes":case["routes"],"threads_requested":case["threads"],
              "threads_applied":case["threads"],"timing_scope":timing_scope_for(case),"port_digest_algorithm":"fnv1a64_native_tensor_bytes"}
    for key,value in expected.items():
        if native.get(key)!=value:
            raise ValueError(f"native {key} disagrees with declared case/contract")
    if sampling_mode(case)=="interleaved_joint_context" and native.get("sampling_mode")!="interleaved_joint_context":
        raise ValueError("native interleaved sampling mode missing")
    _int(native.get("warmup"),"warmup",0)
    _int(native.get("samples"),"samples",3,10000)
    _int(native.get("seed"),"seed",0,1000000)
    discovered=native.get("discovered_routes")
    discovery_case={**case,"routes":discovered}
    validate_case(discovery_case)
    stages=native.get("stages")
    if not isinstance(stages,dict) or set(stages)!=set(stage_names(case)):
        raise ValueError("native stage coverage must exactly match active expert groups")
    for stage,value in stages.items():
        if not isinstance(value,dict):
            raise ValueError("native stage must be an object")
        stats=statistics_ns(value.get("samples_ns"))
        if stats["count"] != native["samples"]:
            raise ValueError("native timing sample count mismatch")
        _int(value.get("workspace_bytes"),"workspace_bytes",0)
        if value.get("workspace_source")!=("shared_ggml_gallocr_internal_buffer_plus_cpu_plan_arena" if sampling_mode(case)=="interleaved_joint_context" else "ggml_gallocr_internal_buffer_plus_cpu_graph_plan"):
            raise ValueError("CPU workspace must have measured allocator/plan provenance")
        memory=value.get("memory_accounting")
        if not isinstance(memory,dict):
            raise ValueError("native memory accounting missing")
        for key in ("resident_logical_bytes","resident_allocated_bytes","internal_buffer_bytes","cpu_graph_plan_work_bytes"):
            _int(memory.get(key),key,0)
        if memory["resident_logical_bytes"]!=stage_memory(case,stage)["resident_logical_bytes"] or memory["resident_allocated_bytes"]<memory["resident_logical_bytes"]:
            raise ValueError("native resident inventory does not match declared ports")
        if memory.get("metadata_bytes_included") is not False or value["workspace_bytes"]!=memory["internal_buffer_bytes"]+memory["cpu_graph_plan_work_bytes"]:
            raise ValueError("native workspace accounting mismatch")
        if sampling_mode(case)=="interleaved_joint_context":
            for key in ("shared_resident_logical_bytes","shared_resident_allocated_bytes","stage_cpu_graph_plan_required_bytes"):
                _int(memory.get(key),key,0)
            if (memory.get("shared_allocator") is not True or
                memory["shared_resident_logical_bytes"]!=stage_memory(case,"joint")["resident_logical_bytes"] or
                memory["shared_resident_allocated_bytes"]!=memory["resident_allocated_bytes"] or
                memory["shared_resident_allocated_bytes"]<memory["shared_resident_logical_bytes"] or
                memory["stage_cpu_graph_plan_required_bytes"]>memory["cpu_graph_plan_work_bytes"]):
                raise ValueError("shared graph memory context mismatch")
        reference=value.get("reference_check")
        if value.get("all_outputs_finite") is not True or not isinstance(reference,dict) or reference.get("passed") is not True:
            raise ValueError("native stage did not pass independent numerical reference")
        for key in ("max_abs_error","max_rel_error","atol","rtol"):
            x=reference.get(key)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0:
                raise ValueError("native numerical reference fields must be finite nonnegative numbers")
        if reference["atol"] != 0.0001 or reference["rtol"] != 0.0001:
            raise ValueError("native numerical reference tolerance changed")
        ports=value.get("ports")
        declared=stage_ports(case,stage)
        if not isinstance(ports,dict) or set(ports)!=set(declared):
            raise ValueError("native port inventory mismatch")
        for name,port in ports.items():
            if not isinstance(port,dict):
                raise ValueError("native port must be an object")
            ne=port.get("ne")
            if not isinstance(ne,list) or len(ne)!=4:
                raise ValueError("native port shape must have four integer dimensions")
            for dimension in ne:
                _int(dimension,"native port dimension")
            _int(port.get("bytes"),"native port bytes",0)
            if not isinstance(port,dict) or {key:port.get(key) for key in ("dtype","ne","bytes","category")}!=declared[name]:
                raise ValueError(f"native port geometry mismatch: {stage}.{name}")
            digest=port.get("digest")
            if not isinstance(digest,str) or len(digest)!=16 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("native port digest must be lowercase FNV1a64")
    if sampling_mode(case)=="interleaved_joint_context":
        shared_keys=("shared_resident_logical_bytes","shared_resident_allocated_bytes","internal_buffer_bytes","cpu_graph_plan_work_bytes")
        baseline=stages["joint"]["memory_accounting"]
        if any(any(value["memory_accounting"][key]!=baseline[key] for key in shared_keys) for value in stages.values()):
            raise ValueError("stages do not share the joint allocation")
    joint=stages["joint"]["ports"]
    for stage, value in stages.items():
        for name,port in value["ports"].items():
            if port["digest"]!=joint[name]["digest"]:
                raise ValueError(f"materialized stage lineage mismatch: {stage}.{name}")
    return native
