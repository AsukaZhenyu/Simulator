"""Generate an explicitly batched probe from immutable historical C++ sources.

Only anchored timing/metadata changes are allowed. The original graph builders,
mathematical references, precision contract and numerical checks are retained.
No native program is executed by this generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONTEXT_SHA256_LF = "37ed5525074d2653050daa433a7b938a6f7f751a1afdeaac17a8c21cc2173cd4"
HELPER_SHA256_LF = "2556bed47f9f45ffef7884f5f58b452ec69a5c036b98a1e137afc9c55085c327"

WRAPPER = r'''
// One sample is the TOTAL time of N completed invocations, never a divided
// per-invocation estimate. Original measure owns warmup and raw sample timing.
static int batch_repetitions() {
    const char*value=std::getenv("GGML_CALIBRATION_BATCH_REPETITIONS");
    if(!value || !*value)throw std::runtime_error("explicit GGML_CALIBRATION_BATCH_REPETITIONS required; no default");
    for(const char*p=value;*p;++p)if(*p<'0'||*p>'9')throw std::runtime_error("GGML_CALIBRATION_BATCH_REPETITIONS must be an integer in [1,1024]");
    const int count=integer(value);
    if(count<1||count>1024)throw std::runtime_error("GGML_CALIBRATION_BATCH_REPETITIONS must be an integer in [1,1024]");
    return count;
}
template<class F>static std::vector<int64_t> measure_batched(const Options&o,F&&f) {
    const int count=batch_repetitions();
    return measure(o,[&](){for(int iteration=0;iteration<count;++iteration)f();});
}
'''

SINGLE_TRANSFERS = r'''
    // Independently time each complete pinned-staging H2D operation. These
    // samples are not obtained by dividing the aggregate three-weight time.
    for(auto name:{"gate_weight","up_weight","down_weight"}){
        times["hybrid/preload_"+std::string(name)+"_h2d"]=measure_batched(o,[&](){transfer.h2d(gd.at(name).data(),gpu.ports.at(name));});
    }
'''


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_exact(text, anchor, replacement, count, label):
    observed = text.count(anchor)
    if observed != count:
        raise ValueError(f"source anchor drift for {label}: expected {count}, observed {observed}")
    return text.replace(anchor, replacement)


def generate(context, helper):
    context, helper = Path(context).resolve(), Path(helper).resolve()
    context_raw, helper_raw = context.read_bytes(), helper.read_bytes()
    source = context_raw.decode("utf-8").replace("\r\n", "\n")
    helper_text = helper_raw.decode("utf-8").replace("\r\n", "\n")
    # Pin the semantic baseline as well as individual edits. LF/CRLF checkout
    # normalization is allowed; any other upstream drift requires an audit.
    for label, text, expected in (("context", source, CONTEXT_SHA256_LF),
                                  ("shared helper", helper_text, HELPER_SHA256_LF)):
        if sha256(text.encode("utf-8")) != expected:
            raise ValueError(f"{label} source identity drift; re-audit anchors before updating the pinned SHA256")
    helper_anchor = "template<class F>static std::vector<int64_t> measure(const Options&o,F&&f) {"
    if helper_text.count(helper_anchor) != 1:
        raise ValueError("shared helper measure anchor drift")
    source = replace_exact(source, '#include "../matched_mechanisms/compute.cpp"',
        f'#include "{helper.as_posix()}"', 1, "absolute shared helper include")
    source = replace_exact(source, "#include <thread>\n", "#include <thread>\n" + WRAPPER,
                           1, "batch wrapper after thread include")
    source = replace_exact(source, "measure(o,", "measure_batched(o,", 20,
                           "nineteen original timing sites plus wrapper")
    # The wrapper alone must call the original unmodified helper. Its own text
    # was intentionally included in the counted replacement above.
    source = replace_exact(source, "return measure_batched(o,[&](){for(int iteration=0;iteration<count;++iteration)f();});",
        "return measure(o,[&](){for(int iteration=0;iteration<count;++iteration)f();});",
        1, "wrapper calls original measure")
    source = replace_exact(source,
        "static TransformOptions transform_options(int argc, char **argv) {\n",
        "static TransformOptions transform_options(int argc, char **argv) {\n    (void)batch_repetitions(); // fail before any allocation when the batch contract is absent\n",
        1, "early explicit environment validation")
    aggregate = '    times["hybrid/preload_weights"]=measure_batched(o,preload);\n'
    source = replace_exact(source, aggregate, aggregate + SINGLE_TRANSFERS, 1,
                           "independent per-weight transfers")
    source = replace_exact(source,
        r'\"schema\":\"execution-transform-native-0.1\",\"family\":\"',
        r'\"schema\":\"execution-transform-native-batch-0.1\",\"batch_repetitions\":"<<batch_repetitions()<<",\"timing_unit\":\"completed_work_batch_total\",\"family\":\"',
        1, "distinct raw batch schema and unit")
    source = replace_exact(source, r'\"wall_ns\":{', r'\"batch_wall_ns\":{',
                           1, "raw batch totals key")
    source = replace_exact(source, '"execution-transform-probe: "',
                           '"execution-transform-batch-probe: "', 1, "error identity")
    header = "// Generated by execution_transforms_batch/generate_probe.py. Do not edit.\n"
    generated = (header + source).encode("utf-8")
    manifest = dict(schema="execution-transform-batch-source-0.1",
        generator_sha256=sha256(Path(__file__).read_bytes()),
        context_source=str(context), context_sha256=sha256(context_raw),
        context_canonical_lf_sha256=CONTEXT_SHA256_LF,
        helper_source=str(helper), helper_sha256=sha256(helper_raw),
        helper_canonical_lf_sha256=HELPER_SHA256_LF,
        generated_sha256=sha256(generated), generated_bytes=len(generated),
        original_transform_timing_sites=19,
        new_per_weight_metrics=[f"hybrid/preload_{name}_weight_h2d" for name in ("gate", "up", "down")],
        schema_output="execution-transform-native-batch-0.1",
        timing_unit="completed_work_batch_total", raw_duration_key="batch_wall_ns",
        batch_repetitions_environment="GGML_CALIBRATION_BATCH_REPETITIONS",
        batch_repetitions_range=[1, 1024], raw_times_divided=False,
        graph_reference_precision_or_tolerance_changed=False,
        historical_sources_modified=False, hardware_measurement_performed=False)
    return generated, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=HERE.parent / "execution_transforms_dual/compute_context.cpp")
    parser.add_argument("--helper", type=Path, default=HERE.parent / "matched_mechanisms/compute.cpp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    protected = {args.context.resolve(), args.helper.resolve(), Path(__file__).resolve()}
    if args.output.resolve() == args.manifest.resolve() or any(
            path.resolve() in protected for path in (args.output, args.manifest)):
        parser.error("generated outputs must be distinct and must not overwrite any source")
    generated, manifest = generate(args.context, args.helper)
    for destination in (args.output, args.manifest):
        destination.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(generated)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8", newline="\n")
    print(json.dumps(dict(generated_source=str(args.output.resolve()),
                          generated_sha256=manifest["generated_sha256"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
