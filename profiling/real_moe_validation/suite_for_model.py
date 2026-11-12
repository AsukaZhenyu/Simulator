"""The tokenizer pass and frozen request suite for one pinned model.

`published/request-suite-v3.json` answers the six domain/length windows with
OLMoE's tokenizer. The windows are part of the workload, so a second model has
to answer them with its own tokenizer: this module runs the same pinned probe in
vocab-only mode over candidate prompts, keeps only its actual token IDs, and
lets `dataset.select` choose the eighteen requests from them.

Nothing here reads a timing, a price, a prediction or a held-out result. The
model is the GGUF the pinned manifest names, verified by byte count and SHA-256,
and its declared contract is the pinned geometry -- so the probe checks the
tokenizer that ran against the file the campaign will price.

    python -m profiling.real_moe_validation.suite_for_model \\
        --model ~/work/models/Qwen3-0.6B-Q4_K_M.gguf --model-id qwen3-0.6b \\
        --build build/runtime-linux4 --candidates qwen-candidates.json \\
        --out build/qwen3-0.6b-suite --runtime-dir build/runtime-linux4/bin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiling import host
from profiling.state_calibration import run as common_run

from . import dataset
from .prepare_portable import contract_of, pinned_model, verify_model
from .run import child_environment
from .runtime_window_io import require, sha

HERE = Path(__file__).resolve().parent
PROBE_STEM = "real-moe-runtime-window-probe"
# The published cohort's mask: the probe narrows itself to these CPUs and the
# tokenizer pass runs under the same guard as the measurement it prepares for.
DEFAULT_MASK = 4294967295


def tokenize_request(model: dict, contract: dict, source_sha256: str, rows: list,
                     threads: int = 16, context_size: int = 2048, batch_size: int = 256,
                     mask: int = DEFAULT_MASK) -> dict:
    """One vocab-only request over the candidate prompts, in the probe's schema."""
    require(isinstance(source_sha256, str) and len(source_sha256) == 64,
            "the probe build's native source SHA is required")
    require(1 <= threads <= 32 and 128 <= context_size <= 16384 and 1 <= batch_size <= 4096,
            "explicit context envelope exceeded")
    return dict(schema="real-moe-runtime-window-request-1", source_sha256=source_sha256,
                mode="tokenize-only",
                model_contract=contract, tokenization_add_special=False,
                threadpool_policy="metadata_only_no_compute",
                model_path=model["path"], model_sha256=model["sha256"], model_bytes=model["bytes"],
                placement="cpu", threads=threads, context_size=context_size, batch_size=batch_size,
                expected_process_mask=mask, warmup_replays=0, minimum_free_host_bytes=0,
                minimum_free_gpu_bytes=0, cuda_graph_policy="runtime_default_enabled",
                tf32_override="0", runtime_window_observer=False,
                requests=[dict(r, max_output_tokens=8) for r in rows])


def build(model_path: Path, model_id: str, build_dir: Path, candidates: Path, out: Path,
          runtime_dirs, threads: int = 16, mask: int = DEFAULT_MASK,
          timeout: int = 1800) -> dict:
    out = Path(out).resolve()
    require(not out.exists(), "a new suite root is required; never overwrite tokenizer evidence")
    build_dir = Path(build_dir).resolve()
    prepared = json.loads((build_dir / "PREPARED.json").read_text(encoding="utf-8"))
    binary = build_dir / "bin" / host.executable_name(PROBE_STEM)
    require(binary.is_file(), f"the probe is not built at {binary}")
    directories = [Path(d).resolve() for d in runtime_dirs]
    for directory in directories:
        require(directory.is_dir(), f"runtime directory missing: {directory}")

    pinned = pinned_model(model_id)
    model = verify_model(Path(model_path), pinned)
    contract = contract_of(pinned)
    data = dataset.read(candidates)
    rows = data.get("requests")
    require(isinstance(rows, list) and rows, "candidate prompt inventory missing")

    out.mkdir(parents=True)
    req = tokenize_request(model, contract, prepared["native_source_sha256"], rows,
                           threads=threads, mask=mask)
    request_path = out / "tokenize-request.json"
    dataset.write(request_path, req)
    result_path = out / "tokenized.json"
    command = [str(binary), str(request_path), str(result_path)]
    env = child_environment(directories, req)
    # The process mask is narrowed before the child starts, exactly as the
    # measurement launcher does it: a host whose allowed CPU set is wider than
    # one mask word (128 CPUs here) cannot report that set, and the probe would
    # then refuse the declared mask for a reason that is about the parent.
    with common_run.measurement_lock():
        process, stdout, stderr, affinity = common_run.launch(
            command, env, out / "stdout.txt", out / "stderr.txt", mask)
        try:
            returncode = process.wait(timeout=timeout)
        finally:
            stdout.close()
            stderr.close()
    require(returncode == 0, "the vocab-only tokenizer pass failed; nothing was substituted")

    native = dataset.read(result_path)
    require(native.get("schema") == "real-moe-runtime-window-native-1"
            and native.get("mode") == "tokenize-only",
            "the native answer is not a vocab-only tokenizer pass")
    require(native["source_sha256"] == req["source_sha256"],
            "the tokenizer pass ran a different native source")
    require(native["model_sha256_verified_by_parent"] == model["sha256"],
            "the tokenizer pass ran a different model file")
    require(native.get("model_contract_verified") is True
            and native.get("observed_model_contract") == contract,
            "the tokenizer pass did not verify the declared model contract")
    require(native.get("model_weights_loaded") is False, "vocab-only mode loaded weights")
    require(len(native["tokenized_requests"]) == len(rows), "tokenizer inventory differs")

    suite_path = out / "request-suite.json"
    suite = dataset.select(result_path, suite_path)
    proof = dataset.audit(suite_path, out / "suite-audit.json")
    receipt = dict(schema="real-moe-model-suite-1", model=model, model_id=model_id,
                   model_contract=contract,
                   probe=dict(path=str(binary), sha256=sha(binary)),
                   prepared=dict(path=str(build_dir / "PREPARED.json"),
                                 sha256=sha(build_dir / "PREPARED.json"),
                                 native_source_sha256=prepared["native_source_sha256"]),
                   candidates=dict(path=str(Path(candidates).resolve()),
                                   sha256=sha(Path(candidates).resolve()),
                                   prompts=len(rows)),
                   request=dict(path=str(request_path), sha256=sha(request_path)),
                   tokenized=dict(path=str(result_path), sha256=sha(result_path)),
                   suite=dict(path=str(suite_path), sha256=sha(suite_path),
                              suite_sha256=suite["suite_sha256"]),
                   audit=dict(path=str(out / "suite-audit.json"), sha256=sha(out / "suite-audit.json"),
                              passed=proof["passed"]),
                   runtime_dirs=[str(d) for d in directories], command=command,
                   returncode=returncode, expected_process_mask=mask,
                   observed_process_affinity=affinity, tokenizer_pass_used_prompt_lengths=True,
                   performance_measured=False, prior_result_read=False)
    dataset.write(out / "suite-receipt.json", receipt)
    return dict(out=str(out), model_id=model_id, model_contract=contract,
                requests=proof["requests"], suite_sha256=suite["suite_sha256"],
                audit=proof["passed"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--mask", type=int, default=DEFAULT_MASK)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.model, args.model_id, args.build, args.candidates, args.out,
                           args.runtime_dir, threads=args.threads, mask=args.mask,
                           timeout=args.timeout), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
