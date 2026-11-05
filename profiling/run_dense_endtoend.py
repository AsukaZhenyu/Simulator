"""Measure the declared shapes with llama-bench and record the protocol.

This tool only measures.  What a token step is *predicted* to cost is assembled
in ``composing/`` (``python -m composing predict``), and the comparison between
the two is written there too (``python -m composing compare``).  The split is
deliberate: a file that both predicts and times cannot show that the prediction
came first.

``measure`` reads ``predicted.json`` from the output directory and refuses to run
without it.  That is the pre-registration rule: the prediction is fixed before
the clock, and a row measured against another prediction is refused rather than
quietly mixed in.

The prediction is one document with one sha256, and it can be measured on any
device: ``--models-dir`` re-resolves each model by file name, because the path
``predict`` recorded is where the weights sat on the machine that made the
prediction.  Nothing in ``predicted.json`` is rewritten for that -- a prediction
edited to fit a machine would not be the one that was registered.

The declared shapes themselves come from ``composing.spec`` -- a wall clock and
a prediction have to be numbers for the same question, so the vocabulary is
defined once, on the composition side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from composing.spec import SHAPES  # noqa: E402

# What llama-bench echoes back in its JSON row, and therefore what can be
# *checked* instead of trusted.  Each entry is a way the wall clock stops
# meaning what the cost table means:
#
#   flash_attn  a fused attention kernel is a different implementation, and on
#               this machine it moves the decode slope from 20.52 to 36.09
#               us/KV token (1.76x).  The table prices the materialized graph,
#               so the wall clock must run the same one -- but llama-bench's
#               default is a per-build choice, not a constant.
#   type_k/v    the declared block prices an f16 cache (dense_contract.KV_DTYPE).
#               A quantized cache is a different byte count, so the pair would
#               be two answers to two questions.
ECHOED_PROTOCOL = {"flash_attn": False, "type_k": "f16", "type_v": "f16"}
# Flags that change the wall clock but never appear in llama-bench's JSON at
# all.  Pinned so the command is reproducible, and copied into measured.json as
# a declaration: llama.cpp cannot prove them from its output, so the file has to
# say they were requested.
UNECHOED_FLAGS = (("-mmp", "1"), ("-b", "2048"), ("-ub", "512"), ("--poll", "50"))
# What a grouped decode invocation throws away and a per-shape one does not.  One
# process's first decode costs extra (OLMoE 2026-11-05: +13.2 / +10.8 / +4.4 ms
# over the mean of the rest, three runs, same sign); every later shape started
# inside the ordinary spread, so only the first shape loses a sample.
WARMUP_GROUPED = "first sample of the first shape in the invocation"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def _require_prediction(output: Path) -> dict:
    path = output / "predicted.json"
    if not path.is_file():
        raise ValueError("predicted.json must exist before any wall clock is measured")
    return {"path": str(path), "sha256": _sha(path),
            "mtime_ns": path.stat().st_mtime_ns, "content": json.loads(path.read_text(encoding="utf-8"))}


def resolve_model(entry: dict, models_dir: Path | None = None) -> Path:
    """The weights this prediction names, as they exist on *this* machine.

    ``predict`` records the absolute path it resolved, which is a record of what
    was priced and not an input another device can rely on.  ``--models-dir``
    re-resolves the same file name under a directory that exists here; the
    document, and therefore its sha256, stays the one that was registered before
    the clock.  Without the flag the recorded path is used as-is, and a missing
    file is named rather than handed to llama-bench to fail on.
    """
    recorded = Path(entry["gguf"])
    if models_dir is None:
        if not recorded.is_file():
            raise ValueError(
                f"the prediction names {recorded}, which is not on this machine; pass "
                f"--models-dir pointing at a directory that holds {recorded.name}")
        return recorded
    resolved = (Path(models_dir) / recorded.name).resolve()
    if not resolved.is_file():
        raise ValueError(f"--models-dir {models_dir} holds no {recorded.name}")
    return resolved


def expected_row(config: str) -> tuple[int, int, int]:
    """The (n_prompt, n_gen, n_depth) triple llama-bench must report for a shape."""
    phase, tokens, history = SHAPES[config]
    return (tokens, 0, 0) if phase == "prefill" else (0, tokens, history)


def _invocation_targets(configs) -> tuple[tuple[str, ...], str, int, list[int]]:
    """The (configs, phase, tokens, histories) one llama-bench invocation asks for.

    llama-bench accepts a comma list for ``-d``, so several decode shapes can be
    timed inside one process.  That matters for the decode slope: four separate
    invocations give four independent process states, and the state band
    (12-18% between runs of one command) then enters the intercept and the
    slope.  One invocation gives the four points one state, which is the same
    argument R13 makes for the CPU mask.
    """
    if isinstance(configs, str):
        configs = (configs,)
    configs = tuple(configs)
    if not configs:
        raise ValueError("one invocation must ask for at least one shape")
    if len(set(configs)) != len(configs):
        raise ValueError(f"a shape was asked for twice in one invocation: {list(configs)}")
    targets = {SHAPES[name][:2] for name in configs}
    if len(targets) != 1:
        raise ValueError(f"one invocation cannot ask for several phases or token counts: "
                         f"{sorted(targets)}")
    phase, tokens = targets.pop()
    histories = [SHAPES[name][2] for name in configs]
    if len(set(histories)) != len(histories):
        raise ValueError(f"two shapes ask for the same depth: {list(configs)}")
    return configs, phase, tokens, histories


def bench_command(binary: Path, gguf: Path, configs, threads: int, cpu_mask: list[int],
                  repetitions: int) -> list[str]:
    """The command that measures the declared shapes on the CPUs of record.

    ``configs`` is one shape name or a tuple of them; several are only allowed
    when they share a phase and a token count and differ in depth, which is what
    a comma list on ``-d`` can express.

    Three flags are load bearing, and each was missing from an earlier attempt:

    ``-nopo 1`` disables op offload. llama.cpp offloads host-tensor operations to
    the CUDA device by default even at ``-ngl 0``, so without it the prompt GEMMs
    run on the GPU: prefill came back ~25x faster than the same table's 2-thread
    price and did not scale with ``-t`` at all.

    ``-d`` sets the decode history. llama-bench's generation test always reports
    ``n_prompt=0``; giving it ``-p <history>`` only appends an unused prefill test
    at depth 0, which is why every decode shape once came back at the same time.

    ``-fa`` picks the attention implementation.  The first two were caught by
    their effect on the numbers; this one leaves the numbers plausible while
    moving the decode slope by 1.76x, so it is pinned explicitly and checked
    against the echoed row (:data:`ECHOED_PROTOCOL`) rather than inherited from
    whatever the build happens to default to.  The KV element type is pinned for
    the same reason.
    """
    configs, phase, tokens, histories = _invocation_targets(configs)
    prompt, generate = (tokens, 0) if phase == "prefill" else (0, tokens)
    mask = 0
    for cpu in cpu_mask:
        mask |= 1 << cpu
    command = [str(binary), "-m", str(gguf), "-p", str(prompt), "-n", str(generate),
               "-d", ",".join(str(history) for history in histories),
               "-r", str(repetitions), "-ngl", "0", "-t", str(threads),
               "-C", hex(mask), "--cpu-strict", "1", "-nopo", "1",
               "-fa", "0", "-ctk", "f16", "-ctv", "f16"]
    for flag, value in UNECHOED_FLAGS:
        command += [flag, value]
    return command + ["-o", "json"]


def _bench_row(config: str, row: dict, threads: int, cpu_mask: list[int], repetitions: int,
               discard_warmup: bool = False) -> dict:
    """One llama-bench row, checked against the protocol it claims to have run.

    ``repetitions`` is what the invocation asked for with ``-r``: a row with a
    different number of samples is not the run the command describes.

    ``discard_warmup`` drops the first sample.  In a grouped decode invocation
    the first shape timed is the first shape this process ever decodes, and that
    first decode costs extra: measured on OLMoE at 128 tokens the first sample
    came back 62.25 / 58.89 / +13.2 / +10.8 ms above the mean of the rest, three
    runs, same sign, while the first sample of a later shape in the same process
    was +0.5 / -0.1 / -4.5 / +4.6 / +3.9 / -2.9, i.e. ordinary spread.  The
    discarded value stays on the record, and the mean is recomputed from the
    samples that are kept, so ``avg_ns`` and ``samples_ns`` never disagree.
    """
    mask = 0
    for cpu in cpu_mask:
        mask |= 1 << cpu
    protocol = {"n_threads": threads, "cpu_mask": hex(mask), "cpu_strict": True,
                "no_op_offload": True, "n_gpu_layers": 0, **ECHOED_PROTOCOL}
    for key, value in protocol.items():
        if row.get(key) != value:
            raise ValueError(f"{config}: llama-bench row reports {key}={row.get(key)!r}, "
                             f"not the requested {value!r}")
    samples = row.get("samples_ns")
    if not isinstance(samples, list) or not samples or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in samples):
        raise ValueError(f"{config}: llama-bench did not report positive integral samples")
    if len(samples) != repetitions:
        raise ValueError(f"{config}: llama-bench reported {len(samples)} samples but the "
                         f"command asked for {repetitions}")
    record = {"avg_ns": row["avg_ns"], "samples_ns": list(samples),
              "n_prompt": row.get("n_prompt"),
              "n_gen": row.get("n_gen"), "n_depth": row.get("n_depth")}
    if discard_warmup:
        kept = list(samples[1:])
        if not kept:
            raise ValueError(f"{config}: discarding the warm-up sample leaves nothing measured; "
                             f"ask for at least two repetitions")
        record["discarded_warmup_ns"] = samples[0]
        record["reported_avg_ns"] = row["avg_ns"]     # what llama-bench itself averaged
        record["samples_ns"] = kept
        record["avg_ns"] = round(sum(kept) / len(kept))
    record["llama_bench"] = {key: row.get(key) for key in
                             ("build_commit", "model_filename", "n_threads", "cpu_mask", "cpu_strict",
                              "no_op_offload", "n_gpu_layers", "n_depth", "type_k", "type_v",
                              "flash_attn", "test_time")}
    return record


def parse_bench_rows(stdout: str, configs, threads: int, cpu_mask: list[int],
                     repetitions: int, discard_warmup_on: str | None = None) -> dict[str, dict]:
    """Every declared shape asked for by one invocation, keyed by shape name.

    The triple ``(n_prompt, n_gen, n_depth)`` has to match exactly, one row per
    shape: a row at another depth is another measurement, and a duplicate means
    llama-bench ran the shape twice under one label.

    ``discard_warmup_on`` names at most one shape whose first sample is a warm-up
    rather than a measurement (see :func:`_bench_row`).  It is the shape timed
    first in a grouped invocation, because that is the only position where the
    extra cost was reproducible.
    """
    if isinstance(configs, str):
        configs = (configs,)
    rows = json.loads(stdout)
    if not isinstance(rows, list) or not rows:
        raise ValueError("llama-bench produced no test rows")
    found = {}
    for config in configs:
        wanted = [row for row in rows
                  if (row.get("n_prompt"), row.get("n_gen"), row.get("n_depth")) == expected_row(config)]
        if len(wanted) != 1:
            raise ValueError(f"{config}: expected exactly one row with "
                             f"(n_prompt, n_gen, n_depth)={expected_row(config)}, found {len(wanted)}")
        found[config] = _bench_row(config, wanted[0], threads, cpu_mask, repetitions,
                                   discard_warmup=(config == discard_warmup_on))
    return found


def parse_bench(stdout: str, config: str, threads: int, cpu_mask: list[int],
                repetitions: int = 1) -> dict:
    """One declared shape from one invocation."""
    return parse_bench_rows(stdout, config, threads, cpu_mask, repetitions)[config]


def _invocations(configurations: dict, group_decode: bool) -> list[tuple[str, ...]]:
    """The llama-bench invocations one model needs, in declared order.

    With ``group_decode`` every decode shape of the model goes to one process,
    which is the only way the four points of a slope can share a machine state.
    Prefill shapes cannot join it: they differ in token count, which ``-d``
    cannot express.  Every shape in one invocation also has to ask for the same
    number of repetitions, because ``-r`` is per invocation.
    """
    if not group_decode:
        return [(config,) for config in configurations]
    decode = tuple(config for config in configurations if SHAPES[config][0] == "decode")
    if len({configurations[config] for config in decode}) > 1:
        raise ValueError("a grouped decode invocation needs one repetition count for all "
                         f"its shapes, got {[(c, configurations[c]) for c in decode]}")
    groups = [(config,) for config in configurations if SHAPES[config][0] != "decode"]
    if decode:
        groups.append(decode)
    return groups


def measure(output: Path, binary: Path, configurations: dict, threads: int, cpu_mask: list[int],
            timeout_s: float, models: list[str] | None = None,
            group_decode: bool = False, models_dir: Path | None = None) -> dict:
    recorded = _require_prediction(output)
    prediction = recorded["content"]
    declared = prediction["models"]
    selected = list(declared) if not models else list(dict.fromkeys(models))
    unknown = [name for name in selected if name not in declared]
    if unknown:
        raise ValueError(f"prediction declares no model named {unknown}; "
                         f"it has {sorted(declared)}")
    invocations = _invocations(configurations, group_decode)
    measured = {"schema_version": "dense_endtoend_measurement_v1", "created_at_utc": _now(),
                "llama_bench": str(binary.resolve()), "llama_bench_sha256": _sha(binary),
                "threads": threads, "cpu_mask": list(cpu_mask),
                "thread_placement": "explicit_strict_cpu_mask",
                "op_offload": False, "decode_history_knob": "llama-bench -d",
                # Which implementation and cache the wall clock ran.  The echoed
                # half is checked against every row; the rest is a declaration,
                # because llama-bench prints neither.
                "bench_protocol": {"echoed_and_checked": dict(ECHOED_PROTOCOL),
                                   "pinned_not_echoed": dict(UNECHOED_FLAGS)},
                # One process per shape gives each decode point its own machine
                # state; one process per model gives all of them the same one.
                # A file whose rows come from two groupings is not one protocol.
                "decode_grouping": "one_process_per_model_decode" if group_decode
                                   else "one_process_per_shape",
                # In a grouped invocation the first shape timed pays the process's
                # first decode; that sample is asked for and thrown away.  The
                # per-shape path keeps its historical -r meaning: every archived
                # campaign was taken with it, and nothing here may silently
                # redefine those rows.
                "warmup": {"discarded": WARMUP_GROUPED if group_decode else None},
                "prediction": {"path": recorded["path"], "sha256": recorded["sha256"],
                               "mtime_ns": recorded["mtime_ns"]},
                # Where this machine found the weights.  A prediction may be
                # measured on several devices; the document names the file, this
                # field names the directory each run read it from.
                "models_directory": (str(Path(models_dir).resolve())
                                     if models_dir is not None else None),
                # Each model row carries its own time because a campaign can span
                # several processes, and measured latency carries a per-process
                # state (see the MoE report §3).  A file whose rows come from
                # different processes must show that instead of looking uniform.
                "measured_at_utc": {}, "models": {}}
    previous_path = output / "measured.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        same_prediction = (previous.get("prediction", {}).get("sha256")
                           == measured["prediction"]["sha256"])
        kept = [name for name in previous.get("models", {}) if name not in selected]
        # A kept row is only comparable to a new one under the protocol that
        # produced it (R13): threads, affinity and the bench binary itself are
        # part of what the wall clock means, so rows from two protocols cannot
        # share one file's header.  A run that replaces every row mixes nothing
        # and is allowed to change the protocol.
        if kept:
            for field, mine in (("threads", threads), ("cpu_mask", list(cpu_mask)),
                                ("llama_bench_sha256", measured["llama_bench_sha256"]),
                                ("bench_protocol", measured["bench_protocol"]),
                                ("decode_grouping", measured["decode_grouping"]),
                                ("warmup", measured["warmup"])):
                theirs = previous.get(field)
                if theirs is not None and theirs != mine:
                    raise ValueError(f"measured.json was taken with {field}={theirs!r} but this "
                                     f"run uses {mine!r}; rows measured under two protocols "
                                     f"cannot share one file")
        if not same_prediction and set(selected) != set(declared):
            raise ValueError("measured.json was taken against another prediction; a partial "
                             "re-measurement cannot be mixed with it")
        if same_prediction:
            # A partial run must not delete a model measured earlier in the campaign.
            measured["models"] = {name: rows for name, rows in previous.get("models", {}).items()
                                  if name in kept}
            measured["measured_at_utc"] = {name: taken for name, taken in
                                           previous.get("measured_at_utc", {}).items()
                                           if name in kept}
    for model_id in selected:
        entry = declared[model_id]
        model_rows = {}
        for group in invocations:
            repetitions = configurations[group[0]]
            # ``-r`` counts llama-bench's recorded samples, and there is one value
            # per invocation, so the warm-up has to be asked for as an extra
            # sample of every shape and dropped afterwards from the first only.
            requested = repetitions + 1 if len(group) > 1 else repetitions
            command = bench_command(binary, resolve_model(entry, models_dir), group, threads,
                                    cpu_mask, requested)
            process = subprocess.run(command, capture_output=True, text=True, errors="replace",
                                     timeout=timeout_s)
            if process.returncode:
                raise ValueError(f"{model_id} {','.join(group)}: llama-bench returned "
                                 f"{process.returncode}: {process.stderr[-800:]}")
            rows = parse_bench_rows(process.stdout, group, threads, cpu_mask, requested,
                                    discard_warmup_on=group[0] if len(group) > 1 else None)
            for config in group:
                model_rows[config] = {"command": command, **rows[config]}
                print(f"[{model_id}] {config}: {model_rows[config]['avg_ns']/1e6:.3f} ms",
                      flush=True)
            if len(group) > 1:
                print(f"[{model_id}] {len(group)} depths measured in one process "
                      f"(one machine state for the slope)", flush=True)
        measured["models"][model_id] = model_rows
        measured["measured_at_utc"][model_id] = _now()
        # Keep completed models on disk: a later failure must not discard hours of runs.
        _dump(output / "measured.json", measured)
    measured["finished_at_utc"] = _now()
    return measured


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    measuring = sub.add_parser("measure")
    measuring.add_argument("--output", type=Path, required=True)
    measuring.add_argument("--llama-bench", type=Path, required=True)
    measuring.add_argument("--config", action="append", required=True,
                           help="name=repetitions, repeatable")
    measuring.add_argument("--threads", type=int, required=True)
    measuring.add_argument("--cpu-mask", required=True, help="comma separated logical CPUs")
    measuring.add_argument("--timeout-s", type=float, default=3600)
    measuring.add_argument("--model", action="append", default=None,
                           help="restrict to these model ids (repeatable); rows already on "
                                "disk for other models are kept, with their own timestamps")
    measuring.add_argument("--group-decode", action="store_true",
                           help="time every decode depth of one model inside a single "
                                "llama-bench process, so the points of the slope share one "
                                "machine state instead of four")
    measuring.add_argument("--models-dir", type=Path, default=None,
                           help="where this machine keeps the weights: each model is taken "
                                "by file name from here instead of from the absolute path "
                                "the prediction recorded on the machine that made it")

    args = parser.parse_args(argv)
    configurations = {}
    for item in args.config:
        name, _, count = item.partition("=")
        if name not in SHAPES:
            parser.error(f"unknown config {name!r}; known: {sorted(SHAPES)}")
        try:
            configurations[name] = int(count or 5)
        except ValueError:
            parser.error("--config needs name=repetitions")
    cpu_mask = [int(value) for value in args.cpu_mask.split(",")]
    measured = measure(args.output, args.llama_bench, configurations, args.threads, cpu_mask,
                       args.timeout_s, args.model, group_decode=args.group_decode,
                       models_dir=args.models_dir)
    _dump(args.output / "measured.json", measured)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"run_dense_endtoend: {error}", file=sys.stderr)
        raise SystemExit(1)
