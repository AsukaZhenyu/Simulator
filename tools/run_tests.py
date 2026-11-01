# -*- coding: utf-8 -*-
"""Run every suite that CI runs, from one command.

The suite table below is the single source of truth: the regression job in
.github/workflows/ci.yml calls this script, and docs/REPRODUCTION.md tells a
human to run the same thing. A suite is a directory, an optional file pattern
and an optional top-level directory; PYTHONPATH is set for every suite so that
the checkout works with or without an editable install.

No suite requires a GPU, CUDA, model weights or network access. Suites that
would need them live under profiling/results and are not run here.

Usage:
    python tools/run_tests.py                # everything
    python tools/run_tests.py --list         # show the suites
    python tools/run_tests.py -k mapping     # only suites whose name matches
    python tools/run_tests.py --keep-going   # do not stop at the first failure
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (name, start directory, pattern, top-level directory)
SUITES: list[tuple[str, str, str, str | None]] = [
    ("public-library", "tests", "test_library.py", None),
    ("analytical-inputs", "tests", "test_analytical_inputs.py", None),
    ("dense-subbatches", "tests", "test_dense_subbatches.py", None),
    ("ir-region-acceptance", "tests", "test_ir_region_acceptance.py", None),
    ("request-flow", "tests", "test_request_flow.py", None),
    ("serving-acceptance", "tests", "test_serving_acceptance.py", None),
    ("strategy-fidelity", "tests", "test_strategy_fidelity.py", None),
    ("policy-no-progress", "tests", "test_policy_no_progress.py", None),
    ("hardware-sensitivity", "tests", "test_hardware_sensitivity.py", None),
    ("performance-boundaries", "tests", "test_performance_boundaries.py", None),
    ("execution-calibration", "profiling/execution_calibration/tests", "test*.py", None),
    ("batched-calibration", "profiling/batched_calibration/tests", "test*.py", None),
    ("composition-calibration", "profiling/composition_calibration/tests", "test*.py", None),
    ("accuracy-suite", "profiling/accuracy_suite/tests", "test*.py", None),
    ("chain-calibration", "profiling/chain_calibration/tests", "test*.py", None),
    ("profiling", "profiling/tests", "test*.py", None),
    ("workload-accuracy", "profiling/workload_accuracy/tests", "test*.py", None),
    ("overlap-accuracy", "profiling/overlap_accuracy/tests", "test*.py", None),
    ("native-execution-transforms", "profiling/native/execution_transforms_batch", "test_*.py", None),
    ("state-calibration", "profiling/state_calibration/tests", "test*.py", None),
    ("native-state-calibration", "profiling/native/state_calibration", "test_*.py", None),
    ("resource-calibration", "profiling/resource_calibration/tests", "test*.py", None),
    ("parameter-calibration", "profiling/parameter_calibration/tests", "test*.py", None),
    ("available-service", "profiling/available_service/tests", "test*.py", None),
    ("service-response", "profiling/service_response/tests", "test*.py", None),
    ("state-response", "profiling/state_response/tests", "test*.py", None),
    ("native-state-response", "profiling/native/state_response", "test_build.py", None),
    ("single-layer-calibration", "profiling/single_layer_calibration/tests", "test*.py", None),
    ("single-layer-context-pair", "profiling/single_layer_context_pair/tests", "test*.py", None),
    ("single-layer-wait-diagnosis", "profiling/single_layer_wait_diagnosis/tests", "test*.py", None),
    ("native-single-layer", "profiling/native/single_layer_calibration", "test_build.py", None),
    ("native-context-pair", "profiling/native/single_layer_context_pair", "test_build.py", None),
    ("native-wait-diagnosis", "profiling/native/single_layer_wait_diagnosis", "test_build.py", None),
    ("tools", "tools/tests", "test*.py", None),
    ("modeling", "modeling/tests", "test*.py", None),
    ("mapping", "mapping/tests", "test*.py", "mapping"),
]


def build_command(start: str, pattern: str, top: str | None) -> list[str]:
    command = [sys.executable, "-m", "unittest", "discover", "-s", start, "-p", pattern]
    if top is not None:
        command += ["-t", top]
    return command


def build_env() -> dict[str, str]:
    """Source roots first, so the runner works from a checkout without an install.

    A freshly created virtual environment that never ran
    ``pip install -e ".[profiling]"`` has no ``llm_simulator``,
    ``tensor_mapping``, ``llm_infer_model`` or ``llm_infer_profile``; pointing
    PYTHONPATH at the four source directories makes every suite behave the same
    installed or not, which is also what `docs/REPRODUCTION.md` tells a human to
    reproduce by hand.
    """
    env = dict(os.environ)
    parts = [str(ROOT / name) for name in ("src", "mapping", "modeling", "profiling")]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    # Tests write nothing outside the source tree; keep bytecode out of it.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the suite names and exit")
    parser.add_argument("-k", "--filter", default="", help="run only suites whose name contains this text")
    parser.add_argument("--keep-going", action="store_true", help="run every suite even after a failure")
    parser.add_argument("-q", "--quiet", action="store_true", help="one line per suite")
    args = parser.parse_args(argv)

    suites = [s for s in SUITES if args.filter in s[0]]
    if not suites:
        print("no suite matches %r" % args.filter, file=sys.stderr)
        return 2
    if args.list:
        for name, start, pattern, top in suites:
            print("%-30s %s  %s%s" % (name, start, pattern, "  -t " + top if top else ""))
        return 0

    failures: list[tuple[str, float]] = []
    started_all = time.time()
    for name, start, pattern, top in suites:
        if not (ROOT / start).is_dir():
            print("SKIP %-30s missing directory %s" % (name, start))
            failures.append((name, 0.0))
            continue
        command = build_command(start, pattern, top)
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, env=build_env(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        output = completed.stdout.decode("utf-8", "replace")
        ok = completed.returncode == 0
        if not ok:
            failures.append((name, elapsed))
        if ok and args.quiet:
            print("PASS %-30s %6.1fs" % (name, elapsed))
        else:
            print("%s %-30s %6.1fs" % ("PASS" if ok else "FAIL", name, elapsed))
            if output.strip():
                print(output.rstrip())
        if not ok and not args.keep_going:
            print("\nstopping at the first failure (use --keep-going to continue)")
            break

    total = time.time() - started_all
    print("\n%d/%d suites passed in %.1fs" % (len(suites) - len(failures), len(suites), total))
    if failures:
        print("failed: " + ", ".join(name for name, _ in failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
