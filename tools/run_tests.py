# -*- coding: utf-8 -*-
"""Run every suite that CI runs, from one command.

The suite table below is the single source of truth: the regression job in
.github/workflows/ci.yml calls this script, and docs/REPRODUCTION.md tells a
human to run the same thing. A suite is a directory, an optional file pattern
and an optional top-level directory; PYTHONPATH is set for every suite so that
the checkout works with or without an editable install.

No suite requires a GPU, CUDA, model weights or network access. The
measurement apparatus that would need them is kept with the paper, not in this
repository.

Usage:
    python tools/run_tests.py                # everything
    python tools/run_tests.py --list         # show the suites
    python tools/run_tests.py -k mapping     # only suites whose name matches
    python tools/run_tests.py --keep-going   # do not stop at the first failure
"""
from __future__ import annotations

import argparse
import os
import platform
import re
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
    ("request-flow", "tests", "test_request_flow.py", None),
    ("strategy-fidelity", "tests", "test_strategy_fidelity.py", None),
    ("strategy-ir", "tests", "test_strategy_ir.py", None),
    ("policy-no-progress", "tests", "test_policy_no_progress.py", None),
    ("shared-memory-facade", "tests", "test_shared_memory_facade.py", None),
    ("expert-recipes", "tests", "test_expert_recipes.py", None),
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


def _git(*arguments: str) -> str:
    try:
        done = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def repository_state() -> dict[str, str]:
    return {
        "remote": _git("config", "--get", "remote.origin.url"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": "有未提交改动" if _git("status", "--porcelain") else "干净",
    }


def dependency_versions() -> str:
    from importlib import metadata

    wanted = ("numpy", "scipy", "jinja2", "PyYAML", "gguf")
    parts = []
    for name in wanted:
        try:
            parts.append("%s %s" % (name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            parts.append("%s 未安装" % name)
    return " · ".join(parts)


def machine_facts() -> dict[str, str]:
    facts = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cores": str(os.cpu_count() or "?"),
        "processor": platform.processor() or "未知",
        "gpu": "未检测到 nvidia-smi",
    }
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        if done.returncode == 0 and done.stdout.strip():
            facts["gpu"] = "；".join(line.strip() for line in done.stdout.strip().splitlines())
    except (OSError, subprocess.SubprocessError):
        pass
    return facts


def parse_counts(output: str) -> dict[str, int | float | None]:
    counts: dict[str, int | float | None] = {"tests": None, "seconds": None, "skipped": None}
    match = re.search(r"^Ran (\d+) tests? in ([\d.]+)s", output, re.MULTILINE)
    if match:
        counts["tests"] = int(match.group(1))
        counts["seconds"] = float(match.group(2))
    skipped = re.search(r"skipped=(\d+)", output)
    if skipped:
        counts["skipped"] = int(skipped.group(1))
    return counts


def fence(text: str) -> tuple[str, str]:
    """Pick a backtick fence longer than any run inside the block."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    marker = "`" * max(3, longest + 1)
    return marker, marker


def write_report(path: Path, *, command_line: str,
                 results: list[dict], total_seconds: float) -> None:
    passed = sum(1 for r in results if r["status"] == "PASS")
    verdict = "通过" if passed == len(results) else "未通过"
    facts = machine_facts()
    repository = repository_state()

    lines = [
        "# 回归报告 / Regression report",
        "",
        "- 生成时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "- 命令：`%s`" % command_line,
        "- 结论：**%s**，%d/%d 套件通过，总耗时 %.1f s" % (verdict, passed, len(results), total_seconds),
        "- 仓库：%s @ `%s`（%s）" % (repository["remote"] or "未知远端", repository["commit"] or "无提交",
                                    repository["dirty"]),
        "- 主机：%s；%s；逻辑核 %s" % (facts["platform"], facts["processor"], facts["cores"]),
        "- GPU：%s" % facts["gpu"],
        "- Python：%s（`%s`）" % (platform.python_version(), sys.executable),
        "- 依赖：%s" % dependency_versions(),
        "",
        "## 逐套件 / Suites",
        "",
        "| # | 套件 | 目录 | 结果 | 耗时 | 用例 | 跳过 |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, result in enumerate(results, 1):
        counts = result["counts"]
        lines.append("| %d | %s | `%s` | %s | %.1f s | %s | %s |" % (
            index, result["name"], result["start"], result["status"], result["elapsed"],
            counts["tests"] if counts["tests"] is not None else "–",
            counts["skipped"] if counts["skipped"] is not None else "–"))

    failed = [r for r in results if r["status"] != "PASS"]
    lines += ["", "## 失败详情 / Failures", ""]
    if not failed:
        lines += ["无。", ""]
    for result in failed:
        lines += [
            "### %s" % result["name"],
            "",
            "- 目录：`%s`" % result["start"],
            "- 命令：`%s`" % result["command"],
            "- 退出码：`%s`" % result["returncode"],
            "",
        ]
        marker, closing = fence(result["output"] or "(无输出)")
        lines += [marker + "text", (result["output"] or "(无输出)").rstrip(), closing, ""]

    lines += [
        "## 复现",
        "",
        "```bash",
        "git clone %s" % (repository["remote"] or "<仓库地址>"),
        "cd Simulator && git checkout %s" % (repository["commit"] or "<commit>"),
        "python -m venv .venv && .venv\\Scripts\\activate    # POSIX: source .venv/bin/activate",
        'python -m pip install -e ".[profiling]"',
        command_line,
        "```",
        "",
        "把整份文件原样发回即可，不需要摘录或总结。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the suite names and exit")
    parser.add_argument("-k", "--filter", default="", help="run only suites whose name contains this text")
    parser.add_argument("--keep-going", action="store_true", help="run every suite even after a failure")
    parser.add_argument("-q", "--quiet", action="store_true", help="one line per suite")
    parser.add_argument("--report", metavar="PATH",
                        help="write a Markdown report (environment, per-suite result, full failure output)")
    args = parser.parse_args(argv)

    suites = [s for s in SUITES if args.filter in s[0]]
    if not suites:
        print("no suite matches %r" % args.filter, file=sys.stderr)
        return 2
    if args.list:
        for name, start, pattern, top in suites:
            print("%-30s %s  %s%s" % (name, start, pattern, "  -t " + top if top else ""))
        return 0

    results: list[dict] = []
    started_all = time.time()
    interrupted = False
    for name, start, pattern, top in suites:
        if not (ROOT / start).is_dir():
            print("SKIP %-30s missing directory %s" % (name, start))
            results.append(dict(name=name, start=start, status="SKIP", elapsed=0.0, command="",
                                returncode="", output="目录不存在：%s" % start,
                                counts=dict(tests=None, seconds=None, skipped=None)))
            continue
        command = build_command(start, pattern, top)
        started = time.time()
        completed = subprocess.run(command, cwd=ROOT, env=build_env(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        output = completed.stdout.decode("utf-8", "replace")
        ok = completed.returncode == 0
        results.append(dict(name=name, start=start, status="PASS" if ok else "FAIL", elapsed=elapsed,
                            command=" ".join(command), returncode=str(completed.returncode),
                            output=output, counts=parse_counts(output)))
        if ok and args.quiet:
            print("PASS %-30s %6.1fs" % (name, elapsed))
        else:
            print("%s %-30s %6.1fs" % ("PASS" if ok else "FAIL", name, elapsed))
            if output.strip():
                print(output.rstrip())
        if not ok and not args.keep_going:
            print("\nstopping at the first failure (use --keep-going to continue)")
            interrupted = True
            break

    total = time.time() - started_all
    passed = sum(1 for r in results if r["status"] == "PASS")
    print("\n%d/%d suites passed in %.1fs" % (passed, len(results), total))
    failed = [r["name"] for r in results if r["status"] != "PASS"]
    if failed:
        print("failed: " + ", ".join(failed))

    if args.report:
        report = Path(args.report)
        if not report.is_absolute():
            report = ROOT / report
        command_line = "python tools/run_tests.py " + " ".join(argv if argv is not None else sys.argv[1:])
        write_report(report, command_line=command_line.strip(),
                     results=results, total_seconds=total)
        print("report: %s" % report)

    return 1 if failed or interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
