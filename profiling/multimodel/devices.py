# -*- coding: utf-8 -*-
"""Describe the machine a campaign ran on, using only what can be observed here.

A device branch carries this file's output.  It deliberately contains structural
facts only -- cores, memory, interconnect, software identity -- and never a
measured price: prices belong to the run that produced them, and copying one
machine's numbers into another machine's report is the mistake R3 exists to
prevent.

Every field is either read from the OS or written as ``null``.  Nothing is
inferred: a machine without an NVIDIA driver reports ``"gpu": []`` and a null
driver field, not a guess.

    python profiling/multimodel/devices.py            # print the descriptor
    python profiling/multimodel/devices.py --out p.json
    python profiling/multimodel/devices.py --check p.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "multimodel-device-1"
REQUIRED = ("schema", "device_id", "collected_utc", "os", "machine_architecture",
            "logical_processors", "memory_bytes", "gpu", "unknown_fields_are_null_not_inferred")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _run(*command: str) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return finished.stdout.strip() if finished.returncode == 0 else None


def cpu_model() -> str | None:
    if sys.platform == "win32":
        return os.environ.get("PROCESSOR_IDENTIFIER") or _run("powershell", "-NoProfile", "-Command",
                                                             "(Get-CimInstance Win32_Processor).Name") or None
    if sys.platform == "darwin":
        return _run("sysctl", "-n", "machdep.cpu.brand_string")
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def physical_cores() -> int | None:
    if sys.platform == "win32":
        raw = _run("powershell", "-NoProfile", "-Command",
                   "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
    elif sys.platform == "darwin":
        raw = _run("sysctl", "-n", "hw.physicalcpu")
    else:
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        pairs = {(m.group(1), m.group(2)) for m in
                 re.finditer(r"^physical id\s*:\s*(\d+)$[\s\S]*?^core id\s*:\s*(\d+)$", text, re.MULTILINE)}
        return len(pairs) or None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value or None


def memory_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total)
        return None
    if sys.platform == "darwin":
        raw = _run("sysctl", "-n", "hw.memsize")
        return int(raw) if raw and raw.isdigit() else None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def gpus() -> list[dict]:
    """NVIDIA only, and only when the driver answers.  An absent answer is []."""
    raw = _run("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
               "--format=csv,noheader,nounits")
    if not raw:
        return []
    found = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        found.append({"vendor": "nvidia", "name": parts[0],
                      "memory_bytes": int(float(parts[1]) * 1024 ** 2) if parts[1] else None,
                      "driver": parts[2] or None})
    return found


def topology_note() -> dict:
    """What a scheduler would need to pin threads.  Not a measurement."""
    return {"platform_processor": platform.processor() or None,
            "python": platform.python_version(),
            "hostname": socket.gethostname(),
            "cpu_count_reported_by_python": os.cpu_count()}


def survey() -> dict:
    host = socket.gethostname()
    system = platform.system() or "unknown"
    return {
        "schema": SCHEMA,
        "device_id": slug(f"{host}-{system}") or "unnamed",
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "machine_architecture": platform.machine() or None,
        "processor_identifier": cpu_model(),
        "physical_cores": physical_cores(),
        "logical_processors": os.cpu_count(),
        "memory_bytes": memory_bytes(),
        "gpu": gpus(),
        "topology": topology_note(),
        "unknown_fields_are_null_not_inferred": True,
    }


def check(document: dict) -> list[str]:
    problems = [f"missing {field}" for field in REQUIRED if field not in document]
    if document.get("schema") != SCHEMA:
        problems.append(f"schema is {document.get('schema')!r}, expected {SCHEMA!r}")
    if document.get("unknown_fields_are_null_not_inferred") is not True:
        problems.append("descriptor does not promise unobserved fields stay null")
    for key in ("memory_bytes", "logical_processors", "physical_cores"):
        value = document.get(key)
        if value is not None and not isinstance(value, int):
            problems.append(f"{key} is not an integer or null")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args(argv)

    if args.check:
        document = json.loads(args.check.read_text(encoding="utf-8"))
        problems = check(document)
        for problem in problems:
            print(problem)
        print(f"{args.check}: {'ok' if not problems else 'invalid'}")
        return 1 if problems else 0

    document = survey()
    text = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if args.out:
        # Never overwrite a descriptor that a previous run already recorded.
        with args.out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
