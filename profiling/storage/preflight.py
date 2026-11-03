"""Before a campaign: is this directory a disk we are allowed to price?

`ssd_measure.py` records the mechanism it actually achieved, so it never lies
about cache bypass. It does *not* check where ``--directory`` lives. On a
laptop that is fine -- the user knows which disk they picked -- but a rented
UMA box (DGX Spark, RTX Spark, or any cloud VM) usually hands out a container
whose working directory is an overlay, a tmpfs, or a network mount. Measuring
there still produces a valid-looking ``ssd_storage_measurements_v1`` document
whose numbers belong to no device, and the two-tier write threshold measured
on an overlay is meaningless.

So this tool asks the two questions the campaign cannot answer later:

* **Is the path on a local block device?**  ``findmnt``/``df`` on POSIX,
  ``GetDriveTypeW``/``GetVolumeInformationW`` on Windows. Virtual filesystems
  are refused, not warned about: a wrong-looking number is worse than no number.
* **What bypass will the campaign actually get?**  It runs the same
  ``open_backends`` selection the campaign uses on a 4 KiB file in that
  directory and reports the mechanism. ``buffered`` is refused; the Linux
  ``fadvise_dontneed`` fallback (O_DIRECT unavailable on that filesystem) is
  reported as a warning because it is honestly recorded but is a weaker claim
  than ``unbuffered``.

Nothing here times anything; the 4 KiB trial is about *which API answered*, not
how fast it was.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

try:
    from . import ssd_measure
except ImportError:  # direct script / flat sys.path execution
    import ssd_measure  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]
TRIAL_SIZE = 4096
DEFAULT_MIN_FREE_BYTES = 2 << 30  # 1 GiB backing file + 1 GiB aligned buffer

# A filesystem whose numbers describe something other than the drive the user
# thinks they rented. ``fuse.*`` is included because the common cloud cases
# (blobfuse, gcsfuse, sshfs) are all network or object storage.
VIRTUAL_FILESYSTEMS = frozenset({
    "overlay", "overlayfs", "tmpfs", "ramfs", "squashfs", "9p", "virtiofs",
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "afs", "ceph", "glusterfs",
})

REMOVABLE_DRIVE = 2
FIXED_DRIVE = 3
REMOTE_DRIVE = 4
RAMDISK_DRIVE = 6


def classify_filesystem(fstype: str | None) -> str:
    """``local``, ``virtual`` or ``unknown`` for a filesystem type name."""
    if not fstype:
        return "unknown"
    name = fstype.strip().lower()
    if not name:
        return "unknown"
    if name in VIRTUAL_FILESYSTEMS or name.startswith("fuse.") or name.startswith("fuse"):
        return "virtual"
    return "local"


def _posix_mount_info(directory: Path) -> dict[str, Any]:
    findmnt = shutil.which("findmnt")
    if findmnt:
        result = subprocess.run(
            [findmnt, "--target", str(directory), "--json",
             "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            capture_output=True, text=True, errors="replace")
        if result.returncode == 0 and result.stdout.strip():
            try:
                rows = json.loads(result.stdout).get("filesystems") or []
            except json.JSONDecodeError:
                rows = []
            if rows:
                row = rows[0]
                return {
                    "source": row.get("source"),
                    "fstype": row.get("fstype"),
                    "mount_point": row.get("target"),
                    "options": row.get("options"),
                    "probe": "findmnt",
                }
    df = shutil.which("df")
    if df:
        result = subprocess.run([df, "-T", "-P", str(directory)],
                                capture_output=True, text=True, errors="replace")
        lines = result.stdout.strip().splitlines()
        if result.returncode == 0 and len(lines) >= 2:
            fields = lines[1].split()
            if len(fields) >= 7:
                return {
                    "source": fields[0],
                    "fstype": fields[1],
                    "mount_point": fields[6],
                    "options": None,
                    "probe": "df",
                }
    return {"source": None, "fstype": None, "mount_point": None,
            "options": None, "probe": "unavailable"}


def _windows_mount_info(directory: Path) -> dict[str, Any]:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR,
                                            wintypes.DWORD]
    kernel32.GetVolumePathNameW.restype = wintypes.BOOL
    kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetDriveTypeW.restype = wintypes.UINT
    kernel32.GetVolumeInformationW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD, wintypes.LPDWORD,
        wintypes.LPDWORD, wintypes.LPDWORD, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetVolumeInformationW.restype = wintypes.BOOL

    volume = ctypes.create_unicode_buffer(261)
    if not kernel32.GetVolumePathNameW(str(directory), volume, 261):
        raise ctypes.WinError(ctypes.get_last_error())
    root = volume.value
    drive_type = kernel32.GetDriveTypeW(root)
    filesystem = ctypes.create_unicode_buffer(261)
    kernel32.GetVolumeInformationW(root, None, 0, None, None, None,
                                   filesystem, 261)
    mapping = {
        FIXED_DRIVE: "local",
        REMOVABLE_DRIVE: "removable",
        REMOTE_DRIVE: "virtual",
        RAMDISK_DRIVE: "virtual",
    }
    return {
        "source": root,
        "fstype": filesystem.value or None,
        "mount_point": root,
        "options": None,
        "drive_type": drive_type,
        "filesystem_class": mapping.get(drive_type, "unknown"),
        "probe": "GetDriveTypeW",
    }


def mount_info(directory: Path) -> dict[str, Any]:
    """Where does ``directory`` actually live? Never raises for a probe failure."""
    if os.name == "nt":
        try:
            info = _windows_mount_info(directory)
        except OSError as exc:
            info = {"source": None, "fstype": None, "mount_point": None,
                    "options": None, "probe": "unavailable", "error": str(exc)}
    else:
        info = _posix_mount_info(directory)
    info.setdefault("filesystem_class", classify_filesystem(info.get("fstype")))
    return info


def free_bytes(directory: Path) -> int | None:
    try:
        return shutil.disk_usage(directory).free
    except OSError:
        return None


def trial_bypass(directory: Path) -> dict[str, Any]:
    """Run the campaign's own backend selection on a 4 KiB file there.

    Returns the mechanism that answered. It deliberately does not time the
    transfer: this answers "which API is available here", not "how fast".
    """
    path = directory / f"ssd_preflight_{os.getpid()}.bin"
    try:
        ssd_measure._prepare_backing_file(path, TRIAL_SIZE)
        reader, writer, mechanism, backend = ssd_measure.open_backends(
            path, TRIAL_SIZE, "required")
        try:
            writer.fill(TRIAL_SIZE)
            writer.write(TRIAL_SIZE, 0)
            writer.flush()
            reader.read(TRIAL_SIZE, 0)
        finally:
            reader.close()
            writer.close()
        return {"mechanism": mechanism, "backend": backend, "error": None}
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
        return {"mechanism": None, "backend": None,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        path.unlink(missing_ok=True)


def commit_info() -> dict[str, Any]:
    """The commit the campaign will be labelled with, plus whether it is dirty."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "error": None,
    }


def probe(directory: Path) -> dict[str, Any]:
    report = {
        "schema_version": "ssd_measurement_preflight_v1",
        "directory": str(directory),
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "mount": mount_info(directory),
        "free_bytes": free_bytes(directory),
        "bypass": trial_bypass(directory),
        "source": commit_info(),
    }
    return report


def commits_match(actual: str | None, expected: str | None) -> bool:
    """Accept a short SHA against a full one, but never a one-character prefix."""
    if not actual or not expected:
        return actual == expected
    left, right = actual.strip().lower(), expected.strip().lower()
    if min(len(left), len(right)) < 7:
        return left == right
    return left.startswith(right) or right.startswith(left)


def verdict(report: dict[str, Any], *, expected_commit: str | None = None,
            min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
            allow_non_local: bool = False) -> dict[str, Any]:
    """Turn a probe report into ``ok`` / ``warn`` / ``refuse`` plus reasons."""
    refusals: list[str] = []
    warnings: list[str] = []

    if sys.version_info < (3, 10):
        refusals.append(f"python {platform.python_version()} is older than 3.10")

    mount = report.get("mount") or {}
    filesystem_class = mount.get("filesystem_class")
    fstype = mount.get("fstype")
    if filesystem_class == "virtual":
        message = (f"{report.get('directory')} is on filesystem {fstype!r} "
                   f"({mount.get('source')}), which is not a local disk")
        (warnings if allow_non_local else refusals).append(message)
    elif filesystem_class == "removable":
        warnings.append(f"{fstype!r} on removable media {mount.get('source')}: "
                        "confirm this is the drive under test")
    elif filesystem_class == "unknown":
        warnings.append("could not identify the filesystem; the numbers will still "
                        "be recorded but nobody can say which device they describe")

    free = report.get("free_bytes")
    if free is None:
        warnings.append("free space unavailable")
    elif free < min_free_bytes:
        refusals.append(f"{free} bytes free, below the {min_free_bytes} byte floor "
                        "(backing file plus aligned buffer)")

    bypass = report.get("bypass") or {}
    mechanism = bypass.get("mechanism")
    if mechanism is None:
        refusals.append(f"no cache bypass available here: {bypass.get('error')}")
    elif mechanism == "buffered":
        refusals.append("only buffered I/O is available: this would price the page "
                        "cache, and the exported table would be validation_only")
    elif mechanism == "fadvise_dontneed":
        warnings.append("O_DIRECT was refused by this filesystem; the campaign will "
                        "record fadvise_dontneed, which is best-effort")
    elif mechanism == "f_nocache":
        warnings.append("Darwin F_NOCACHE is per-handle and does not prove the drive "
                        "cache was bypassed")

    source = report.get("source") or {}
    if expected_commit:
        actual = source.get("commit")
        if not commits_match(actual, expected_commit):
            refusals.append(f"commit {actual} does not match the expected "
                            f"{expected_commit}; the two tables would not be comparable")
    if source.get("dirty"):
        warnings.append("the working tree is dirty: the commit label alone does not "
                        "identify the code that produced these numbers")

    status = "refuse" if refusals else ("warn" if warnings else "ok")
    return {
        "status": status,
        "refusals": refusals,
        "warnings": warnings,
        "mechanism": mechanism,
        "filesystem_class": filesystem_class,
        "commit": source.get("commit"),
    }


def _format(report: dict[str, Any], result: dict[str, Any]) -> str:
    mount = report["mount"]
    lines = [
        f"directory : {report['directory']}",
        f"platform  : {report['platform']} {report['machine']} / "
        f"python {report['python']}",
        f"filesystem: {mount.get('fstype')} on {mount.get('source')} "
        f"({mount.get('filesystem_class')})",
        f"free      : {report['free_bytes']}" if report["free_bytes"] is not None
        else "free      : unknown",
        f"bypass    : {report['bypass'].get('mechanism')}"
        + (f" ({report['bypass'].get('backend')})"
           if report["bypass"].get("backend") else ""),
        f"commit    : {report['source'].get('commit')}"
        + (" (dirty)" if report["source"].get("dirty") else ""),
        "",
        f"VERDICT   : {result['status'].upper()}",
    ]
    for refusal in result["refusals"]:
        lines.append(f"  REFUSE  {refusal}")
    for warning in result["warnings"]:
        lines.append(f"  WARN    {warning}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check that a directory is a measurable local disk before a campaign.")
    parser.add_argument("--directory", required=True,
                        help="the --directory the campaign will use")
    parser.add_argument("--expected-commit",
                        help="refuse if HEAD differs (use on the second machine)")
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--allow-non-local", action="store_true",
                        help="downgrade virtual filesystems to a warning")
    parser.add_argument("--json", action="store_true",
                        help="print the probe report and verdict as JSON")
    args = parser.parse_args(argv)

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2
    report = probe(directory)
    result = verdict(report, expected_commit=args.expected_commit,
                     min_free_bytes=args.min_free_bytes,
                     allow_non_local=args.allow_non_local)
    if args.json:
        print(json.dumps({"report": report, "verdict": result},
                         ensure_ascii=False, indent=2))
    else:
        print(_format(report, result))
    return 2 if result["status"] == "refuse" else 0


if __name__ == "__main__":
    raise SystemExit(main())
