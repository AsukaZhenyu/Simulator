"""Verify a recorded source version without rewriting historical measurements.

Current-source verifiers should reject an older experiment after source changes.
This separate check verifies archived source bytes against the original campaign
bindings and reports current changes explicitly. It does not execute an archived
runtime, revalidate numerical outputs, or establish hardware accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts):
        raise ValueError("archive entries must be relative POSIX paths")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("archive entry escapes its root")
    return target


def verify_archive(archive, workspace):
    archive, workspace = Path(archive).resolve(), Path(workspace).resolve()
    record = json.loads((archive / "archive.json").read_text(encoding="utf-8"))
    if record.get("schema") != "performance-boundary-source-archive-0.1" or record.get("raw_measurements_rewritten") is not False:
        raise ValueError("unsupported source archive")
    expected = record.get("sources")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("source archive has no bindings")
    changed, size = [], 0
    for name, digest in expected.items():
        saved = _inside(archive / "sources", name)
        if _sha(saved) != digest:
            raise ValueError("archived source does not match recorded bytes: " + name)
        size += saved.stat().st_size
        current = _inside(workspace, name)
        if not current.is_file() or _sha(current) != digest:
            changed.append(name)
    seen, campaigns = {}, []
    for name, binding in record["campaigns"].items():
        manifest_path = _inside(workspace, name)
        if _sha(manifest_path) != binding["manifest_sha256"]:
            raise ValueError("historical campaign manifest changed: " + name)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_rows = manifest["sources"]
        resolved_sources = {}
        for path, value in source_rows.items():
            resolved_sources.setdefault(Path(path).resolve(), []).append(value)
        for relative, digest in binding["sources"].items():
            # Historical manifests contain absolute paths on their recorded host.
            current_path = _inside(workspace, relative)
            matches = resolved_sources.get(current_path, [])
            if matches != [digest] or expected.get(relative) != digest:
                raise ValueError("archive source differs from original campaign binding")
            if relative in seen and seen[relative] != digest:
                raise ValueError("inconsistent source versions across campaigns")
            seen[relative] = digest
        campaigns.append({"manifest": name, "manifest_sha256": binding["manifest_sha256"],
                          "archived_bindings": len(binding["sources"])})
    if seen != expected:
        raise ValueError("archive contains unbound or missing recorded source files")
    return {"schema": "recorded-source-version-verification-0.1", "verified": True,
            "recorded_git_commit": record["git_commit"], "archive_sha256": _sha(archive / "archive.json"),
            "source_files": len(expected), "source_bytes": size,
            "current_source_changes": sorted(changed), "campaigns": campaigns,
            "raw_measurements_rewritten": False, "current_source_equivalence_claimed": not changed,
            "archived_runtime_executed": False, "hardware_accuracy_claimed": False,
            "scope": "Archived mutable source bytes and original manifest bindings only; not runtime assets or numerical replay"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify_archive(args.archive, args.workspace)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("verified", "source_files", "current_source_changes")}), flush=True)


if __name__ == "__main__":
    main()
