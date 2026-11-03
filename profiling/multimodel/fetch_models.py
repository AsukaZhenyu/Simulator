# -*- coding: utf-8 -*-
"""Fetch the validation model set declared in ``models.json``.

Weights are never committed: a 4-bit GGUF is 0.4-9.5 GB, and GitHub rejects
files over 100 MB, release assets over 2 GB and free LFS over 1 GB.  A device
therefore materialises the same bytes from the declared publisher revision and
proves they are the same bytes by SHA-256 before anything is measured.

The manifest pins repository, filename, revision, byte count, SHA-256 and the
geometry read from the header, so a device branch records *which* model it ran
without carrying the model.

This module only downloads.  It does not read tensor payloads, run inference or
write into the repository.

    python profiling/multimodel/fetch_models.py --list
    python profiling/multimodel/fetch_models.py                 # all models
    python profiling/multimodel/fetch_models.py --only qwen3-0.6b
    python profiling/multimodel/fetch_models.py --check         # verify, no network
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "models.json"
BLOCK = 8 * 1024 * 1024
BASE = "https://huggingface.co"


def default_root() -> Path:
    declared = os.environ.get("LLM_INFER_MODELS_DIR")
    if declared:
        return Path(declared).expanduser()
    return Path.home() / ".cache" / "llm-infer-simulator" / "models"


def read_manifest(path: Path = MANIFEST) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "multimodel-models-1":
        raise SystemExit(f"unexpected manifest schema in {path}")
    seen = set()
    for model in document["models"]:
        for field in ("id", "repository", "filename", "revision", "bytes", "sha256"):
            if field not in model:
                raise SystemExit(f"{model.get('id', '?')}: missing {field}")
        if model["id"] in seen:
            raise SystemExit(f"duplicate model id {model['id']}")
        seen.add(model["id"])
        if len(model["sha256"]) != 64:
            raise SystemExit(f"{model['id']}: sha256 is not a hex digest")
    return document


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, model: dict) -> str | None:
    """Return None when the file matches the manifest, else the reason."""
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size != model["bytes"]:
        return f"size {size} != {model['bytes']}"
    actual = digest(path)
    if actual != model["sha256"]:
        return f"sha256 {actual} != {model['sha256']}"
    return None


def target(root: Path, model: dict) -> Path:
    # Flat, because every declared filename is unique and a user pointing
    # --root at an existing model directory should find the files where they
    # already are rather than in a layout this tool invented.
    return root / model["filename"]


def url(model: dict) -> str:
    # The revision is a commit hash, so this resolves to immutable bytes even if
    # the publisher later rewrites main.
    return f"{BASE}/{model['repository']}/resolve/{model['revision']}/{model['filename']}"


def _open(url_: str, offset: int):
    headers = {"User-Agent": "llm-infer-simulator-fetch/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url_, headers=headers), timeout=120)


def download(model: dict, root: Path, retries: int = 5) -> Path:
    out = target(root, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    reason = verify(out, model)
    if reason is None:
        print(f"  ok       {model['id']}")
        return out
    if reason != "missing":
        # A file that is present but wrong is never silently overwritten: that
        # would hide a corrupted or substituted download.
        raise SystemExit(f"{out} exists but {reason}; move it away and retry")

    part = out.with_name(out.name + ".part")
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if offset > model["bytes"]:
            part.unlink()
            offset = 0
        try:
            with _open(url(model), offset) as response, part.open("ab") as sink:
                if offset and response.status != 206:
                    sink.close()
                    part.unlink()
                    raise SystemExit("server ignored the resume range; refusing to append")
                total = int(response.headers.get("Content-Length", 0)) + offset
                done = offset
                started = time.monotonic()
                while True:
                    block = response.read(BLOCK)
                    if not block:
                        break
                    sink.write(block)
                    done += len(block)
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = (done - offset) / elapsed / 1e6
                    sys.stderr.write(
                        f"\r  {model['id']:<24} {done / 1e9:7.2f}/{total / 1e9:6.2f} GB"
                        f"  {rate:6.1f} MB/s")
                    sys.stderr.flush()
            sys.stderr.write("\n")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt + 1 == retries:
                raise SystemExit(f"{model['id']}: giving up after {retries} attempts: {error}")
            wait = 2 ** attempt
            print(f"  retry    {model['id']} in {wait}s ({error})")
            time.sleep(wait)
            continue

        reason = verify(part, model)
        if reason is not None:
            part.unlink()
            print(f"  retry    {model['id']}: downloaded bytes {reason}")
            continue
        part.replace(out)
        print(f"  ok       {model['id']}")
        return out
    raise SystemExit(f"{model['id']}: no attempt produced matching bytes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--only", action="append", default=[], metavar="ID")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true", help="verify local files, do not download")
    args = parser.parse_args(argv)

    document = read_manifest(args.manifest)
    models = document["models"]
    if args.only:
        known = {model["id"] for model in models}
        unknown = sorted(set(args.only) - known)
        if unknown:
            raise SystemExit(f"unknown model id(s): {', '.join(unknown)}")
        models = [model for model in models if model["id"] in set(args.only)]

    if args.list:
        for model in models:
            state = "present" if verify(target(args.root, model), model) is None else "absent"
            print(f"{model['id']:<24} {model['role']:<5} {model['bytes'] / 1e9:6.2f} GB"
                  f"  {state:<7} {model['filename']}")
        print(f"\nroot: {args.root}")
        return 0

    if args.check:
        bad = 0
        for model in models:
            reason = verify(target(args.root, model), model)
            print(f"{model['id']:<24} {'ok' if reason is None else reason}")
            bad += reason is not None
        return 1 if bad else 0

    print(f"root: {args.root}")
    for model in models:
        download(model, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
