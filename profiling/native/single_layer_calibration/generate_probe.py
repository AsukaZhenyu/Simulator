"""Generate only a renamed-main copy of the pinned historical executor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HELPER_SHA256 = "ceeebc81c00a4fb760923c7b1286a7470558250488f93119c7d628dfed0872eb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generated_helper(content):
    if hashlib.sha256(content).hexdigest() != HELPER_SHA256:
        raise ValueError("immutable historical helper bytes differ")
    marker = b"int main(int argc,char **argv)"
    if content.count(marker) != 1:
        raise ValueError("ambiguous historical main definition")
    return content.replace(marker, b"static int single_layer_historical_main(int argc,char **argv)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bound-file", action="append", type=Path, default=[])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    original = here.parent / "highlevel_accuracy/highlevel_probe.cpp"
    generated = generated_helper(original.read_bytes())
    files = {original, *here.glob("*.py"), *here.glob("*.cpp"), here / "CMakeLists.txt", *args.bound_file}
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError("explicit build input missing: " + repr(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(generated)
    record = dict(format_version="single-layer-generated-source-v1", immutable_helper_sha256=HELPER_SHA256,
                  source_mutation="exact historical main symbol rename only", generated_file=str(args.output.resolve()),
                  generated_sha256=sha(args.output), bound_files={str(path.resolve()): sha(path) for path in sorted(files)})
    args.manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
