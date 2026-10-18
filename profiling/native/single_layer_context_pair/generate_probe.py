"""Copy two pinned executors with exact main-symbol renames, without edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HIGHLEVEL_SHA256 = "ceeebc81c00a4fb760923c7b1286a7470558250488f93119c7d628dfed0872eb"
SINGLE_LAYER_SHA256 = "27f8360fc6e54a090dec777c21a835d1ec29db2810aa4e453e6fb020401cb9a6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rename_main(content, expected_sha256, replacement):
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("immutable executor bytes differ")
    marker = b"int main(int argc,char **argv)"
    if content.count(marker) != 1:
        raise ValueError("ambiguous executor main definition")
    return content.replace(marker, replacement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bound-file", action="append", type=Path, default=[])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    historical = here.parent / "highlevel_accuracy/highlevel_probe.cpp"
    single = here.parent / "single_layer_calibration/probe.cpp"
    generated = {
        "highlevel_generated.inc": rename_main(historical.read_bytes(), HIGHLEVEL_SHA256,
            b"static int single_layer_historical_main(int argc,char **argv)"),
        "single_layer_generated.inc": rename_main(single.read_bytes(), SINGLE_LAYER_SHA256,
            b"static int context_pair_single_layer_main(int argc,char **argv)"),
    }
    files = {historical, single, *here.glob("*.py"), *here.glob("*.cpp"), here / "CMakeLists.txt", *args.bound_file}
    if any(not path.is_file() for path in files):
        raise ValueError("explicit required build input is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        (args.output_dir / name).write_bytes(content)
    record = dict(format_version="single-layer-context-pair-generated-v1",
        executor_mutations="exact main symbols renamed only; source executor files unchanged",
        immutable_executors_sha256={str(historical.resolve()): HIGHLEVEL_SHA256, str(single.resolve()): SINGLE_LAYER_SHA256},
        generated_files={str((args.output_dir / name).resolve()): hashlib.sha256(content).hexdigest() for name, content in generated.items()},
        bound_files={str(path.resolve()): sha(path) for path in sorted(files)})
    (args.output_dir / "source-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
