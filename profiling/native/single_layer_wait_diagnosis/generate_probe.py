"""Generate pinned executor copies with only declared NVTX/main insertions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PINS = {
    "highlevel_accuracy/highlevel_probe.cpp": "ceeebc81c00a4fb760923c7b1286a7470558250488f93119c7d628dfed0872eb",
    "single_layer_calibration/probe.cpp": "27f8360fc6e54a090dec777c21a835d1ec29db2810aa4e453e6fb020401cb9a6",
    "single_layer_context_pair/paired.cpp": "b4690ff3828c5a3fb99c34690b9ffcdbc2c91bf7045c36558e89488d6b609a25",
}
MAIN = b"int main(int argc,char **argv)"
ACTION_MARKER = b"    void action(const json &a){\n"
ACTION_INSERTION = b"        wait_diag::ActionRange wait_action_range(a);\n"
BLOCK_MARKER = b"const char label=order.at(index);"
BLOCK_INSERTION = b"wait_diag::BlockRange wait_block_range(index);"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(relative, content):
    if relative not in PINS or hashlib.sha256(content).hexdigest() != PINS[relative]:
        raise ValueError("pinned executor bytes differ: " + relative)
    if content.count(MAIN) != 1:
        raise ValueError("ambiguous main definition")
    names = {"highlevel_accuracy/highlevel_probe.cpp": b"static int single_layer_historical_main(int argc,char **argv)",
             "single_layer_calibration/probe.cpp": b"static int context_pair_single_layer_main(int argc,char **argv)",
             "single_layer_context_pair/paired.cpp": b"static int wait_diag_pair_main(int argc,char **argv)"}
    result = content.replace(MAIN, names[relative])
    if relative.startswith("highlevel_accuracy/"):
        if result.count(ACTION_MARKER) != 1:
            raise ValueError("ambiguous action entry")
        result = result.replace(ACTION_MARKER, ACTION_MARKER + ACTION_INSERTION)
    if relative.startswith("single_layer_context_pair/"):
        if result.count(BLOCK_MARKER) != 1:
            raise ValueError("ambiguous pair block entry")
        result = result.replace(BLOCK_MARKER, BLOCK_MARKER + BLOCK_INSERTION)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bound-file", action="append", type=Path, default=[])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    names = {"highlevel_accuracy/highlevel_probe.cpp": "highlevel_generated.inc",
             "single_layer_calibration/probe.cpp": "single_layer_generated.inc",
             "single_layer_context_pair/paired.cpp": "pair_generated.inc"}
    originals = {relative: here.parent / relative for relative in PINS}
    generated = {names[relative]: generate(relative, path.read_bytes()) for relative, path in originals.items()}
    files = {*originals.values(), *here.glob("*.py"), *here.glob("*.cpp"), here / "CMakeLists.txt", *args.bound_file}
    if any(not path.is_file() for path in files):
        raise ValueError("explicit required build input is missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        (args.output_dir / name).write_bytes(content)
    record = dict(format_version="single-layer-wait-diagnosis-generated-v1",
        allowed_changes=["three exact main-symbol renames", "one NVTX action-entry RAII insertion", "one NVTX pair-block RAII insertion"],
        immutable_executors_sha256={str(path.resolve()): PINS[relative] for relative, path in originals.items()},
        generated_files={str((args.output_dir / name).resolve()): hashlib.sha256(content).hexdigest() for name, content in generated.items()},
        bound_files={str(path.resolve()): sha(path) for path in sorted(files)})
    (args.output_dir / "source-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
