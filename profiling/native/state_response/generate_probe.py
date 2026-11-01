"""Reuse byte-pinned execution and event helpers without changing old files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

PINS = {
    "highlevel_accuracy/highlevel_probe.cpp": "24b9fec69f56050e92e5219ba6b59a5519b8ac2af839496d1894f1b79707d56d",
    "single_layer_calibration/probe.cpp": "27f8360fc6e54a090dec777c21a835d1ec29db2810aa4e453e6fb020401cb9a6",
    "state_calibration/compute.cpp": "bb2ace018e543c706d72563f7c61ccd260a58a4cde79354b76d5dd950959fc78",
}
EVENT_ABI_FILES = {
    "ggml/src/ggml-backend-impl.h": "470f4ede95a1a66edb7259981c1e8c98e2827de765323fe1672badf8f99f4546",
    "ggml/src/ggml-backend.cpp": "20e449bd92cec4c0c16ab91dc103ee6871ac6c45f22fd70f1d43bca33f171500",
    "ggml/src/ggml-cuda/ggml-cuda.cu": "65389d299a774b8a777ac07b1a6558a46322638b3f99d337452d3f6f3d0cd50b",
}
MAIN = b"int main(int argc,char **argv)"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def generated_sources(originals):
    if set(originals) != set(PINS):
        raise ValueError("pinned executor set differs")
    for name, content in originals.items():
        if name not in PINS or hashlib.sha256(content).hexdigest() != PINS[name]:
            raise ValueError("pinned executor differs: " + name)
    high = originals["highlevel_accuracy/highlevel_probe.cpp"]
    single = originals["single_layer_calibration/probe.cpp"]
    state = originals["state_calibration/compute.cpp"]
    if high.count(MAIN) != 1 or single.count(MAIN) != 1:
        raise ValueError("ambiguous main symbol")
    def extract(first, last):
        if state.count(first) != 1 or state.count(last) != 1:
            raise ValueError("ambiguous pinned helper boundary")
        return state[state.index(first):state.index(last)]
    return {
        "highlevel_generated.inc": high.replace(MAIN, b"static int single_layer_historical_main(int argc,char **argv)"),
        "single_layer_generated.inc": single.replace(MAIN, b"static int state_response_historical_main(int argc,char **argv)"),
        "timing_events.inc": extract(b"static void cuda_ok(", b"struct TargetTiming {"),
        "numeric_helpers.inc": extract(b"static json compare_ports(", b"static json graph_description("),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--bound-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    originals = {name: (here.parent / name).read_bytes() for name in PINS}
    generated = generated_sources(originals)
    for name, expected in EVENT_ABI_FILES.items():
        if sha(args.llama_root / name) != expected:
            raise ValueError("pinned event ABI differs: " + name)
    cuda_headers = [*sorted((args.cuda_root / "include").rglob("*.h")), *sorted((args.cuda_root / "include").rglob("*.hpp"))]
    if not (args.cuda_root / "include/cuda_runtime_api.h").is_file():
        raise ValueError("explicit CUDA runtime headers missing")
    files = {*(here.parent / name for name in PINS), *(args.llama_root / name for name in EVENT_ABI_FILES), *cuda_headers,
             *here.glob("*.py"), *here.glob("*.cpp"), here / "CMakeLists.txt", *args.bound_file}
    if any(not path.is_file() for path in files):
        raise ValueError("explicit required build input missing")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        (args.output_dir / name).write_bytes(content)
    record = dict(format_version="state-response-generated-v1", immutable_executors_sha256=PINS,
        event_abi_sha256=EVENT_ABI_FILES,
        changes="two exact main-symbol renames; byte-exact extraction of event and numerical helpers; Executor.action/reset unchanged",
        generated_files={str((args.output_dir / name).resolve()): hashlib.sha256(content).hexdigest() for name, content in generated.items()},
        bound_files={str(path.resolve()): sha(path) for path in sorted(files)})
    (args.output_dir / "source-manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")

if __name__ == "__main__":
    main()
