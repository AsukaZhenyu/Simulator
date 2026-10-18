"""CMake post-build provenance; emitted only after a successful native link."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


parser=argparse.ArgumentParser()
parser.add_argument("--binary",required=True,type=Path)
parser.add_argument("--source-dir",required=True,type=Path)
parser.add_argument("--compiler",required=True)
parser.add_argument("--flags",required=True)
parser.add_argument("--link-library",action="append",default=[])
args=parser.parse_args()
files=[args.source_dir/"highlevel_probe.cpp",args.source_dir/"CMakeLists.txt",Path(__file__).resolve()]
record=dict(format_version="highlevel-native-successful-build-v1",linked_at=datetime.now(timezone.utc).isoformat(),
    binary=str(args.binary.resolve()),binary_sha256=digest(args.binary),
    source_sha256={str(path.resolve()):digest(path) for path in files},
    compiler=args.compiler,flags=args.flags,
    linked_library_sha256={str(Path(path).resolve()):digest(path) for path in args.link_library})
(args.binary.parent/"build-manifest.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
