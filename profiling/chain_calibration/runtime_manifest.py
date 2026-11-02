"""Pure-stdlib binding for optional private strict-F32 backend manifests.

This validates declared build bytes, not operating-system loader resolution.
It does not claim a clean rebuild or cover unchanged MUL_MAT_ID arithmetic.
"""
import hashlib
import json
from pathlib import Path
import re

GROUPS=("original_source_sha256","inputs_sha256","output_sha256","evidence_sha256")


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb")as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):value.update(block)
    return value.hexdigest()


def discover(runtime_dirs):
    records=[];assets={}
    manifests=sorted({(Path(folder).resolve()/"strict-f32-runtime.json")for folder in runtime_dirs if(Path(folder)/"strict-f32-runtime.json").is_file()})
    for path in manifests:
        document=json.loads(path.read_text(encoding="utf-8"))
        if document.get("format_version")!="ggml-private-strict-f32-runtime-v1"or type(document.get("patch_hunks"))is not int or document["patch_hunks"]!=1:
            raise ValueError("unsupported private strict-F32 build manifest: "+str(path))
        bound={str(path):sha(path)}
        for group in GROUPS:
            mapping=document.get(group)
            if not isinstance(mapping,dict)or not mapping:raise ValueError("private backend manifest omits "+group)
            for name,expected in mapping.items():
                file=Path(name)
                if not file.is_absolute()or not isinstance(expected,str)or not re.fullmatch("[0-9a-f]{64}",expected):raise ValueError("invalid absolute-path/SHA256 runtime binding")
                file=file.resolve();actual=sha(file)
                if actual!=expected:raise ValueError("private backend manifest byte mismatch: "+str(file))
                if str(file)in bound and bound[str(file)]!=expected:raise ValueError("conflicting private runtime manifest bindings")
                bound[str(file)]=actual
        cuda=path.parent/"ggml-cuda.dll"
        if document["output_sha256"].get(str(cuda.resolve()))!=sha(cuda):raise ValueError("manifest does not bind supplied ggml-cuda.dll output")
        for name,value in bound.items():
            if name in assets and assets[name]!=value:raise ValueError("runtime manifests disagree about shared input bytes")
            assets[name]=value
        records.append(dict(manifest_path=str(path),manifest_sha256=bound[str(path)],manifest=document,
            validated_file_sha256=bound,loader_paths_observed=False,
            scope="one patched regular-MUL_MAT router plus byte-bound reused objects; MUL_MAT_ID unchanged; not a clean GGML rebuild"))
    return records,assets


def verify_recorded(records,inventory):
    current,assets=discover([Path(r["manifest_path"]).parent for r in records])
    if current!=records:raise ValueError("recorded private strict-F32 manifests differ from current byte-bound evidence")
    if any(inventory.get(path)!=value for path,value in assets.items()):raise ValueError("private backend manifest/input/evidence/output absent from campaign source inventory")
    return True


def hash_bindings(records):return{r["manifest_path"]:r["manifest_sha256"]for r in records}
