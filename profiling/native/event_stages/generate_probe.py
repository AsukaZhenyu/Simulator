"""Reuse pinned helpers and rename only the old main function declaration."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def generated_state_source(source):
    declaration=b"int main(int argc,char **argv)"
    if source.count(declaration)!=1:
        raise ValueError("historical state main declaration is ambiguous")
    return source.replace(declaration,b"static int physical_stages_unused_state_response_main(int argc,char **argv)")


def main():
    here=Path(__file__).resolve().parent
    source=here.parent/'state_response'/'generate_probe.py'
    spec=importlib.util.spec_from_file_location('physical_stages_pinned_source_generator',source)
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    parent.main() # Same explicit CLI; emits the unchanged pinned helper includes.
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args,_=parser.parse_known_args()
    original=here.parent/'state_response'/'probe.cpp'
    target=args.output_dir/'state_response_generated.inc'
    target.write_bytes(generated_state_source(original.read_bytes()))
    manifest=args.output_dir/'source-manifest.json'
    record=json.loads(manifest.read_text(encoding='utf-8'))
    record['generated_files'][str(target.resolve())]=hashlib.sha256(target.read_bytes()).hexdigest()
    record['bound_files'][str(original.resolve())]=hashlib.sha256(original.read_bytes()).hexdigest()
    record['bound_files'][str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['changes']+='; exact state_response main function declaration rename only; no preprocessor token rename'
    manifest.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
