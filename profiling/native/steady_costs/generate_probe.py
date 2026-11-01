"""Reuse byte-pinned historical helpers; never edit an old execution source."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def state_source(content):
    declaration=b'int main(int argc,char **argv)'
    if content.count(declaration)!=1:
        raise ValueError('historical state main declaration is ambiguous')
    return content.replace(declaration,b'static int steady_costs_unused_state_main(int argc,char **argv)')


def main():
    here=Path(__file__).resolve().parent
    parent_path=here.parent/'state_response/generate_probe.py'
    spec=importlib.util.spec_from_file_location('steady_pinned_generator',parent_path)
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    parent.main()
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True)
    args,_=parser.parse_known_args()
    original=here.parent/'state_response/probe.cpp'
    generated=args.output_dir/'state_response_generated.inc'
    generated.write_bytes(state_source(original.read_bytes()))
    path=args.output_dir/'source-manifest.json'
    value=json.loads(path.read_text(encoding='utf-8'))
    value['generated_files'][str(generated.resolve())]=hashlib.sha256(generated.read_bytes()).hexdigest()
    for source in (original,Path(__file__)):
        value['bound_files'][str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest()
    value['changes']+='; one exact unused state-main declaration rename; original Executor remains unchanged; separate new steady/component harness'
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
