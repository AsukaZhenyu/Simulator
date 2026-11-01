"""Mechanical JSON scalar serialization repair for retained analysis v1.

The statistical algorithm, candidate selection and all measurements are
unchanged. Keep analyze.py and its interrupted output as executed evidence.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import analyze

def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def save(path,value):
    # Serialize before creating the exclusive file, avoiding partial artifacts.
    result=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(result+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--rounds',type=int,default=0)
    a=ap.parse_args();analyze.save=save
    analyze.json=SimpleNamespace(dumps=lambda obj,**kw:json.dumps(obj,default=scalar,**kw),loads=json.loads)
    analyze.run(a.campaign,a.out,a.rounds)

if __name__=='__main__':main()
