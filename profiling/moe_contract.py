"""Shared physical MoE FFN measurement contract, located in modeling."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
from llm_infer_model.m2.moe_ffn_contract import *  # noqa: F401,F403
