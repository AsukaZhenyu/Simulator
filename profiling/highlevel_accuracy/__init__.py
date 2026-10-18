"""Exact, bounded ModelPlan -> GGML accuracy bridge; no historical fixtures mutate."""
from .bridge import BridgeError, export_plan, validate_program, verify_native

__all__ = ["BridgeError", "export_plan", "validate_program", "verify_native"]
