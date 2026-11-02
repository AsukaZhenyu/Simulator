"""Optional profiling of physical action efficiency and coherent trace evidence.

No execution-time predictor lives here. Modeling consumes the measured bindings.
Legacy inclusive traces are retained only as correlated diagnostic ensembles.
"""

from .calibration import freeze_efficiencies, load_efficiency_bindings
from .coherent import freeze_coherent_ensemble, validate_coherent_ensemble
from .validation import make_barrier, evaluate_available_service

__all__ = [
    "freeze_efficiencies", "load_efficiency_bindings",
    "freeze_coherent_ensemble", "validate_coherent_ensemble",
    "make_barrier", "evaluate_available_service",
]
