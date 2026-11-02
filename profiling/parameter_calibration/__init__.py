"""Raw evidence and parameter estimation; performance equations live in modeling.

This optional package does not collect hardware counters or launch benchmarks.
It preserves every supplied observation, estimates physical parameters only from
compatible timing scopes, and calls the model-owned predictor for validation.
"""

from .observations import ParameterObservation, derive_observation, load_observations
from .parameters import estimate_parameters, load_frozen_parameters

__all__ = ["ParameterObservation", "derive_observation", "load_observations",
           "estimate_parameters", "load_frozen_parameters"]
