"""How priced actions add up into a request.

This package is the assembly layer.  It answers one question:

    given an exported cost table and a declared shape, what does one token step
    cost, and what did the wall clock say?

It never measures anything.  Wall clocks live in ``profiling/``; what a device
costs lives in a cost table (exported by ``profiling/``, read through
``modeling.llm_infer_model.m2.costs``).  What lives here is the summation rule,
the declared terms that the table does not price, and the attribution of a
disagreement between the sum and the clock.

Dependency rule: ``composing`` may read ``modeling`` (the physical format of a
cost table, its contract signatures).  ``modeling`` must never import
``composing``, and ``composing`` must never import ``profiling`` -- a
composition that needed the measurement apparatus would not be a model.

The four modules ask four different questions, and each is the only place its
question is answered:

===================  ============================================================
``mapping``          what should be done
``modeling``         how an action executes
``profiling``        what one action costs on a device
``composing``        how those costs add up into a request
===================  ============================================================
"""

from .compose import compose_total
from .terms import KNOWN_TERMS, Term, TermSpec

__all__ = ["KNOWN_TERMS", "Term", "TermSpec", "compose_total"]
