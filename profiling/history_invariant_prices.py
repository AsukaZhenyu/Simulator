"""One rule for the decode prices of stages that read no history.

A decode price is filed under ``(kind, device, phase, batch, tokens,
visible_history)``.  Three of the declared stages -- the projection and
whichever feed-forward the contract declares -- read one token and no KV, so
their work at visible_history 128, 512 and 1024 is the same work.  Priced
separately, those entries are three replicates of one physical task, and the
differences between them are measurement and machine state.  Filing them as
three different prices turns that noise into a depth slope the model then
multiplies by 128, 512 and 1024 tokens of history.

This module holds the arithmetic for both callers: the exporter, which prices
a fresh measurement run, and the repricer, which applies the same rule to a
table that has already been exported.  The two must never drift.

What the rule does NOT do is decide that the entries are replicates.  That is a
claim about the contract, and the caller gets it from the contract's own stage
list (``history_invariant_kinds``), never from the data.
"""
from __future__ import annotations

import math
import statistics

# A pooled price is only defensible while the replicates look like replicates.
# Every model measured on this machine spreads 1.1-17.3% across depths, inside
# the 12-18% band that separates two processes producing the same table; past
# half the price the differences are no longer replicate noise and the table
# has a different defect to investigate.
MAX_HISTORY_REPLICATE_SPREAD = 0.5


class ReplicateError(ValueError):
    """The entries offered as history replicates cannot support one pooled price."""


# The dimensions along which replicates are allowed to differ. Everything else
# -- the kind, the device, the shape, the workspace, the protocol fields, and
# the whole signature except its depth -- must be identical, or the entries are
# not the same physical task and a pooled price would be a fabrication.
REPLICATE_IGNORED_SIGNATURE_KEYS = ("visible_history", "history_invariant_price_policy")
REPLICATE_IGNORED_ENTRY_KEYS = ("duration_ns", "contract_hash", "visible_history")


def replicate_key(entry: dict) -> dict:
    """Everything about a cost entry that a history replicate must share."""
    signature = {key: value for key, value in entry["physical_signature"].items()
                 if key not in REPLICATE_IGNORED_SIGNATURE_KEYS}
    return {key: value for key, value in entry.items() if key not in REPLICATE_IGNORED_ENTRY_KEYS} | {
        "physical_signature": signature}


def require_replicates(entries: list, label: str) -> None:
    """Refuse a group that differs in anything except the depth it was priced at."""
    if not isinstance(entries, list) or len(entries) < 2:
        raise ReplicateError(f"{label}: pooling needs at least two history replicates")
    reference = replicate_key(entries[0])
    for entry in entries[1:]:
        if replicate_key(entry) != reference:
            raise ReplicateError(f"{label}: entries priced at different depths are not "
                                 "replicates -- they differ in more than the depth")


def pool_history_invariant(prices: list[tuple[int, object, int]]) -> dict:
    """Pool ``(visible_history, tag, duration_ns)`` replicates into one price.

    Returns the pooled median, the relative spread it was pooled across, and
    every reading it was pooled from, so a caller can record what it replaced.
    Raises when the readings are not replicates at all: fewer than two depths,
    a repeated depth, or a spread past :data:`MAX_HISTORY_REPLICATE_SPREAD`.
    """
    if not isinstance(prices, list) or len(prices) < 2:
        raise ReplicateError("pooling needs at least two history replicates")
    depths = [history for history, _, _ in prices]
    if len(set(depths)) != len(depths):
        raise ReplicateError(f"history replicates repeat a depth: {sorted(depths)}")
    if any(isinstance(ns, bool) or not isinstance(ns, int) or ns < 1 for _, _, ns in prices):
        raise ReplicateError("history replicate prices must be positive integer nanoseconds")
    observed = sorted(prices)
    values = [ns for _, _, ns in observed]
    pooled = math.ceil(statistics.median(values))
    spread = (max(values) - min(values)) / pooled
    if spread > MAX_HISTORY_REPLICATE_SPREAD:
        raise ReplicateError(f"history replicates spread {spread:.3f} of their median, "
                             f"past {MAX_HISTORY_REPLICATE_SPREAD}; this is not replicate noise")
    return {"pooled_ns": pooled, "relative_spread": spread,
            "replicates": [{"visible_history": history, "tag": tag, "duration_ns": ns}
                           for history, tag, ns in observed]}
