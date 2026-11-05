"""The vocabulary composition shares with measurement.

Two things are defined once here and imported by both sides:

* the **shapes** a token step is priced at (``pp256``, ``tg128``, ``tg512``,
  ``tg1024``), because a wall clock and a prediction must be numbers for the
  same question;
* the **reader of an exported cost table**.  Composition does not parse the
  physical format itself: ``modeling.llm_infer_model.m2.costs.parse_cost_table``
  is the only place in the repository that knows it, and a second reader here
  would drift from the one the simulator loads.

The import below puts ``modeling/`` on the path only when this package is run
straight from a checkout (``python -m composing``).  An installed environment
already resolves ``llm_infer_model`` and the insert is a no-op.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

_MODELING = Path(__file__).resolve().parent.parent / "modeling"
if _MODELING.is_dir() and str(_MODELING) not in sys.path:
    sys.path.insert(0, str(_MODELING))

from llm_infer_model.m2.costs import MacroShape, M2CostTable, parse_cost_table  # noqa: E402
from llm_infer_model.tensor.spec import SpecError  # noqa: E402

# name -> (phase, batch tokens, visible history).  A decode step is one token
# with a history; a prefill step is a batch of tokens with an empty cache.
SHAPES = {
    "pp256": ("prefill", 256, 0),
    "tg128": ("decode", 1, 128),
    "tg512": ("decode", 1, 512),
    "tg1024": ("decode", 1, 1024),
}

# The regime a table was measured under.  Absent means the historical resident
# protocol, which is what every table exported before the regime was declarable
# carries.  A prediction may only use entries that all declare one regime.
RESIDENT = "resident_warm_cache_shared_across_samples"
STREAMING = "streamed_cold_cache_before_each_sample"

STAGE_KINDS = ("projection", "attention", "dense_ffn")
# model geometry.ffn -> the stage name that declares it in the exported table.
FFN_STAGES = {"dense": "dense_ffn", "moe": "moe_ffn"}
BOUNDARY_KINDS = ("embedding", "head")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def shape_key(phase: str, tokens: int, history: int) -> str:
    return f"{phase}/B1/q{tokens}/hist{history}"


def table_regime(report: dict) -> str:
    """The one residency regime every entry of an exported table was priced under."""
    regimes = {(entry.get("physical_signature") or {})
               .get("residency_regime", RESIDENT) for entry in report["entries"]}
    if len(regimes) != 1:
        raise ValueError("an exported table mixes residency regimes: "
                         f"{sorted(regimes)}; its prices are not one protocol")
    return regimes.pop()


def entry_signature(entry: dict) -> dict:
    """The physical signature an exported record claims, or ``{}`` if it has none.

    ``export_report.json`` keeps it inside the measurement signature it
    re-derived from the raw probe output; a ``costs.json`` entry carries it at
    the top level.  Both name the same physical conditions.
    """
    signature = (entry.get("physical_signature")
                 or (entry.get("measurement_signature") or {}).get("physical_signature"))
    return signature if isinstance(signature, dict) else {}


def parse_exported_table(document: dict, *, where: str) -> M2CostTable:
    """A ``costs.json`` document as the simulator itself reads it.

    One device only: the price key has no device dimension, so a table that
    priced two devices would silently collapse them into one number.
    """
    try:
        table = parse_cost_table(document, where=where)
    except SpecError as error:
        raise ValueError(f"{where}: the simulator refuses this cost table: {error}") from error
    devices = {device for (_, device, _) in table.entries}
    if len(devices) > 1:
        raise ValueError(f"{where}: the table prices several devices {sorted(devices)}; "
                         "one prediction is one device")
    return table


def table_prices(table: M2CostTable) -> dict:
    """Every price a table actually exports, keyed by (kind, phase, tokens, history).

    The report is the evidence; the table is the artifact the simulator loads,
    and the two are not the same document once a price policy has replaced a
    reading.  A prediction is made from the table, never from the raw medians a
    repriced table no longer exports.
    """
    prices = {}
    for (kind, _device, shape), cost in table.entries.items():
        # The price key has no batch dimension, so a table that priced a batch
        # would collapse several entries into one number here.
        if shape.batch != 1:
            raise ValueError(f"the table prices {kind} at batch {shape.batch}; one token "
                             "step is one request")
        key = (kind, shape.phase, int(shape.tokens), int(shape.visible_history))
        if key in prices:
            raise ValueError(f"the table exports {key} twice; it is not one price")
        prices[key] = int(cost.duration_ns)
    return prices


def declared_policies(table: M2CostTable) -> list[str]:
    """The history-invariant price policies the table's own entries declare."""
    policies = {(cost.physical_signature or {}).get("history_invariant_price_policy")
                for cost in table.entries.values()}
    policies.discard(None)
    return sorted(policies)


__all__ = [
    "BOUNDARY_KINDS", "FFN_STAGES", "MacroShape", "M2CostTable", "RESIDENT", "SHAPES",
    "STAGE_KINDS", "STREAMING", "declared_policies", "dump_json", "entry_signature",
    "now_utc", "parse_exported_table", "sha256_file", "shape_key", "table_prices",
    "table_regime",
]
