"""Finite-grid warm block-average transfer qualification, with unchanged prices.

Receipt contents and parameter/source provenance are verified by profiling.
This runtime performs no filesystem IO and makes no sampling-equivalence claim.
"""
from dataclasses import dataclass
import json
import re

from .prepared_transfer import validate_contract, validate_parameters
from .transfer_response import inclusive_response_ns


@dataclass(frozen=True)
class ValidatedTransferCatalog:
    _manifest_json: str

    @classmethod
    def from_manifest(cls, manifest):
        value = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
        if value.get("schema") != "prepared-transfer-validated-catalog-1":
            raise ValueError("validated transfer catalog schema differs")
        if not isinstance(value.get("evidence_scope"), str) or not value["evidence_scope"].strip():
            raise ValueError("explicit finite qualification scope required")
        entries = value.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("memory-mode keyed qualified entries required")
        for memory, entry in entries.items():
            source = validate_contract(entry["source_contract"])
            if source["memory_mode"] != memory or source["preparation_contract"]["timed_replays_per_batch"] != 256:
                raise ValueError("source memory mode or frozen 256-replay contract differs")
            prep = source["preparation_contract"]
            if (prep["reset_before_batch"] or prep["untimed_transfer_prime_passes"] < 1
                    or prep["warmup_iterations_min"] < 1 or prep["warmup_ms_min"] < 1):
                raise ValueError("only explicitly warmed and primed component-source contracts are supported")
            validate_parameters(entry["parameters"], entry["minimum_bytes"], entry["maximum_bytes"])
            if entry.get("qualified") is not True:
                raise ValueError("independent finite-grid qualification required")
            receipt = entry.get("qualification_receipt")
            if (not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str)
                    or not receipt["path"].strip() or not isinstance(receipt.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"])):
                raise ValueError("qualification receipt path and SHA256 required")
            grid = entry.get("validated_replays")
            if not isinstance(grid, dict) or not grid:
                raise ValueError("explicit validated byte/replay grid required")
            for size, repeats in grid.items():
                if (not re.fullmatch(r"[1-9][0-9]*", size)
                        or not entry["minimum_bytes"] <= int(size) <= entry["maximum_bytes"]
                        or type(repeats) is not int or repeats < 2):
                    raise ValueError("invalid validated block-average byte/replay coordinate")
        return cls(json.dumps(value, sort_keys=True, allow_nan=False))

    def prediction(self, size_bytes, *, contract):
        target = validate_contract(contract)
        value = json.loads(self._manifest_json)
        entry = value["entries"].get(target["memory_mode"])
        if entry is None or type(size_bytes) is not int:
            raise ValueError("requested memory/byte coordinate lacks qualification")
        repeats = entry["validated_replays"].get(str(size_bytes))
        if repeats is None:
            raise ValueError("byte size outside the independently validated finite grid")
        expected = entry["source_contract"]
        expected["preparation_contract"]["timed_replays_per_batch"] = repeats
        if target != expected:
            raise ValueError("requested transfer API, preparation, sampling or runtime differs")
        return dict(duration_ns=inclusive_response_ns(entry["parameters"], size_bytes),
            qualified_for_requested_contract=True,
            price_scope="warm block-average completed response", source_replays=256,
            requested_replays=repeats, coefficients_changed=False,
            sampling_equivalence_claimed=False, physicalbandwidthidentified=False,
            diagnostic_phase_or_event_cost_added=False,
            qualification_receipt=entry["qualification_receipt"], evidence_scope=value["evidence_scope"])

