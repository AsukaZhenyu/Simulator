"""Re-price a declared cost table with one pooled median per history-invariant stage.

An already exported table files the projection and the feed-forward at decode
once per visible_history, although those stages read one token and no history.
This tool applies the same rule the exporter now offers at export time
(``--history-invariant-price median_of_history_replicates``) to a table that
already exists, so a campaign measured before the rule can be re-priced without
another hardware run.

It is a declaration, not a measurement: every pooled price comes from entries
already in the table, the entries it replaced stay in the report, and each
re-priced entry says in its signature which estimator produced its price.  A
table this tool has not touched is byte-identical to its input.

Refusals (never a silent guess):
  * the table mixes contracts;
  * the offered entries are not replicates -- they differ in anything except
    visible_history, their duration and their own signature;
  * fewer than two depths, or a spread past the pooling limit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

MODELING = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))

import contracts
from history_invariant_prices import (ReplicateError, pool_history_invariant,
                                      require_replicates)
from llm_infer_model.m2.costs import canonical_contract_hash, parse_cost_table
from llm_infer_model.tensor.spec import InvalidInput


class RepriceError(ValueError):
    """The table cannot be re-priced under the pooled-median rule."""


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _signature_inputs(signature: dict, contract) -> dict:
    """Recover the arguments that produced a stored signature, verbatim."""
    residency = signature.get("residency_regime")
    if residency is not None:
        short = [name for name, text in contract.RESIDENCY_REGIMES.items() if text == residency]
        if len(short) != 1:
            raise RepriceError(f"unknown residency regime in the stored signature: {residency!r}")
        residency = short[0]
    return {"config": signature["config"], "kind": signature["kind"],
            "phase": signature["phase"], "tokens": signature["tokens"],
            "visible_history": signature["visible_history"], "threads": signature["threads_requested"],
            "batch": signature["batch"], "residency": residency,
            "process_scope": contract.declared_process_scope(
                signature.get("measurement_process_scope")),
            "cpu_ids": signature.get("cpu_mask")}


def _rebuild(signature: dict, contract, *, price_policy=None) -> dict:
    inputs = _signature_inputs(signature, contract)
    threads = inputs.pop("threads")
    return contract.signature(inputs.pop("config"), inputs.pop("kind"), inputs.pop("phase"),
                              inputs.pop("tokens"), inputs.pop("visible_history"), threads,
                              price_policy=price_policy, **inputs)


def reprice(table: dict, policy: str) -> tuple[dict, dict]:
    """Return (re-priced table, report).  Never mutates ``table``."""
    parse_cost_table(table)
    entries = copy.deepcopy(table["entries"])
    contracts_seen = {entry["physical_signature"]["contract"] for entry in entries}
    if len(contracts_seen) != 1:
        raise RepriceError(f"table mixes contracts: {sorted(contracts_seen)}")
    contract = contracts.for_contract(contracts_seen.pop())
    kinds = set(contract.history_invariant_kinds())
    label = contract.PRICE_POLICIES[policy]
    for entry in entries:
        stored = entry["physical_signature"]
        if _json(_rebuild(stored, contract)) != _json(stored):
            raise RepriceError(f"{entry['kind']}/{entry['visible_history']}: the stored signature "
                               "is not the one its own declarations produce")
        if entry.get("contract_hash") != canonical_contract_hash(stored):
            raise RepriceError(f"{entry['kind']}/{entry['visible_history']}: contract hash "
                               "does not match its signature")

    groups: dict[tuple[str, int, str], list[dict]] = {}
    for entry in entries:
        if entry["phase"] != "decode" or entry["kind"] not in kinds:
            continue
        groups.setdefault((entry["kind"], entry["tokens"], entry["device"]), []).append(entry)

    replaced = []
    for key in sorted(groups):
        members = groups[key]
        if len({entry["visible_history"] for entry in members}) < 2:
            continue
        try:
            require_replicates(members, f"{key[0]}/tokens {key[1]}")
            pooled = pool_history_invariant(
                [(entry["visible_history"], entry["visible_history"], entry["duration_ns"])
                 for entry in members])
        except ReplicateError as error:
            raise RepriceError(f"{key[0]}/{key[1]}: {error}") from None
        replacements = []
        for entry in members:
            stored = entry["physical_signature"]
            repriced = _rebuild(stored, contract, price_policy=policy)
            replacements.append({"visible_history": entry["visible_history"],
                                 "duration_ns_replaced": entry["duration_ns"],
                                 "workspace_bytes": entry["workspace_bytes"]})
            entry["duration_ns"] = pooled["pooled_ns"]
            entry["physical_signature"] = repriced
            entry["contract_hash"] = canonical_contract_hash(repriced)
        replaced.append({"kind": key[0], "tokens": key[1], "device": key[2],
                         "pooled_duration_ns": pooled["pooled_ns"],
                         "relative_spread": pooled["relative_spread"],
                         "replacements": replacements})

    result = copy.deepcopy(table)
    result["entries"] = entries
    parse_cost_table(result)
    digest = hashlib.sha256(_json(entries).encode("utf-8")).hexdigest()
    result["source"] = f"{table['source']}-replicated-median-{digest[:16]}"
    result["comment"] = (table.get("comment", "") + " Re-priced by profiling/reprice_history_invariant.py: "
                         f"{len(replaced)} history-invariant decode groups priced by the pooled median of "
                         "their depth replicates, declared in each signature; no new measurement.").strip()
    report = {"schema_version": "m2_history_invariant_reprice_v1",
              "input_source": table["source"], "output_source": result["source"],
              "price_policy": label, "history_invariant_kinds": sorted(kinds),
              "contract": contract.CONTRACT, "groups": replaced,
              "entries_changed": sum(len(group["replacements"]) for group in replaced),
              "measurement_performed": False,
              "entries_canonical_sha256": digest}
    return result, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", default="median_of_history_replicates",
                        choices=("median_of_history_replicates",))
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RepriceError(f"output directory must be empty: {output}")
    table = json.loads(args.table.resolve().read_text(encoding="utf-8"))
    reprice_result, report = reprice(table, args.policy)
    report["input"] = {"table": str(args.table.resolve()),
                       "sha256": hashlib.sha256(args.table.resolve().read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    for name, document in (("costs.json", reprice_result), ("reprice_report.json", report)):
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"Re-priced {report['entries_changed']} entries in {len(report['groups'])} groups "
          f"to {output}; no measurement performed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, InvalidInput) as error:
        print(f"reprice_history_invariant: {error}", file=sys.stderr)
        raise SystemExit(1)
