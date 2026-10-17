"""Explicit, causal resource response; this is a controller, not a searcher.

The environment owns notices and future events. This module can only act on
the model's already public request, physical ledger and safe release list.
"""
from dataclasses import asdict, dataclass

from llm_infer_model.tensor.spec import InvalidInput, Unsupported


@dataclass(frozen=True)
class ReconfigurationSpec:
    reclaim_response: str
    release_order: str
    cpu_response: str


@dataclass(frozen=True)
class ParsedEnvironment:
    events: tuple
    requests: tuple
    control_costs: object


def _exact(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise InvalidInput(f"{label} requires exactly {sorted(fields)}")


def parse_reconfiguration(document):
    _exact(document, {"reclaim_response", "release_order", "cpu_response"}, "reconfiguration")
    for field, choices in (
            ("reclaim_response", {"release_clean_weights", "decline"}),
            ("release_order", {"lru", "largest_first", "history_frequency"}),
            ("cpu_response", {"fixed", "current_cost"})):
        if not isinstance(document[field], str) or document[field] not in choices:
            raise Unsupported(f"reconfiguration.{field} must be one of {sorted(choices)}")
    return ReconfigurationSpec(**document)


def reconfiguration_document(spec):
    if not isinstance(spec, ReconfigurationSpec):
        raise InvalidInput("reconfiguration requires a parsed ReconfigurationSpec")
    return asdict(spec)


def parse_environment_trace(document):
    """Keep legacy events exact; new requests require an explicit cost contract."""
    from .causal_cli import parse_environment
    from llm_infer_model.tensor.environment import ControlCosts, EnvironmentReclaimRequest
    if isinstance(document, dict) and document.get("format_version") == "environment-trace-0.1":
        return ParsedEnvironment(parse_environment(document), (), None)
    _exact(document, {"format_version", "events", "requests", "control_costs"}, "environment-trace-0.2")
    if document["format_version"] != "environment-trace-0.2":
        raise Unsupported("environment requires environment-trace-0.1 or environment-trace-0.2")
    events = parse_environment({"format_version": "environment-trace-0.1", "events": document["events"]})
    if not isinstance(document["requests"], list):
        raise InvalidInput("environment.requests must be an explicit array")
    requests = []
    for index, row in enumerate(document["requests"]):
        _exact(row, {"id", "domain", "t_notice_ns", "target_external_bytes", "deadline_ns"},
               f"environment.requests[{index}]")
        requests.append(EnvironmentReclaimRequest(**row))
    costs = document["control_costs"]
    _exact(costs, {"ack_ns", "decline_ns", "decision_ns", "source", "timing_scope"}, "environment.control_costs")
    return ParsedEnvironment(events, tuple(requests), ControlCosts(**costs))
