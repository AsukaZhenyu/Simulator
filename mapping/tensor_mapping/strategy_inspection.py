"""Non-executable strategy inspection, independent of performance prices.

This is a declaration and static-binding checker, not a second compiler. It
never constructs a tensor runtime, opens a cost table, or substitutes prices.
The report states the checks it performed and leaves graph-lowering,
performance coverage and runtime feasibility explicitly unproved.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path

from llm_infer_model.m2.spec import parse_invocation_document, parse_layer_document, validate_invocation
from llm_infer_model.m2.execution_pipeline import parse_transform
from llm_infer_model.tensor.spec import InvalidInput, Unsupported

from .composition_policy import parse_composition_policy
from .policy_ir import OPERATION_KINDS


FORMAT_VERSION = "strategy-inspection-0.1"
_FIELDS = {"format_version", "id", "model", "execution", "policy", "cache", "prefetch",
           "transport", "services_file", "environment_file", "history_file",
           "routing_scores_file", "max_decisions"}
_LABELS = {"layer_id", "invocation_id", "request_id", "kind", "phase"}


@dataclass(frozen=True)
class InspectionIssue:
    code: str
    path: str
    message: str


@dataclass(frozen=True)
class InspectionStage:
    status: str
    checked: tuple[str, ...] = ()
    issues: tuple[InspectionIssue, ...] = ()


@dataclass(frozen=True)
class StaticOperation:
    """Declared macro context, not a revealed/expanded executable operation."""
    layer_id: str
    invocation_id: str
    request_id: str
    kind: str
    phase: str
    allowed_devices: tuple[str, ...]


@dataclass(frozen=True)
class StrategyInspection:
    strategy_id: str | None
    declarations: InspectionStage
    static_bindings: InspectionStage
    cost_coverage: InspectionStage
    lowering: InspectionStage
    runtime_feasibility: InspectionStage
    operations: tuple[StaticOperation, ...]
    sources: tuple[dict, ...]
    cost_inputs: tuple[dict, ...]
    format_version: str = FORMAT_VERSION

    @property
    def executable(self):
        return False

    @property
    def semantic_validity(self):
        """A successful declaration check is not a compiler proof.

        This inspector never reports globally valid semantics. Invalid or
        unsupported declarations can be identified, but acceptance still needs
        the strict compiler's applicability/port/dependence checks.
        """
        if self.declarations.status in {"invalid", "unsupported"}:
            return self.declarations.status
        if self.static_bindings.status == "invalid":
            return "invalid"
        return "unknown"

    def document(self):
        return dict(asdict(self), executable=False, semantic_validity=self.semantic_validity,
                    claim_boundary="declarations and static macro bindings only; not compiler preservation or performance qualification")

    def render_text(self):
        """Explain checked and unchecked boundaries without treating unknown as pass."""
        rows = [f"Strategy inspection: {self.strategy_id or '<unidentified>'}",
                f"Semantic validity: {self.semantic_validity}",
                f"Declarations: {self.declarations.status}",
                f"Static macro bindings: {self.static_bindings.status}",
                f"Cost coverage: {self.cost_coverage.status}",
                f"Lowering: {self.lowering.status.replace('_', ' ')}",
                f"Runtime feasibility: {self.runtime_feasibility.status.replace('_', ' ')}",
                "Executable: no"]
        checks = tuple(dict.fromkeys((*self.declarations.checked, *self.static_bindings.checked)))
        if checks:
            rows.append("Checks performed:")
            rows.extend(f"  - {item}" for item in checks)
        issues = (*self.declarations.issues, *self.static_bindings.issues,
                  *self.cost_coverage.issues, *self.lowering.issues,
                  *self.runtime_feasibility.issues)
        if issues:
            rows.append("Findings and checks still required:")
            rows.extend(f"  - {issue.code} [{issue.path}]: {issue.message}" for issue in issues)
        if self.cost_inputs:
            rows.append("Cost inputs (presence only, not price verification):")
            rows.extend(f"  - {row['scope_id']}: {row['status']} [{row['source']}]" for row in self.cost_inputs)
        rows.append("A declaration pass does not establish graph-lowering correctness, measured accuracy, or runtime feasibility.")
        rows.append("Use the strict compiler before execution; no missing cost has been replaced by zero or an estimate.")
        return "\n".join(rows)

    def explain(self):
        return self.render_text()


def _exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise InvalidInput(f"{label} requires exactly {sorted(keys)}")


def _identity(value, label):
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n"):
        raise InvalidInput(f"{label} requires a nonempty single-line string")
    return value


def _mapping(value):
    return asdict(value) if is_dataclass(value) else deepcopy(value)


def _selector(value):
    value = _mapping(value)
    if not isinstance(value, dict) or set(value) - _LABELS:
        raise InvalidInput("selector contains unsupported labels")
    result = {key: value.get(key) for key in sorted(_LABELS)}
    for key, item in result.items():
        if item is not None:
            _identity(item, "selector." + key)
    if result["kind"] is not None and result["kind"] not in OPERATION_KINDS:
        raise Unsupported("unsupported selector.kind")
    if result["phase"] not in {None, "prefill", "decode", "synthetic_generation"}:
        raise Unsupported("unsupported selector.phase")
    return result


def _matches(selector, operation):
    return all(value is None or getattr(operation, key) == value for key, value in selector.items())


def _specializes(own, parent):
    return all(value is None or own[key] == value for key, value in parent.items())


def _parse_json(raw, path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise InvalidInput(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(InvalidInput(f"nonfinite JSON {value}")))
    except (UnicodeError, ValueError) as error:
        raise InvalidInput(f"invalid JSON in {path}: {error}") from error


def _without_prices(raw):
    """Keep choices and geometry; inline price values belong to another check."""
    if isinstance(raw, dict):
        return {key: [] if key in {"entries", "transfer_entries"} else _without_prices(value)
                for key, value in raw.items()}
    if isinstance(raw, list):
        return [_without_prices(value) for value in raw]
    return deepcopy(raw)


def inspect_strategy_document(document, *, base_dir, placements=(), modules=(),
                              default_device=None, visibility_mode="data_dependencies",
                              scoped_transport=()):
    """Check declared meaning without requiring performance data.

    ``placements``/``modules`` accept frontend dataclasses or their ``asdict``
    documents. For refined, MoE or block models, selector applicability may be
    incomplete and is reported as unknown. No report has a run() method, and
    ``executable`` is always false; the existing strict compiler is required.
    Missing model inputs prevent inspection, whereas missing cost inputs do not.
    Existing cost files are deliberately not opened, so presence is never called
    full cost coverage. This function performs no profiling or execution.
    ``scoped_transport`` is accepted as a frontend extension but its applicability
    and precedence are explicitly deferred to strict compilation.
    """
    root = Path(base_dir).resolve()
    snapshots, costs, operations, checked, deferred, bindings = {}, [], [], [], [], []
    catalogue_limits = []
    declaration_issue, binding_issue = [], []
    status = "valid"
    strategy_id = document.get("id") if isinstance(document, dict) else None
    if scoped_transport:
        deferred.append("scoped transport rule applicability, precedence and engine/service consistency")

    def read(path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if path in snapshots and snapshots[path] != digest:
            raise InvalidInput(f"source changed during inspection: {path}")
        snapshots[path] = digest
        return _parse_json(raw, path)

    def cost_input(invocation, path, scope):
        if invocation.costs_file is None:
            costs.append(dict(scope_id=scope, source="inline", status="present_unverified"))
        else:
            source = (path.parent / invocation.costs_file).resolve()
            costs.append(dict(scope_id=scope, source=str(source),
                              status="present_unverified" if source.is_file() else "missing"))

    def invoke(path, scope):
        raw = read(path)
        invocation = parse_invocation_document(raw, where=str(path))
        if invocation.layer_file is None:
            raise Unsupported("inspection requires an explicit layer_file")
        layer = parse_layer_document(read(path.parent / invocation.layer_file), where=invocation.layer_file)
        validate_invocation(layer, invocation)
        cost_input(invocation, path, scope)
        if layer.ffn == "moe":
            deferred.append("MoE dynamic task and router-bound selector applicability")
            catalogue_limits.append("MoE dynamic task catalogue")
        for request in invocation.requests:
            for task in layer.tasks:
                operations.append(StaticOperation(layer.id, scope, request.request_id, task.kind,
                                                  invocation.phase, invocation.devices))
        return invocation

    try:
        if not isinstance(document, dict):
            raise InvalidInput("strategy must be an object")
        version = document.get("format_version")
        if version not in {"unified-strategy-0.2", "unified-strategy-0.3"}:
            raise Unsupported("inspection supports unified-strategy-0.2 and 0.3")
        _exact(document, _FIELDS | ({"reconfiguration"} if version.endswith("0.3") else set()), "strategy")
        _identity(document["id"], "strategy.id")
        if type(document["max_decisions"]) is not int or document["max_decisions"] < 1:
            raise InvalidInput("max_decisions must be a positive integer")
        for field in ("services_file", "environment_file", "history_file", "routing_scores_file"):
            if document[field] is not None or field == "environment_file":
                _identity(document[field], field)
        if visibility_mode not in {"scope_barriers", "data_dependencies"}:
            raise Unsupported("unsupported visibility_mode")
        if default_device not in {None, "cpu", "gpu"}:
            raise Unsupported("default_device must be cpu, gpu or None")
        checked.append("strategy document declaration schema")
        descriptor = document["model"]
        if not isinstance(descriptor, dict):
            raise InvalidInput("model must be an object")
        kind = descriptor.get("kind")
        _exact(descriptor, {"kind", "file", "override_model_policy"} if kind == "block" else {"kind", "file"}, "model")
        _identity(descriptor["file"], "model.file")
        model_path = root / descriptor["file"]
        scopes = {}
        if kind == "m2":
            scopes["single"] = (model_path.resolve(), invoke(model_path, "single"))
        elif kind == "global_m2":
            model = read(model_path)
            model_version=model.get("format_version")
            if model_version=="global-execution-0.1":
                _exact(model, {"format_version", "id", "rounds"}, "global model")
                if not isinstance(model["rounds"],list) or not model["rounds"]:
                    raise InvalidInput("global model requires nonempty rounds")
                round_rows=[(i,round_,None) for i,round_ in enumerate(model["rounds"])]
            elif model_version=="global-execution-0.2":
                from llm_infer_model.m2.request_flow import parse_request_flow
                parse_request_flow(model)
                round_rows=[(i,step,request["request_id"]) for request in model["requests"]
                            for i,step in enumerate(request["steps"])]
                checked.append("finite request arrival and independent step declaration schema")
                deferred.append("request arrival allocation, retirement, token gates and all-layer P/D handoff lowering")
            else:
                raise Unsupported("unsupported global model")
            _identity(model["id"], "global model.id")
            round_ids = set()
            for i, round_,request_id in round_rows:
                _exact(round_, {"id", "input_mode", "gate_cost" if model_version=="global-execution-0.1" else "gate_costs", "invocations"}, "round")
                rid = _identity(round_["id"], "round.id")
                identity=(request_id,rid)
                if identity in round_ids:
                    raise InvalidInput("duplicate global round id")
                round_ids.add(identity)
                mode = "initial_supplied_hidden" if i == 0 else "trace_supplied_next_token"
                if round_["input_mode"] != mode:
                    raise InvalidInput("global round has an inconsistent input_mode")
                if not isinstance(round_["invocations"], list) or not round_["invocations"]:
                    raise InvalidInput("global round requires nonempty invocations")
                for entry in round_["invocations"]:
                    _exact(entry, {"id", "invocation_file", "execution_file"}, "global invocation")
                    sid = _identity(entry["id"], "scope id")
                    if sid in scopes:
                        raise InvalidInput("duplicate invocation scope")
                    path = model_path.parent / _identity(entry["invocation_file"], "invocation_file")
                    scopes[sid] = (path.resolve(), invoke(path, sid))
                    # An inline binding replaces the inherited transform. Do
                    # not read an obsolete source that strict compilation also
                    # ignores (its path may intentionally no longer exist).
                    overridden = isinstance(document["execution"], list) and any(
                        isinstance(row, dict) and row.get("scope_id") == sid
                        for row in document["execution"])
                    if entry["execution_file"] is not None and not overridden:
                        parse_transform(_without_prices(read(model_path.parent / _identity(entry["execution_file"], "execution_file"))))
                        deferred.append("inherited execution transform applicability and lowered dataflow")
                        catalogue_limits.append("inherited execution transform")
            deferred.append("cross-invocation hidden/KV/trace-gate connection preservation")
        elif kind == "block":
            if descriptor["override_model_policy"] is not True or visibility_mode != "scope_barriers":
                raise Unsupported("block inspection requires explicit override and scope_barriers")
            read(model_path)
            deferred.append("block task catalogue and traversal legality")
            catalogue_limits.append("block catalogue")
        else:
            raise Unsupported("model.kind must be m2, global_m2 or block")
        if kind in {"m2", "global_m2"}:
            checked.extend(("strict model/layer/invocation declarations", "layer dependency DAG and per-invocation state contract"))
        else:
            checked.append("block source JSON syntax only; block schema remains unchecked")
        if not isinstance(document["execution"], list):
            raise InvalidInput("execution must be an explicit list")
        seen = set()
        for row in document["execution"]:
            _exact(row, {"scope_id", "spec"}, "execution binding")
            sid = row["scope_id"]
            if sid not in scopes or sid in seen:
                raise InvalidInput("execution target must be a distinct existing invocation scope")
            seen.add(sid)
            parse_transform(_without_prices(row["spec"]))
            deferred.append("execution transform target applicability, ports and lowered dataflow")
            catalogue_limits.append("execution transform")
        checked.append("execution transform choice/geometry schema, without inline price validation")
        for module in modules:
            raw = _mapping(module)
            _exact(raw, {"path", "selector"}, "module")
            _selector(raw["selector"])
        for binding in placements:
            raw = _mapping(binding)
            _exact(raw, {"selector", "device", "priority", "module_path"}, "placement")
            if raw["device"] not in {"cpu", "gpu"} or type(raw["priority"]) is not int:
                raise InvalidInput("placement requires cpu/gpu and an integer priority")
            bindings.append((raw, _selector(raw["selector"])))
        # The normal policy parser is price-independent. It requires a nonempty
        # placement table, so derive only explicit default/scoped declarations
        # for static macro tasks; never invent a device to satisfy the parser.
        policy = deepcopy(document["policy"])
        _exact(policy, {"placements", "request_placements", "experts", "release"}, "policy")
        if not isinstance(policy["placements"], dict):
            raise InvalidInput("policy.placements must be an object")
        if not isinstance(policy["request_placements"], dict) or any(
                not isinstance(value, dict) for value in policy["request_placements"].values()):
            raise InvalidInput("policy.request_placements must contain objects")
        derived = dict(policy["placements"])
        if not catalogue_limits:
            for op in operations:
                matches = [(row, sel) for row, sel in bindings if _matches(sel, op)]
                if matches:
                    priority = max(row["priority"] for row, _ in matches)
                    matches = [(row, sel) for row, sel in matches if row["priority"] == priority]
                    winners = [(row, sel) for row, sel in matches if not any(
                        other != sel and _specializes(other, sel) for _, other in matches)]
                    devices = {row["device"] for row, _ in winners}
                    if len(devices) != 1:
                        raise InvalidInput("ambiguous scoped placements; supply explicit priority")
                    selected = next(iter(devices))
                    derived.setdefault(op.kind, "scoped_binding")
                else:
                    selected = policy.get("request_placements", {}).get(op.request_id, {}).get(op.kind,
                                policy["placements"].get(op.kind, default_device))
                    if selected is not None:
                        derived.setdefault(op.kind, selected)
                if selected not in op.allowed_devices:
                    raise InvalidInput(f"no explicit admissible placement for {op.invocation_id}/{op.kind}")
            for row, selector in bindings:
                if not any(_matches(selector, op) for op in operations):
                    raise InvalidInput(f"placement selector matches no declared task: {selector}")
        elif bindings:
            deferred.append("selector coverage and precedence after refinement/dynamic expansion")
        policy["placements"] = derived
        if derived:
            parse_composition_policy(dict(policy, cache=document["cache"], prefetch=document["prefetch"], transport=document["transport"]))
            checked.append("policy/cache/prefetch/transport declaration contracts")
        else:
            deferred.append("placement completion and policy parsing after lowering")
        if version == "unified-strategy-0.3":
            from .reconfiguration import parse_reconfiguration
            parse_reconfiguration(document["reconfiguration"])
        for path, digest in snapshots.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise InvalidInput(f"source changed during inspection: {path}")
    except FileNotFoundError as error:
        status = "unknown"
        declaration_issue.append(InspectionIssue("missing_model_input", str(error.filename), str(error)))
    except Unsupported as error:
        status = "unsupported"
        declaration_issue.append(InspectionIssue("unsupported_declaration", "strategy", str(error)))
    except (InvalidInput, TypeError) as error:
        status = "invalid"
        declaration_issue.append(InspectionIssue("invalid_declaration", "strategy", str(error)))
    missing = tuple(InspectionIssue("missing_cost_input", row["source"], "Referenced cost input is absent; no price was substituted")
                    for row in costs if row["status"] == "missing")
    binding_status = "not_checked" if status != "valid" else "unknown" if catalogue_limits else "valid"
    binding_checks = () if binding_status != "valid" else ("static macro selector coverage, specificity, priority and device admissibility",)
    return StrategyInspection(strategy_id, InspectionStage(status, tuple(checked), tuple(declaration_issue)),
        InspectionStage(binding_status, binding_checks, tuple(binding_issue)),
        InspectionStage("missing" if missing else "unknown", (), missing),
        InspectionStage("not_checked", (), tuple(InspectionIssue("requires_strict_compile", "compiler", item)
                                                 for item in dict.fromkeys(deferred))),
        InspectionStage("not_run"), tuple(operations),
        tuple(dict(path=str(path), sha256=digest) for path, digest in sorted(snapshots.items(), key=lambda row: str(row[0]))),
        tuple(costs))
