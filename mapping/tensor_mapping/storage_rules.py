"""Static tensor-owner rules over the existing shared storage controller.

This first scope extension selects PCIe engines for immutable model weights.
It does not assign a request/phase owner to a reused physical tensor, create
per-module capacity, or make a copy to satisfy a rule. All work still goes
through the same modeling inventory, service arbitration and cache budget.
"""
from dataclasses import asdict, dataclass, field

from llm_infer_model.tensor.spec import InvalidInput, Unsupported


def _name(value, label, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value or any(c in value for c in "\r\n"):
        raise InvalidInput(f"{label} must be a nonempty single-line string")


@dataclass(frozen=True)
class TensorSelect:
    """Conjunction over a public (tensor, layer, stage, expert) weight row.

    None means no constraint. Expert identity describes static ownership,
    never whether that expert will be selected by a future router.
    """
    layer_id: str | None = None
    stage: str | None = None
    tensor_id: str | None = None
    expert_id: int | str | None = None

    def __post_init__(self):
        for key in ("layer_id", "stage", "tensor_id"):
            _name(getattr(self, key), f"tensor selector.{key}", optional=True)
        if self.expert_id is not None and not (
                type(self.expert_id) is int and self.expert_id >= 0 or self.expert_id == "shared"):
            raise InvalidInput("tensor selector.expert_id must be a nonnegative integer, 'shared', or None")

    def document(self):
        return asdict(self)

    def intersect(self, other):
        if not isinstance(other, TensorSelect):
            raise InvalidInput("tensor selectors must be TensorSelect objects")
        values = {}
        for key, value in asdict(self).items():
            added = getattr(other, key)
            if value is not None and added is not None and value != added:
                raise InvalidInput(f"contradictory tensor selector {key}: {value!r}/{added!r}")
            values[key] = value if added is None else added
        return TensorSelect(**values)

    def specializes(self, other):
        if not isinstance(other, TensorSelect):
            raise InvalidInput("tensor selectors must be TensorSelect objects")
        return all(value is None or getattr(self, key) == value for key, value in asdict(other).items())

    def matches(self, row):
        if not isinstance(row, tuple) or len(row) != 4:
            raise InvalidInput("tensor owner row must be a frozen (tensor,layer,stage,expert) tuple")
        tensor_id, layer_id, stage, expert_id = row
        return all(selected is None or selected == actual for selected, actual in (
            (self.layer_id, layer_id), (self.stage, stage),
            (self.tensor_id, tensor_id), (self.expert_id, expert_id)))


@dataclass(frozen=True)
class ScopedTransportRule:
    role: str
    src: str
    dst: str
    engine: str | None
    selector: TensorSelect = field(default_factory=TensorSelect)
    priority: int = 0
    module_path: tuple[str, ...] = ()

    def __post_init__(self):
        if self.role != "weight":
            raise Unsupported("scoped transport currently selects static weight ownership only; "
                              "other roles retain the global role/stage transport rules")
        if (self.src, self.dst) not in (("dram", "vram"), ("vram", "dram")):
            raise Unsupported("scoped transport binds dram/vram PCIe directions only")
        if self.engine is not None and self.engine not in ("ce", "sm"):
            raise Unsupported("scoped transport engine must be explicit None, ce or sm")
        if not isinstance(self.selector, TensorSelect):
            raise InvalidInput("scoped transport requires a TensorSelect")
        if type(self.priority) is not int:
            raise InvalidInput("scoped transport priority must be an integer")
        if not isinstance(self.module_path, tuple):
            raise InvalidInput("scoped transport module_path must be an immutable tuple")
        for name in self.module_path:
            _name(name, "scoped transport module path")

    def document(self):
        result = asdict(self)
        result["module_path"] = list(self.module_path)
        return result

    @classmethod
    def from_document(cls, document):
        if not isinstance(document, dict) or set(document) != {
                "role", "src", "dst", "engine", "selector", "priority", "module_path"}:
            raise InvalidInput("scoped transport requires role, src, dst, engine, selector, priority and module_path")
        selected = document["selector"]
        if not isinstance(selected, dict) or set(selected) != {"layer_id", "stage", "tensor_id", "expert_id"}:
            raise InvalidInput("tensor selector requires exactly layer_id, stage, tensor_id and expert_id")
        if not isinstance(document["module_path"], list):
            raise InvalidInput("serialized scoped transport module_path must be an array")
        return cls(**(document | {"selector": TensorSelect(**selected),
                                  "module_path": tuple(document["module_path"])}))


def normalize_scoped_transport(rules):
    if not isinstance(rules, (tuple, list)):
        raise InvalidInput("scoped transport must be a finite tuple or list of typed rules")
    if any(not isinstance(rule, ScopedTransportRule) for rule in rules):
        raise InvalidInput("scoped transport entries must be ScopedTransportRule objects")
    # Revalidate frozen objects through one canonical serialized contract.
    return tuple(ScopedTransportRule.from_document(rule.document()) for rule in rules)


def resolve_scoped_transport(rules, *, tensor_id, role, src, dst,
                             layer_id=None, stage=None, expert_id=None):
    """Return one authoritative rule, or None to use the global fallback.

    Priority is explicit. At equal priority, strict specialization wins. The
    number of labels is deliberately not a tie-breaker for incomparable scopes.
    """
    row = (tensor_id, layer_id, stage, expert_id)
    matches = [rule for rule in rules if rule.role == role and rule.src == src and rule.dst == dst
               and rule.selector.matches(row)]
    if not matches:
        return None
    priority = max(rule.priority for rule in matches)
    matches = [rule for rule in matches if rule.priority == priority]
    winners = [rule for rule in matches if not any(
        other.selector != rule.selector and other.selector.specializes(rule.selector)
        for other in matches)]
    if len({rule.engine for rule in winners}) != 1:
        raise InvalidInput(f"ambiguous scoped transport for {tensor_id}/{src}->{dst}; use explicit priority")
    return winners[0]


def validate_scoped_transport_bindings(rules, catalogue):
    """Check static declarations without publishing future task/route data."""
    if not isinstance(catalogue, tuple):
        raise InvalidInput("scoped transport requires a frozen public weight catalogue")
    for row in catalogue:
        if not isinstance(row, tuple) or len(row) != 4:
            raise InvalidInput("public weight catalogue rows must be frozen four-tuples")
    for rule in rules:
        if not any(rule.selector.matches(row) for row in catalogue):
            raise InvalidInput(f"scoped transport selector matches no public model weight: {rule.selector.document()}")
    for tensor_id, layer_id, stage, expert_id in catalogue:
        for src, dst in (("dram", "vram"), ("vram", "dram")):
            resolve_scoped_transport(rules, tensor_id=tensor_id, role="weight", src=src, dst=dst,
                                     layer_id=layer_id, stage=stage, expert_id=expert_id)
