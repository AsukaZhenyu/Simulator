"""Structural feature permissions for inclusive native execution windows.

These declarations retain actual node metadata. They identify which measured
response terms have a structural justification; they do not identify isolated
device service or invent cache/OS state. Unclassified work remains explicit.
The GGML numeric operation IDs below are the pinned b8705 metadata contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from .parameters import InvalidModel, nonempty


SCHEMA = "actual-window-work-1"
INTERCEPT = "intercept"
QUERY = "query_tokens"
ATTENTION = "query_tokens_times_KV"
PAYLOAD = "payload_bytes"
FEATURES = (INTERCEPT, QUERY, ATTENTION, PAYLOAD)
WORK_SCOPES = frozenset({"CPU_NODE", "RUNTIME_BRIDGE", "HOST", "SAMPLER"})
HOST_STAGES = frozenset({"ordinary_host", "attention_mask_prepare"})
# A bridge window can carry the activation produced by a host-resident stage it
# is serialized behind. The dependency names that stage class; the evidence
# names the actual producer node, so the permission is never granted by name
# alone or by the caller's strategy label.
DEPENDENCIES = frozenset({"cpu_attention_stage"})
CPU_ATTENTION_STAGE = "cpu_attention_stage"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidModel(name + " must be an observed integer >= " + str(minimum))


def _shape(value, name):
    if type(value) is not tuple or len(value) != 4:
        raise InvalidModel(name + " requires all four observed GGML dimensions")
    for item in value:
        _integer(item, name)


@dataclass(frozen=True)
class WindowOperator:
    name: str
    op: int
    ne: tuple[int, ...]
    type: int
    weight_ne: tuple[int, ...]
    weight_type: int

    def __post_init__(self):
        nonempty(self.name, "actual node name")
        _integer(self.op, "actual GGML op")
        _integer(self.type, "actual node dtype")
        _integer(self.weight_type, "actual source-zero dtype", -1)
        _shape(self.ne, "actual node shape")
        _shape(self.weight_ne, "actual source-zero shape")
        if self.op in (29, 30) and (not all(self.weight_ne) or self.weight_type < 0):
            raise InvalidModel("matrix work requires observed nonempty source-zero geometry/dtype")

    @property
    def role(self):
        # Normal non-final Prefill chunks can retain zero-row pruned nodes in
        # the GPU graph metadata. Keep their occurrence/shape evidence without
        # assigning matrix or attention work that was not dispatched.
        if 0 in self.ne:
            return "pruned_zero_output"
        # Names alone are insufficient: require their actual dispatched op.
        if (self.op == 29 and self.name.startswith(("kq-", "kqv-"))
                or self.op == 46 and self.name.startswith("kq_soft_max-")):
            return "attention_core"
        if self.op in (29, 30):
            return "matrix"
        if (self.op == 95 and self.name.startswith("ffn_moe_swiglu-")
                or self.op == 25 and self.name.startswith("norm-")):
            return "pointwise"
        return "other_compute"

    @property
    def query_axis(self):
        if self.role in {"matrix", "attention_core"}:
            return 2 if self.op == 30 else 1
        if self.role == "pointwise":
            if self.op == 95:
                return 2
            # A norm reduces along axis 0 and is applied once per row of the
            # axes above it. An ordinary layer norm carries those rows on axis
            # 1 ([hidden, rows, 1, 1]); a per-head QK norm carries the heads on
            # axis 1 and the rows on axis 2 ([head_dim, heads, rows, 1]). The
            # row axis is the outermost axis that is not a unit: naming any
            # other axis makes the row count of one request -- its own prompt
            # chunk length -- part of the work signature, and no completed
            # window of another length can then be priced.
            for axis in (3, 2, 1):
                if self.ne[axis] > 1:
                    return axis
            return 1
        return None

    def structural_signature(self):
        value = dict(name=self.name, op=self.op, type=self.type,
                     weight_type=self.weight_type, role=self.role)
        # Ordinary matrix source-zero is the static weight. Attention source
        # geometry contains dynamic K; retain it as evidence, not a key axis.
        if self.role == "matrix":
            value["weight_ne"] = list(self.weight_ne)
            value["fixed_output_axes"] = list(self.ne[2:] if self.op == 29 else (self.ne[1], self.ne[3]))
        if self.role == "pointwise":
            value["fixed_output_axes"] = [v for axis, v in enumerate(self.ne) if axis != self.query_axis]
        if self.role == "attention_core":
            value["output_heads_batch"] = list(self.ne[2:])
            if self.op == 29:
                static_axis = 1 if self.name.startswith("kqv-") else 0
                value["source_static_axes"] = [self.weight_ne[static_axis], *self.weight_ne[2:]]
                if self.name.startswith("kqv-"):
                    value["output_head_dimension"] = self.ne[0]
        if self.role == "pruned_zero_output":
            value["zero_output_shape"] = list(self.ne)
            if self.op in (29, 30):
                value["weight_ne"] = list(self.weight_ne)
        return value

    def to_dict(self):
        return dict(name=self.name, op=self.op, ne=list(self.ne), type=self.type,
                    weight_ne=list(self.weight_ne), weight_type=self.weight_type)

    @classmethod
    def from_metadata(cls, value):
        if not isinstance(value, dict) or set(value) != {"name", "op", "ne", "type", "weight_ne", "weight_type"}:
            raise InvalidModel("complete actual node metadata required; unknown fields are not ignored")
        obj = dict(value)
        for key in ("ne", "weight_ne"):
            if not isinstance(obj[key], (list, tuple)):
                raise InvalidModel("actual node dimensions must be an explicit sequence")
            obj[key] = tuple(obj[key])
        return cls(**obj)


@dataclass(frozen=True)
class WindowWorkSpec:
    scope: str
    operators: tuple[WindowOperator, ...]
    copy_calls: int
    host_stage: str | None = None
    host_stage_evidence: tuple[str, ...] = ()
    dependency: str | None = None
    dependency_evidence: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.scope, str) or self.scope not in WORK_SCOPES:
            raise InvalidModel("unknown actual-window work scope")
        if type(self.operators) is not tuple or any(not isinstance(v, WindowOperator) for v in self.operators):
            raise InvalidModel("actual operators must be an explicit WindowOperator tuple")
        _integer(self.copy_calls, "actual copy-call count")
        if self.scope == "CPU_NODE" and len(self.operators) != 1:
            raise InvalidModel("CPU_NODE work requires exactly one actual node")
        if self.scope == "CPU_NODE" and self.operators[0].role == "pruned_zero_output":
            raise InvalidModel("zero-output graph metadata is not an actual dispatched CPU node")
        if self.scope == "RUNTIME_BRIDGE" and not self.operators:
            raise InvalidModel("runtime bridge work requires its actual GPU nodes")
        if self.scope in {"HOST", "SAMPLER"} and self.operators:
            raise InvalidModel("host/sampler work cannot hide covered compute nodes")
        if self.scope not in {"RUNTIME_BRIDGE", "HOST"} and self.copy_calls:
            raise InvalidModel("copy response terms require a measured bridge or host-copy window")
        if type(self.host_stage_evidence) is not tuple:
            raise InvalidModel("host-stage evidence must be an explicit tuple")
        for value in self.host_stage_evidence:
            nonempty(value, "host-stage evidence")
        if self.host_stage is not None:
            if self.scope != "HOST" or not isinstance(self.host_stage, str) or self.host_stage not in HOST_STAGES or not self.host_stage_evidence:
                raise InvalidModel("identified host work requires an allowed stage and actual stage evidence")
        elif self.host_stage_evidence:
            raise InvalidModel("host-stage evidence requires an identified stage")
        if type(self.dependency_evidence) is not tuple:
            raise InvalidModel("ordering-dependency evidence must be an explicit tuple")
        for value in self.dependency_evidence:
            nonempty(value, "ordering-dependency evidence")
        if self.dependency is not None:
            if not isinstance(self.dependency, str) or self.dependency not in DEPENDENCIES:
                raise InvalidModel("unknown ordering dependency for a completed window")
            if self.scope != "RUNTIME_BRIDGE":
                raise InvalidModel("ordering dependency requires an inclusive runtime bridge")
            if not self.copy_calls:
                raise InvalidModel("ordering dependency requires the actual transfer that carries the producer's output")
            if not self.dependency_evidence:
                raise InvalidModel("ordering dependency requires actual producer-node evidence")
        elif self.dependency_evidence:
            raise InvalidModel("ordering-dependency evidence requires a declared dependency")

    @property
    def kind(self):
        if self.scope == "HOST":
            return ("host_transfer" if self.copy_calls else "host_unclassified") if self.host_stage is None else self.host_stage
        if self.scope == "SAMPLER":
            return "sampler_tail"
        if self.scope == "RUNTIME_BRIDGE":
            return "mixed_bridge"
        return self.operators[0].role

    @property
    def allowed_features(self):
        result = [INTERCEPT]
        roles = {v.role for v in self.operators}
        if self.query_basis in {"actual_attention_rows", "actual_matrix_rows", "actual_pointwise_rows"}:
            result.append(QUERY)
        if ("attention_core" in roles or self.host_stage == "attention_mask_prepare"
                or self.dependency == CPU_ATTENTION_STAGE):
            result.append(ATTENTION)
        if self.copy_calls:
            result.append(PAYLOAD)
        return tuple(result)

    @property
    def query_basis(self):
        if self.operators and all(v.role == "pruned_zero_output" for v in self.operators):
            return "pruned_no_query_work"
        if any(v.role == "attention_core" for v in self.operators):
            return "actual_attention_rows"
        rows = {v.ne[v.query_axis] for v in self.operators if v.role in {"matrix", "pointwise"}}
        if len(rows) != 1:
            return "unidentified"
        return "actual_matrix_rows" if any(v.role == "matrix" for v in self.operators) else "actual_pointwise_rows"

    @property
    def unknown_mechanisms(self):
        if self.kind in {"host_unclassified", "host_transfer"}:
            return ("host work is not identified by the current event boundaries",)
        if self.kind == "other_compute":
            return ("node work law is not classified; only an observed call response is declared",
                    "shape scope is the frozen model/backend and observed query/K coordinates, not arbitrary new tensor shapes")
        if self.kind == "mixed_bridge":
            result = []
            if any(v.role == "other_compute" for v in self.operators):
                result.append("non-matrix/non-attention node work remains inside the inclusive bridge response")
            if self.query_basis == "unidentified":
                result.append("a single query-row coordinate is not identified for the mixed nodes")
            if self.dependency == CPU_ATTENTION_STAGE:
                result.append("the bridge is serialized behind a host attention stage; its elapsed response is not isolated device work")
            return tuple(result)
        return ()

    @property
    def domain_axes(self):
        axes = []
        if QUERY in self.allowed_features:
            axes.append("query_tokens")
        if ATTENTION in self.allowed_features:
            axes.append("kv_padded")
        if PAYLOAD in self.allowed_features:
            axes.append("payload_bytes")
        # A constant response for unidentified work is not evidence that its
        # true work is independent of dimensions. Retain conservative guards.
        if (self.kind in {"host_unclassified", "host_transfer", "other_compute"}
                or self.kind == "mixed_bridge" and self.query_basis == "unidentified"):
            for name in ("query_tokens", "kv_padded"):
                if name not in axes:
                    axes.append(name)
        return tuple(axes)

    def validate_coordinates(self, query_tokens, kv_padded, payload_bytes):
        """Reject a claimed local CPU matrix shape that contradicts its record."""
        if payload_bytes and not self.copy_calls:
            raise InvalidModel("positive requested bytes require actual observed copy calls")
        if self.scope == "RUNTIME_BRIDGE":
            if self.query_basis == "actual_attention_rows":
                rows = {v.ne[1] for v in self.operators if v.role == "attention_core"}
                histories = {v.weight_ne[0] if v.name.startswith("kqv-") else v.ne[0]
                             for v in self.operators if v.role == "attention_core"}
                if rows != {query_tokens} or histories != {kv_padded}:
                    raise InvalidModel("bridge coordinates disagree with its actual attention geometry")
            elif self.query_basis in {"actual_matrix_rows", "actual_pointwise_rows"}:
                rows = {v.ne[v.query_axis] for v in self.operators if v.role in {"matrix", "pointwise"}}
                if rows != {query_tokens}:
                    raise InvalidModel("bridge query coordinate disagrees with its actual matrix rows")
            return
        if self.scope != "CPU_NODE":
            return
        node = self.operators[0]
        if node.query_axis is not None:
            if query_tokens != node.ne[node.query_axis]:
                raise InvalidModel("query coordinate disagrees with actual CPU matrix rows")
        if node.role == "attention_core":
            actual_k = node.weight_ne[0] if node.name.startswith("kqv-") else node.ne[0]
            if kv_padded != actual_k:
                raise InvalidModel("KV coordinate disagrees with actual CPU attention geometry")

    def structural_signature(self):
        value = dict(scope=self.scope, kind=self.kind,
            operators=[v.structural_signature() for v in self.operators],
            transfer_observed=bool(self.copy_calls), host_stage=self.host_stage,
            query_basis=self.query_basis, allowed_features=list(self.allowed_features))
        # Kept out of the signature while absent so previously frozen work
        # inventories keep reproducing their exact price keys.
        if self.dependency is not None:
            value["dependency"] = self.dependency
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def to_dict(self):
        return dict(schema=SCHEMA, scope=self.scope, kind=self.kind,
            operators=[v.to_dict() for v in self.operators], copy_calls=self.copy_calls,
            host_stage=self.host_stage, host_stage_evidence=list(self.host_stage_evidence),
            dependency=self.dependency, dependency_evidence=list(self.dependency_evidence),
            allowed_features=list(self.allowed_features), domain_axes=list(self.domain_axes),
            unknown_mechanisms=list(self.unknown_mechanisms))

    @classmethod
    def from_dict(cls, value):
        keys = {"schema", "scope", "kind", "operators", "copy_calls", "host_stage",
                "host_stage_evidence", "allowed_features", "domain_axes", "unknown_mechanisms"}
        optional = {"dependency", "dependency_evidence"}
        if (not isinstance(value, dict) or set(value) - optional != keys
                or value["schema"] != SCHEMA):
            raise InvalidModel("unsupported or incomplete actual-window work schema")
        if type(value["operators"]) is not list or type(value["host_stage_evidence"]) is not list:
            raise InvalidModel("explicit operator and host evidence arrays required")
        evidence = value.get("dependency_evidence", [])
        if type(evidence) is not list:
            raise InvalidModel("explicit ordering-dependency evidence array required")
        obj = cls(value["scope"], tuple(WindowOperator.from_metadata(v) for v in value["operators"]),
                  value["copy_calls"], value["host_stage"], tuple(value["host_stage_evidence"]),
                  value.get("dependency"), tuple(evidence))
        for field in ("kind", "allowed_features", "domain_axes", "unknown_mechanisms"):
            expected = getattr(obj, field)
            if value[field] != (list(expected) if isinstance(expected, tuple) else expected):
                raise InvalidModel("work mechanism declaration disagrees with its actual metadata: " + field)
        return obj

    @classmethod
    def from_metadata(cls, scope, *, copy_calls, cpu_node=None, gpu_nodes=(),
                      host_stage=None, host_stage_evidence=(),
                      dependency=None, dependency_evidence=()):
        if not isinstance(host_stage_evidence, (tuple, list)):
            raise InvalidModel("host-stage evidence must be an explicit sequence")
        if not isinstance(dependency_evidence, (tuple, list)):
            raise InvalidModel("ordering-dependency evidence must be an explicit sequence")
        if scope == "CPU_NODE":
            if cpu_node is None or gpu_nodes:
                raise InvalidModel("CPU work requires one actual CPU node and no hidden GPU node list")
            operators = (WindowOperator.from_metadata(cpu_node),)
        elif scope == "RUNTIME_BRIDGE":
            if cpu_node is not None or not isinstance(gpu_nodes, (tuple, list)):
                raise InvalidModel("runtime bridge requires actual GPU node metadata")
            operators = tuple(WindowOperator.from_metadata(v) for v in gpu_nodes)
        else:
            if cpu_node is not None or gpu_nodes:
                raise InvalidModel("host/sampler work cannot contain hidden actual nodes")
            operators = ()
        return cls(scope, operators, copy_calls, host_stage, tuple(host_stage_evidence),
                   dependency, tuple(dependency_evidence))
