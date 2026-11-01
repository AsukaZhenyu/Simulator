"""Causal, model-owned history of a declared serial action stream.

This module tracks *logical* accesses and predicted wall time.  It does not
observe the machine, identify a cache hit, infer a driver queue, or turn memory
capacity into bandwidth.  In particular, a COPY reading DRAM is not labelled a
CPU read: hidden pageable staging is not present in the declared graph.

The caller supplies an independently established response function.  No cost
coefficient is invented here, and this optional state machine changes no default
performance provider.  Reset/prime operations must be replayed explicitly if
their history is to be represented; a warm initial cache is never assumed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from .parameters import InvalidModel, UnknownParameter, integer, rational
from .reusable_action_costs import semantic_action


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise InvalidModel('history scope requires an explicit SHA256 runtime/boundary scope')
    return value


def _json_value(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


@dataclass(frozen=True)
class HistoryScope:
    """Explicit model boundary and optional declared history envelope.

    A limit bounds the model's accepted inputs, not calibrated accuracy.  A
    response provider must additionally reject its own unmeasured coordinates.
    The scope identifies hardware/runtime/memory/timing conditions, not a
    strategy, layer number, program identity, or observed target outcome.
    """
    sha256: str
    max_elapsed_ns: int | Fraction | None = None
    max_completed_actions: int | None = None

    def __post_init__(self):
        _digest(self.sha256)
        if self.max_elapsed_ns is not None:
            object.__setattr__(self, 'max_elapsed_ns', rational(self.max_elapsed_ns, 'history horizon'))
        if self.max_completed_actions is not None:
            integer(self.max_completed_actions, 'history action limit')


@dataclass(frozen=True)
class HistorySnapshot:
    scope_sha256: str
    semantic: dict
    features: dict

    def to_dict(self):
        return _json_value(dict(schema='causal-logical-execution-history-1',
            scope_sha256=self.scope_sha256, semantic=deepcopy(self.semantic),
            features=deepcopy(self.features),
            physical_cache_queue_or_clock_inferred=False,
            target_outcome_used=False))


class SerialExecutionHistory:
    """Evolve history using declared actions and already predicted durations.

    Handles and ordinals are used internally to recognize repeated buffers.
    They never occur in exported features.  Distinct footprint is the union of
    declared tensor handles in DRAM, not physical arena pages or cache lines;
    aliased/view descriptors are rejected rather than silently counted twice.
    Action legality and tensor liveness remain the graph executor's job.
    """
    def __init__(self, program: dict, scope: HistoryScope):
        if not isinstance(scope, HistoryScope):
            raise InvalidModel('explicit HistoryScope required')
        self.scope = scope
        try:
            # Retain only declarations used by semantic_action. Numerical
            # initial values and captured output ports can contain millions of
            # numbers; they are neither history state nor predictor inputs.
            self._program = deepcopy(dict(
                graph=dict(workload=dict(tensors=program['graph']['workload']['tensors'])),
                actions=program['actions'], bindings=program['bindings'],
                physical=dict(config=program['physical']['config'], threads=program['physical']['threads']),
                boundary_contract=program['boundary_contract']))
            descriptors = self._program['graph']['workload']['tensors']
            actions = self._program['actions']
            bindings = self._program['bindings']
        except (KeyError, TypeError) as exc:
            raise InvalidModel('history requires a declared highlevel action program') from exc
        if not isinstance(descriptors, list) or not isinstance(actions, list) or not actions:
            raise InvalidModel('history requires nonempty actions and tensor descriptors')
        self._tensors = {}
        for descriptor in descriptors:
            tid = descriptor.get('id')
            if not isinstance(tid, str) or not tid or tid in self._tensors:
                raise InvalidModel('missing or duplicate logical tensor handle')
            for field in ('alias_of', 'view_src', 'storage_alias'):
                if descriptor.get(field) is not None:
                    raise InvalidModel('aliased logical footprint needs an explicit storage mapping')
            integer(descriptor.get('storage_bytes'), 'declared tensor bytes', 1)
            for field in ('dtype', 'ne', 'nb'):
                if field not in descriptor:
                    raise InvalidModel('history geometry is incomplete: ' + field)
            self._tensors[tid] = descriptor
        self._bindings = {}
        for binding in bindings:
            tid = binding.get('operation_id')
            if not isinstance(tid, str) or not tid or tid in self._bindings:
                raise InvalidModel('missing or duplicate compute binding')
            self._bindings[tid] = binding
        self._accesses = []
        self._semantics = []
        for i, action in enumerate(actions):
            accesses = self._action_accesses(action)
            try:
                semantic = semantic_action(self._program, i)
            except (KeyError, TypeError, StopIteration, ValueError) as exc:
                raise InvalidModel('unsupported/incomplete semantic action') from exc
            self._accesses.append(accesses)
            self._semantics.append(semantic)
        self.time_ns = Fraction()
        self.completed_actions = 0
        self._event = 0
        self._buffers = {}
        self._last_host_touch = {}
        self._previous_service = None
        self._last_compute = {'cpu': None, 'gpu': None}
        self._last_route = None
        self._last_service_end = None
        self._busy = {'cpu': Fraction(), 'gpu': Fraction(), 'copy': Fraction()}
        self._total_idle = Fraction()
        self._host_bytes = {'cpu_read': 0, 'cpu_write': 0, 'copy_read': 0, 'copy_write': 0}

    def _action_accesses(self, action):
        kind = action.get('kind')
        accesses = []
        if kind == 'COPY':
            src, dst, tid = action.get('src'), action.get('dst'), action.get('target_id')
            if src not in ('dram', 'vram') or dst not in ('dram', 'vram') or src == dst:
                raise InvalidModel('history supports explicit DRAM/VRAM copies only')
            accesses = [(tid, src, 'copy_read'), (tid, dst, 'copy_write')]
        elif kind == 'COMPUTE':
            device = action.get('device')
            if device not in ('cpu', 'gpu'):
                raise InvalidModel('history supports CPU/GPU compute only')
            binding = self._bindings.get(action.get('target_id'))
            if binding is None:
                raise InvalidModel('compute has no semantic binding')
            location = 'dram' if device == 'cpu' else 'vram'
            read = set(binding.get('operands', {}).values())
            read.update(b['id'] for b in binding.get('history', []))
            written = set(binding.get('outputs', {}).values())
            written.update(b['id'] for b in binding.get('new_kv', []))
            accesses = [(tid, location, device + '_read') for tid in sorted(read)]
            accesses += [(tid, location, device + '_write') for tid in sorted(written)]
        elif kind not in ('ADVANCE', 'EVICT'):
            raise InvalidModel('unsupported action kind in logical history')
        if any(tid not in self._tensors for tid, _, _ in accesses):
            raise InvalidModel('action accesses an undeclared tensor')
        return tuple(accesses)

    def _index(self, index):
        integer(index, 'declared action index')
        if index >= len(self._semantics):
            raise InvalidModel('history action is outside the declared program')
        return index

    def _envelope(self, time_ns, count):
        if self.scope.max_elapsed_ns is not None and time_ns > self.scope.max_elapsed_ns:
            raise UnknownParameter('predicted history exceeds the declared time envelope')
        if self.scope.max_completed_actions is not None and count > self.scope.max_completed_actions:
            raise UnknownParameter('predicted history exceeds the declared action envelope')

    def _buffer_features(self, tid, location, role):
        state = self._buffers.get((tid, location), {})
        last = state.get('any_event')
        if location == 'dram' and last is not None:
            distance = sum(self._tensors[other]['storage_bytes']
                for other, touched in self._last_host_touch.items()
                if other != tid and touched > last)
        else:
            distance = None
        result = dict(role=role, location=location,
            storage_bytes=self._tensors[tid]['storage_bytes'],
            distinct_other_host_bytes_since_last_access=distance)
        for access in ('any', 'cpu_read', 'cpu_write', 'gpu_read', 'gpu_write', 'copy_read', 'copy_write'):
            at = state.get(access + '_ns')
            result['age_since_' + access + '_ns'] = None if at is None else self.time_ns - at
        return result

    def snapshot(self, index: int) -> HistorySnapshot:
        index = self._index(index)
        self._envelope(self.time_ns, self.completed_actions)
        semantic = deepcopy(self._semantics[index])
        route = ((semantic['src'], semantic['dst']) if semantic['kind'] == 'COPY' else None)
        features = dict(elapsed_predicted_ns=self.time_ns,
            completed_actions=self.completed_actions,
            previous_service=deepcopy(self._previous_service),
            last_cpu_compute=self._last_compute['cpu'],
            last_gpu_compute=self._last_compute['gpu'],
            previous_copy_route=self._last_route,
            copy_route_switch=None if route is None or self._last_route is None else route != self._last_route,
            idle_since_last_service_ns=None if self._last_service_end is None else self.time_ns - self._last_service_end,
            explicit_idle_ns=self._total_idle,
            predicted_busy_ns=dict(self._busy),
            logical_host_distinct_bytes=sum(self._tensors[tid]['storage_bytes'] for tid in self._last_host_touch),
            logical_host_access_bytes=dict(self._host_bytes),
            accesses=[self._buffer_features(tid, loc, role) for tid, loc, role in self._accesses[index]])
        # Roles are read/write classes, never buffer names. Sort equal-geometry
        # roles deterministically so renaming internal handles changes no key.
        features['accesses'].sort(key=lambda row: repr(_json_value(row)))
        return HistorySnapshot(self.scope.sha256, semantic, features)

    def commit(self, index: int, predicted_duration_ns: int | Fraction):
        """Complete one modeled action; no observed target-duration input exists."""
        index = self._index(index)
        duration = rational(predicted_duration_ns, 'predicted completed-action duration')
        end = self.time_ns + duration
        self._envelope(end, self.completed_actions + 1)
        semantic = self._semantics[index]
        self._event += 1
        for tid, location, role in self._accesses[index]:
            state = self._buffers.setdefault((tid, location), {})
            state['any_ns'] = state[role + '_ns'] = end
            state['any_event'] = self._event
            if location == 'dram':
                self._last_host_touch[tid] = self._event
                self._host_bytes[role] += self._tensors[tid]['storage_bytes']
        kind = semantic['kind']
        if kind == 'COMPUTE':
            device = semantic['device']
            self._busy[device] += duration
            self._last_compute[device] = semantic['semantic_op']
            self._previous_service = dict(kind=kind, device=device, semantic_op=semantic['semantic_op'])
            self._last_service_end = end
        elif kind == 'COPY':
            self._busy['copy'] += duration
            self._last_route = (semantic['src'], semantic['dst'])
            self._previous_service = dict(kind=kind, route=self._last_route,
                storage_bytes=semantic['tensor']['storage_bytes'])
            self._last_service_end = end
        self.time_ns = end
        self.completed_actions += 1

    def idle(self, predicted_duration_ns: int | Fraction):
        """Apply an explicit idle interval; never infer an idle gap from jitter."""
        duration = rational(predicted_duration_ns, 'predicted idle duration')
        end = self.time_ns + duration
        self._envelope(end, self.completed_actions)
        self.time_ns = end
        self._total_idle += duration


def evaluate_with_history(program: dict, scope: HistoryScope,
                          response: Callable[[HistorySnapshot], int | Fraction], *, repeats: int = 1):
    """Serial causal recurrence; the response sees no future measured duration.

    A provider returning None is unknown and rejected.  A provider must enforce
    its own calibration envelope; this function does not claim qualification.
    Inclusive response times are used once, without another switching penalty.
    """
    integer(repeats, 'modeled replay count', 1)
    if not callable(response):
        raise InvalidModel('explicit scoped history response provider required')
    history = SerialExecutionHistory(program, scope)
    rows = []
    for _ in range(repeats):
        for index in range(len(program['actions'])):
            state = history.snapshot(index)
            duration = response(state)
            if duration is None:
                raise UnknownParameter('unmeasured history-conditioned response')
            duration = rational(duration, 'history-conditioned response')
            rows.append(dict(state=state, duration_ns=duration))
            history.commit(index, duration)
    return dict(duration_ns=history.time_ns, action_rows=tuple(rows),
        target_outcome_used=False, history_is_logical_not_microarchitectural=True,
        hardware_accuracy_qualified=False)
