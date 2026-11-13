"""Optional scoped inclusive responses conditioned on causal logical history.

Profiling supplies frozen prices and evidence.  Modeling selects a declared
response cell, advances predicted time, and updates logical accesses.  An H2D
cell may depend on distinct *declared* DRAM footprint since source reuse.  This
is a piecewise-constant empirical approximation, not a cache/staging mechanism.

Static cells from the same calibration set are retained for a fair comparison
and to advance the untimed logical prime.  A missing conditional timed cell is
never replaced with a static price.  No complete-target correction is added.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json

from .execution_history import HistoryScope, HistorySnapshot, SerialExecutionHistory
from .parameters import InvalidModel, UnknownParameter, integer
from .reusable_action_costs import digest


SCHEMA = 'history-conditioned-inclusive-response-1'
BUCKETING = 'power_of_two_floor_bytes_v1'
INITIALIZATION = 'fresh_logical_history_then_one_full_prime'
ALLOWED_FIELDS = frozenset({'h2d_source_reuse_bucket', 'previous_service', 'previous_copy_route',
    'destination_busy_bucket'})
FORBIDDEN_PREDICTOR_KEYS = frozenset({'program_id', 'program_sha256', 'strategy', 'strategy_id',
    'strategy_name', 'layer_id', 'layer_index', 'action_index', 'target_id', 'operation_id',
    'seed', 'observed_ns', 'actual_target_ns', 'measured_target_ns', 'end_to_end_ns'})


def _json(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InvalidModel('response profile must contain finite JSON values') from exc


def _no_ids(value):
    if isinstance(value, dict):
        if set(value) & FORBIDDEN_PREDICTOR_KEYS:
            raise InvalidModel('program, strategy, layer, target identity or outcome cannot be a response predictor')
        for child in value.values():
            _no_ids(child)
    elif isinstance(value, list):
        for child in value:
            _no_ids(child)


def _fields(fields):
    if not isinstance(fields, (tuple, list)) or len(fields) != len(set(fields)):
        raise InvalidModel('unique declared history condition fields required')
    if set(fields) - ALLOWED_FIELDS:
        raise InvalidModel('unsupported or unmodeled reset-sensitive history field')
    return tuple(sorted(fields))


def reuse_bucket(distance_bytes):
    """Declared floor-power-of-two intervals; zero and unknown stay separate."""
    if distance_bytes is None:
        return 'unknown'
    integer(distance_bytes, 'distinct logical DRAM reuse footprint')
    return 'zero' if distance_bytes == 0 else str(1 << (distance_bytes.bit_length() - 1))


def _is_h2d(semantic):
    return semantic.get('kind') == 'COPY' and semantic.get('src') == 'dram' and semantic.get('dst') == 'vram'


def response_coordinate(snapshot: HistorySnapshot, condition_fields, *, use_history=True):
    """ID-free semantic coordinate; snapshots contain prior modeled state only."""
    if not isinstance(snapshot, HistorySnapshot):
        raise InvalidModel('explicit model-owned HistorySnapshot required')
    fields = _fields(condition_fields)
    condition = {}
    if use_history and _is_h2d(snapshot.semantic):
        for name in fields:
            if name == 'h2d_source_reuse_bucket':
                sources = [row for row in snapshot.features['accesses']
                    if row['location'] == 'dram' and row['role'] == 'copy_read']
                if len(sources) != 1:
                    raise InvalidModel('H2D history needs one declared DRAM source')
                condition[name] = reuse_bucket(sources[0]['distinct_other_host_bytes_since_last_access'])
            elif name == 'destination_busy_bucket':
                # What the destination device is already doing, as the model itself
                # predicts it: the busy time it has accumulated when the copy is
                # issued.  It is state, not identity, so it stays inside the scope
                # contract; a copy issued onto an idle device and one issued behind
                # queued work are different coordinates.
                busy = (snapshot.features.get('predicted_busy_ns') or {}).get(
                    snapshot.semantic.get('dst'))
                condition[name] = reuse_bucket(int(busy) if busy else 0)
            else:
                condition[name] = deepcopy(snapshot.features[name])
    coordinate = _json(dict(semantic=deepcopy(snapshot.semantic), condition=condition,
                           scope_sha256=snapshot.scope_sha256))
    _no_ids(coordinate)
    return coordinate


def _duration(value):
    if type(value) not in (int, str):
        raise InvalidModel('response mean must be an exact integer or rational string')
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise InvalidModel('response mean must be finite and known') from exc
    if result < 0:
        raise InvalidModel('response mean must be nonnegative')
    return result


def _semantic(value):
    if not isinstance(value, dict):
        raise InvalidModel('semantic response description required')
    kind = value.get('kind')
    if kind == 'COPY':
        fields = {'kind', 'src', 'dst', 'tensor'}
        if value.get('src') not in ('dram', 'vram') or value.get('dst') not in ('dram', 'vram') or value['src'] == value['dst']:
            raise InvalidModel('unknown copy route')
        geometry = value['tensor']
        if not isinstance(geometry, dict) or set(geometry) != {'dtype', 'ne', 'nb', 'storage_bytes'}:
            raise InvalidModel('exact copy geometry required')
        integer(geometry['storage_bytes'], 'copy bytes', 1)
    elif kind == 'COMPUTE':
        fields = {'kind', 'device', 'semantic_op', 'shape', 'op_params', 'operands', 'outputs',
                  'history', 'new_kv', 'config', 'threads', 'port_boundary'}
        if value.get('device') not in ('cpu', 'gpu'):
            raise InvalidModel('unsupported compute device')
    elif kind in ('ADVANCE', 'EVICT'):
        fields = {'kind', 'native_behavior'}
        if value.get('native_behavior') != 'logical_control_no_allocation_or_wait':
            raise InvalidModel('unsupported logical control execution boundary')
    else:
        raise InvalidModel('unknown semantic response kind')
    if set(value) != fields:
        raise InvalidModel('unexpected or incomplete semantic response fields')
    _no_ids(value)


class HistoryConditionedResponseTable:
    """Frozen exact semantic prices with explicitly selected H2D history bins.

    ``entries`` contains static condition={} cells for every semantic coordinate
    and additional conditioned H2D cells.  All prices must be supplied by
    profiling.  Constructor validation is a software contract, not a claim that
    the chosen features identify a physical mechanism or generalize accurately.
    """
    def __init__(self, profile):
        value = _json(profile)
        if value.get('schema') != SCHEMA or value.get('bucketing') != BUCKETING:
            raise InvalidModel('history response profile schema/bucketing differs')
        if value.get('initialization') != INITIALIZATION or value.get('unmodeled_reset_accesses') is not True:
            raise InvalidModel('explicit logical prime and unmodeled reset-access contract required')
        self.scope = HistoryScope(value.get('scope_sha256'))
        self.condition_fields = _fields(value.get('condition_fields'))
        entries = value.get('entries')
        if not isinstance(entries, list) or not entries:
            raise InvalidModel('nonempty frozen response entries required')
        self._rows, self._prices = {}, {}
        for row in entries:
            if not isinstance(row, dict) or not {'coordinate', 'key', 'mean_ns', 'sample_count', 'evidence'} <= set(row):
                raise InvalidModel('response entry lacks price/evidence/coordinate')
            coordinate = row['coordinate']
            if not isinstance(coordinate, dict) or set(coordinate) != {'semantic', 'condition', 'scope_sha256'}:
                raise InvalidModel('response coordinates cannot contain program identities or extra fields')
            if coordinate['scope_sha256'] != self.scope.sha256:
                raise InvalidModel('response scope differs')
            _semantic(coordinate['semantic'])
            condition = coordinate['condition']
            if not isinstance(condition, dict):
                raise InvalidModel('explicit condition mapping required')
            if condition:
                if not _is_h2d(coordinate['semantic']) or set(condition) != set(self.condition_fields):
                    raise InvalidModel('conditions must match declared H2D fields exactly')
                if 'h2d_source_reuse_bucket' in condition:
                    bucket = condition['h2d_source_reuse_bucket']
                    if bucket not in ('unknown', 'zero'):
                        try:
                            size = int(bucket)
                        except (TypeError, ValueError) as exc:
                            raise InvalidModel('unknown logical reuse bucket') from exc
                        if type(bucket) is not str or size < 1 or size & (size - 1) or str(size) != bucket:
                            raise InvalidModel('reuse bin must be a positive floor power of two')
            _no_ids(coordinate)
            key = digest(coordinate)
            if row['key'] != key or key in self._rows:
                raise InvalidModel('duplicate or changed response coordinate')
            integer(row['sample_count'], 'response sample count', 1)
            if not isinstance(row['evidence'], list) or not row['evidence']:
                raise InvalidModel('independent response measurement evidence required')
            if any(not ((isinstance(item, str) and item.strip())
                    or (isinstance(item, dict) and item)) for item in row['evidence']):
                raise InvalidModel('response evidence references must be explicit and nonempty')
            self._prices[key] = _duration(row['mean_ns'])
            self._rows[key] = deepcopy(row)
        # Every conditioned semantic needs an explicitly measured static cell.
        for row in self._rows.values():
            static = dict(row['coordinate'], condition={})
            if digest(static) not in self._rows:
                raise InvalidModel('same-calibration static baseline is missing')
        self._profile = value

    @classmethod
    def from_profile(cls, profile):
        return cls(profile)

    def lookup(self, snapshot: HistorySnapshot, *, use_history=True):
        if snapshot.scope_sha256 != self.scope.sha256:
            raise UnknownParameter('history response runtime/memory/boundary scope differs')
        coordinate = response_coordinate(snapshot, self.condition_fields, use_history=use_history)
        key = digest(coordinate)
        row = self._rows.get(key)
        if row is None or row['coordinate'] != coordinate:
            raise UnknownParameter('unmeasured exact semantic/history response: ' + key)
        return self._prices[key]

    def predict_blocks(self, program, *, blocks, repeats=8, use_history=True):
        return predict_blocks(program, self, blocks=blocks, repeats=repeats, use_history=use_history)


def predict_blocks(program, table: HistoryConditionedResponseTable, *, blocks, repeats=8, use_history=True):
    """Reset logical history, model one complete prime, then timed replays.

    Each block intentionally starts a new *logical* history, not a physical
    cache/clock reset. Native reset writes and prior-block hardware recovery are
    not represented. Consequently reset-sensitive ages/busy totals cannot be
    selected as response features. Prime timing uses explicit static cells from
    the same calibration; it is excluded from reported timed execution.
    """
    integer(blocks, 'predicted block count', 1)
    integer(repeats, 'timed replays per block', 1)
    if type(use_history) is not bool:
        raise InvalidModel('history/static comparison must be explicit bool')
    rows, batch_ns = [], []
    for block in range(blocks):
        history = SerialExecutionHistory(program, table.scope)
        for index in range(len(program['actions'])):
            state = history.snapshot(index)
            history.commit(index, table.lookup(state, use_history=False))
        start = history.time_ns
        for replay in range(repeats):
            for index in range(len(program['actions'])):
                state = history.snapshot(index)
                duration = table.lookup(state, use_history=use_history)
                rows.append(dict(block=block, replay=replay, action_index=index,
                    state=state.to_dict(), duration_ns=duration,
                    coordinate=response_coordinate(state, table.condition_fields, use_history=use_history)))
                history.commit(index, duration)
        batch_ns.append(history.time_ns - start)
    per_replay = tuple(value / repeats for value in batch_ns)
    return dict(duration_ns=sum(per_replay, Fraction()) / blocks,
        block_ns=per_replay, timed_batch_ns=tuple(batch_ns), action_rows=tuple(rows),
        provider='logical-history-conditioned inclusive responses' if use_history else 'same-calibration static inclusive responses',
        initialization=INITIALIZATION, prime_response='same-calibration static cells; excluded from timed total',
        unmodeled_reset_accesses=True, physical_cache_reset_assumed=False,
        target_outcome_used=False, hardware_accuracy_qualified=False,
        causal_mechanism_identified=False, extra_transition_ns=0)
