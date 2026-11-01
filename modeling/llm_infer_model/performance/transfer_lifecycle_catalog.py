"""Coordinate qualification for an immutable completed-transfer response.

This adapter does not change the calibrated formula or any parameter. It keeps
the original full-grid first-response flag and adds independent qualification
only for explicitly validated (bytes, declared idle) coordinates. Complete COPY
windows continue to use their separate qualification contract.
"""
from __future__ import annotations
from dataclasses import dataclass

from .parameters import InvalidModel,UnknownParameter,integer
from .transfer_lifecycle import CompletedTransferLifecycle

@dataclass(frozen=True)
class ScopedTransferLifecycle:
    lifecycle:CompletedTransferLifecycle
    qualified_first_coordinates:tuple[tuple[int,int],...]
    qualification_evidence:tuple[str,...]

    def __post_init__(self):
        if not isinstance(self.lifecycle,CompletedTransferLifecycle):
            raise InvalidModel('explicit immutable completed-transfer response required')
        if type(self.qualified_first_coordinates) is not tuple:
            raise InvalidModel('explicit qualified first-COPY coordinate tuple required')
        idles={r.requested_idle_ns for r in self.lifecycle.first_impulses}
        seen=set()
        for coordinate in self.qualified_first_coordinates:
            if type(coordinate) is not tuple or len(coordinate)!=2:
                raise InvalidModel('first-COPY coordinates are (bytes, declared idle ns) pairs')
            size,idle=coordinate;integer(size,'qualified COPY bytes',1);integer(idle,'qualified declared idle ns')
            if coordinate in seen:raise InvalidModel('duplicate qualified first-COPY coordinate')
            seen.add(coordinate)
            if size not in self.lifecycle.qualified_byte_coordinates or idle not in idles:
                raise InvalidModel('first-COPY allowlist exceeds independently measured coordinates')
        if type(self.qualification_evidence) is not tuple or not self.qualification_evidence or any(not isinstance(v,str) or not v.strip() for v in self.qualification_evidence):
            raise InvalidModel('explicit independent coordinate-qualification evidence required')

    @property
    def first_qualified(self):
        """Preserve the original full-grid flag; the allowlist is separate."""
        return self.lifecycle.first_qualified

    @property
    def window_qualified(self):return self.lifecycle.window_qualified

    def first_response_ns(self,size_bytes,requested_idle_ns,*,context_sha256,preparation_sha256,require_qualified=True):
        if type(require_qualified) is not bool:raise InvalidModel('explicit boolean qualification requirement required')
        # The original model still validates the immutable geometry/context and
        # computes the same result, even for an explicit unqualified evaluation.
        value=self.lifecycle.first_response_ns(size_bytes,requested_idle_ns,context_sha256=context_sha256,
            preparation_sha256=preparation_sha256,require_qualified=False)
        if require_qualified and (size_bytes,requested_idle_ns) not in self.qualified_first_coordinates:
            raise UnknownParameter('first COPY lacks independent qualification at this bytes/idle coordinate')
        return value

    def window_ns(self,size_bytes,requested_idle_ns,copies,*,context_sha256,preparation_sha256,require_qualified=True):
        if type(require_qualified) is not bool:raise InvalidModel('explicit boolean qualification requirement required')
        return self.lifecycle.window_ns(size_bytes,requested_idle_ns,copies,context_sha256=context_sha256,
            preparation_sha256=preparation_sha256,require_qualified=require_qualified)

    def describe(self):
        return dict(self.lifecycle.describe(),qualified_first_coordinates=[list(c) for c in self.qualified_first_coordinates],
            original_full_grid_first_flag_preserved=True,qualification_evidence=list(self.qualification_evidence),
            formula_or_parameters_changed=False)

    @classmethod
    def from_profile(cls,profile):
        if 'qualified_first_coordinates' not in profile or 'first_coordinate_qualification_evidence' not in profile:
            raise InvalidModel('explicit independent first-coordinate qualification metadata required')
        try:
            return cls(CompletedTransferLifecycle.from_profile(profile),
                tuple(tuple(v) for v in profile['qualified_first_coordinates']),
                tuple(profile['first_coordinate_qualification_evidence']))
        except (KeyError,TypeError) as error:
            raise InvalidModel('complete explicit scoped transfer parameters required') from error
