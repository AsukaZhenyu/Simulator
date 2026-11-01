"""Additional read-only PCIe observations for new diagnostic protocols only."""
from __future__ import annotations

import ctypes

from profiling.state_calibration.telemetry import NVMLProvider


class PCIeNVMLProvider(NVMLProvider):
    EXTRA_FIELDS = {
        "pcie_current_gen": "nvmlDeviceGetCurrPcieLinkGeneration",
        "pcie_max_gen": "nvmlDeviceGetMaxPcieLinkGeneration",
        "pcie_current_width": "nvmlDeviceGetCurrPcieLinkWidth",
        "pcie_max_width": "nvmlDeviceGetMaxPcieLinkWidth",
    }

    def metadata(self):
        return {**super().metadata(), "pcie_semantics": "Read-only driver queries of negotiated/current and device maximum link generation/width; not instantaneous effective bandwidth. Unsupported symbols/queries stay unknown."}

    def read(self):
        result = super().read()
        for field, symbol in self.EXTRA_FIELDS.items():
            try:
                if self._error or self._closed:
                    raise RuntimeError(self._error or "NVML provider closed")
                value = ctypes.c_uint()
                function = self._function((symbol,), [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)])
                self._check(function(self._handle, ctypes.byref(value)), symbol)
                result["values"][field] = value.value
            except Exception as error:
                result["unknown"][field] = f"{type(error).__name__}: {error}"
        result["status"] = "ok" if not result["unknown"] else ("partial" if result["values"] else "unavailable")
        return result
