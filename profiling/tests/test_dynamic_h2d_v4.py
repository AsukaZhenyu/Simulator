import unittest
from unittest.mock import patch

from profiling.dynamic_validation.h2d_v4 import direct_associations
from profiling.dynamic_validation.pcie_telemetry import PCIeNVMLProvider


class CopyAssociations(unittest.TestCase):
    def records(self, external=17):
        return {"records": [
            {"kind": "external_correlation", "correlation_id": 8, "external_id": external},
            {"kind": "memcpy", "correlation_id": 8, "timestamp_valid": True, "bytes": 2097152, "copy_kind": 1},
        ]}

    def test_exact_direct_association(self):
        self.assertTrue(direct_associations(self.records(), [{"external_id": 17}])[0]["supported"])

    def test_missing_correlation_rejected(self):
        self.assertFalse(direct_associations({"records": self.records()["records"][1:]}, [{"external_id": 17}])[0]["supported"])

    def test_wrong_external_id_not_temporally_substituted(self):
        self.assertFalse(direct_associations(self.records(99), [{"external_id": 17}])[0]["supported"])

    def test_ambiguous_correlation_rejected(self):
        data = self.records()
        data["records"].append({"kind": "external_correlation", "correlation_id": 8, "external_id": 99})
        self.assertFalse(direct_associations(data, [{"external_id": 17}])[0]["supported"])

    def test_duplicate_or_wrong_payload_rejected(self):
        data = self.records()
        data["records"].append(dict(data["records"][1]))
        self.assertFalse(direct_associations(data, [{"external_id": 17}])[0]["supported"])
        data = self.records()
        data["records"][1]["bytes"] = 4096
        self.assertFalse(direct_associations(data, [{"external_id": 17}])[0]["supported"])

    def test_timestamp_unknown_rejected(self):
        data = self.records()
        data["records"][1]["timestamp_valid"] = False
        self.assertFalse(direct_associations(data, [{"external_id": 17}])[0]["supported"])

    def test_unsupported_pcie_fields_stay_unknown(self):
        provider = PCIeNVMLProvider.__new__(PCIeNVMLProvider)
        provider._error, provider._closed = "unsupported test provider", False
        with patch("profiling.state_calibration.telemetry.NVMLProvider.read", return_value={"values": {"pstate": 3}, "unknown": {}, "status": "ok"}):
            result = provider.read()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(set(result["unknown"]), set(provider.EXTRA_FIELDS))
        self.assertFalse(any(key in result["values"] for key in provider.EXTRA_FIELDS))


if __name__ == "__main__":
    unittest.main()
