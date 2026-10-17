from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import experiment as e
from export_profiles import PORTS
from replay_serial import build_cpu_binding


def fixture():
    c = e.document(HERE / "suite.json")["cases"][0]
    h, i, q = c["hidden"], c["intermediate"], c["tokens"]
    shape = {"input": [h, q, 1], "gate_weight": [h, i, 1], "up_weight": [h, i, 1],
             "down_weight": [i, h, 1], "gate": [i, q, 1], "up": [i, q, 1],
             "activated": [i, q, 1], "output": [h, q, 1]}
    entries = []
    for n, stage in enumerate(("gate", "up", "activation", "down")):
        inputs, outputs = PORTS[stage]
        port = lambda name: {"ne": shape[name], "bytes": shape[name][0] * shape[name][1] * 4, "input_fixture_fnv1a64": name}
        entries.append({"case_id": c["id"], "stage": stage, "device": "cpu", "duration_ns": (n + 1) * 100,
                        "workspace_bytes": 128, "latency_quality_accepted": True,
                        "inputs": {x: port(x) for x in inputs}, "outputs": {x: port(x) for x in outputs}})
    return c, entries


class Replay(unittest.TestCase):
    def test_engine_runs_all_four_stage_port_dependencies(self):
        c, entries = fixture()
        workload, scenario, result = build_cpu_binding(c, entries, diagnostic=False)
        self.assertEqual(result.evaluation.status, "valid")
        self.assertEqual(result.evaluation.makespan_ns, 1000)
        self.assertEqual([o["semantic_op"] for o in workload["operations"]], ["expert_gate", "expert_up", "expert_swiglu", "expert_down"])
        self.assertEqual(scenario["costs"]["compute"]["activation"]["cpu"]["workspace_bytes"], 128)

    def test_rejected_price_cannot_be_accepted(self):
        c, entries = fixture(); entries[0]["latency_quality_accepted"] = False
        with self.assertRaisesRegex(ValueError, "quality-rejected"):
            build_cpu_binding(c, entries, diagnostic=False)

    def test_rejected_price_only_runs_explicit_diagnostic(self):
        c, entries = fixture(); entries[0]["latency_quality_accepted"] = False
        _, scenario, result = build_cpu_binding(c, entries, diagnostic=True)
        self.assertEqual(result.evaluation.status, "valid")
        self.assertEqual(scenario["costs"]["source"], "quality_rejected_measured_diagnostic")

    def test_unknown_workspace_never_zero(self):
        c, entries = fixture(); entries[0]["workspace_bytes"] = None
        with self.assertRaisesRegex(ValueError, "unknown workspace"):
            build_cpu_binding(c, entries, diagnostic=True)

    def test_boundary_port_identity_mismatch_refused(self):
        c, entries = fixture(); entries[2]["inputs"]["gate"]["input_fixture_fnv1a64"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity"):
            build_cpu_binding(c, entries, diagnostic=True)

    def test_missing_stage_not_replaced_by_whole_ffn(self):
        c, entries = fixture(); entries.pop(2)
        with self.assertRaisesRegex(ValueError, "coverage"):
            build_cpu_binding(c, entries, diagnostic=True)


if __name__ == "__main__": unittest.main()
