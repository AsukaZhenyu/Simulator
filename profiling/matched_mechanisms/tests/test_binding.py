import copy
from pathlib import Path
import sys
import unittest
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import experiment as e
from binding import expected_ports, validate_ports, fnv_f32_bytes


def payload(case):
    shapes = expected_ports(case)
    return {"ports": {n: {"ne": v, "bytes": v[0] * v[1] * v[2] * 4} for n, v in shapes.items()},
            "source_values": {n: [0.] * (v[0] * v[1] * v[2]) for n, v in shapes.items()}}


class Binding(unittest.TestCase):
    def test_exact_all_family_shapes(self):
        for case in e.document(HERE / "suite.json")["cases"]:
            validate_ports(case, payload(case))

    def test_ne_echo_alone_not_sufficient(self):
        case = e.document(HERE / "suite.json")["cases"][0]; p = payload(case)
        p["ports"]["input"] = {"ne": [1, 1, 1], "bytes": 4}
        p["source_values"]["input"] = [0.]
        with self.assertRaisesRegex(ValueError, "geometry"): validate_ports(case, p)

    def test_q_layout_not_confused_with_token_major_ffn(self):
        case = e.document(HERE / "suite.json")["cases"][3]; p = payload(case)
        p["ports"]["query"]["ne"] = [case["hidden"], case["tokens"], 1]
        with self.assertRaisesRegex(ValueError, "geometry"): validate_ports(case, p)

    def test_missing_down_matrix_not_whole_expert_replacement(self):
        case = e.document(HERE / "suite.json")["cases"][0]; p = payload(case)
        p["ports"].pop("down_weight")
        with self.assertRaisesRegex(ValueError, "coverage"): validate_ports(case, p)

    def test_byte_identity_detects_change(self):
        raw = e._float_bytes([.1, .2]); mutated = e._float_bytes([.1, .3])
        self.assertNotEqual(fnv_f32_bytes(raw), fnv_f32_bytes(mutated))
        self.assertEqual(fnv_f32_bytes(b""), "cbf29ce484222325")


if __name__ == "__main__": unittest.main()
