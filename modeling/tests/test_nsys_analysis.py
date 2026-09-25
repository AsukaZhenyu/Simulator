from __future__ import annotations

import json
import unittest
from pathlib import Path

from llm_infer_model.nsys_analysis import (
    analyze_nsys_sqlite,
    expected_graph_fragments,
)


ROOT = Path(__file__).resolve().parents[1]


class NsightAnalysisTests(unittest.TestCase):
    def test_qwen35_graph_fragment_formula(self) -> None:
        self.assertEqual(expected_graph_fragments(24, 6), 79)
        self.assertEqual(expected_graph_fragments(32, 8), 105)

    def test_2b_trace_finds_embedding_fallback(self) -> None:
        sqlite_path = ROOT / "outputs" / "nsys" / "llminfer_2b_k24.sqlite"
        if not sqlite_path.exists():
            self.skipTest("Nsight artifact is not present")
        manifest_path = (
            ROOT / "outputs" / "gguf" / "qwen35_2b_q4_k_m_manifest.json"
        )
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        embedding_bytes = next(
            tensor["n_bytes"]
            for tensor in manifest["global_tensors"]
            if tensor["name"] == "token_embd.weight"
        )
        result = analyze_nsys_sqlite(
            sqlite_path,
            generation_tokens=64,
            layer_count=24,
            attention_layer_count=6,
            ssm_layer_count=18,
            embedding_bytes=embedding_bytes,
        )
        self.assertEqual(result["graph_fragmentation"]["graphs_per_token"], 79)
        self.assertTrue(
            result["graph_fragmentation"]["matches_expected_formula"]
        )
        self.assertTrue(result["embedding_fallback"]["matches_once_per_token"])
        self.assertGreater(
            result["copy_summary"]["non_embedding_d2h_bytes_per_token"],
            18 * 1024**2,
        )

    def test_architecture_metadata_selects_decode_suffix(self) -> None:
        sqlite_path = ROOT / "outputs" / "nsys" / "llminfer_2b_k24.sqlite"
        if not sqlite_path.exists():
            self.skipTest("Nsight artifact is not present")
        result = analyze_nsys_sqlite(
            sqlite_path,
            generation_tokens=63,
            layer_count=24,
            attention_layer_count=6,
        )
        self.assertEqual(
            result["graph_fragmentation"]["excluded_prefix_graph_launches"],
            79,
        )
        self.assertEqual(
            result["graph_fragmentation"]["selected_decode_graph_launches"],
            63 * 79,
        )


if __name__ == "__main__":
    unittest.main()
