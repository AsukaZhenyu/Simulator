"""Hardware-free strict route import tests; fixtures below are parser examples."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moe_routes import (COLLECTOR_CONTRACT, RouteError, convert_llama_files,
                        dump, load, parse_llama_trace, select_m2_routes, validate_bundle,
                        invocation_slice)


def stderr():
    return ("print_info: arch = example-moe\n"
            "print_info: general.name = Parser Example\n"
            "print_info: n_layer = 2\n"
            "print_info: n_expert = 4\n"
            "print_info: n_expert_used = 2\n"
            "number of input tokens = 3\n"
            "llama_perf_context_print: total time = 12.0 ms\n")


def trace():
    return ("debug tensor preview\n"
            "MOETOPK ffn_moe_topk-0 2 2 | 3 0 | 1 2\n"
            "MOETOPK ffn_moe_topk-1 2 2 | 2 0 | 3 1\n"
            "MOETOPK ffn_moe_topk-0 2 1 | 3 1\n"
            "MOETOPK ffn_moe_topk-1 2 1 | 2 3\n"
            "GENTOK 101\n"
            "MOETOPK ffn_moe_topk-0 2 1 | 1 0\n"
            "MOETOPK ffn_moe_topk-1 2 1 | 3 0\n"
            "GENTOK 102\n"
            "MOETOPK ffn_moe_topk-0 2 1 | 2 0\n"
            "MOETOPK ffn_moe_topk-1 2 1 | 0 2\n")


def bundle():
    return parse_llama_trace(trace(), stderr(), request_id="r", collector_contract=COLLECTOR_CONTRACT)


class LlamaRouteTests(unittest.TestCase):
    def parse(self, out=None, err=None, **kw):
        return parse_llama_trace(trace() if out is None else out,
                                 stderr() if err is None else err,
                                 request_id=kw.get("request_id", "r"),
                                 collector_contract=kw.get("collector_contract", COLLECTOR_CONTRACT))

    def test_prefill_chunks_and_decode_positions_are_exact(self):
        result = self.parse()
        self.assertEqual(result["requests"][0]["generated_token_ids"], [101, 102])
        rows = [row for row in result["records"] if row["layer_id"] == "0"]
        self.assertEqual([row["absolute_position"] for row in rows], [0, 1, 2, 3, 4])
        self.assertEqual([row["phase"] for row in rows], ["prefill"] * 3 + ["decode"] * 2)
        self.assertEqual(rows[0]["expert_ids"], [3, 0])
        self.assertIsNone(result["model"]["model_file_sha256"])
        self.assertIsNone(result["verification"]["process_exit_code"])

    def test_prefill_only_is_not_relabelled_decode(self):
        result = self.parse(trace().split("GENTOK")[0])
        self.assertEqual(result["requests"][0]["decode_tokens"], 0)
        self.assertTrue(all(row["phase"] == "prefill" for row in result["records"]))

    def test_crlf_input_preserves_semantics(self):
        self.assertEqual(self.parse(), self.parse(trace().replace("\n", "\r\n"), stderr().replace("\n", "\r\n")))

    def test_collector_contract_and_request_name_are_explicit(self):
        for kwargs in ({"collector_contract": "training"}, {"request_id": ""}, {"request_id": "__moe_batch__"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RouteError):
                self.parse(**kwargs)

    def test_final_partial_line_is_rejected(self):
        with self.assertRaises(RouteError): self.parse(trace().rstrip("\n"))

    def test_no_missing_layer_is_dropped(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("MOETOPK ffn_moe_topk-1 2 1 | 2 3\n", ""))

    def test_out_of_order_layer_is_rejected(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("ffn_moe_topk-0", "ffn_moe_topk-1", 1))

    def test_top_k_is_not_truncated(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("-0 2 2 |", "-0 1 2 |", 1))

    def test_missing_expert_id_is_rejected(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("| 3 0 |", "| 3 |", 1))

    def test_duplicate_expert_is_rejected(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("| 3 0 |", "| 3 3 |", 1))

    def test_out_of_range_and_negative_expert_are_rejected(self):
        for replacement in ("| 4 0 |", "| -1 0 |"):
            with self.subTest(replacement=replacement), self.assertRaises(RouteError):
                self.parse(trace().replace("| 3 0 |", replacement, 1))

    def test_no_token_is_dropped_at_last_layer(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("-1 2 2 | 2 0 | 3 1", "-1 2 1 | 2 0", 1))

    def test_unmarked_post_prompt_routes_are_rejected(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("GENTOK 101\n", "", 1))

    def test_generation_before_prompt_ends_is_rejected(self):
        with self.assertRaises(RouteError): self.parse("GENTOK 101\n" + trace())

    def test_pending_generated_token_requires_complete_forward(self):
        with self.assertRaises(RouteError): self.parse(trace() + "GENTOK 103\n")

    def test_duplicate_generated_marker_is_rejected(self):
        with self.assertRaises(RouteError): self.parse(trace().replace("GENTOK 101", "GENTOK 101\nGENTOK 102", 1))

    def test_metadata_is_not_guessed(self):
        for line in ("print_info: n_layer = 2\n", "print_info: n_expert = 4\n", "print_info: n_expert_used = 2\n", "number of input tokens = 3\n"):
            with self.subTest(line=line), self.assertRaises(RouteError):
                self.parse(err=stderr().replace(line, ""))

    def test_repeated_metadata_is_ambiguous(self):
        with self.assertRaises(RouteError): self.parse(err=stderr() + "number of input tokens = 3\n")

    def test_unfinished_or_failed_capture_is_rejected(self):
        for value in (stderr().split("llama_perf")[0], stderr() + "failed to eval generated token 0\n"):
            with self.subTest(value=value), self.assertRaises(RouteError): self.parse(err=value)

    def test_raw_source_hashes_and_output_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out, err, target = root / "out", root / "err", root / "result.json"
            out.write_bytes(trace().replace("\n", "\r\n").encode())
            err.write_bytes(stderr().encode())
            result = convert_llama_files(out, err, request_id="r", collector_contract=COLLECTOR_CONTRACT)
            self.assertEqual(result["sources"]["stdout"]["sha256"], hashlib.sha256(out.read_bytes()).hexdigest())
            dump(target, result)
            self.assertNotIn(b"\r\n", target.read_bytes())
            self.assertEqual(load(target), result)


class RouteSelectionTests(unittest.TestCase):
    def select(self, source=None, **overrides):
        args = {"request_id": "r", "source_layer": 1, "m2_layer_id": "model.layer1",
                "phase": "prefill", "position_start": 0, "tokens": 3, "information_mode": "causal_replay"}
        args.update(overrides)
        return select_m2_routes(bundle() if source is None else source, **args)

    def test_selected_layer_ids_and_expert_ids_are_preserved(self):
        result = self.select()
        self.assertEqual([row["expert_ids"] for row in result["route_trace"]], [[2, 0], [3, 1], [2, 3]])
        self.assertTrue(all(row["layer_id"] == "model.layer1" for row in result["route_trace"]))
        self.assertFalse(result["provenance"]["expert_ids_remapped"])
        self.assertEqual(result["request"]["tokens"], 3)
        self.assertFalse(result["provenance"]["initial_state_required"])

    def test_exact_decode_span_is_one_token(self):
        result = self.select(phase="decode", position_start=3, tokens=1)
        self.assertEqual(result["route_trace"][0]["expert_ids"], [3, 0])
        self.assertTrue(result["provenance"]["initial_state_required"])
        with self.assertRaises(RouteError): self.select(phase="decode", position_start=3, tokens=2)

    def test_phase_cannot_be_guessed_from_n_equal_one(self):
        with self.assertRaises(RouteError): self.select(phase="decode", position_start=2, tokens=1)

    def test_missing_span_is_rejected(self):
        for overrides in ({"request_id": "missing"}, {"tokens": 4}, {"source_layer": 2}, {"position_start": True}, {"information_mode": "online"}):
            with self.subTest(overrides=overrides), self.assertRaises(RouteError): self.select(**overrides)

    def test_no_duplicate_bundle_token(self):
        source = bundle(); source["records"].append(copy.deepcopy(source["records"][0]))
        with self.assertRaises(RouteError): validate_bundle(source)

    def test_no_missing_bundle_token(self):
        source = bundle(); source["records"].pop()
        with self.assertRaises(RouteError): validate_bundle(source)

    def test_boolean_expert_is_rejected(self):
        source = bundle(); source["records"][0]["expert_ids"][0] = True
        with self.assertRaises(RouteError): validate_bundle(source)

    def test_training_arrays_and_unknown_bundle_are_rejected(self):
        for source in ([0, 1], {"expert_ids": [0, 1]}, {"schema_version": "training"}):
            with self.subTest(source=source), self.assertRaises(RouteError): validate_bundle(source)

    def test_selection_does_not_mutate_input(self):
        source = bundle(); before = copy.deepcopy(source)
        result = self.select(source)
        result["route_trace"][0]["expert_ids"][0] = 99
        self.assertEqual(source, before)

    def test_invocation_fragment_fields_are_ready_for_m2(self):
        result = invocation_slice(bundle(), request_id="r", source_layer=1, m2_layer_id="model.layer1",
                                  phase="decode", position_start=4, tokens=1, information_mode="causal_replay")
        self.assertEqual(result["requests"], [{"request_id": "r", "tokens": 1, "position_start": 4}])
        self.assertEqual(result["route_trace"][0]["expert_ids"], [0, 2])

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"phase":"prefill","phase":"decode"}', encoding="utf-8")
            with self.assertRaises(RouteError): load(path)


class FreeTokenRangeTests(unittest.TestCase):
    @staticmethod
    def fixture():
        import struct
        raw = json.dumps({"model_geometry": {"num_experts": 16, "num_moe_layers": 2, "top_k": 10},
                          "quant": "source-only", "uninterpreted_metadata": {"value": "preserved"}}).encode()
        hbytes = (32 + len(raw) + 7) // 8 * 8
        header = struct.pack("<8sHHIHHHHII", b"FTEXTRC\0", 3, 0x0102, hbytes, 176, 210, 2, 10, 129, len(raw)) + raw
        header += bytes(hbytes - len(header))
        block_bytes = 176 + 2 * 210
        def block(uid, phase, index):
            out = bytearray(block_bytes)
            struct.pack_into("<IBB2xIQIII", out, 0, uid, phase, 0, index, 1234, 32, 16, 0)
            struct.pack_into("<4I", out, 32, 100 + index, 99 + index, 0xFFFFFFFF, 0xFFFFFFFF)
            for layer in range(2):
                start = 176 + layer * 210
                struct.pack_into("<IBBHI", out, start, uid, phase, layer, 0, index)
                struct.pack_into("<10H", out, start + 12, *range(10))
                struct.pack_into("<10e", out, start + 32, *([0.1] * 10))
                struct.pack_into("<10H", out, start + 52, *([65535] * 10))
                struct.pack_into("<32H", out, start + 72, *(list(range(10)) + list(range(22))))
            return bytes(out)
        return header, block_bytes, block

    def parse(self, **overrides):
        from moe_routes import parse_freetoken_ranges
        header, block_bytes, block = self.fixture()
        args = {"header": header, "chunks": [(len(header), block(7, 0, 0) + block(7, 0, 1) + block(8, 0, 0)),
                                              (len(header) + 20 * block_bytes, block(7, 1, 0) + block(7, 1, 1))],
                "request_uid": 7, "request_id": "uid7", "source_metadata": {"model": "Parser Model", "publisher": {"caveats": ["metadata preserved"]}}}
        args.update(overrides)
        return parse_freetoken_ranges(**args)

    def test_v3_route_prefix_records_and_p_d_absolute_positions(self):
        result = self.parse()
        self.assertEqual(result["requests"][0]["prefill_tokens"], 2)
        self.assertEqual(result["requests"][0]["decode_tokens"], 2)
        rows = [row for row in result["records"] if row["layer_id"] == "0"]
        self.assertEqual([row["absolute_position"] for row in rows], [0, 1, 2, 3])
        self.assertEqual(rows[2]["source_tok_idx"], 0)
        self.assertEqual(result["source_run_header"]["uninterpreted_metadata"], {"value": "preserved"})
        self.assertEqual(result["source_metadata"]["publisher"]["caveats"], ["metadata preserved"])
        self.assertFalse(result["verification"]["publisher_file_sha256_verified"])

    def test_partial_ranges_and_misalignment_are_rejected(self):
        header, width, block = self.fixture()
        for chunks in ([(len(header) + 1, block(7, 0, 0))], [(len(header), block(7, 0, 0)[:-1])]):
            with self.subTest(chunks=chunks), self.assertRaises(RouteError): self.parse(chunks=chunks)

    def test_overlapping_ranges_are_rejected(self):
        header, width, block = self.fixture()
        chunks = [(len(header), block(7, 0, 0)), (len(header), block(7, 0, 0))]
        with self.assertRaises(RouteError): self.parse(chunks=chunks)

    def test_missing_prefill_origin_or_decode_origin_is_rejected(self):
        header, width, block = self.fixture()
        for blocks in (block(7, 0, 1) + block(7, 1, 0), block(7, 0, 0) + block(7, 1, 1), block(7, 0, 0)):
            with self.subTest(blocks=blocks), self.assertRaises(RouteError): self.parse(chunks=[(len(header), blocks)])

    def test_step_and_layer_identity_must_agree(self):
        import struct
        header, width, block = self.fixture()
        damaged = bytearray(block(7, 0, 0)); struct.pack_into("<I", damaged, 176, 8)
        with self.assertRaises(RouteError): self.parse(chunks=[(len(header), bytes(damaged))])

    def test_top32_prefix_must_be_execution_experts(self):
        import struct
        header, width, block = self.fixture()
        damaged = bytearray(block(7, 0, 0)); struct.pack_into("<H", damaged, 176 + 72, 11)
        with self.assertRaises(RouteError): self.parse(chunks=[(len(header), bytes(damaged))])

    def test_out_of_range_expert_and_invalid_gate_are_rejected(self):
        import struct
        header, width, block = self.fixture()
        for offset, fmt, value in ((176 + 12, "<H", 16), (176 + 32, "<e", float("nan"))):
            damaged = bytearray(block(7, 0, 0)); struct.pack_into(fmt, damaged, offset, value)
            with self.subTest(offset=offset), self.assertRaises(RouteError): self.parse(chunks=[(len(header), bytes(damaged))])

    def test_header_magic_width_geometry_and_padding_are_strict(self):
        import struct
        header, width, block = self.fixture()
        for offset, fmt, value in ((0, "<8s", b"BADMAGIC"), (16, "<H", 32), (20, "<H", 3), (22, "<H", 8)):
            damaged = bytearray(header); struct.pack_into(fmt, damaged, offset, value)
            with self.subTest(offset=offset), self.assertRaises(RouteError): self.parse(header=bytes(damaged))
        with self.assertRaises(RouteError): self.parse(header=header[:-1])

    def test_phase_2_is_prefill_with_source_semantics_preserved(self):
        header, width, block = self.fixture()
        result = self.parse(chunks=[(len(header), block(7, 2, 0) + block(7, 1, 0))])
        self.assertEqual(result["records"][0]["phase"], "prefill")
        self.assertEqual(result["records"][0]["source_phase"], 2)

    def test_scores_are_exported_as_observed_metadata(self):
        source = self.parse()
        result = select_m2_routes(source, request_id="uid7", source_layer=1, m2_layer_id="target",
                                 phase="decode", position_start=2, tokens=1, information_mode="causal_replay")
        self.assertEqual(result["route_trace"][0]["scores"], source["records"][4]["scores"])
        self.assertFalse(result["provenance"]["performance_costs_provided"])


class BoundedDownloadTests(unittest.TestCase):
    def response(self, *, status=206, content_range="bytes 4-6/100", payload=b"abc"):
        from unittest.mock import MagicMock
        response = MagicMock()
        response.status = status
        response.headers = {"Content-Range": content_range, "ETag": '"source"'}
        response.read.return_value = payload
        response.__enter__.return_value = response
        return response

    def test_download_retains_range_digest_and_never_claims_full_file(self):
        from unittest.mock import patch
        from moe_routes import download_range
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch("urllib.request.urlopen", return_value=self.response()) as opener:
                result = download_range("https://public.example/pinned/file", offset=4, size_bytes=3,
                                        output=path / "raw.bin", metadata_output=path / "meta.json")
            self.assertEqual((path / "raw.bin").read_bytes(), b"abc")
            self.assertEqual(result["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertFalse(result["full_object_sha256_verified"])
            self.assertEqual(opener.call_args.args[0].headers["Range"], "bytes=4-6")

    def test_ignored_misaligned_or_truncated_response_does_not_write(self):
        from unittest.mock import patch
        from moe_routes import download_range
        for response in (self.response(status=200), self.response(content_range="bytes 0-2/100"),
                         self.response(payload=b"ab"), self.response(payload=b"abcd")):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                with patch("urllib.request.urlopen", return_value=response), self.assertRaises(RouteError):
                    download_range("https://public.example/file", offset=4, size_bytes=3,
                                   output=path / "raw", metadata_output=path / "meta")
                self.assertFalse((path / "raw").exists())

    def test_budget_bad_ranges_and_non_https_do_not_start_network(self):
        from unittest.mock import patch
        from moe_routes import download_range
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for overrides in ({"size_bytes": 4, "max_bytes": 3}, {"offset": -1}, {"size_bytes": True}, {"url": "http://source/file"}):
                args = {"url": "https://public.example/file", "offset": 0, "size_bytes": 3,
                        "output": root / "raw", "metadata_output": root / "meta"}
                args.update(overrides)
                with patch("urllib.request.urlopen") as opener, self.subTest(overrides=overrides), self.assertRaises(RouteError):
                    download_range(**args)
                opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
