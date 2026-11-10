"""Software-only contracts for the additive v5 GPU observation transforms.

Tests read original/frozen sources and transform strings. They do not compile,
construct a backend, load a model, initialize CUDA, or collect performance data.
"""
from pathlib import Path
import re
import unittest

from . import tap_v5


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CUDA_SOURCE = ROOT.parent / "llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml/src/ggml-cuda/ggml-cuda.cu"
FROZEN_NATIVE = ROOT / "build/real-moe-validation-20261027/tap-runtime-v4b/source/real_moe_tap_probe.cpp"


class GPUConsumerTapSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Both inputs are materialised outside the checkout: the pinned llama.cpp
        # tree and the frozen tap source, which is a build product archived under
        # paper-write/simulator-archive/build/. Skip rather than error when a
        # fresh clone cannot supply them; the transform itself is untested then,
        # not wrong.
        for path in (CUDA_SOURCE, FROZEN_NATIVE):
            if not path.is_file():
                raise unittest.SkipTest("%s is not in the checkout" % path)
        cls.original_cuda = CUDA_SOURCE.read_text(encoding="utf-8")
        cls.original_native = FROZEN_NATIVE.read_text(encoding="utf-8")
        cls.cuda = tap_v5.patched_cuda(cls.original_cuda)
        cls.native = tap_v5.native_v5(cls.original_native)
        cls.pod = (HERE / "tap_native_v5/gpu_tap_pod.hpp").read_text(encoding="utf-8")
        cls.api = (HERE / "tap_native_v5/gpu_tap_api.h").read_text(encoding="utf-8")

    def test_cuda_transform_has_exactly_the_three_declared_changes(self):
        restored = self.cuda.replace(tap_v5.HOOK + "\n", "", 1)
        restored = restored.replace(tap_v5.DISPATCH_NEW, tap_v5.DISPATCH, 1)
        restored = restored.replace(tap_v5.GRAPH_NEW, tap_v5.GRAPH, 1)
        self.assertEqual(restored, self.original_cuda)

    def test_cuda_missing_each_pinned_anchor_fails_closed(self):
        for anchor in (tap_v5.DECL, tap_v5.DISPATCH, tap_v5.GRAPH):
            with self.subTest(anchor=anchor[:60]):
                with self.assertRaises(ValueError):
                    tap_v5.patched_cuda(self.original_cuda.replace(anchor, "// removed anchor", 1))

    def test_cuda_duplicated_each_pinned_anchor_fails_closed(self):
        for anchor in (tap_v5.DECL, tap_v5.DISPATCH, tap_v5.GRAPH):
            with self.subTest(anchor=anchor[:60]):
                with self.assertRaises(ValueError):
                    tap_v5.patched_cuda(self.original_cuda + "\n" + anchor)

    def test_cuda_altered_anchor_does_not_apply_fuzzily(self):
        altered = self.original_cuda.replace(tap_v5.DECL, tap_v5.DECL.replace(" & ctx", "& ctx"), 1)
        with self.assertRaises(ValueError):
            tap_v5.patched_cuda(altered)

    def test_observation_is_once_at_consumer_before_fast_path_dispatch(self):
        consumer = self.cuda.split(tap_v5.DECL, 1)[1].split("static bool ggml_cuda_compute_forward", 1)[0]
        self.assertEqual(consumer.count("sim_gpu_reserve_v5(dst, ids,"), 1)
        self.assertLess(consumer.index("sim_gpu_reserve_v5(dst, ids,"),
                        consumer.index("if (src1->type == GGML_TYPE_F32"))
        self.assertEqual(self.cuda.count("cudaMemcpy2DAsync("), self.original_cuda.count("cudaMemcpy2DAsync(") + 1)

    def test_async_copy_uses_actual_ids_strides_and_same_consumer_stream(self):
        self.assertIn("cudaMemcpy2DAsync(host, row_bytes, ids->data, ids->nb[1], row_bytes, size_t(ids->ne[1]), cudaMemcpyDeviceToHost, ctx.stream())", self.cuda)
        self.assertIn("sim_gpu_reserve_v5(dst, ids, (void *) ctx.stream(), row_bytes", self.cuda)
        self.assertIn("sim_gpu_status_v5(int(status), sim_gpu_user_v5)", self.cuda)

    def test_no_new_sync_event_or_fusion_disable_call(self):
        for token in ("cudaStreamSynchronize(", "cudaDeviceSynchronize(", "cudaEventCreate(",
                      "cudaEventRecord(", "cudaStreamBeginCapture(", "cudaStreamEndCapture(",
                      "cudaGraphLaunch(", "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_DISABLE_GRAPHS"):
            with self.subTest(token=token):
                self.assertEqual(self.cuda.count(token), self.original_cuda.count(token))
        self.assertNotRegex(tap_v5.DISPATCH_NEW, r"\b(?:cudaStreamSynchronize|cudaDeviceSynchronize|cudaEvent\w+|cupti\w+)\s*\(")

    def test_graph_decisions_are_observed_before_capture_without_new_assignment(self):
        graph = self.cuda.split("static enum ggml_status ggml_backend_cuda_graph_compute", 1)[1]
        self.assertLess(graph.index("sim_gpu_graph_v5(cgraph,"), graph.index("cudaStreamBeginCapture("))
        self.assertIn("sim_gpu_graph_v5(cgraph, simulator_graph_compiled, int(use_cuda_graph), int(cuda_graph_update_required), sim_gpu_user_v5)", graph)
        for name in ("use_cuda_graph", "cuda_graph_update_required"):
            pattern = r"\b" + name + r"\s*="
            self.assertEqual(len(re.findall(pattern, self.cuda)), len(re.findall(pattern, self.original_cuda)))

    def test_native_schema_is_new_and_capture_timings_cannot_be_bare(self):
        self.assertIn('"real-moe-tap-request-5"', self.native)
        self.assertIn('"real-moe-tap-native-5"', self.native)
        self.assertNotIn('"real-moe-tap-request-4"', self.native)
        self.assertNotIn('"real-moe-tap-native-4"', self.native)
        self.assertIn('{"timing_usable_as_bare_reference",false}', self.native)
        self.assertIn('"observer_extra_flush_ns",0', self.native)
        self.assertIn('"observer_completion_boundary","existing llama_synchronize only"', self.native)

    def test_native_missing_or_duplicate_pinned_replacements_fail_closed(self):
        anchors = ('#include "route_context_inspect.hpp"',
                   "const json &request,Capture &capture,bool record)",
                   "TapBinding tap_binding(capture,mode==\"capture\");",
                   "real-moe-tap-native-4")
        for anchor in anchors:
            with self.subTest(anchor=anchor):
                with self.assertRaises(ValueError):
                    tap_v5.native_v5(self.original_native.replace(anchor, "// removed pinned text", 1))
                with self.assertRaises(ValueError):
                    tap_v5.native_v5(self.original_native + "\n" + anchor)

    def test_native_does_not_add_sync_callback_or_tensor_graph_flags(self):
        for token in ("llama_synchronize(", "ggml_backend_synchronize(", "ggml_set_output(", "ggml_set_input(",
                      "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_DISABLE_GRAPHS"):
            with self.subTest(token=token):
                self.assertEqual(self.native.count(token), self.original_native.count(token))
        self.assertNotRegex(self.native, r"cp\.cb_eval\s*=(?!=)")
        self.assertIn("require(cp.cb_eval==nullptr", self.native)
        self.assertIn('"runtime_default_enabled"', self.native)
        self.assertIn('result["scheduler_callback_enabled"]=false', self.native)

    def test_gpu_graph_partition_capture_or_replay_is_rejected(self):
        graph_hook = self.pod.split("static void graph(", 1)[1].split("static void *reserve(", 1)[0]
        self.assertIn("if(g.enabled&&use&&experts)g.error=10;", graph_hook)
        self.assertIn("GGML_OP_MUL_MAT_ID", graph_hook)
        self.assertIn('"unsupported if any GPU expert partition uses CUDA graph; no cached route reuse"', self.native)

    def test_graph_events_report_submission_not_device_completion(self):
        self.assertIn('"capture_update_launch"', self.pod)
        self.assertIn('"replay_launch"', self.pod)
        self.assertIn('"device_completion_observed_here",false', self.pod)
        self.assertIn('"USE_CUDA_GRAPH_compiled"', self.pod)

    def test_pinned_unique_slots_and_allocation_envelope_are_explicit(self):
        self.assertIn("cudaHostAlloc(ptr, bytes, cudaHostAllocDefault)", tap_v5.HOOK)
        self.assertIn("cudaFreeHost(ptr)", tap_v5.HOOK)
        self.assertIn("bytes<=80000000", self.pod)
        self.assertIn("if(bytes>pinned_bytes)", self.pod)
        self.assertIn("g.data.ids_used+count>g.data.ids.size()", self.pod)
        self.assertIn("(g.data.ids_used+count)*4>g.pinned_bytes", self.pod)
        self.assertIn("static_cast<char*>(g.pinned)+g.data.ids_used*4", self.pod)
        self.assertIn("g.data.ids_used+=count", self.pod)

    def test_reserve_requires_contiguous_id_slots_and_bounded_pitch(self):
        reserve = self.pod.split("static void *reserve(", 1)[1].split("static void status(", 1)[0]
        for token in ("ids->type!=GGML_TYPE_I32", "ids->nb[0]!=sizeof(int32_t)",
                      "ids->ne[0]!=g.data.top_k", "ids->ne[1]>4096", "ids->ne[2]!=1",
                      "ids->ne[3]!=1", "row_bytes!=size_t(g.data.top_k)*4", "ids->nb[1]<row_bytes"):
            self.assertIn(token, reserve)
        for forbidden in ("push_back(", "resize(", "assign(", "json", "throw ", "cudaMemcpy", "Synchronize("):
            self.assertNotIn(forbidden, reserve)

    def test_each_execution_clears_prior_route_and_event_ledgers(self):
        prepare = self.pod.split("void prepare(", 1)[1].split("static void graph(", 1)[0]
        self.assertIn("error=0", prepare)
        self.assertIn("event_count=0", prepare)
        self.assertIn("copy_count=0", prepare)
        self.assertIn("data.prepare(forwards,rows,layers,experts,k)", prepare)
        output = self.pod.split("json output(){", 1)[1].split("struct GPUBinding", 1)[0]
        self.assertIn("if(error)return json::array();", output)
        self.assertIn("std::memcpy(data.ids.data(),pinned,data.ids_used*sizeof(int32_t))", output)
        self.assertNotIn("Synchronize(", output)

    def test_unknown_gpu_stages_and_async_failures_remain_errors(self):
        self.assertIn("if(stage<0){g.error=12;return nullptr;}", self.pod)
        self.assertIn("if(status&&!g.error)g.error=1000+status", self.pod)
        self.assertIn('"gpu_capture_error_code",gpu.error', self.native)
        self.assertIn('"gpu_capture_status",gpu.error?"unsupported":"observed"', self.native)

    def test_observer_abi_is_explicit_and_not_a_scheduler_eval_callback(self):
        self.assertIn("simulator_gpu_tap_set_v5", self.api)
        self.assertIn("sim_gpu_graph_event", self.api)
        self.assertIn("sim_gpu_copy_status", self.api)
        self.assertNotIn("cb_eval", self.api)
        self.assertNotIn("ggml_backend_sched_set_eval_callback", self.cuda)


if __name__ == "__main__":
    unittest.main()
