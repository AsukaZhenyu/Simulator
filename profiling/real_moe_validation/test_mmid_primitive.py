"""Source/fixture contracts only; never launches a compiler or CUDA process."""
from pathlib import Path
import unittest
from .mmid_primitive import extract,function_text,make_suite,ANCHOR,FENCE,ROOT,TOKENS

class MMIDPrimitivePreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT.parent/'llama.cpp-implement/release-b8705/llama.cpp-b8705/ggml/src/ggml-cuda'
        cls.mmid=(source/'mmid.cu').read_text(encoding='utf-8');cls.common=(source/'common.cuh').read_text(encoding='utf-8')
        cls.header,cls.metadata,cls.original,cls.fixed=extract(cls.mmid,cls.common)

    def test_kernel_variants_restore_exact_pinned_original(self):
        marker='template <int n_expert_used_template>\n__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)\nstatic __global__ void mm_ids_helper('
        actual=function_text(self.mmid,marker)
        self.assertEqual(self.original.replace('void mm_ids_helper_original(','void mm_ids_helper('),actual)
        self.assertEqual(self.fixed.replace('void mm_ids_helper_syncwarp(','void mm_ids_helper(').replace(FENCE,''),actual)

    def test_real_warp_helpers_copied_without_reimplementation(self):
        for marker in ('static constexpr __device__ int ggml_cuda_get_physical_warp_size()','template<int width = WARP_SIZE>\nstatic __device__ __forceinline__ int warp_reduce_sum(int x)','template<int width = WARP_SIZE>\nstatic __device__ __forceinline__ int warp_reduce_any(int x)'):
            self.assertIn(function_text(self.common,marker),self.header)
        self.assertIn('__reduce_add_sync(0xffffffff, x)',self.header);self.assertIn('#define GGML_CUDA_CC_AMPERE',self.header)

    def test_only_one_fullmask_fence_after_reduction_before_read_or_return(self):
        self.assertEqual(self.header.count('__syncwarp('),1)
        self.assertGreater(self.fixed.index(FENCE),self.fixed.index('nex_prev = warp_reduce_sum'))
        self.assertLess(self.fixed.index(FENCE),self.fixed.index('for (int itc = threadIdx.x;'))
        self.assertLess(self.fixed.index(FENCE),self.fixed.index('if (threadIdx.x != 0)'))

    def test_changed_or_duplicate_kernel_anchor_refused(self):
        for source in (self.mmid.replace(ANCHOR,'changed reduction\n',1),self.mmid+self.mmid):
            with self.assertRaises(ValueError):extract(source,self.common)

    def test_changed_device_macro_or_helper_refused(self):
        with self.assertRaises(ValueError):extract(self.mmid,self.common.replace('#define WARP_SIZE 32','#define WARP_SIZE 64',1))
        with self.assertRaises(ValueError):extract(self.mmid,self.common.replace('int warp_reduce_sum(int x)','int altered_reduce_sum(int x)',1))

    def test_fixed_42_cases_cover_tails_patterns_layouts_and_exact_oracles(self):
        suite=make_suite();self.assertEqual(len(suite['cases']),42);self.assertEqual(suite['planned_launches'],252)
        self.assertEqual({c['n_tokens'] for c in suite['cases']},set(TOKENS));self.assertEqual(len({c['id'] for c in suite['cases']}),42)
        for c in suite['cases']:
            self.assertEqual(len(c['reference']['ids_src1']),c['n_tokens']*8);self.assertEqual(len(c['reference']['ids_dst']),c['n_tokens']*8)
            self.assertEqual(c['reference']['expert_bounds'][-1],c['n_tokens']*8)
            for token in range(c['n_tokens']):self.assertEqual(len(set(c['ids_storage'][token*c['si1']:token*c['si1']+8])),8)

    def test_own_device_buffers_and_dynamic_shared_are_bounded(self):
        for c in make_suite()['cases']:
            length=c['n_tokens']*8;bounds=c['n_experts']+1
            owned=(len(c['ids_storage'])+2*(2*(length+32)+(bounds+32)))*4
            self.assertLessEqual(owned,1024*1024);self.assertLessEqual(c['n_tokens']*4,1024)

    def test_probe_has_distinct_backing_poison_and_complete_output_oracles(self):
        text=(Path(__file__).parent/'mmid_native/mmid_probe.cu').read_text()
        self.assertIn('variant=="original"?a_src:b_src',text);self.assertIn('variant=="original"?0xa5:0x5a',text)
        self.assertIn('redzones_unchanged',text);self.assertIn('std::memcmp',text);self.assertIn('CPU references differ',text)
        self.assertIn('CREATE_NEW',text);self.assertNotIn('llama_model_load',text)

if __name__=='__main__':unittest.main()
