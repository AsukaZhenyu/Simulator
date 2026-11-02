"""No-hardware contracts for host-only numerical diagnostic transformation."""
from pathlib import Path
import tempfile
import unittest
from .numerical_v6 import native_v6,ORDER,ROOT,prepare_cohort
from .dataset import read

class NumericalDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=(ROOT/'build/real-moe-validation-20261027/tap-runtime-v5b/source/real_moe_tap_probe.cpp').read_text(encoding='utf-8')
        cls.result=native_v6(cls.original)

    def test_new_schema_and_host_export_after_end_tick(self):
        self.assertIn('real-moe-numerical-request-6',self.result);self.assertIn('real-moe-numerical-native-6',self.result)
        self.assertGreater(self.result.index('const auto logit_artifact=save_host_logits'),self.result.index('const auto end=tick();'))
        self.assertIn('logit_snapshots.data(),generated.size(),vocab_size',self.result)

    def test_no_new_GPU_copy_sync_callback_or_math_flags(self):
        for name in ('llama_decode(','llama_synchronize(','llama_memory_clear(','ggml_backend_tensor_get','cudaMemcpy','cudaStreamSynchronize','cp.op_offload','cp.cb_eval'):
            self.assertEqual(self.result.count(name),self.original.count(name),name)
        self.assertIn('CPU/CUDA/llama DLLs unchanged from frozen v5b',self.result)

    def test_actual_batches_before_forward_and_no_all_token_outputs(self):
        self.assertEqual(self.result.count('actual_batches.record(batch.value,forward);'),2)
        self.assertIn('b.logits[i]=last_output&&i==count-1',self.result)
        self.assertIn('actual_batch_inputs',self.result);self.assertIn('context_reset_index',self.result)

    def test_environment_is_recorded_and_checked_not_silently_changed(self):
        self.assertIn('diagnostic_environment()==request.at("expected_runtime_environment")',self.result)
        self.assertNotIn('setenv(',self.result);self.assertNotIn('_putenv',self.result)
        self.assertIn('runtime_default_enabled',self.result)

    def test_pinned_transform_rejects_changed_source(self):
        for anchor in ('#include "gpu_tap_pod.hpp"','json graph_observations=json::array();int64_t logit_copy_ns=0,graph_observer_ns=0;','real-moe-tap-request-5'):
            with self.assertRaises(ValueError):native_v6(self.original.replace(anchor,'CHANGED_ANCHOR',1))

    def test_fixed_six_processes_balance_modes_and_sequences(self):
        self.assertEqual(len(ORDER),6)
        self.assertEqual({(mode,seq) for _,mode,seq in ORDER},{(mode,seq) for mode in ('generate','replay','capture') for seq in ('SSLS','LSSS')})
        self.assertEqual([mode for _,mode,_ in ORDER],['generate','replay','capture','generate','replay','capture'])

    def test_host_writer_exclusive_finite_and_bounded(self):
        header=(Path(__file__).parent/'numerical_native_v6/host_snapshots.hpp').read_text()
        self.assertIn('CREATE_NEW',header);self.assertIn('std::isfinite',header);self.assertIn('new_GPU_reads',header)
        self.assertIn('export_in_target_interval",false',header);self.assertIn('b.n_seq_id[i]==1',header)

    def test_prepared_cohort_is_balanced_and_has_globally_unique_observations(self):
        root=(ROOT/'build/real-moe-validation-20261027/numerical-cohort-test-fixtures').resolve()
        if ROOT.resolve() not in root.parents:raise ValueError('fixture outside workspace')
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='case-',dir=root) as td:
            out=Path(td)/'cohort';prepare_cohort(out,ROOT/'build/real-moe-validation-20261027/numerical-driver-v6',ROOT/'build/real-moe-validation-20261027/tap-smoke-v5b-gpu-consumer-v1')
            m=read(out/'manifest.json');self.assertEqual(m['fixed_order'],[x[0] for x in ORDER]);ids=[]
            for job in m['jobs']:
                req=read(job['request']['path']);self.assertEqual(len(req['requests']),4);self.assertEqual(req['threads'],16);self.assertEqual(req['batch_size'],256)
                for i,row in enumerate(req['requests']):
                    ids.append(row['id']);self.assertEqual(row['context_reset_index'],i);self.assertIn('origin_case_id',row)
                    self.assertEqual('output_token_ids' in row,req['mode']!='generate')
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(len(ids),24)
            self.assertEqual(m['parent_preflight_bytes'],6076332864);self.assertEqual(m['native_guard_bytes'],5824125760);self.assertEqual(m['resident_peak_limit_bytes'],5287254848)
            self.assertFalse((out/'jobs').exists());self.assertFalse(m['whole_model_accuracy_evaluated'])

if __name__=='__main__':unittest.main()
