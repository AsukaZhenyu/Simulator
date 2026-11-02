"""Small source/protocol tests; no model, backend or performance measurement."""
import unittest
import re
from pathlib import Path
from .tap_v4 import ANCHOR, HOOK, REPLACEMENT, patched_cpu

class ConsumerTapSourceTests(unittest.TestCase):
    def test_observer_precedes_repack_and_preserves_dispatch(self):
        source='// ggml_compute_forward_mul_mat_id\nvoid before() {}\n'+ANCHOR+'\n return;\n}'
        changed=patched_cpu(source)
        self.assertLess(changed.index('simulator_moe_observer(tensor'),changed.index('ggml_cpu_extra_compute_forward'))
        self.assertIn('params->ith == 0 && tensor->op == GGML_OP_MUL_MAT_ID',changed)
        self.assertEqual(changed.replace(HOOK+'\n','',1).replace(REPLACEMENT,ANCHOR,1),source)

    def test_changed_upstream_fails_loudly(self):
        for source in ('',ANCHOR, '// ggml_compute_forward_mul_mat_id\n'+ANCHOR+ANCHOR):
            with self.assertRaises(ValueError):patched_cpu(source)

    def test_no_scheduler_callback_or_tensor_mutation(self):
        root=Path(__file__).parent/'tap_native'
        source=(root/'real_moe_tap_probe.cpp').read_text(encoding='utf-8')
        self.assertIsNone(re.search(r'cp\.cb_eval\s*=(?!=)',source))
        self.assertIn('require(cp.cb_eval==nullptr',source)
        self.assertNotIn('ggml_set_output',source)
        self.assertNotIn('ggml_set_input',source)
        self.assertIn('!record&&!observe_logits&&!observe_graph',source)

    def test_callback_has_no_allocation_json_or_synchronization(self):
        source=(Path(__file__).parent/'tap_native/route_tap_pod.hpp').read_text(encoding='utf-8')
        callback=source.split('static void callback(',1)[1].split('\n json output()',1)[0]
        for forbidden in ('push_back','resize(','assign(','json','throw ','synchronize','tensor_get','new '):
            self.assertNotIn(forbidden,callback)
        self.assertIn('source->nb[1]',callback)
        self.assertIn('source->nb[0]',callback)

if __name__=='__main__':unittest.main()
