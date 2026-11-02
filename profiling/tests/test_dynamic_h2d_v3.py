from copy import deepcopy
from unittest import TestCase
from profiling.dynamic_validation.h2d_v3 import validate_stages


def stages():
    names=('begin_tick','copy_API_enter_tick','copy_API_return_tick','extra_GPU_sync_enter_tick','extra_GPU_sync_return_tick','CPU_sync_enter_tick','CPU_sync_return_tick','end_tick')
    value=dict(zip(names,(0,1,11,12,15,16,18,20)))
    return dict(value,host_wall_ns=2000,copy_API_inclusive_ns=1000,extra_GPU_sync_ns=300,CPU_sync_ns=200,QPC_wrapper_gaps_ns=500,device_span_added_to_host_wall=False)


class H2DStageLedgerTests(TestCase):
    def test_wall_partition_closes_on_one_QPC_clock(self):validate_stages(stages(),10000000)

    def test_nested_device_span_is_not_an_additive_wall_phase(self):
        value=stages();value['device_span_added_to_host_wall']=True
        with self.assertRaisesRegex(ValueError,'added twice'):validate_stages(value,10000000)

    def test_missing_host_gap_is_rejected(self):
        value=stages();value['QPC_wrapper_gaps_ns']=0
        with self.assertRaisesRegex(ValueError,'does not close'):validate_stages(value,10000000)

    def test_reverse_host_boundary_is_rejected(self):
        value=stages();value['copy_API_return_tick']=0
        with self.assertRaisesRegex(ValueError,'phase order'):validate_stages(value,10000000)
