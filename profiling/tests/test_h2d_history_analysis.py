import unittest

from profiling.h2d_history.analysis import correlation_map, one_physical, union_ns


class HistoryActivityAccountingTests(unittest.TestCase):
    def records(self):
        return [
            dict(kind='external_correlation',correlation_id=10,external_id=1),
            dict(kind='runtime_api',timestamp_valid=True,correlation_id=10,
                 process_id=7,thread_id=9,begin_ns=100,end_ns=300,name='cudaMemcpyAsync'),
            dict(kind='driver_api',timestamp_valid=True,correlation_id=20,
                 process_id=7,thread_id=9,begin_ns=120,end_ns=200,name='cuMemcpyHtoDAsync'),
            dict(kind='memcpy',timestamp_valid=True,correlation_id=20,
                 begin_ns=160,end_ns=190,bytes=2097152,copy_kind=1),
        ]

    def test_union_does_not_add_overlapping_intervals(self):
        self.assertEqual(union_ns([(1,5),(3,8),(10,12)]),9)
        self.assertEqual(union_ns([(1,5),(3,8),(10,12)],(2,11)),7)

    def test_nested_driver_can_bridge_explicit_runtime_anchor(self):
        mapped,methods,graph=correlation_map(self.records())
        self.assertEqual(mapped[20],{1})
        self.assertEqual(methods[20],'nested_runtime_driver_bridge')
        self.assertEqual(len(graph['bridges']),1)

    def test_temporal_overlap_on_different_thread_is_not_correlation(self):
        records=self.records(); records[2]['thread_id']=11
        mapped,_,graph=correlation_map(records)
        self.assertNotIn(20,mapped)
        self.assertFalse(graph['bridges'])

    def test_conflicting_explicit_anchors_are_not_forced_together(self):
        records=self.records()+[dict(kind='external_correlation',correlation_id=20,external_id=2)]
        mapped,_,graph=correlation_map(records)
        self.assertEqual(mapped[10],{1}); self.assertEqual(mapped[20],{2})
        self.assertTrue(graph['conflicts']); self.assertFalse(graph['bridges'])

    def test_payload_coverage_allows_disjoint_envelope_accounting(self):
        actual=one_physical(dict(begin_ns=100,end_ns=350),
            dict(kind='COPY',src='dram',dst='vram',bytes=2097152),self.records(),True)
        self.assertTrue(actual['exact_payload_activity_coverage'])
        self.assertEqual(actual['device_memcpy_union_ns'],30)
        self.assertEqual(actual['pre_device_wall_ns'],60)
        self.assertEqual(actual['post_device_wall_ns'],160)
        self.assertEqual(actual['host_wall_outside_memcpy_union_ns'],220)
        # Runtime and driver overlap; these are separate diagnostic domains.
        self.assertEqual(actual['api_union_ns_by_name']['runtime_api:cudaMemcpyAsync'],200)
        self.assertEqual(actual['api_union_ns_by_name']['driver_api:cuMemcpyHtoDAsync'],80)
        self.assertFalse(actual['host_device_times_additive'])

    def test_missing_or_outside_payload_does_not_create_partition(self):
        semantic=dict(kind='COPY',src='dram',dst='vram',bytes=2097152)
        missing=one_physical(dict(begin_ns=100,end_ns=350),semantic,[],True)
        self.assertFalse(missing['exact_payload_activity_coverage'])
        self.assertIsNone(missing['device_memcpy_union_ns'])
        self.assertIsNone(missing['pre_device_wall_ns'])
        outside=one_physical(dict(begin_ns=100,end_ns=180),semantic,self.records(),True)
        self.assertFalse(outside['exact_payload_activity_coverage'])
        self.assertIsNone(outside['pre_device_wall_ns'])

    def test_wrong_copy_direction_or_incomplete_capture_cannot_qualify(self):
        semantic=dict(kind='COPY',src='dram',dst='vram',bytes=2097152)
        records=self.records(); records[-1]['copy_kind']=2
        wrong=one_physical(dict(begin_ns=100,end_ns=350),semantic,records,True)
        self.assertFalse(wrong['exact_payload_activity_coverage'])
        invalid=one_physical(dict(begin_ns=100,end_ns=350),semantic,self.records(),False)
        self.assertFalse(invalid['exact_payload_activity_coverage'])


if __name__=='__main__': unittest.main()
