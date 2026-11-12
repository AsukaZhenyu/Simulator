from fractions import Fraction
from pathlib import Path
import unittest

from profiling.real_moe_validation.runtime_windows import patched_base,patched_cpu,patched_native
from profiling.real_moe_validation.runtime_window_analysis import extract_windows,fit_nonnegative,fit_prices,predict_inventory,price_fit_diagnostics,decode_history
from llm_infer_model.performance.window_work import WindowWorkSpec


def fixture():
    def n(name,op=1):return dict(name=name,op=op,type=0,ne=[256,1,1,1],weight_type=0,weight_ne=[32,32,1,1])
    nodes=[n('kq-0'),n('ffn_moe_up-0'),n('ffn_moe_gate-0')]
    def e(kind,t,first=0,count=0,cpu=False,bytes=0):
        return dict(kind=kind,tick=t,forward=0,query_tokens=1,position=12,phase='prefill',ordinal=0,cpu=cpu,first_node=first,node_count=count,requested_bytes=bytes)
    events=[e(20,0),e(21,1),e(1,2,0,1),e(5,3,0,0,bytes=64),e(2,4),e(1,9,1,2,True),e(10,10,1,1,True),e(10,30,2,1,True),e(2,50,cpu=True),e(22,52),e(23,60),e(24,65),e(25,70),e(26,80)]
    return dict(id='synthetic-clock-contract',begin_tick=0,end_tick=80,prefill_end_tick=60,
        runtime_windows=dict(schema='real-moe-runtime-windows-1',events=events,nodes=nodes,added_GPU_synchronizations=0))


class RuntimeWindowsTest(unittest.TestCase):
    def test_exact_clock_partition_and_no_GPU_time_double_add(self):
        ledger=extract_windows(fixture(),placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000)
        self.assertEqual(ledger['total_ns'],80)
        self.assertEqual(ledger['CPU_node_windows'],2)
        cpu=[w for w in ledger['windows'] if w['inventory']['scope']=='CPU_NODE']
        self.assertEqual([w['observed_ns'] for w in cpu],[20,20])
        bridge=[w for w in ledger['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE'][0]
        self.assertEqual(bridge['observed_ns'],10)
        self.assertEqual(bridge['inventory']['payload_bytes'],64)
        self.assertTrue(all('observed_ns' not in w for w in ledger['workload_inventory']))
        prices=fit_prices([ledger],['software test fixture, not measured hardware'])
        forecast=predict_inventory(ledger['workload_inventory'],prices)
        self.assertEqual(forecast['total_ns'],80)
        self.assertEqual(forecast['phase_ns'],dict(prefill=60,decode=20))

    def test_captured_ticks_carry_no_blocked_time_when_the_probe_has_no_cpu_stamp(self):
        ledger=extract_windows(fixture(),placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000,capture_ticks=True)
        self.assertTrue(all('ticks' in w for w in ledger['windows']))
        self.assertTrue(all('blocked_ns' not in w['ticks'] for w in ledger['windows']))
        self.assertTrue(all('observed_ns' not in w for w in ledger['workload_inventory']))

    def test_blocked_time_is_wall_minus_the_thread_cpu_delta(self):
        row=fixture();stamps={0:1_000,10:1_002,30:1_002,50:1_500}
        for event in row['runtime_windows']['events']:
            if event['tick'] in stamps:event['cpu_ns']=stamps[event['tick']]
        ledger=extract_windows(row,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000,capture_ticks=True)
        for w in ledger['windows']:
            ticks=w['ticks']
            if 'blocked_ns' not in ticks:continue
            self.assertEqual(ticks['blocked_ns']+Fraction(ticks['cpu_ns']),w['observed_ns'])
        bridge=next(w for w in ledger['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE')
        self.assertEqual(bridge['ticks']['blocked_ns'],8)

    def test_cycle_stamps_need_one_thread_and_a_measured_rate(self):
        row=fixture();row['runtime_windows']['thread_cycles_per_second']=1_000_000_000.0
        for event in row['runtime_windows']['events']:
            event['thread_cycles']=event['tick'];event['thread_id']=7
        ledger=extract_windows(row,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000,capture_ticks=True)
        bridge=next(w for w in ledger['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE')
        # One cycle per nanosecond: the thread consumed exactly its wall time.
        self.assertEqual(bridge['ticks']['cpu_ns'],bridge['observed_ns'])
        self.assertEqual(bridge['ticks']['blocked_ns'],0)
        # A boundary pair that crossed threads has no CPU division.
        row=fixture();row['runtime_windows']['thread_cycles_per_second']=1_000_000_000.0
        for index,event in enumerate(row['runtime_windows']['events']):
            event['thread_cycles']=event['tick'];event['thread_id']=7 if index<3 else 8
        ledger=extract_windows(row,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000,capture_ticks=True)
        first=next(w for w in ledger['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE')
        self.assertNotIn('blocked_ns',first['ticks'])
        self.assertTrue(any('blocked_ns' in w['ticks'] for w in ledger['windows']))
        # Without a measured rate the cycle stamp is not time.
        row=fixture()
        for event in row['runtime_windows']['events']:
            event['thread_cycles']=event['tick'];event['thread_id']=7
        ledger=extract_windows(row,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000,capture_ticks=True)
        self.assertTrue(all('blocked_ns' not in w['ticks'] for w in ledger['windows']))

    def test_bad_clock_rejected(self):
        row=fixture();row['runtime_windows']['events'][4]['tick']=2
        with self.assertRaisesRegex(ValueError,'nonmonotonic'):extract_windows(row,placement='cpu',frequency_hz=1_000_000_000)

    def test_existing_GPU_sync_closes_bridge_without_added_sync(self):
        row=fixture();template=dict(row['runtime_windows']['events'][4])
        row['runtime_windows']['events'][5:5]=[dict(template,kind=3,tick=5),dict(template,kind=4,tick=8)]
        ledger=extract_windows(row,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000)
        first=next(w for w in ledger['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE')
        self.assertEqual(first['observed_ns'],8)
        self.assertEqual(ledger['total_ns'],80)

    def test_missing_actual_KV_rejected(self):
        row=fixture();row['runtime_windows']['nodes'][0]['name']='not_attention'
        with self.assertRaisesRegex(ValueError,'KV dimension missing'):extract_windows(row,placement='cpu',frequency_hz=1_000_000_000)

    def test_simple_affine_estimator_has_no_negative_terms(self):
        samples=[dict(inventory=dict(query_tokens=q,kv_padded=32,payload_bytes=0),observed_ns=Fraction(10+2*q)) for q in (1,2,4,8)]
        a,b,c,d=fit_nonnegative(samples)
        self.assertTrue(all(v>=0 for v in (a,b,c,d)))
        for s in samples:
            q=s['inventory']['query_tokens'];self.assertAlmostEqual(float(a+b*q+c*q*32),float(s['observed_ns']))

    def test_patch_requires_exact_anchors(self):
        for patch in (patched_base,patched_cpu,patched_native):
            with self.assertRaises(ValueError):patch('unrelated source')

    def test_the_observer_driver_comes_from_the_repository_alone(self):
        """One frozen driver, composed from repository source, on every device.

        The driver used to be read out of whichever build directory had produced
        the first observer. That made a result the only copy of the instrument's
        own code: two checkouts of one commit could compile two different
        observers, and a checkout without that directory could not build the
        instrument at all. The composition below is the whole driver -- the
        frozen base plus the two patch stages -- so the hash moving means the
        core logic moved, which is exactly what a device port must not do.
        """
        import hashlib
        from profiling.real_moe_validation import tap_v5
        here=Path(__file__).resolve().parent
        driver=patched_native(tap_v5.native_v5(
            (here/'tap_native/real_moe_tap_probe.cpp').read_text(encoding='utf-8')))
        self.assertEqual(hashlib.sha256(driver.encode()).hexdigest(),
                         '54dceda5221e2dd74ab8b9d870e63c184b7a9caec12933ec4ae35e812f636f32')
        self.assertNotIn(str(here.parents[1]/'build'),driver,
                         'the driver may not name a build directory')

    def test_distinct_same_name_nodes_keep_distinct_actual_occurrences(self):
        row=fixture();row['runtime_windows']['nodes'][2]=dict(row['runtime_windows']['nodes'][1])
        ledger=extract_windows(row,placement='cpu',frequency_hz=1_000_000_000)
        cpu=[w['inventory'] for w in ledger['windows'] if w['inventory']['scope']=='CPU_NODE']
        self.assertNotEqual(cpu[0]['covered_nodes'],cpu[1]['covered_nodes'])
        self.assertNotEqual(cpu[0]['key'],cpu[1]['key'])
        self.assertEqual(predict_inventory(ledger['workload_inventory'],fit_prices([ledger],['synthetic']))['total_ns'],80)

    def test_pruned_matrix_query_rows_use_actual_shapes(self):
        row=fixture()
        for event in row['runtime_windows']['events']:event['query_tokens']=125
        row['runtime_windows']['nodes'][1].update(op=30,ne=[1024,8,1,1])
        row['runtime_windows']['nodes'][2].update(op=29,ne=[50304,1,1,1])
        ledger=extract_windows(row,placement='cpu',frequency_hz=1_000_000_000)
        cpu=[w['inventory'] for w in ledger['windows'] if w['inventory']['scope']=='CPU_NODE']
        self.assertEqual([w['query_tokens'] for w in cpu],[1,1])
        self.assertEqual([w['metadata']['actual_forward_query_tokens'] for w in cpu],[125,125])

    def test_decode_history_is_a_position_within_a_padded_kv_depth(self):
        # Two forwards at one padded depth, then a generation that crosses into
        # the next depth: the depth, not the request, is what the position is
        # read inside, because the excess cost returns at every depth change.
        rank={4:0,5:1,6:2,7:0,8:1}
        self.assertEqual([decode_history(f,rank,4) for f in (4,5,6,7,8)],
            ['decode:first','decode:settling','decode:continuation','decode:first','decode:settling'])
        # A window whose preceding graph is a Prefill forward keeps the previous
        # fallback so its key does not move.
        self.assertEqual(decode_history(3,rank,4),'decode:first')
        self.assertEqual(decode_history(9,rank,4),'decode:continuation')
        # The request-closing window keeps the request position even when its
        # forward enters a depth: its identity exists once per request, and a
        # depth class there leaves a key no calibration request can produce.
        self.assertEqual(decode_history(7,rank,4,closes_request=True),'decode:continuation')
        self.assertEqual(decode_history(4,rank,4,closes_request=True),'decode:first')

    def test_semantic_history_does_not_turn_forward_ordinal_into_price_identity(self):
        original=fixture();other=fixture()
        for event in other['runtime_windows']['events']:event['ordinal']=7
        a=extract_windows(original,placement='cpu',frequency_hz=1_000_000_000)
        b=extract_windows(other,placement='cpu',frequency_hz=1_000_000_000)
        self.assertEqual([v['inventory']['key'] for v in a['windows']],[v['inventory']['key'] for v in b['windows']])
        self.assertEqual(b['windows'][0]['inventory']['metadata']['actual_forward_ordinal'],7)

    def test_sparse_design_support_is_disclosed(self):
        ledger=extract_windows(fixture(),placement='cpu',frequency_hz=1_000_000_000)
        ledger['source_process_id']=123
        prices=fit_prices([ledger],['synthetic'])
        diagnostics=price_fit_diagnostics([ledger],prices)
        for value in diagnostics.values():
            self.assertEqual(value['scaled_design_rank'],1)
            self.assertEqual(value['calibration_process_count'],1)
            self.assertEqual(value['individual_terms_identifiable'],value['scaled_design_rank']==value['design_columns'])
            self.assertFalse(value['calibration_fit_is_independent_accuracy'])

    def test_GPU_tail_keeps_submission_forward_query_at_next_forward_boundary(self):
        row=fixture()
        for event in row['runtime_windows']['events']:
            event['query_tokens']=125
            if event['tick']>=9:event.update(forward=1,query_tokens=3)
        row['runtime_windows']['events'][5].update(first_node=0,node_count=3)
        ledger=extract_windows(row,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000)
        bridge=next(w['inventory'] for w in ledger['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE')
        self.assertEqual(bridge['query_tokens'],125)
        self.assertEqual(bridge['forward_index'],0)
        self.assertTrue(all(node.startswith('forward0:') for node in bridge['covered_nodes']))
        self.assertEqual(bridge['metadata']['GPU_source_forwards'],[dict(forward_index=0,query_tokens=125,kv_padded=256)])

    def test_fitted_coefficient_denominators_share_fixed_grid(self):
        samples=[dict(inventory=dict(query_tokens=q,kv_padded=32,payload_bytes=q*7),observed_ns=Fraction(11*q+13,7)) for q in (1,3,5,9)]
        coefficients=fit_nonnegative(samples)
        self.assertTrue(all(1_000_000_000 % value.denominator == 0 for value in coefficients))

    def test_intercept_policy_defaults_to_the_per_key_table(self):
        ledger=extract_windows(fixture(),placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000)
        default=fit_prices([ledger],['synthetic'])
        explicit=fit_prices([ledger],['synthetic'],intercept_policy='per_key')
        self.assertEqual([p.to_dict() for p in default],[p.to_dict() for p in explicit])
        with self.assertRaises(ValueError):fit_prices([ledger],['synthetic'],intercept_policy='whatever')

    def test_shared_intercept_policy_prices_every_key_from_one_value(self):
        ledger=extract_windows(fixture(),placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000)
        prices=fit_prices([ledger],['synthetic'],intercept_policy='shared_within_placement')
        self.assertEqual(len({p.intercept_ns for p in prices}),1)
        self.assertTrue(all(p.intercept_ns >= 0 for p in prices))

    def test_ordinary_FFN_cost_never_uses_KV_as_a_proxy(self):
        samples=[]
        for q,kv in ((1,256),(1,1024),(4,256),(4,1024)):
            node=dict(name='ffn_moe_up-0',op=30,type=0,ne=[1024,8,q,1],weight_ne=[2048,1024,64,1],weight_type=12)
            spec=WindowWorkSpec.from_metadata('CPU_NODE',cpu_node=node,copy_calls=0)
            samples.append(dict(inventory=dict(query_tokens=q,kv_padded=kv,payload_bytes=0,work_spec=spec.to_dict()),observed_ns=Fraction(5+3*q*kv)))
        a,b,c,d=fit_nonnegative(samples)
        self.assertEqual(c,0)
        self.assertEqual(d,0)

    def test_only_actual_attention_work_admits_qK(self):
        samples=[]
        for q,kv in ((1,256),(1,1024),(4,256),(4,1024)):
            node=dict(name='kq-0',op=29,type=0,ne=[kv,q,16,1],weight_ne=[128,kv,16,1],weight_type=0)
            spec=WindowWorkSpec.from_metadata('CPU_NODE',cpu_node=node,copy_calls=0)
            samples.append(dict(inventory=dict(query_tokens=q,kv_padded=kv,payload_bytes=0,work_spec=spec.to_dict()),observed_ns=Fraction(5+3*q*kv)))
        a,b,c,d=fit_nonnegative(samples)
        self.assertGreater(c,0)
        self.assertEqual(d,0)

    def test_unclassified_HOST_is_not_fitted_with_q_or_KV(self):
        spec=WindowWorkSpec.from_metadata('HOST',copy_calls=0)
        samples=[dict(inventory=dict(query_tokens=q,kv_padded=kv,payload_bytes=0,work_spec=spec.to_dict()),observed_ns=Fraction(7+q*kv)) for q,kv in ((1,256),(4,1024))]
        a,b,c,d=fit_nonnegative(samples)
        self.assertEqual((b,c,d),(0,0,0))

    def test_pruned_pointwise_uses_its_recorded_rows(self):
        row=fixture()
        for event in row['runtime_windows']['events']:event['query_tokens']=125
        row['runtime_windows']['nodes'][1].update(name='ffn_moe_swiglu-15',op=95,ne=[1024,8,1,1])
        row['runtime_windows']['nodes'][2].update(name='norm-15',op=25,ne=[2048,1,1,1])
        ledger=extract_windows(row,placement='cpu',frequency_hz=1_000_000_000)
        cpu=[w['inventory'] for w in ledger['windows'] if w['inventory']['scope']=='CPU_NODE']
        self.assertEqual([w['query_tokens'] for w in cpu],[1,1])
        self.assertTrue(all('query_tokens_times_KV' not in w['work_spec']['allowed_features'] for w in cpu))


def host_attention_row():
    """CPU attention node, then a bridge whose transfer carries its output."""
    def n(name,op,ne,wne,wtype=0):return dict(name=name,op=op,type=0,ne=ne,weight_type=wtype,weight_ne=wne)
    nodes=[n('kq-0',29,[256,1,16,1],[128,256,16,1],1),n('ffn_moe_up-0',30,[1024,8,1,1],[2048,1024,64,1],12)]
    def e(kind,t,first=0,count=0,cpu=False,bytes=0):
        return dict(kind=kind,tick=t,forward=0,query_tokens=1,position=12,phase='prefill',ordinal=0,cpu=cpu,first_node=first,node_count=count,requested_bytes=bytes)
    events=[e(20,0),e(21,1),e(1,2,0,1,cpu=True),e(10,10,0,1,cpu=True),e(2,30,cpu=True),
            e(1,40,1,1),e(5,42,bytes=64),e(2,44),e(4,46),e(23,60),e(22,62),e(24,65),e(25,70),e(26,80)]
    return dict(id='synthetic-host-attention-dependency',begin_tick=0,end_tick=80,prefill_end_tick=60,
        runtime_windows=dict(schema='real-moe-runtime-windows-1',events=events,nodes=nodes,added_GPU_synchronizations=0))


class HostAttentionDependencyTest(unittest.TestCase):
    def test_only_the_declared_policy_reads_the_host_attention_stage(self):
        from llm_infer_model.performance.moe_execution import MECHANISM_POLICY,MECHANISM_POLICY_V2
        old=extract_windows(host_attention_row(),placement='cpu-attn-gpu-other',frequency_hz=1_000_000_000)
        new=extract_windows(host_attention_row(),placement='cpu-attn-gpu-other',frequency_hz=1_000_000_000,feature_policy=MECHANISM_POLICY_V2)
        self.assertEqual(old['total_ns'],new['total_ns'])
        attn=[w['inventory'] for w in new['windows'] if w['inventory']['work_spec']['kind']=='attention_core']
        self.assertEqual(len(attn),1)
        self.assertIn('query_tokens_times_KV',attn[0]['work_spec']['allowed_features'])
        bridge_old=[w['inventory'] for w in old['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE' and w['inventory']['payload_bytes']][0]
        bridge_new=[w['inventory'] for w in new['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE' and w['inventory']['payload_bytes']][0]
        self.assertEqual(bridge_old['work_spec']['dependency'],None)
        self.assertNotIn('query_tokens_times_KV',bridge_old['work_spec']['allowed_features'])
        self.assertEqual(bridge_new['work_spec']['dependency'],'cpu_attention_stage')
        self.assertIn('query_tokens_times_KV',bridge_new['work_spec']['allowed_features'])
        self.assertTrue(bridge_new['work_spec']['dependency_evidence'])
        self.assertEqual(bridge_new['metadata']['mechanism_dependency'],'cpu_attention_stage')
        self.assertNotEqual(bridge_old['key'],bridge_new['key'])
        # A placeable outcome must survive the round trip through prices.
        price=fit_prices([new],['software test fixture, not measured hardware'],MECHANISM_POLICY_V2)
        self.assertEqual(predict_inventory(new['workload_inventory'],price)['total_ns'],80)

    def test_a_placement_without_a_host_attention_stage_keeps_the_v1_permission(self):
        # The published two-placement results must not move: with attention on
        # the GPU there is no host attention stage to inherit.
        from llm_infer_model.performance.moe_execution import MECHANISM_POLICY_V2
        row=host_attention_row();row['runtime_windows']['nodes'][0]['op']=1
        ledger=extract_windows(row,placement='cpu-experts-gpu-other',frequency_hz=1_000_000_000,feature_policy=MECHANISM_POLICY_V2)
        bridges=[w['inventory'] for w in ledger['windows'] if w['inventory']['scope']=='RUNTIME_BRIDGE']
        self.assertTrue(bridges)
        self.assertTrue(all(b['work_spec']['dependency'] is None for b in bridges))
        self.assertTrue(all('query_tokens_times_KV' not in b['work_spec']['allowed_features'] for b in bridges))


if __name__=='__main__':unittest.main()
