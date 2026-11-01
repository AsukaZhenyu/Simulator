"""No hardware execution: exact-clock, memory and causal freeze guards."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from profiling.dynamic_validation.idle_campaign import orders,validate_protocol,verify


def fixture():
    target=dict(kind='COPY',src='dram',dst='vram',target_id='weight-up')
    request=dict(schema='dynamic-idle-request-1',source_sha256='a'*64,memory_mode='pageable',
        sampling=dict(trials=16,repeats=16),idle_ms=[0,1,5,20],orders=orders(0))
    binding=dict(program_sha256='b'*64,target_action=target,target_action_index=42,
                 reference=dict(path='unused.json'))
    rows=[];last=0
    for trial in range(16):
        for slot in range(4):
            index=request['orders'][trial][slot];idle=request['idle_ms'][index]
            ib=last+100;ie=ib+idle*10000;bb=ie+1;samples=[];previous=ie
            for ordinal in range(16):
                begin=bb if ordinal==0 else previous+1;end=begin+10
                samples.append(dict(ordinal=ordinal,begin_tick=begin,end_tick=end,
                    wall_ns=1000,gap_since_previous_completion_ns=(begin-previous)*100))
                previous=end
            be=previous+1
            rows.append(dict(trial=trial,condition_slot=slot,condition_index=index,requested_idle_ms=idle,
                idle_begin_tick=ib,idle_end_tick=ie,actual_idle_ns=(ie-ib)*100,
                burst_begin_tick=bb,burst_end_tick=be,burst_wall_ns=(be-bb)*100,copies=samples))
            last=be
    payload=dict(bytes=2097152,full_byte_equality=True,source_sha256='c'*64,target_sha256='c'*64)
    native=dict(schema='dynamic-idle-native-1',request=request,process_id=123,source_sha256='a'*64,
        caller_source_sha256_match=True,program_sha256='b'*64,target_action=target,target_action_index=42,target_bytes=2097152,
        main_thread_affinity=dict(requested_mask=4,observed_mask=4,final_mask=4,restored=True),
        process_affinity=dict(requested_mask=4294967295,observed_mask=4294967295,final_mask=4294967295,
                              applied_before_backend_initialization=True),
        arena_registration=dict(mode='pageable',during=dict(registered_host=False),restored=True,after=dict(unregistered_pageable=True)),
        source_pointer_during_measurement=dict(registered_host=False),payload_before=payload,payload_after=deepcopy(payload),
        native_repeat_consistency=dict(passed=True),verification_native=dict(copy_values_equal=True,ports=dict(port=[1.,-1.])),
        instrumentation=dict(cupti=False,logging_inside_timer=False),clock=dict(frequency_hz=10000000),sequences=rows)
    return native,request,binding


class DynamicIdleCampaignTests(TestCase):
    def verified(self,native,request,binding):
        with patch('profiling.dynamic_validation.idle_campaign.read',return_value=dict(port=[1.,-1.])):
            return verify(native,request,binding,123)

    def test_balanced_rotations_all_rounds(self):
        for round_id in range(3):
            _,request,_=fixture();request['orders']=orders(round_id);validate_protocol(request)

    def test_full_1024_interval_ledger(self):
        native,request,binding=fixture();result=self.verified(native,request,binding)
        self.assertTrue(result['passed']);self.assertEqual(result['copy_intervals'],1024)

    def test_copy_clock_tamper_rejected(self):
        native,request,binding=fixture();native['sequences'][0]['copies'][0]['wall_ns']+=100
        with self.assertRaisesRegex(ValueError,'copy QPC'):self.verified(native,request,binding)

    def test_sleep_is_separate_from_copy_timer(self):
        native,request,binding=fixture();row=native['sequences'][1]
        self.assertGreater(row['actual_idle_ns'],row['burst_wall_ns'])
        self.assertTrue(self.verified(native,request,binding)['requested_idle_is_not_actual_idle'])

    def test_wrong_host_memory_not_silently_replaced(self):
        native,request,binding=fixture();native['arena_registration']['during']['registered_host']=True
        with self.assertRaisesRegex(ValueError,'actual memory'):self.verified(native,request,binding)

    def test_unbalanced_within_process_positions_rejected(self):
        _,request,_=fixture();request['orders']=[[0,1,2,3] for _ in range(16)]
        with self.assertRaisesRegex(ValueError,'positions'):validate_protocol(request)

    def test_incomplete_or_nonfinite_numerical_ports_rejected(self):
        native,request,binding=fixture();native['verification_native']['ports']['port'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'independent port'):self.verified(native,request,binding)

    def test_zero_is_control_not_sleep_claim(self):
        _,request,_=fixture();request['idle_ms']=[1,2,5,20]
        with self.assertRaisesRegex(ValueError,'controls'):validate_protocol(request)
