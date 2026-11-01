"""Pure profile-only cell support and transition handling checks."""
from fractions import Fraction
import unittest

from .runtime_stage_profile import aggregate_cells, offered_state, profile_observations


class LocalCellTests(unittest.TestCase):
    def native(self):
        return dict(clock=dict(frequency_hz=1000),target_expected_origin_tick=0,
                    request=dict(cpu_actor=dict(intervals_ms=[[10,20]]),gpu_actor=dict(intervals_ms=[[15,25]])))

    def test_exact_boundary_is_not_forced_into_wrong_state(self):
        n=self.native()
        self.assertEqual(offered_state(n,10,15)['state'],(1,0))
        self.assertEqual(offered_state(n,15,20)['state'],(1,1))
        self.assertEqual(offered_state(n,20,25)['state'],(0,1))

    def test_crossing_stage_retains_transition_not_steady_price(self):
        r=offered_state(self.native(),9,16)
        self.assertIsNone(r['state'])
        self.assertEqual(len(r['crossings']),2)

    def test_heldout_cannot_enter_profile_extractor(self):
        with self.assertRaises(ValueError):
            profile_observations({},context_id='h',dataset_role='heldout',stream_roles={},expected_API_counts={})

    def datasets(self):
        scope=dict(target_thread_mask=4,CPU_actor_thread_mask=16)
        output=[]
        for index in range(3):
            name=f'fresh-{index}';rows=[]
            for state in ((0,0),(1,0),(0,1),(1,1)):
                rows.append(dict(context_id=name,scope=scope,coordinate_sha256='a'*64,role='api',state=state,response_ns=Fraction(10+index)))
            output.append(dict(dataset_role='profile',context_id=name,scope=scope,stage_sequence=[('a'*64,'api')],observations=rows,
                               independence_identity=(100+index,1000+index),source_evidence=dict(native_sha256=str(index)*64),
                               cross_state_observations=[],whole_target_response_used_for_fitting=False))
        return output

    def test_only_local_observations_define_exact_cell_mean(self):
        r=aggregate_cells(self.datasets(),minimum_samples_per_context=1)
        self.assertTrue(r['all_cells_supported'])
        self.assertEqual(r['declared_cells'],4)
        self.assertTrue(all(x['mean_ns']==11 for x in r['cells']))
        self.assertFalse(r['qualified'])
        self.assertFalse(r['whole_target_duration_used'])

    def test_missing_cell_is_unknown_not_zero(self):
        rows=self.datasets();rows[0]['observations'].pop()
        r=aggregate_cells(rows,minimum_samples_per_context=1)
        self.assertFalse(r['all_cells_supported'])
        failed=[x for x in r['cells'] if not x['supported']]
        self.assertEqual(len(failed),1)
        self.assertIsNone(failed[0]['mean_ns'])

    def test_affinity_scope_and_duplicate_contexts_cannot_pool(self):
        rows=self.datasets();rows[1]['scope']=dict(target_thread_mask=4,CPU_actor_thread_mask=4)
        with self.assertRaises(ValueError):aggregate_cells(rows,minimum_samples_per_context=1)
        rows=self.datasets();rows[1]['context_id']=rows[0]['context_id']
        with self.assertRaises(ValueError):aggregate_cells(rows,minimum_samples_per_context=1)

    def test_same_native_process_cannot_get_three_labels(self):
        rows=self.datasets();rows[1]['independence_identity']=rows[0]['independence_identity']
        with self.assertRaises(ValueError):aggregate_cells(rows,minimum_samples_per_context=1)


class ConstantStateCellTests(unittest.TestCase):
    def constant_datasets(self):
        """12 distinct processes: three explicitly declared contexts per state."""
        scope=dict(target_thread_mask=4,CPU_actor_thread_mask=4)
        output=[]
        for state_index,state in enumerate(((0,0),(1,0),(0,1),(1,1))):
            for index in range(3):
                identity=3*state_index+index;name=f'state-{state_index}-fresh-{index}'
                output.append(dict(dataset_role='profile',context_id=name,scope=scope,
                    stage_sequence=[('a'*64,'api')],state_protocol='constant_state_v8',declared_state=state,
                    independence_identity=(100+identity,1000+identity),
                    source_evidence=dict(native_sha256=f'{identity:064x}'),
                    observations=[dict(context_id=name,scope=scope,coordinate_sha256='a'*64,
                                       role='api',state=state,response_ns=Fraction(100*state_index+10+index))],
                    cross_state_observations=[],whole_target_response_used_for_fitting=False))
        return output

    def test_three_matching_state_contexts_support_cell_without_requiring_other_nine(self):
        result=aggregate_cells(self.constant_datasets(),minimum_samples_per_context=1)
        self.assertEqual(result['state_protocol'],'constant_state_v8')
        self.assertTrue(result['all_cells_supported'])
        self.assertEqual(result['supported_cells'],4)
        bystate={row['state']:row for row in result['cells']}
        for index,state in enumerate(((0,0),(1,0),(0,1),(1,1))):
            row=bystate[state]
            self.assertEqual(row['mean_ns'],100*index+11)
            self.assertEqual(len(row['expected_contexts']),3)
            self.assertEqual(len(row['counts_by_context']),3)
            self.assertEqual(row['missing_contexts'],[])

    def test_missing_coordinate_in_a_matching_context_stays_unknown(self):
        datasets=self.constant_datasets();missing=datasets[-1]['context_id']
        datasets[-1]['observations']=[]
        result=aggregate_cells(datasets,minimum_samples_per_context=1)
        self.assertFalse(result['all_cells_supported'])
        unsupported=[row for row in result['cells'] if not row['supported']]
        self.assertEqual(len(unsupported),1)
        self.assertEqual(unsupported[0]['state'],(1,1))
        self.assertIsNone(unsupported[0]['mean_ns'])
        self.assertEqual(unsupported[0]['missing_contexts'],[missing])

    def test_missing_state_group_or_fewer_than_three_contexts_cannot_support_prices(self):
        for dropped in (1,3):
            with self.assertRaises(ValueError):
                aggregate_cells(self.constant_datasets()[:-dropped],minimum_samples_per_context=1)

    def test_duplicate_native_identity_across_state_groups_is_not_independence(self):
        datasets=self.constant_datasets();datasets[3]['independence_identity']=datasets[0]['independence_identity']
        with self.assertRaises(ValueError):aggregate_cells(datasets,minimum_samples_per_context=1)

    def test_row_state_is_not_allowed_to_relabel_constant_context(self):
        datasets=self.constant_datasets();datasets[0]['observations'][0]['state']=(1,1)
        with self.assertRaises(ValueError):aggregate_cells(datasets,minimum_samples_per_context=1)

    def test_declared_state_and_no_transition_are_explicit_requirements(self):
        for invalid in (None,[True,0],[0],[-1,0]):
            datasets=self.constant_datasets();datasets[0]['declared_state']=invalid
            with self.assertRaises(ValueError):aggregate_cells(datasets,minimum_samples_per_context=1)
        datasets=self.constant_datasets();datasets[0]['cross_state_observations']=[dict(state=None)]
        with self.assertRaises(ValueError):aggregate_cells(datasets,minimum_samples_per_context=1)

    def test_protocols_cannot_be_pooled_to_fill_missing_state(self):
        datasets=self.constant_datasets();datasets[0]['state_protocol']='mixed_state_v6'
        with self.assertRaises(ValueError):aggregate_cells(datasets,minimum_samples_per_context=1)

    def test_new_state_grouping_does_not_weaken_legacy_mixed_state_support(self):
        datasets=LocalCellTests().datasets()
        # A mixed protocol promises every context can contribute every state;
        # one absent state observation remains unsupported, not excluded from
        # that state's denominator by looking only at observed rows.
        datasets[0]['observations'].pop()
        result=aggregate_cells(datasets,minimum_samples_per_context=1)
        failed=[row for row in result['cells'] if not row['supported']]
        self.assertEqual(len(failed),1)
        self.assertEqual(failed[0]['missing_contexts'],['fresh-0'])
        self.assertEqual(len(failed[0]['expected_contexts']),3)


class ConstantStateExtractionTests(unittest.TestCase):
    def fixture(self):
        metadata=[dict(name='cudaMemcpyAsync_v3020',parameters_available=True,bytes=1024,
                       copy_kind_cudaMemcpyKind=3,stream_handle_opaque_process_local=123),
                  dict(name='cudaStreamSynchronize_v3020',parameters_available=True,
                       stream_handle_opaque_process_local=123)]
        callbacks=dict(all_selected_calls_retained=True,activity_driver_kernel_collection_enabled=False,
                       metadata=metadata,callback_records=[dict(ordinal=i,phase_id=1,enter_tick=a,exit_tick=b,
                           exit_received=True,return_status=0,metadata_index=i,correlation_id=100+i)
                           for i,(a,b) in enumerate(((20,40),(70,90)))])
        native=dict(schema='dynamic-external-stage-native-8',clock=dict(frequency_hz=1000),
            target_expected_origin_tick=0,profile_window_contained=True,target_entry_controller_passed=True,
            instrumentation='runtime_callbacks',performance_parameters_fitted=False,process_id=123,
            source_sha256='a'*64,program_sha256='b'*64,init_policy='pinned_init',
            target_thread_affinity=dict(observed_mask=4),actors=dict(
                cpu=dict(thread_affinity=dict(observed_mask=4)),gpu=dict(thread_affinity=dict(observed_mask=65536))),
            request=dict(split='profile',offered_state=[1,0],callback_source_sha256='c'*64,
                cpu_actor=dict(device='cpu',component=dict(kind='fixture'),intervals_ms=[[0,30000]]),
                gpu_actor=dict(device='gpu',component=dict(kind='fixture'),intervals_ms=[])),
            runtime_callbacks=callbacks,attention_action_index=22,continuous_replays=1,
            blocks=[dict(actions=[dict(action_index=22,callback_phase_id=1,begin_tick=10,end_tick=100)])])
        binding={key:'d'*64 for key in ('native_sha256','execution_sha256','runtime_binding_sha256','context_sha256','topology_sha256')}
        binding.update(process_id=123,process_creation_time_100ns=1000,
                       target_physical_cores=[1],CPU_actor_physical_cores=[1],GPU_actor_physical_cores=[8])
        return native,binding

    def extract(self,native,binding):
        return profile_observations(native,context_id='fixed-10',dataset_role='profile',
            stream_roles={123:'target_compute'},source_binding=binding,
            expected_API_counts={'cudaMemcpyAsync_v3020':1,'cudaStreamSynchronize_v3020':1})

    def test_extractor_records_explicit_v8_membership_after_validating_every_stage(self):
        dataset=self.extract(*self.fixture())
        self.assertEqual(dataset['state_protocol'],'constant_state_v8')
        self.assertEqual(dataset['declared_state'],(1,0))
        self.assertEqual(len(dataset['observations']),5)
        self.assertTrue(all(row['state']==(1,0) for row in dataset['observations']))
        self.assertEqual(dataset['cross_state_observations'],[])

    def test_declared_state_mismatch_is_rejected_even_if_native_containment_flag_is_true(self):
        native,binding=self.fixture();native['request']['offered_state']=[0,0]
        with self.assertRaises(ValueError):self.extract(native,binding)

    def test_phase_crossing_an_actor_boundary_cannot_supply_steady_state_cells(self):
        native,binding=self.fixture();native['request']['cpu_actor']['intervals_ms']=[[0,50]]
        with self.assertRaises(ValueError):self.extract(native,binding)

    def test_missing_declared_state_is_not_inferred_from_observed_timings(self):
        native,binding=self.fixture();native['request'].pop('offered_state')
        with self.assertRaises(ValueError):self.extract(native,binding)

    def test_legacy_v6_can_retain_transition_observations(self):
        native,binding=self.fixture();native['schema']='dynamic-external-callback-native-6'
        native['request']['cpu_actor']['intervals_ms']=[[0,50]]
        dataset=self.extract(native,binding)
        self.assertEqual(dataset['state_protocol'],'mixed_state_v6')
        self.assertIsNone(dataset['declared_state'])
        self.assertEqual(len(dataset['cross_state_observations']),1)


if __name__=='__main__':unittest.main()
