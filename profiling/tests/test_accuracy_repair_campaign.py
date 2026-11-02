"""Software-only admission fixtures. These tests never launch a native process."""
import copy
import unittest

from profiling.dac_accuracy_repair import campaign


class RepairCampaignAdmissionTests(unittest.TestCase):
    PID = 12345
    SOURCE = "a" * 64
    PROGRAM = "b" * 64
    BYTE_HASH = "c" * 64
    PROCESS_MASK = 20

    def request(self, mode, **kwargs):
        request = campaign.request(mode, self.SOURCE, **kwargs)
        request["expected_process_mask"] = self.PROCESS_MASK
        return request

    @staticmethod
    def frame(request, *, stamped=False, block=0):
        return dict(begin_tick=1000, end_tick=1080, wall_ns=8000,
            repeats=request["sampling"]["repeats"], actual_stamped=stamped,
            block=block)

    def common(self, request):
        return dict(schema="dac-accuracy-repair-native-1", request=copy.deepcopy(request),
            process_id=self.PID, source_sha256=self.SOURCE,
            caller_source_sha256_match=True, program_sha256=self.PROGRAM,
            clock=dict(frequency_hz=10000000),
            main_thread_affinity=dict(requested_mask=4, observed_mask=4,
                final_mask=4, restored=True),
            process_affinity=dict(requested_mask=self.PROCESS_MASK,
                observed_mask=self.PROCESS_MASK, final_mask=self.PROCESS_MASK,
                enforced=True, applied_before_backend_initialization=True))

    def dense_fixture(self, mode="dense-primary"):
        request = self.request(mode)
        action = dict(kind="COMPUTE", target_id="synthetic-op", device="cpu")
        program = dict(program_sha256=self.PROGRAM, actions=[action])
        oracle = {"synthetic-output": [1.0, 2.0]}
        native = self.common(request)
        prep = dict(reset_before_batch=True, untimed_complete_prime_passes=1,
            timed_replays_per_batch=8, fixed_input_replay=True,
            warmup_iterations_min=20, warmup_ms_min=500)
        native.update(preparation_contract=prep, native_repeat_consistency=dict(passed=True),
            verification_native=dict(copy_values_equal=True, actions=[action],
                ports=copy.deepcopy(oracle)))
        native["trials"] = []
        for _ in range(request["sampling"]["trials"]):
            for block in range(4 if mode == "dense-observer" else 1):
                frame = self.frame(request, stamped=mode == "dense-observer" and block in (1, 2), block=block)
                frame["preparation_contract"] = copy.deepcopy(prep)
                native["trials"].append(frame)
        return program, oracle, request, native

    def transfer_fixture(self, memory_mode="pageable"):
        request = self.request("transfer-mode", memory_mode=memory_mode)
        program = dict(program_sha256=self.PROGRAM)
        oracle = dict(bytes=64, sha256=self.BYTE_HASH)
        native = self.common(request)
        native.update(measurement_api=campaign.API, completion_semantic=campaign.COMPLETION,
            host_registration_restored=True,
            visits=[dict(memory_mode=memory_mode,
                host_pointer=dict(status="observed", registered_host=memory_mode == "registered",
                    unregistered_pageable=memory_mode == "pageable"),
                full_byte_verification=dict(full_byte_equality=True, checked_bytes=64,
                    source_sha256=self.BYTE_HASH, output_sha256=self.BYTE_HASH))],
            trials=[self.frame(request, stamped=block in (1, 2), block=block)
                for _ in range(request["sampling"]["trials"]) for block in range(4)])
        return program, oracle, request, native

    def verify(self, fixture):
        return campaign.verify(*fixture, self.PID)

    def test_request_uses_explicit_source_sampling_and_memory_mode(self):
        dense = self.request("dense-primary")
        transfer = self.request("transfer-mode", memory_mode="registered")
        self.assertEqual(dense["source_sha256"], self.SOURCE)
        self.assertEqual((dense["sampling"]["mode"], dense["sampling"]["repeats"]), ("full", 8))
        self.assertEqual((transfer["sampling"]["mode"], transfer["sampling"]["repeats"]), ("action", 256))
        self.assertEqual(transfer["memory_mode"], "registered")

    def test_valid_dense_primary_and_observer_software_fixtures_pass(self):
        for mode in ("dense-primary", "dense-observer"):
            with self.subTest(mode=mode):
                self.assertTrue(self.verify(self.dense_fixture(mode))["passed"])

    def test_valid_pageable_and_registered_software_fixtures_pass(self):
        for mode in ("pageable", "registered"):
            with self.subTest(mode=mode):
                self.assertTrue(self.verify(self.transfer_fixture(mode))["passed"])

    def test_wrong_process_mask_or_late_backend_enforcement_rejects(self):
        cases = (("requested_mask", 4), ("observed_mask", 4),
                 ("final_mask", 4), ("enforced", False),
                 ("applied_before_backend_initialization", False))
        for field, value in cases:
            fixture = self.dense_fixture()
            fixture[3]["process_affinity"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_missing_backend_setup_proof_rejects(self):
        fixture = self.dense_fixture()
        del fixture[3]["process_affinity"]["applied_before_backend_initialization"]
        with self.assertRaises((KeyError, ValueError)):
            self.verify(fixture)

    def test_wrong_main_thread_or_unrestored_affinity_rejects(self):
        for field, value in (("requested_mask", 20), ("observed_mask", 20),
                             ("final_mask", 20), ("restored", False)):
            fixture = self.dense_fixture()
            fixture[3]["main_thread_affinity"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_missing_or_wrong_source_and_process_identity_rejects(self):
        cases = (("source_sha256", "d" * 64), ("caller_source_sha256_match", False),
                 ("process_id", self.PID + 1), ("program_sha256", "d" * 64))
        for field, value in cases:
            fixture = self.dense_fixture()
            fixture[3][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(fixture)
        fixture = self.dense_fixture()
        del fixture[3]["source_sha256"]
        with self.assertRaises((KeyError, ValueError)):
            self.verify(fixture)

    def test_missing_or_wrong_host_registration_proof_rejects(self):
        fixture = self.transfer_fixture("registered")
        fixture[3]["visits"][0]["host_pointer"]["registered_host"] = False
        with self.assertRaises(ValueError):
            self.verify(fixture)
        fixture = self.transfer_fixture()
        del fixture[3]["visits"][0]["host_pointer"]["registered_host"]
        with self.assertRaises((KeyError, ValueError)):
            self.verify(fixture)
        fixture = self.transfer_fixture()
        fixture[3]["host_registration_restored"] = False
        with self.assertRaises(ValueError):
            self.verify(fixture)

    def test_wrong_transfer_completion_or_byte_oracle_rejects(self):
        for field, value in (("measurement_api", "enqueue_only"),
                             ("completion_semantic", "cuda_event_elapsed")):
            fixture = self.transfer_fixture()
            fixture[3][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(fixture)
        for field, value in (("checked_bytes", 32), ("source_sha256", "d" * 64),
                             ("output_sha256", "d" * 64), ("full_byte_equality", False)):
            fixture = self.transfer_fixture()
            fixture[3]["visits"][0]["full_byte_verification"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_changed_dense_preparation_or_missing_trial_rejects(self):
        fixture = self.dense_fixture("dense-observer")
        fixture[3]["trials"][0]["preparation_contract"]["untimed_complete_prime_passes"] = 0
        with self.assertRaises(ValueError):
            self.verify(fixture)
        fixture = self.dense_fixture()
        fixture[3]["trials"].pop()
        with self.assertRaises(ValueError):
            self.verify(fixture)

    def test_qpc_replay_or_elapsed_ledger_mismatch_rejects(self):
        for field, value in (("wall_ns", 1234), ("repeats", 1), ("end_tick", 1000)):
            fixture = self.dense_fixture()
            fixture[3]["trials"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(fixture)

    def test_wrong_numerical_value_or_action_sequence_rejects(self):
        fixture = self.dense_fixture()
        fixture[3]["verification_native"]["ports"]["synthetic-output"][0] = 2.0
        with self.assertRaises(ValueError):
            self.verify(fixture)
        fixture = self.dense_fixture()
        fixture[3]["verification_native"]["actions"] = []
        with self.assertRaises(ValueError):
            self.verify(fixture)


if __name__ == "__main__":
    unittest.main()
