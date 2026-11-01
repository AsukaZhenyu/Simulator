"""Synthetic software-only finite qualification fixtures, without filesystem IO."""
import copy
import unittest

from llm_infer_model.performance.prepared_transfer import MEASUREMENT_API, COMPLETION_SEMANTIC
from llm_infer_model.performance.validated_transfer import ValidatedTransferCatalog


class ValidatedTransferTests(unittest.TestCase):
    def setUp(self):
        self.source = dict(route="vram->dram", memory_mode="pageable", measurement_api=MEASUREMENT_API,
            completion_semantic=COMPLETION_SEMANTIC, scope_sha256="a" * 64,
            preparation_contract=dict(reset_before_batch=False, untimed_transfer_prime_passes=1,
                timed_replays_per_batch=256, warmup_iterations_min=20, warmup_ms_min=500,
                fixed_input_replay=True, equal_warmup_per_memory_mode_visit=True))
        entry = dict(source_contract=self.source,
            parameters=dict(fixed_ns=2000., small_ns_per_byte=.1, large_ns_per_byte=.2,
                breakpoint_bytes=65536), minimum_bytes=4, maximum_bytes=67108864,
            validated_replays={"64":4096, "1024":4096, "262144":256}, qualified=True,
            qualification_receipt=dict(path="synthetic-software-qualification.json", sha256="b" * 64))
        self.manifest = dict(schema="prepared-transfer-validated-catalog-1",
            evidence_scope="Synthetic unit-test finite-grid declaration only", entries={"pageable":entry})

    def target(self, repeats=4096):
        value = copy.deepcopy(self.source)
        value["preparation_contract"]["timed_replays_per_batch"] = repeats
        return value

    def test_known_tiny_and_large_grid_coordinates_reuse_same_formula(self):
        model = ValidatedTransferCatalog.from_manifest(self.manifest)
        tiny = model.prediction(64, contract=self.target())
        large = model.prediction(262144, contract=self.target(256))
        self.assertAlmostEqual(tiny["duration_ns"], 2006.4)
        self.assertAlmostEqual(large["duration_ns"], 2000 + .1 * 65536 + .2 * (262144 - 65536))
        self.assertTrue(tiny["qualified_for_requested_contract"])
        self.assertEqual((tiny["source_replays"], tiny["requested_replays"]), (256,4096))
        self.assertFalse(tiny["coefficients_changed"])
        self.assertFalse(tiny["sampling_equivalence_claimed"])
        self.assertFalse(tiny["physicalbandwidthidentified"])

    def test_r1_wrong_validated_window_and_interpolation_are_rejected(self):
        model = ValidatedTransferCatalog.from_manifest(self.manifest)
        for size, repeats in ((64,1), (64,256), (262144,4096), (65,4096), (True,4096)):
            with self.subTest(size=size,repeats=repeats), self.assertRaises(ValueError):
                model.prediction(size, contract=self.target(repeats))

    def test_cold_memory_api_scope_and_completion_mismatches_reject(self):
        model = ValidatedTransferCatalog.from_manifest(self.manifest)
        for changes in (dict(memory_mode="registered"), dict(scope_sha256="c" * 64),
                        dict(route="dram->vram"), dict(measurement_api="cuda_event_elapsed"),
                        dict(completion_semantic="enqueue_only")):
            target = dict(self.target(), **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model.prediction(64, contract=target)
        for name, value in (("untimed_transfer_prime_passes",0), ("reset_before_batch",True),
                            ("warmup_ms_min",0)):
            target = self.target(); target["preparation_contract"][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                model.prediction(64, contract=target)

    def test_missing_qualification_or_invalid_parameters_cannot_create_catalog(self):
        for field, value in (("qualified",False), ("qualification_receipt",{}),
                             ("validated_replays",{"64":1})):
            bad = copy.deepcopy(self.manifest);bad["entries"]["pageable"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ValidatedTransferCatalog.from_manifest(bad)
        bad = copy.deepcopy(self.manifest);bad["entries"]["pageable"]["parameters"]["fixed_ns"] = -1
        with self.assertRaises(ValueError):ValidatedTransferCatalog.from_manifest(bad)
        bad = copy.deepcopy(self.manifest)
        bad["entries"]["pageable"]["source_contract"]["preparation_contract"]["timed_replays_per_batch"] = 4096
        with self.assertRaises(ValueError):ValidatedTransferCatalog.from_manifest(bad)
        bad = copy.deepcopy(self.manifest)
        bad["entries"]["pageable"]["source_contract"]["preparation_contract"]["untimed_transfer_prime_passes"] = 0
        with self.assertRaises(ValueError):ValidatedTransferCatalog.from_manifest(bad)

    def test_caller_mutation_does_not_change_frozen_catalog(self):
        model = ValidatedTransferCatalog.from_manifest(self.manifest)
        self.manifest["entries"]["pageable"]["parameters"]["fixed_ns"] = 1e12
        self.manifest["entries"]["pageable"]["validated_replays"]["64"] = 1
        self.assertAlmostEqual(model.prediction(64, contract=self.target())["duration_ns"],2006.4)


if __name__ == "__main__":
    unittest.main()
