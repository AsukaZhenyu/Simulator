import unittest

from profiling.real_moe_validation import device_state_control as state


def provider(**values):
    """A flat provider: the module accepts one, so no NVML is needed to test it."""
    return state.Provider(read=lambda: dict(values))


def series(temperatures, clock=2200, memory=8001, step_s=1.0, start_ns=0):
    return [{"temperature_c": temperature, "sm_clock_mhz": clock,
             "memory_clock_mhz": memory, "power_w": 60.0,
             "taken_monotonic_ns": start_ns + int(index * step_s * 1e9)}
            for index, temperature in enumerate(temperatures)]


class BandTest(unittest.TestCase):
    def test_a_band_refuses_a_field_no_reading_provides(self):
        with self.assertRaises(ValueError):
            state.Band.parse({"clock_speed": [1, 2]})

    def test_an_empty_band_admits_every_state_and_is_refused(self):
        with self.assertRaises(ValueError):
            state.Band.parse({})

    def test_bounds_must_be_an_interval(self):
        with self.assertRaises(ValueError):
            state.Band.parse({"temperature_c": [90, 40]})

    def test_a_reading_outside_the_band_names_the_field(self):
        band = state.Band.parse({"temperature_c": [40, 80]})
        ok, reason = band.check({"temperature_c": 91.0})
        self.assertFalse(ok)
        self.assertIn("temperature_c is 91", reason)

    def test_a_field_the_reading_lacks_is_not_zero(self):
        band = state.Band.parse({"temperature_c": [40, 80]})
        ok, reason = band.check({})
        self.assertFalse(ok)
        self.assertIn("not reported", reason)

    def test_bounds_are_inclusive(self):
        band = state.Band.parse({"temperature_c": [40, 80]})
        self.assertTrue(band.check({"temperature_c": 40})[0])
        self.assertTrue(band.check({"temperature_c": 80})[0])


class PlateauTest(unittest.TestCase):
    def test_a_flat_window_settles(self):
        settled, why = state.Plateau(window_s=5).verdict(series([60, 60, 60, 60, 60, 60]))
        self.assertTrue(settled, why)

    def test_a_window_shorter_than_declared_does_not_settle(self):
        settled, why = state.Plateau(window_s=30).verdict(series([60, 60, 60]))
        self.assertFalse(settled)
        self.assertIn("window is 30", why)

    def test_temperature_still_rising_does_not_settle(self):
        settled, why = state.Plateau(window_s=5, temperature_c=1.5).verdict(
            series([60, 61, 62, 63, 64, 65]))
        self.assertFalse(settled)
        self.assertIn("temperature moved", why)

    def test_a_fixed_temperature_with_ramping_clocks_does_not_settle(self):
        rows = series([70] * 6)
        for index, row in enumerate(rows):
            row["sm_clock_mhz"] = 1500 + 200 * index
        settled, why = state.Plateau(window_s=5, clock_fraction=0.02).verdict(rows)
        self.assertFalse(settled)
        self.assertIn("SM clock moved", why)

    def test_readings_without_a_clock_cannot_settle(self):
        rows = series([70] * 6)
        for row in rows:
            row.pop("sm_clock_mhz")
        settled, why = state.Plateau(window_s=5).verdict(rows)
        self.assertFalse(settled)
        self.assertIn("lacks", why)


class RecordTest(unittest.TestCase):
    def test_a_reading_outside_the_band_excludes_the_cohort(self):
        record = state.Record(samples=series([50, 50, 95]))
        ok, why = record.review(state.Band.parse({"temperature_c": [40, 90]}))
        self.assertFalse(ok)
        self.assertEqual(len(record.out_of_band), 1)
        self.assertIn("1 of 3 readings", why)

    def test_drift_inside_the_band_is_its_own_refusal(self):
        record = state.Record(samples=series([60, 65, 70, 75]))
        ok, why = record.review(state.Band.parse({"temperature_c": [40, 90]}),
                                {"temperature_c": 6.0})
        self.assertFalse(ok)
        self.assertIn("temperature_c moved 15", why)

    def test_clock_drift_is_judged_as_a_fraction_of_the_median(self):
        rows = series([70] * 4)
        for row, clock in zip(rows, (2000, 2200, 2400, 2600)):
            row["sm_clock_mhz"] = clock
        record = state.Record(samples=rows)
        ok, why = record.review(None, {"sm_clock_mhz_fraction": 0.10})
        self.assertFalse(ok)
        self.assertIn("sm_clock_mhz moved", why)

    def test_a_stable_cohort_inside_the_band_is_admitted(self):
        record = state.Record(samples=series([70, 70.5, 70.2, 70.4]))
        ok, why = record.review(state.Band.parse({"temperature_c": [40, 90]}),
                                {"temperature_c": 6.0})
        self.assertTrue(ok, why)

    def test_no_readings_is_a_refusal_not_an_empty_pass(self):
        ok, why = state.Record(samples=[]).review(None)
        self.assertFalse(ok)
        self.assertIn("no device-state readings", why)


class SamplerTest(unittest.TestCase):
    def test_the_sampler_collects_readings_while_it_runs(self):
        import time
        sampler = state.Sampler(provider(temperature_c=60, sm_clock_mhz=2200), interval_s=0.01)
        sampler.start()
        time.sleep(0.12)
        samples = sampler.stop()
        self.assertGreaterEqual(len(samples), 3)
        self.assertTrue(all("taken_monotonic_ns" in row for row in samples))

    def test_an_unreadable_sample_is_recorded_and_does_not_end_the_sampler(self):
        import time

        def read():
            raise OSError("driver went away")

        sampler = state.Sampler(state.Provider(read=read), interval_s=0.01)
        sampler.start()
        time.sleep(0.06)
        samples = sampler.stop()
        self.assertEqual(samples, [])
        self.assertTrue(sampler.problems)

    def test_a_nested_reading_is_flattened(self):
        nested = state.Provider(read=lambda: dict(status="ok", unknown={},
                                                 values={"temperature_c": 55}))
        self.assertEqual(state.sample(nested)["temperature_c"], 55)


class LockTest(unittest.TestCase):
    def test_a_missing_tool_is_a_recorded_refusal(self):
        result = state.lock_clocks(2100, 8001, tool="/nonexistent/nvidia-smi")
        self.assertEqual(result["status"], "refused")
        self.assertIn("nvidia-smi", result["reason"])

    def test_a_lock_is_only_reported_verified_after_a_read_back(self):
        held = state.verify_lock(provider(sm_clock_mhz=2100, memory_clock_mhz=8001),
                                 {"sm_clock_mhz": 2100, "memory_clock_mhz": 8001},
                                 attempts=1, wait_s=0)
        self.assertEqual(held["status"], "verified")
        missed = state.verify_lock(provider(sm_clock_mhz=1800, memory_clock_mhz=8001),
                                   {"sm_clock_mhz": 2100, "memory_clock_mhz": 8001},
                                   attempts=1, wait_s=0)
        self.assertEqual(missed["status"], "not_verified")


if __name__ == "__main__":
    unittest.main()
