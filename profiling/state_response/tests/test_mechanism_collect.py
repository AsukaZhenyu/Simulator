"""Matched recovery controls, not implementation-mirroring latency tests."""
import unittest

from profiling.state_response.mechanism_collect import make_jobs
from profiling.state_response.collect import selected_roles, main_instrumentation
from profiling.state_response.schedule import validate_schedule


class MechanismDesignTests(unittest.TestCase):
    def test_every_recovery_pid_has_initial_idle_and_all_three_matched_predecessors(self):
        jobs = make_jobs()
        self.assertEqual(len(jobs), 24)
        for job in jobs:
            validate_schedule(job["schedule"])
            self.assertEqual(job["split"], "diagnostic")
            selected_roles(job)
            if job["condition"].startswith("recovery"):
                phases = job["schedule"]["phases"]
                self.assertEqual(phases[0]["end_ns"], 2_000_000_000)
                predecessors = set()
                for load, off in zip(phases[1::2], phases[2::2]):
                    predecessors.add((load["state"]["cpu_replay"], load["state"]["gpu_replay"]))
                    self.assertEqual((off["state"]["cpu_replay"], off["state"]["gpu_replay"]), (0, 0))
                    self.assertEqual(off["end_ns"] - off["begin_ns"], 2_000_000_000)
                self.assertEqual(predecessors, {(1, 0), (0, 1), (1, 1)})

    def test_thread_count_and_actor_presence_controls_keep_the_numeric_seed_paired(self):
        jobs = make_jobs()
        for repeat in range(3):
            group = [job for job in jobs if job["repeat"] == repeat]
            self.assertEqual(len({job["numeric_seed"] for job in group}), 1)
            controls = {job["condition"]: job for job in group}
            self.assertEqual(selected_roles(controls["solo"]), ["main"])
            self.assertEqual(selected_roles(controls["idle-actors"]), ["cpu-worker", "gpu-worker", "main"])
            self.assertEqual(controls["solo-one-thread"]["cpu_threads"], 1)
            self.assertEqual(main_instrumentation(controls["solo-bare"], {}), "bare")
            self.assertTrue(controls["solo-bare"]["thread_observer"])
            self.assertFalse(controls["solo-bare-observer-off"]["thread_observer"])
        self.assertEqual(jobs, make_jobs())
        self.assertNotEqual(jobs, make_jobs(seed=123456))

    def test_unknown_or_duplicate_condition_cannot_silently_select_a_default(self):
        for conditions in ([], ["solo", "solo"], ["unmeasured"]):
            with self.assertRaises(ValueError):
                make_jobs(conditions=conditions)

    def test_cuda_wait_controls_can_omit_perturbing_thread_query_helper(self):
        jobs = make_jobs(conditions=["solo-gpu-auto", "solo-gpu-spin", "solo-gpu-blocking", "solo-gpu-yield"],
                         disable_thread_observer=True)
        self.assertEqual({job["cuda_schedule"] for job in jobs}, {"auto", "spin", "blocking", "yield"})
        self.assertTrue(all(not job["thread_observer"] for job in jobs))
        self.assertTrue(all(job["instrumentation"] == "diagnostic" for job in jobs))
        with self.assertRaises(ValueError):
            make_jobs(disable_thread_observer="yes")

    def test_main_thread_pinning_keeps_two_cpu_threads_and_matched_recovery_roles(self):
        jobs = make_jobs(conditions=["solo-main-cpu0", "solo-main-cpu2", "recovery-main-cpu0", "recovery-main-cpu2"],
                         disable_thread_observer=True)
        for job in jobs:
            self.assertEqual(job["cpu_threads"], 2)
            self.assertEqual(job["main_thread_affinity_mask"], 1 if job["condition"].endswith("cpu0") else 4)
            self.assertEqual(job["instrumentation"], "diagnostic" if job["condition"].startswith("recovery") else "stamped")
            self.assertEqual(job["owned_roles"], ["cpu-worker", "gpu-worker", "main"]
                             if job["condition"].startswith("recovery") else ["main"])


if __name__ == "__main__":
    unittest.main()
