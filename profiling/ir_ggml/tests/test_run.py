"""The replica rule: a one-replica corpus has no warmup to discard.

Replica 0 of every case pays the CUDA module load, so `analyse` drops it and reads the
others.  With a single replica there is nothing to drop, and a corpus that reaches the
accelerator then measures every non-CPU unit through the load -- the MoE growth corpus's
control family read 2.94 that way (single prefill units 290x their price) against 1.16
when the same corpus was re-run at four replicas.  The guard is what turns that from a
number to be noticed afterwards into a refusal before the run.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from profiling.ir_ggml import run as run_subject


def node(index: int, op: str, machine: str) -> dict:
    return {"index": index, "id": f"g:request0-prefill-layer0:op:layer0:request0:{op}",
            "op": op, "machine": machine, "priced_unit": None, "bytes": 0,
            "tokens": 64, "history": 0}


class UnpriceableSummaryTests(unittest.TestCase):
    def test_a_missing_accelerator_depth_is_named_with_its_device(self):
        # `no_price_at:gpu/decode@1t/64h` is a depth the accelerator table lacks; the CPU
        # form is a depth the campaign's own window lacks. Counting them as one key would
        # put the declared device-axis gap on the depth axis and fail G3 for it.
        summary = run_subject.summarise([{
            "case": "one-case", "unstable": False, "units": {},
            "prefill_pure": False, "makespan_ratio": None, "measured_wall_ns": 1.0,
            "measured_copy_ns": 0.0,
            "unpriceable": {"attention": "no_price_at:gpu/decode@1t/64h",
                            "moe_ffn": "no_price_at:decode@1t/66h"}}])
        self.assertEqual(summary["unpriceable"],
                         {"attention: no_price_at on gpu": 1,
                          "moe_ffn: no_price_at": 1})


class ReplicaGuardTests(unittest.TestCase):
    def cases_dir(self, machines: tuple[str, ...]) -> Path:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name) / "m"
        root.mkdir()
        nodes = [node(index, op, machine)
                 for index, (op, machine) in enumerate(zip(("arrival", "proj", "attn"),
                                                           machines))]
        manifest = {"cases": {"one-case": {"nodes": nodes, "request_tokens": 64,
                                           "price_signature_override": None}}}
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    def invoke(self, root: Path, replicas: int) -> SystemExit:
        # The price table and the executor are both irrelevant to the guard, which is the
        # point: the refusal has to come before a case is started, not from its numbers.
        with mock.patch.object(run_subject, "table_for", return_value={}), \
                mock.patch.object(run_subject, "RUNNER", root / "no-executor"):
            with self.assertRaises(SystemExit) as caught:
                run_subject.main(["--model", "m", "--cases-dir", str(root),
                                  "--replicas", str(replicas),
                                  "--out", str(root / "out.json")])
        return caught.exception

    def test_one_replica_is_refused_when_a_case_reaches_the_accelerator(self):
        error = self.invoke(self.cases_dir(("cpu", "cpu", "gpu")), replicas=1)
        self.assertIn("replica 0 pays the CUDA module load", str(error))

    def test_one_replica_is_allowed_when_every_node_is_on_the_cpu(self):
        error = self.invoke(self.cases_dir(("cpu", "cpu", "cpu")), replicas=1)
        # Past the guard and on to the executor check: the run was allowed to start.
        self.assertIn("no executor at", str(error))

    def test_four_replicas_are_allowed_with_accelerator_nodes(self):
        error = self.invoke(self.cases_dir(("cpu", "cpu", "gpu")), replicas=4)
        self.assertIn("no executor at", str(error))
