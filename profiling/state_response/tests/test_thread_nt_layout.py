"""Byte fixtures executed by an explicitly built helper, with no native queries.

Portable CI retains the source/compiler guards. Set LLM_THREAD_HELPER_TEST_BINARY
to opt into these parser fixtures against an actual source-bound Windows helper.
"""
import base64
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from profiling.state_response import thread_activity as t


def record(pid, threads, *, next_offset=0):
    blob=bytearray(256+80*len(threads))
    struct.pack_into("<II",blob,0,next_offset,len(threads))
    struct.pack_into("<q",blob,80,pid)
    for index,fields in enumerate(threads):
        tid,state,wait,owner=fields
        struct.pack_into("<qq",blob,256+80*index+40,owner,tid)
        struct.pack_into("<II",blob,256+80*index+68,state,wait)
    return blob


@unittest.skipUnless(os.environ.get("LLM_THREAD_HELPER_TEST_BINARY"),"explicit built Windows helper required")
class NtLayoutFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=Path(os.environ["LLM_THREAD_HELPER_TEST_BINARY"]).resolve()
        t.helper_sources(cls.binary)

    def parse(self,blob,*,okay=True):
        with tempfile.TemporaryDirectory(dir=ROOT/"build",prefix="nt-layout-fixture-") as folder:
            folder=Path(folder);src=folder/"input.json";out=folder/"output.json"
            t.save_new(src,dict(buffer_base64=base64.b64encode(blob).decode(),owned_pids=[12]))
            result=subprocess.run([str(self.binary),"--parse-fixture",str(src),str(out)],capture_output=True,timeout=10)
            if okay:
                self.assertEqual(result.returncode,0,result.stderr.decode(errors="replace"))
                doc=t.read(out);self.assertFalse(doc["native_queries_performed"]);return doc
            self.assertNotEqual(result.returncode,0)
            self.assertFalse(out.exists())
            return result.stderr.decode(errors="replace")

    def test_running_ready_wait_are_preserved_and_native_wait_is_not_managed_cast(self):
        doc=self.parse(record(12,[(1,2,0,12),(2,1,0,12),(3,5,13,12)]))
        threads=doc["parsed"][0]["threads"]
        self.assertEqual([row["state_name"] for row in threads],["Running","Ready","Wait"])
        self.assertEqual(threads[2]["native_wait_reason_code"],13)
        self.assertIsNone(threads[2]["wait_reason_name"])
        self.assertIsNone(threads[2]["wait_reason_code"])

    def test_native_extra_state_is_not_classified_ready(self):
        row=self.parse(record(12,[(1,7,0,12)]))["parsed"][0]["threads"][0]
        self.assertEqual(row["state_code"],7);self.assertEqual(row["state_name"],"OtherOrUnknown")

    def test_unowned_entries_are_not_copied_to_output(self):
        foreign=record(99,[(1,2,0,99)],next_offset=336)
        # Fill opaque reserved bytes with a marker; the parser must never emit it.
        foreign[8:29]=b"private-other-process"
        doc=self.parse(foreign+record(12,[(2,2,0,12)]))
        self.assertEqual([row["process_id"] for row in doc["parsed"]],[12])
        self.assertNotIn("private-other-process",json.dumps(doc))

    def test_short_process_header_is_rejected(self): self.parse(bytearray(255),okay=False)
    def test_short_thread_array_is_rejected(self): self.parse(record(12,[(1,2,0,12)])[:-1],okay=False)
    def test_misaligned_next_offset_is_rejected(self): self.parse(record(12,[],next_offset=257),okay=False)
    def test_next_offset_outside_buffer_is_rejected(self): self.parse(record(12,[],next_offset=512),okay=False)

    def test_declared_thread_count_cannot_escape_process_span(self):
        blob=record(12,[(1,2,0,12)]);struct.pack_into("<I",blob,4,2);self.parse(blob,okay=False)

    def test_duplicate_owned_pid_is_rejected(self):
        self.parse(record(12,[(1,2,0,12)],next_offset=336)+record(12,[(2,2,0,12)]),okay=False)

    def test_foreign_owner_and_duplicate_tid_are_rejected(self):
        self.parse(record(12,[(1,2,0,99)]),okay=False)
        self.parse(record(12,[(1,2,0,12),(1,5,0,12)]),okay=False)

    def test_thread_storage_bound_is_not_silently_truncated(self):
        self.parse(record(12,[(i+1,2,0,12) for i in range(129)]),okay=False)


if __name__=="__main__": unittest.main()
