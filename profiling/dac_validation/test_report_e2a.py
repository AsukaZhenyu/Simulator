"""The E2A report may only claim the build it is reporting.

The provenance paragraph of the frozen report names a specific MSVC build of a
specific machine. A campaign re-run on a Linux host or on a Blackwell device
still produces the same report shape, and inheriting that sentence would state,
in a paper-facing artifact, a compiler and an architecture the run never used.
`native_build` is the one place that decides which sentence is true.
"""
from __future__ import annotations

import unittest

from profiling.dac_validation import report_e2a as subject


class NativeBuildTests(unittest.TestCase):
    def test_the_archived_cohort_keeps_its_sentence_word_for_word(self):
        # The frozen E2A_REPORT.md is compared character for character on the
        # machine that produced it, so this string is not free to change.
        self.assertEqual(
            "Native build: MSVC 19.44.35217.0/v143, Release/MaxSpeed, C++17, "
            "CUDA SDK 12.6.0, NVCC 12.6.20, CUDA runtime 12.6.37, "
            "compute architecture 89.",
            subject.ARCHIVED_NATIVE_BUILD)

    def test_the_build_that_produced_the_frozen_numbers_keeps_its_sentence(self):
        record = dict(compiler=r"C:\MSVC\cl.exe", compiler_version="19.44.35217.0")
        self.assertEqual(subject.ARCHIVED_NATIVE_BUILD, subject.native_build(record))

    def test_another_device_gets_its_own_build_and_no_inherited_claims(self):
        record = dict(compiler="/usr/bin/g++", compiler_version="13.3.0")
        sentence = subject.native_build(record)
        self.assertIn("g++ 13.3.0", sentence)
        self.assertNotIn("MSVC", sentence)
        # Neither the flags nor the toolkit were recorded by the manifest, so
        # none of them may be asserted.
        for absent in ("Release/MaxSpeed", "CUDA SDK 12.6.0", "architecture 89"):
            self.assertNotIn(absent, sentence)

    def test_a_manifest_that_records_no_compiler_claims_no_build(self):
        sentence = subject.native_build(dict(compiler=None, compiler_version=None))
        self.assertIn("records no compiler version", sentence)
        self.assertNotIn("MSVC", sentence)

    def test_a_report_without_a_reproducibility_record_is_the_archived_one(self):
        # Re-running the report from the frozen archive must not invent a
        # provenance paragraph just because the record was not kept beside it.
        self.assertEqual(subject.ARCHIVED_NATIVE_BUILD, subject.native_build(None))


if __name__ == "__main__":
    unittest.main()
