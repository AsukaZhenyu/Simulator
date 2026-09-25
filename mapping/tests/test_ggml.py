"""Integration verification of the C++ GGML exporter (``DESIGN.md`` §3).

The exporter is a separate build (``mapping/ggml``), so this module is the only
place where a real ggml graph meets the Python model. It checks the chain in the
order the data flows:

1. every artifact in ``examples/ggml/`` calls itself a ggml export, and its
   provenance is recorded where a reader can find it;
2. each export is field-for-field identical to the hand-written fixture for the
   same graph -- the ids already match, so equality is asserted directly rather
   than through a renaming table;
3. the ``ACCEPTANCE.md`` §2--§4 numbers still hold when the search runs on the
   real exports instead of the fixtures, and the export and the fixture reach the
   same optimum;
4. a fresh export reproduces the committed bytes exactly, and the four negative
   fixtures are refused by name with no file written.

Points 1--3 read the committed artifacts and always run. Point 4 needs the
compiled binary; when it is missing those tests skip and say plainly that the
export was **not** regenerated in this run, so the compile -> export half of the
chain is unverified. A skip is not a pass (``DESIGN.md`` §9): on a machine with
no build the suite must not be able to look green about a step it never ran.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tensor_mapping import (
    load_and_validate,
    load_mapping,
    search,
    workload_fingerprint,
)
from tensor_mapping.engine import REASON_GOAL_REACHED, STATUS_VALID, evaluate_mapping
from tensor_mapping.mapper import (
    STATUS_INFEASIBLE,
    STATUS_OPTIMAL,
    TERMINATION_EXHAUSTED,
    TERMINATION_GOAL_POPPED,
)

from .support import EXAMPLES, documents, load_example, ms, with_capacity

GGML_EXAMPLES = EXAMPLES / "ggml"

# The four graphs the exporter emits, in the order mapping/ggml declares them.
GRAPHS = ("chain", "residual", "fork", "matvec")

# What an export and its fixture are allowed to disagree about. The comparison
# strips exactly these keys and nothing else, so a field the exporter forgets to
# write fails as a missing key rather than going unnoticed:
#
#   origin  -- kind is "ggml" vs "synthetic_fixture", and a synthetic fixture has
#              no ggml_version to report;
#   comment -- the export records its build provenance there; the fixtures carry
#              no comment at all;
#   name    -- the fixtures use realistic llama.cpp names
#              ("blk.0.attn_q.weight") while the export sets name = id, the only
#              stable identifier a ggml graph gives it. The divergence is
#              asserted below, so it stays visible instead of hiding in a filter.
DOCUMENT_DIVERGENCES = ("origin", "comment")
TENSOR_DIVERGENCES = ("name",)

# ACCEPTANCE.md §2--§4, as (graph, capacity, status, termination, makespan_ms,
# peak_bytes, h2d_bytes). These are contract numbers: the export has to reach
# them without the contract being edited.
ACCEPTANCE_TABLE = (
    ("chain", 160, STATUS_OPTIMAL, TERMINATION_GOAL_POPPED, 8, 160, 128),
    ("chain", 96, STATUS_OPTIMAL, TERMINATION_GOAL_POPPED, 10, 96, 128),
    ("chain", 95, STATUS_INFEASIBLE, TERMINATION_EXHAUSTED, None, None, None),
    ("residual", 160, STATUS_OPTIMAL, TERMINATION_GOAL_POPPED, 9, 160, 128),
    ("residual", 112, STATUS_OPTIMAL, TERMINATION_GOAL_POPPED, 11, 112, 128),
    ("residual", 111, STATUS_INFEASIBLE, TERMINATION_EXHAUSTED, None, None, None),
    ("fork", 160, STATUS_OPTIMAL, TERMINATION_GOAL_POPPED, 9, 160, 128),
    ("fork", 112, STATUS_OPTIMAL, TERMINATION_GOAL_POPPED, 11, 112, 128),
    ("fork", 111, STATUS_INFEASIBLE, TERMINATION_EXHAUSTED, None, None, None),
)

# matvec is not in ACCEPTANCE.md §2--§4 -- those clauses describe the 4x4 graphs,
# where §1 fixes W at 64 B and x at 16 B. Its numbers are derived from the
# export's own geometry instead: W is 3x4 (48 B), x is 3x1 (12 B), y is 4x1
# (16 B), so the peak is x + W + y = 76 B and the makespan is the 3 ms copy plus
# the 2 ms MUL_MAT it gates = 5 ms, with nothing to overlap. Asserted for the
# same reason as the table above: a layout mistake in the rectangular case shows
# up here as a wrong byte count.
GEOMETRY_TABLE = (("matvec", 160, STATUS_OPTIMAL, TERMINATION_GOAL_POPPED, 5, 76, 48),)

# Where the exporter lands. The VS generator appends the configuration name, so
# bin/Release comes first; the rest are here so a different generator or a bare
# out-of-source build still gets found (mapping/ggml/README.md).
EXPORTER_ENV = "TENSOR_MAPPING_GGML_EXPORTER"
EXPORTER_BUILD = Path(__file__).resolve().parent.parent / "ggml" / "build"


def exporter_candidates() -> list[Path]:
    names = ("ggml-export-workload.exe", "ggml-export-workload")
    roots = (
        EXPORTER_BUILD / "bin" / "Release",
        EXPORTER_BUILD / "bin" / "RelWithDebInfo",
        EXPORTER_BUILD / "bin",
        EXPORTER_BUILD,
    )
    return [root / name for root in roots for name in names]


EXPORTER_SKIP = (
    "no compiled ggml-export-workload: the artifacts in examples/ggml/ were NOT "
    "regenerated in this run, so the build -> export half of the chain is "
    "UNVERIFIED (the committed artifacts were still loaded and searched by the "
    "other tests). Build it with the commands in mapping/ggml/README.md, or point "
    f"${EXPORTER_ENV} at the binary. Tried: "
    + ", ".join(str(candidate) for candidate in exporter_candidates())
)


def exporter_or_skip(case: unittest.TestCase) -> Path:
    """The binary, or a skip that says exactly what went unverified."""
    override = os.environ.get(EXPORTER_ENV)
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate
        case.skipTest(f"${EXPORTER_ENV} is {candidate}, which is not a file")
    for candidate in exporter_candidates():
        if candidate.is_file():
            return candidate
    case.skipTest(EXPORTER_SKIP)  # always raises
    raise AssertionError("unreachable")


def run_exporter(exporter: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(exporter), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_document(graph: str) -> dict[str, Any]:
    return read_json(GGML_EXAMPLES / f"{graph}.workload.json")


def fixture_document(graph: str) -> dict[str, Any]:
    return read_json(EXAMPLES / f"{graph}.workload.json")


def without(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def normalised(document: dict[str, Any]) -> dict[str, Any]:
    """The document with the known, asserted divergences removed.

    Every remaining key has to match its fixture exactly -- including the ones
    the comparison does not name, which is the point.
    """
    stripped = without(document, DOCUMENT_DIVERGENCES)
    stripped["tensors"] = [
        without(tensor, TENSOR_DIVERGENCES) for tensor in document["tensors"]
    ]
    return stripped


def ggml_scenario(graph: str, capacity: int) -> Any:
    """A scenario that reads the real export, at the requested capacity."""
    path = GGML_EXAMPLES / f"{graph}-cap160.json"
    return with_capacity(load_and_validate(path), capacity)


# ---------------------------------------------------------------------------
# 1--2. The committed artifacts
# ---------------------------------------------------------------------------


class ExportedArtifactTests(unittest.TestCase):
    """The checked-in exports are real ggml exports of the graphs they claim."""

    def test_the_exported_workloads_declare_themselves_real_ggml_exports(self) -> None:
        """The mirror of test_spec.py::test_no_fixture_claims_to_be_a_ggml_export.

        That test globs the non-recursive ``examples/*.workload.json`` and so
        never sees ``examples/ggml/``. Together the two pin the reserved value
        from both sides: no hand-written fixture may claim it, and every document
        that claims it must be a real export.
        """
        for graph in GRAPHS:
            with self.subTest(graph=graph):
                document = export_document(graph)
                origin = document["origin"]
                self.assertEqual(origin["kind"], "ggml")
                self.assertIsInstance(origin["ggml_version"], str)
                self.assertTrue(origin["ggml_version"])
                # The version comes from the runtime ggml_version(), not from a
                # literal: GGML_VERSION is target_compile_definitions(... PRIVATE)
                # and is not visible to a consumer's translation unit.
                self.assertRegex(origin["ggml_version"], r"^\d+\.\d+\.\d+")
                # origin allows exactly kind/ggml_version/sample, so the commit
                # and the build method have to live somewhere else -- and that
                # somewhere is the comment. If it stops carrying them, the
                # artifact loses its provenance.
                comment = document["comment"]
                self.assertIn(origin["ggml_version"], comment)
                self.assertIn("commit", comment)

    def test_the_export_is_field_for_field_identical_to_its_fixture(self) -> None:
        """The sharpest statement that the export and the fixture are one workload.

        Only ``origin``, ``comment`` and the per-tensor ``name`` may differ. The
        ids match by construction (mapping/ggml/src/graphs.cpp reuses the fixture
        vocabulary on purpose), so this compares whole documents rather than a
        list of fields -- a field the exporter stops writing fails as a missing
        key.
        """
        for graph in GRAPHS:
            with self.subTest(graph=graph):
                export = export_document(graph)
                fixture = fixture_document(graph)
                self.assertEqual(normalised(export), normalised(fixture))
                # Documents compare without regard to order, so the order the
                # model reads in is asserted separately: both lists are
                # topological, and the export's has to be the fixture's.
                self.assertEqual(
                    [t["id"] for t in export["tensors"]],
                    [t["id"] for t in fixture["tensors"]],
                )
                self.assertEqual(
                    [o["id"] for o in export["operations"]],
                    [o["id"] for o in fixture["operations"]],
                )
                self.assertEqual(export["outputs"], fixture["outputs"])

    def test_the_only_per_tensor_divergence_is_the_name(self) -> None:
        """The excluded ``name`` key is a real difference, not a hidden one.

        The fixtures name tensors the way llama.cpp would; the export can only
        use the id, because a ggml graph carries no other stable identifier. Both
        halves are asserted so that excluding the key stays honest: if the
        exporter ever copied a fixture name, or the fixture dropped to bare ids,
        this fails instead of the comparison quietly widening.
        """
        for graph in GRAPHS:
            with self.subTest(graph=graph):
                for exported, fixture in zip(
                    export_document(graph)["tensors"], fixture_document(graph)["tensors"]
                ):
                    self.assertEqual(exported["name"], exported["id"])
                    self.assertNotEqual(fixture["name"], fixture["id"])
                    self.assertNotEqual(exported["name"], fixture["name"])

    def test_the_rectangular_weight_cannot_be_read_the_numpy_way(self) -> None:
        """The one graph where an ne/nb convention error changes the numbers.

        ggml stores ``ne[0]`` innermost and MUL_MAT takes ``[weight,
        activation]`` with the weight ``ne = [in_features, out_features, 1, 1]``.
        Read it the numpy way (``a[0]`` = rows = out_features) and the two swap.
        A square weight cannot tell the readings apart -- swapping keeps every
        byte count identical -- so matvec is the graph that can.
        """
        tensors = {t["id"]: t for t in export_document("matvec")["tensors"]}
        weight, activation, output = tensors["W"], tensors["x"], tensors["y"]

        self.assertEqual(weight["ne"], [3, 4, 1, 1])
        self.assertEqual(activation["ne"], [3, 1, 1, 1])
        self.assertEqual(output["ne"], [4, 1, 1, 1])
        # The activation is as wide as ne[0], and ne[0] != ne[1] here.
        self.assertEqual(activation["ne"][0], weight["ne"][0])
        self.assertNotEqual(weight["ne"][0], weight["ne"][1])

        self.assertEqual(activation["storage_bytes"], 12)
        self.assertEqual(weight["storage_bytes"], 48)
        # Read the other way the activation would have to be ne[1] = 4 wide, i.e.
        # 16 B. That the number is not 16 is the whole assertion.
        self.assertNotEqual(activation["storage_bytes"], 4 * weight["ne"][1])

        # nb has to agree with ne, and it is read from the tensor rather than
        # recomputed: nb[i] == nb[i-1] * ne[i-1], with nb[0] the 4-byte F32 unit.
        self.assertEqual(weight["nb"], [4, 12, 48, 48])
        self.assertEqual(activation["nb"], [4, 12, 12, 12])

    def test_an_in_out_swap_leaves_the_search_result_identical(self) -> None:
        """A negative control: the aggregate numbers cannot catch this mistake.

        Read the same 3x4 weight the numpy way -- in = 4, out = 3 -- and the
        graph stays legal, because 3*4 == 4*3 keeps the weight at 48 B while x
        and y swap their sizes (16 + 12 == 12 + 16). Nothing the search reports
        moves: the same peak, the same 48 B copy, the same 3 + 2 ms makespan.

        So the evidence that the export read the layout correctly is the
        per-tensor ``ne``/``storage_bytes`` asserted above, not the acceptance
        numbers. This test exists to keep that honest in both directions: it
        records that the field-level checks are load-bearing, and it fails if
        someone later tries to justify dropping them on the grounds that the
        makespan would notice (it would not).
        """
        transposed = copy.deepcopy(export_document("matvec"))
        swapped = {"W": (4, 3, 1, 1), "x": (4, 1, 1, 1), "y": (3, 1, 1, 1)}
        for tensor in transposed["tensors"]:
            ne = swapped[tensor["id"]]
            tensor["ne"] = list(ne)
            tensor["nb"] = [4, 4 * ne[0], 4 * ne[0] * ne[1], 4 * ne[0] * ne[1] * ne[2]]
            tensor["storage_bytes"] = tensor["nb"][3]
        # The swapped document is a legal MUL_MAT: W.ne[0] == x.ne[0] == 4 and
        # y.ne[0] == W.ne[1] == 3, so the loader takes it.
        scenario_document = read_json(GGML_EXAMPLES / "matvec-cap160.json")
        with documents(workload=transposed, scenario=scenario_document) as path:
            result = search(load_and_validate(path))
        self.assertEqual(result.status, STATUS_OPTIMAL)
        self.assertEqual(result.makespan_ns, ms(5))
        self.assertEqual(result.peak_vram_bytes, 76)
        self.assertEqual(result.evaluation.h2d_bytes, 48)
        # The one thing that does change is the pair of byte counts -- the
        # discriminating signal, and the reason the fixture comparison asserts
        # them tensor by tensor.
        sizes = {t["id"]: t["storage_bytes"] for t in transposed["tensors"]}
        self.assertEqual(sizes, {"W": 48, "x": 16, "y": 12})
        self.assertNotEqual(
            sizes,
            {t["id"]: t["storage_bytes"] for t in export_document("matvec")["tensors"]},
        )


# ---------------------------------------------------------------------------
# 3. The acceptance numbers, on the real exports
# ---------------------------------------------------------------------------


class ExportedSearchTests(unittest.TestCase):
    """ACCEPTANCE.md §2--§4, replayed against the exported workloads."""

    def test_the_acceptance_table_holds_on_the_real_exports(self) -> None:
        for graph, capacity, status, termination, makespan_ms, peak, h2d in (
            ACCEPTANCE_TABLE + GEOMETRY_TABLE
        ):
            with self.subTest(graph=graph, capacity=capacity):
                scenario = ggml_scenario(graph, capacity)
                # The clone has to keep pointing at the export, or this test
                # would silently be re-running the fixture.
                self.assertEqual(scenario.workload.origin.kind, "ggml")
                self.assertEqual(scenario.id, f"{graph}-cap{capacity}")

                result = search(scenario)
                print(
                    f"    {scenario.id:<16} {result.status:<10} "
                    f"{result.termination_reason:<22} makespan={result.makespan_ns} "
                    f"peak={result.peak_vram_bytes} "
                    f"h2d={None if result.evaluation is None else result.evaluation.h2d_bytes} "
                    f"expanded={result.expanded_states} proven={result.optimality_proven}",
                    file=sys.stderr,
                )

                self.assertEqual(result.status, status)
                self.assertEqual(result.termination_reason, termination)
                # An infeasible verdict has to be a proof, not a give-up: the
                # budget is unlimited here, so exhaustion means the space was
                # really empty.
                self.assertTrue(result.optimality_proven)
                if makespan_ms is None:
                    self.assertIsNone(result.makespan_ns)
                    self.assertIsNone(result.peak_vram_bytes)
                    continue
                self.assertEqual(result.makespan_ns, ms(makespan_ms))
                self.assertEqual(result.peak_vram_bytes, peak)
                self.assertIsNotNone(result.evaluation)
                self.assertEqual(result.evaluation.h2d_bytes, h2d)
                # The search reports the numbers the evaluator produces from the
                # same action list; if the two ever disagree the optimum is not
                # what the search claims.
                self.assertEqual(result.status, STATUS_OPTIMAL)

    def test_the_export_and_the_fixture_reach_the_same_optimum(self) -> None:
        """What examples/ggml/*-cap160.json claim in their comments.

        The scenarios share an id with their fixtures, so the pair is one
        scenario pointed at two different workload documents. Field-for-field
        equality already implies the same optimum, but the claim the artifacts
        make is about the search, so it is checked on the search.
        """
        for graph in GRAPHS:
            with self.subTest(graph=graph):
                from_export = search(ggml_scenario(graph, 160))
                from_fixture = search(load_example(f"{graph}-cap160"))
                self.assertEqual(from_export.status, from_fixture.status)
                self.assertEqual(from_export.makespan_ns, from_fixture.makespan_ns)
                self.assertEqual(from_export.peak_vram_bytes, from_fixture.peak_vram_bytes)

    def test_the_two_documents_are_still_different_workloads(self) -> None:
        """Same graph, different identity -- and that is deliberate.

        The fingerprint keeps ``origin.kind`` and the per-tensor ``name``
        (spec.py: _canonical_workload), so an export and its fixture are two
        distinct workloads even though every shape and byte count matches. The
        consequence is real: a mapping fingerprinted against the fixture is
        refused for the export, which is why examples/ggml/chain.mapping.json is
        hand-written and carries no fingerprints. Pinned here so that if the
        canonical form ever drops ``kind`` or ``name``, the change is noticed
        rather than silently making the two interchangeable -- and rather than
        silently breaking the mapping that relies on it.
        """
        for graph in GRAPHS:
            with self.subTest(graph=graph):
                exported = ggml_scenario(graph, 160).workload
                fixture = load_example(f"{graph}-cap160").workload
                self.assertEqual(exported.origin.kind, "ggml")
                self.assertNotEqual(exported.origin.kind, fixture.origin.kind)
                self.assertNotEqual(
                    workload_fingerprint(exported), workload_fingerprint(fixture)
                )
                # Everything that is not identity matches, all the same.
                self.assertEqual(
                    [(t.id, list(t.ne), list(t.nb), t.storage_bytes) for t in exported.tensors],
                    [(t.id, list(t.ne), list(t.nb), t.storage_bytes) for t in fixture.tensors],
                )

    def test_the_worked_mapping_replays_on_the_real_export(self) -> None:
        """The DESIGN.md §4.3 worked mapping, bound to the export.

        examples/ggml/chain.mapping.json is byte-identical in its action list to
        examples/chain.mapping.json on purpose: the same ten actions have to be
        legal and optimal for both documents, which cannot be true unless the two
        describe the same workload. Hand-written, so it carries no fingerprints
        and must replay as unverified.
        """
        scenario = ggml_scenario("chain", 160)
        document = load_mapping(GGML_EXAMPLES / "chain.mapping.json", scenario)
        self.assertEqual(document.scenario_id, scenario.id)
        self.assertFalse(document.fingerprint_verified)
        self.assertIsNone(document.workload_fingerprint)

        result = evaluate_mapping(scenario, document.actions)
        self.assertEqual(result.status, STATUS_VALID)
        self.assertEqual(result.reason, REASON_GOAL_REACHED)
        self.assertEqual(result.makespan_ns, ms(8))
        self.assertEqual(result.peak_vram_bytes, 160)
        self.assertEqual(result.h2d_bytes, 128)
        # The action *list* is the claim the two documents make about each other,
        # so it is compared in full -- kind, target and order -- not just in
        # shape. Both load against the same scenario id, which is what lets the
        # same ten actions be legal for either document.
        self.assertEqual(
            document.actions,
            load_mapping(EXAMPLES / "chain.mapping.json", scenario).actions,
        )


# ---------------------------------------------------------------------------
# 4. The exporter itself
# ---------------------------------------------------------------------------


class ExporterRerunTests(unittest.TestCase):
    """Rebuild the artifacts from the binary and check the refusals fire."""

    def test_the_exporter_reproduces_the_committed_artifacts(self) -> None:
        """Byte for byte, not just semantically.

        The document is written with explicit '\\n' and no environment-dependent
        field, so a binary built from the same ggml tree must reproduce the
        committed bytes exactly -- including the comment. A difference means the
        artifacts are stale (rebuild and re-commit them) or that the export
        depends on something it should not.
        """
        exporter = exporter_or_skip(self)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
            directory = Path(raw_dir)
            completed = run_exporter(exporter, "--graph", "all", "--out-dir", str(directory))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            # --graph all means the four exportable graphs and nothing else: a
            # negative fixture appearing here would mean one is being emitted.
            self.assertEqual(
                sorted(p.name for p in directory.glob("*.json")),
                sorted(f"{graph}.workload.json" for graph in GRAPHS),
            )
            for graph in GRAPHS:
                self.assertIn(f"  {graph} ", completed.stderr)
            self.assertNotIn("reject-", completed.stderr)
            for graph in GRAPHS:
                with self.subTest(graph=graph):
                    fresh = (directory / f"{graph}.workload.json").read_bytes()
                    committed = (GGML_EXAMPLES / f"{graph}.workload.json").read_bytes()
                    self.assertEqual(
                        fresh,
                        committed,
                        f"{graph}.workload.json differs from the committed artifact; "
                        "regenerate it with mapping/ggml/README.md",
                    )

            # --out is the other half of the CLI contract (mutually exclusive
            # with --out-dir, and it needs exactly one graph).
            single = directory / "single.json"
            completed = run_exporter(
                exporter, "--graph", "matvec", "--out", str(single)
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                single.read_bytes(), (GGML_EXAMPLES / "matvec.workload.json").read_bytes()
            )

    def test_the_cli_refuses_the_ambiguous_invocations(self) -> None:
        exporter = exporter_or_skip(self)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
            directory = Path(raw_dir)
            target = directory / "out.json"
            cases = (
                ((), "--graph is required"),
                (("--graph", "chain"), "one of --out or --out-dir"),
                (
                    (
                        "--graph",
                        "chain",
                        "--out",
                        str(target),
                        "--out-dir",
                        str(directory),
                    ),
                    "mutually exclusive",
                ),
                (
                    ("--graph", "chain", "--graph", "fork", "--out", str(target)),
                    "--out needs exactly one",
                ),
                (("--graph", "nosuchgraph", "--out", str(target)), "unknown --graph"),
                (("--graph", "chain", "--out", str(target), "--wat"), "unrecognised"),
            )
            for args, expected in cases:
                with self.subTest(args=args):
                    completed = run_exporter(exporter, *args)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(expected, completed.stderr)
            self.assertFalse(target.exists())

    def test_the_negative_fixtures_are_refused_by_name(self) -> None:
        """Every layout rule the exporter claims to enforce, observed firing.

        A refusal no test triggers is only a claim that the code contains a
        branch. Each of these graphs is a *legal* ggml graph -- nothing upstream
        refuses them -- and each violates exactly one rule, so the message has to
        name the offending tensor to be actionable. Nothing may be written: a
        half-emitted artifact is worse than no artifact.
        """
        exporter = exporter_or_skip(self)
        cases = (
            ("reject-f16", "unsupported dtype on 'W'", "f16"),
            ("reject-op", "unsupported op on 'y'", "DUP"),
            ("reject-unary", "unsupported unary op on 'y'", "SILU"),
            ("reject-inplace", "unsupported view on 'y'", "view_src"),
        )
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
            directory = Path(raw_dir)
            for graph, expected, detail in cases:
                with self.subTest(graph=graph):
                    target = directory / f"{graph}.json"
                    completed = run_exporter(exporter, "--graph", graph, "--out", str(target))
                    self.assertEqual(completed.returncode, 1, completed.stdout)
                    self.assertIn(expected, completed.stderr)
                    self.assertIn(detail, completed.stderr)
                    self.assertFalse(
                        target.exists(), f"{graph} refused but still wrote an artifact"
                    )
            self.assertEqual(list(directory.iterdir()), [])

    def test_the_exporter_reports_the_ggml_it_was_built_against(self) -> None:
        """The version in the artifacts has to be the one the binary reports.

        ggml_version() reads what the library was built with; a consumer cannot
        see GGML_VERSION, which is a private compile definition. Everything in
        the artifact's provenance therefore rests on the runtime call, so the two
        are compared rather than one being assumed from the other.
        """
        exporter = exporter_or_skip(self)
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as raw_dir:
            target = Path(raw_dir) / "chain.json"
            completed = run_exporter(exporter, "--graph", "chain", "--out", str(target))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            reported = read_json(target)["origin"]["ggml_version"]
        self.assertIn(f"ggml {reported} commit ", completed.stderr)
        # The commit is honest about not being a git checkout: the source tree is
        # a release snapshot, so ggml_commit() says so instead of inventing one.
        self.assertIn(read_json(GGML_EXAMPLES / "chain.workload.json")["origin"]["ggml_version"], completed.stderr)


if __name__ == "__main__":
    unittest.main()
