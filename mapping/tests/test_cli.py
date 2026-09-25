"""End-to-end tests for the ``model`` and ``search`` command line.

Exactly one test spawns a real ``python -m tensor_mapping`` process: that is what
proves ``__main__.py`` is wired up and that the exit code reaches the shell.
Every other test calls :func:`tensor_mapping.cli.main` in process and asserts on
its return value, which is the same integer the process would exit with. That
keeps the failure message attached to the assertion instead of to a subprocess's
captured stderr, and skips a ~100 ms interpreter start per case.

The tests follow ``ACCEPTANCE.md`` §7's flat list rather than mirroring the file
structure field by field: each one names a sentence of the contract it pins, so a
failure points at the requirement rather than at a schema.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from tensor_mapping import cli
from tensor_mapping.engine import load_mapping
from tensor_mapping.spec import load_and_validate

from . import support

EXAMPLES = support.EXAMPLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(*argv: str) -> tuple[int, str, str]:
    """Run ``main`` in process and capture what a shell would see."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def read_artifact(directory: Path, name: str) -> dict[str, Any]:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def example_mapping() -> dict[str, Any]:
    """The hand-written ``chain`` mapping, as a fresh mutable document."""
    return json.loads((EXAMPLES / "chain.mapping.json").read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def scenario_copy(scenario: dict[str, Any], **architecture: int) -> dict[str, Any]:
    for key, value in architecture.items():
        scenario["architecture"][key] = value
    return scenario


class ArtifactCase(unittest.TestCase):
    """Base class owning one temp directory per test."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def out(self, name: str = "out") -> Path:
        return self.tmp / name

    def search_example(self, scenario_path: Path, out: Path, *extra: str) -> dict[str, Any]:
        code, _, err = run_cli(
            "search", "--scenario", str(scenario_path), "--out", str(out), *extra
        )
        self.assertEqual(code, cli.EXIT_OK, err)
        return read_artifact(out, cli.STATS_FILE)

    def replay(
        self, scenario_path: Path, mapping_path: Path, out: Path
    ) -> tuple[int, dict[str, Any], str]:
        code, _, err = run_cli(
            "model",
            "--scenario",
            str(scenario_path),
            "--mapping",
            str(mapping_path),
            "--out",
            str(out),
        )
        stats = read_artifact(out, cli.STATS_FILE) if (out / cli.STATS_FILE).exists() else {}
        return code, stats, err


# ---------------------------------------------------------------------------
# ACCEPTANCE §7 item 3-4: the two entry points and their artefacts
# ---------------------------------------------------------------------------


class ModelCommandTests(ArtifactCase):
    """``model`` reproduces the acceptance numbers and writes three artefacts."""

    def test_replays_the_contract_mapping(self) -> None:
        """ACCEPTANCE §2: chain-cap160 replay is 8 ms, peak 160 B, 128 B copied."""
        scenario = EXAMPLES / "chain-cap160.json"
        code, out, err = run_cli(
            "model",
            "--scenario",
            str(scenario),
            "--mapping",
            str(EXAMPLES / "chain.mapping.json"),
            "--out",
            str(self.out()),
        )
        self.assertEqual(code, cli.EXIT_OK, err)

        stats = read_artifact(self.out(), cli.STATS_FILE)
        self.assertEqual(stats["status"], "valid")
        self.assertEqual(stats["reason"], "goal_reached")
        self.assertEqual(stats["makespan_ns"], 8_000_000)
        self.assertEqual(stats["peak_vram_bytes"], 160)
        self.assertEqual(stats["h2d_bytes"], 128)
        self.assertEqual(stats["action_count"], 10)
        self.assertEqual(stats["declared_action_count"], 10)
        self.assertEqual(stats["mode"], "model")
        self.assertNotIn("error", stats)
        self.assertNotIn("search", stats)

        # DESIGN.md §8:274: the graph source and the cost source are disclosed
        # separately, so a synthetic fixture can never read as a real export.
        self.assertEqual(stats["graph_origin"]["kind"], "synthetic_fixture")
        self.assertIsNone(stats["graph_origin"]["ggml_version"])
        self.assertEqual(stats["cost_source"], "synthetic_fixed")

    def test_events_are_indexed_and_well_formed(self) -> None:
        """DESIGN.md §8:266: one event per action, index preserved, bytes split."""
        run_cli(
            "model",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--mapping",
            str(EXAMPLES / "chain.mapping.json"),
            "--out",
            str(self.out()),
        )
        events = read_artifact(self.out(), cli.EVENTS_FILE)["events"]

        self.assertEqual([e["action_index"] for e in events], list(range(10)))
        # DESIGN.md:190 -- bytes *moved* are the tensor's own size, bytes
        # *released* are the aligned allocation. This fixture has alignment 1,
        # so they coincide; the alignment test below separates them.
        for event in events:
            if event["kind"] == "COPY_H2D":
                self.assertEqual(event["bytes_transferred"], 64)
                self.assertEqual(event["resource"], "h2d_copy")
            elif event["kind"] == "COMPUTE":
                self.assertEqual(event["resource"], "gpu_compute")
            elif event["kind"] == "ADVANCE":
                # An advance is not a task, so it has no start: `end_ns` is the
                # time it moved the clock to. Writing a start here would make it
                # look like work with a duration (engine._event_for).
                self.assertIsNone(event["target_id"])
                self.assertIsNone(event["start_ns"])
                self.assertEqual(event["end_ns"], event["t_ns"])

    def test_states_cover_start_and_every_action(self) -> None:
        """DESIGN.md §8:267: the initial state, then one state per action."""
        run_cli(
            "model",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--mapping",
            str(EXAMPLES / "chain.mapping.json"),
            "--out",
            str(self.out()),
        )
        states = read_artifact(self.out(), cli.STATES_FILE)["states"]

        self.assertEqual(len(states), 11)
        self.assertIsNone(states[0]["action_index"])
        self.assertEqual(states[0]["t_ns"], 0)
        self.assertEqual(states[0]["used_vram_bytes"], 16)  # only `x` is resident
        for index, state in enumerate(states[1:]):
            self.assertEqual(state["action_index"], index)
        times = [state["t_ns"] for state in states]
        self.assertEqual(times, sorted(times))

    def test_model_does_not_write_a_mapping(self) -> None:
        """DESIGN.md §8: "search 另外产出 mapping.json" -- model writes three files."""
        run_cli(
            "model",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--mapping",
            str(EXAMPLES / "chain.mapping.json"),
            "--out",
            str(self.out()),
        )
        self.assertEqual(
            sorted(p.name for p in self.out().iterdir()),
            [cli.EVENTS_FILE, cli.STATES_FILE, cli.STATS_FILE],
        )
        # Provenance is not lost: the replayed mapping is named in stats.json.
        stats = read_artifact(self.out(), cli.STATS_FILE)
        self.assertTrue(stats["mapping_file"].endswith("chain.mapping.json"))

    def test_out_directory_is_created(self) -> None:
        """DESIGN.md §9 M0.4: replay must work from a clean output directory."""
        target = self.out("nested/does/not/exist/out")
        code, _, err = run_cli(
            "model",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--mapping",
            str(EXAMPLES / "chain.mapping.json"),
            "--out",
            str(target),
        )
        self.assertEqual(code, cli.EXIT_OK, err)
        self.assertTrue((target / cli.STATS_FILE).is_file())


class SearchAndReplayTests(ArtifactCase):
    """The M0.4 gate: a searched mapping replays to the same numbers, alone."""

    def test_search_output_replays_identically(self) -> None:
        """DESIGN.md §9 M0.4 / ACCEPTANCE.md:106 -- 映射独立回放一致."""
        scenario = EXAMPLES / "chain-cap160.json"
        found = self.out("search")
        self.search_example(scenario, found)
        mapping = found / cli.MAPPING_FILE
        self.assertTrue(mapping.is_file())

        replayed = self.out("replay")
        code, stats, err = self.replay(scenario, mapping, replayed)
        self.assertEqual(code, cli.EXIT_OK, err)
        self.assertEqual(stats["status"], "valid")

        searched = read_artifact(found, cli.STATS_FILE)
        for key in ("makespan_ns", "peak_vram_bytes", "h2d_bytes"):
            self.assertEqual(stats[key], searched[key], key)

        # Event logs must agree field by field, not just in their totals.
        left = read_artifact(found, cli.EVENTS_FILE)
        right = read_artifact(replayed, cli.EVENTS_FILE)
        self.assertEqual(left["events"], right["events"])
        self.assertEqual(len(left["events"]), searched["action_count"])

    def test_search_mapping_carries_verified_fingerprints(self) -> None:
        """DESIGN.md §4.3: both digests, so a replay can prove it belongs."""
        scenario = EXAMPLES / "chain-cap160.json"
        self.search_example(scenario, self.out())
        document = load_mapping(self.out() / cli.MAPPING_FILE, load_and_validate(scenario))
        self.assertTrue(document.fingerprint_verified)

    def test_search_artifacts_satisfy_the_contract(self) -> None:
        """DESIGN.md §8:268: termination reason, optimality and the budget."""
        stats = self.search_example(EXAMPLES / "chain-cap160.json", self.out())
        self.assertEqual(stats["status"], "optimal")
        self.assertEqual(stats["reason"], "goal_popped")
        self.assertEqual(stats["makespan_ns"], 8_000_000)
        self.assertTrue(stats["search"]["optimality_proven"])
        self.assertGreater(stats["search"]["expanded_states"], 0)
        self.assertGreaterEqual(stats["search"]["visited_states"], 1)
        # `limits` must agree with the block printed beside it, or a caller
        # overriding the budget on the command line gets two different answers.
        self.assertEqual(
            stats["limits"]["max_expanded_states"], stats["search"]["max_expanded_states"]
        )
        self.assertEqual(
            stats["limits"]["wall_time_limit_s"], stats["search"]["wall_time_limit_s"]
        )

    def test_command_line_budget_is_recorded_as_effective(self) -> None:
        """DESIGN.md §7: the artefact records the budget in force, not the default."""
        stats = self.search_example(
            EXAMPLES / "chain-cap160.json",
            self.out(),
            "--max-expanded-states",
            "500",
            "--wall-time-limit-s",
            "12.5",
        )
        self.assertEqual(stats["limits"]["max_expanded_states"], 500)
        self.assertEqual(stats["search"]["max_expanded_states"], 500)
        self.assertEqual(stats["limits"]["wall_time_limit_s"], 12.5)
        self.assertEqual(stats["search"]["wall_time_limit_s"], 12.5)

    def test_feasible_is_reported_without_claiming_optimality(self) -> None:
        """DESIGN.md §7 / ACCEPTANCE.md §6:110 -- a budget cut is not a proof.

        A limit of 28 leaves the search one step short of popping the goal it has
        already found, so the plan is real and replayable but nothing about
        optimality has been established.
        """
        stats = self.search_example(
            EXAMPLES / "chain-cap160.json", self.out(), "--max-expanded-states", "28"
        )
        self.assertEqual(stats["status"], "feasible")
        self.assertEqual(stats["reason"], "state_budget_exhausted")
        self.assertFalse(stats["search"]["optimality_proven"])
        self.assertEqual(stats["makespan_ns"], 8_000_000)

        # The generated comment is the only place the strength of the claim
        # travels with the mapping, so it must not say "optimal".
        mapping = read_artifact(self.out(), cli.MAPPING_FILE)
        self.assertIn("NOT proven optimal", mapping["comment"])
        self.assertNotIn("a proven optimum", mapping["comment"])

        # And it is a genuine mapping: it replays as valid, on its own.
        code, replayed, err = self.replay(
            EXAMPLES / "chain-cap160.json", self.out() / cli.MAPPING_FILE, self.out("r")
        )
        self.assertEqual(code, cli.EXIT_OK, err)
        self.assertEqual(replayed["status"], "valid")
        self.assertEqual(replayed["makespan_ns"], stats["makespan_ns"])

    def test_optimal_comment_says_so(self) -> None:
        self.search_example(EXAMPLES / "chain-cap160.json", self.out())
        mapping = read_artifact(self.out(), cli.MAPPING_FILE)
        self.assertIn("a proven optimum", mapping["comment"])
        self.assertNotIn("NOT proven optimal", mapping["comment"])

    def test_real_ggml_export_scenario(self) -> None:
        """ACCEPTANCE.md:113 -- the same two commands on a real export.

        ``examples/ggml/chain-cap160.json`` shares its scenario id with the
        fixture, so this also shows that identity alone does not distinguish two
        graphs; the workload fingerprint in a search-produced mapping does.
        """
        scenario = EXAMPLES / "ggml" / "chain-cap160.json"
        found = self.out("search")
        stats = self.search_example(scenario, found)
        self.assertEqual(stats["status"], "optimal")
        self.assertEqual(stats["makespan_ns"], 8_000_000)
        self.assertEqual(stats["graph_origin"]["kind"], "ggml")
        self.assertTrue(stats["graph_origin"]["ggml_version"])
        # The graph is real; the durations are still made up, and both must be
        # disclosed (DESIGN.md:271).
        self.assertEqual(stats["cost_source"], "synthetic_fixed")

        code, replayed, err = self.replay(scenario, found / cli.MAPPING_FILE, self.out("r"))
        self.assertEqual(code, cli.EXIT_OK, err)
        self.assertEqual(replayed["makespan_ns"], stats["makespan_ns"])
        self.assertEqual(replayed["peak_vram_bytes"], stats["peak_vram_bytes"])
        self.assertEqual(replayed["h2d_bytes"], stats["h2d_bytes"])
        self.assertEqual(
            read_artifact(found, cli.EVENTS_FILE)["events"],
            read_artifact(self.out("r"), cli.EVENTS_FILE)["events"],
        )


# ---------------------------------------------------------------------------
# The verdict paths: invalid, incomplete, infeasible, unknown, unstartable
# ---------------------------------------------------------------------------


class VerdictTests(ArtifactCase):
    """Every refusal keeps its own status, exit code and diagnosis."""

    def test_capacity_exceeded_points_at_the_failing_action(self) -> None:
        """ACCEPTANCE.md:41 -- the cap160 plan on a 96-byte device dies at index 3."""
        with support.documents(scenario=scenario_copy(
            support.example_document("scenario", "chain-cap160"),
            vram_capacity_bytes=96,
        )) as scenario_path:
            mapping = example_mapping()
            # Retargeted at the smaller scenario and stripped of its digests, so
            # the identity check cannot pre-empt the capacity verdict this test
            # is about.
            mapping["scenario_id"] = "chain-cap160"
            mapping.pop("workload_fingerprint", None)
            mapping.pop("scenario_fingerprint", None)
            mapping_path = write_json(scenario_path.parent / "m.json", mapping)

            code, stats, err = self.replay(scenario_path, mapping_path, self.out())

        self.assertEqual(code, cli.EXIT_NO_USABLE_MAPPING)
        self.assertEqual(stats["status"], "invalid_mapping")
        self.assertEqual(stats["error"]["code"], "CAPACITY_EXCEEDED")
        self.assertEqual(stats["error"]["action_index"], 3)
        self.assertEqual(stats["error"]["t_ns"], 3_000_000)
        self.assertIn("over the 96-byte budget", stats["error"]["message"])
        # Applied versus declared: the file lists ten actions, three of them ran.
        self.assertEqual(stats["action_count"], 3)
        self.assertEqual(stats["declared_action_count"], 10)
        self.assertEqual(len(read_artifact(self.out(), cli.EVENTS_FILE)["events"]), 3)
        self.assertIn("CAPACITY_EXCEEDED", err)

    def test_incomplete_mapping_is_diagnosed_without_an_error_code(self) -> None:
        """The one verdict whose ``error_code`` is ``None``.

        ``engine._failure`` leaves the code unset when the mapping simply ran out,
        so the action index and the time are the entire diagnosis. Gating the
        ``error`` block on the code being present would discard exactly those two
        numbers, which is why it is gated on the status instead.
        """
        mapping = example_mapping()
        mapping["actions"] = mapping["actions"][:5]
        mapping_path = write_json(self.tmp / "short.json", mapping)

        code, stats, err = self.replay(
            EXAMPLES / "chain-cap160.json", mapping_path, self.out()
        )
        self.assertEqual(code, cli.EXIT_NO_USABLE_MAPPING, err)
        self.assertEqual(stats["status"], "incomplete_mapping")
        self.assertEqual(stats["reason"], "mapping_ended_before_the_goal: the mapping ran "
                         "out of actions before every requested output was resident and "
                         "every operation done")
        self.assertIsNone(stats["error"]["code"])
        self.assertEqual(stats["error"]["action_index"], 5)
        self.assertEqual(stats["error"]["t_ns"], 5_000_000)
        self.assertEqual(stats["action_count"], 5)
        self.assertEqual(stats["declared_action_count"], 5)
        # The log stops where the run stopped, and says so.
        self.assertEqual(len(read_artifact(self.out(), cli.STATES_FILE)["states"]), 6)

    def test_infeasible_and_unknown_are_distinguishable(self) -> None:
        """ACCEPTANCE.md §6:110 -- a budget cut must never be reported as infeasible."""
        with support.documents(
            scenario=scenario_copy(
                support.example_document("scenario", "chain-cap160"),
                vram_capacity_bytes=95,
            )
        ) as tight:
            code, exhausted, _ = run_cli(
                "search", "--scenario", str(tight), "--out", str(self.out("a"))
            )

        code_b, cut, _ = run_cli(
            "search",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--out",
            str(self.out("b")),
            "--max-expanded-states",
            "1",
        )

        for code_value in (code, code_b):
            self.assertEqual(code_value, cli.EXIT_NO_USABLE_MAPPING)

        a = read_artifact(self.out("a"), cli.STATS_FILE)
        b = read_artifact(self.out("b"), cli.STATS_FILE)
        self.assertEqual((a["status"], a["reason"]), ("infeasible", "search_space_exhausted"))
        self.assertEqual((b["status"], b["reason"]), ("unknown", "state_budget_exhausted"))
        # Infeasible is a proof; unknown is the absence of one.
        self.assertTrue(a["search"]["optimality_proven"])
        self.assertFalse(b["search"]["optimality_proven"])
        # Nothing to replay either way, so no mapping is written -- an empty
        # action list would replay as `incomplete_mapping` and read as a plan.
        for out in (self.out("a"), self.out("b")):
            self.assertFalse((out / cli.MAPPING_FILE).exists())

        # The three artefacts both entry points always write still appear, and
        # the empty logs explain themselves rather than looking like a no-op run.
        for out in (self.out("a"), self.out("b")):
            events = read_artifact(out, cli.EVENTS_FILE)
            states = read_artifact(out, cli.STATES_FILE)
            self.assertEqual(events["events"], [])
            self.assertEqual(states["states"], [])
            self.assertIn("no action was applied", events["note"])
            self.assertIn("no action was applied", states["note"])
        self.assertIsNone(a["action_count"])
        self.assertIsNone(a["makespan_ns"])

    def test_unstartable_scenario_agrees_across_both_entry_points(self) -> None:
        """DESIGN.md §8:270 -- initial capacity is a verdict, not a broken file."""
        with support.documents(
            scenario=scenario_copy(
                support.example_document("scenario", "chain-cap160"),
                vram_capacity_bytes=8,
            )
        ) as scenario_path:
            code_s, search_stats, err_s = self._search(scenario_path, self.out("s"))
            code_m, model_stats, err_m = self.replay(
                scenario_path, EXAMPLES / "chain.mapping.json", self.out("m")
            )

        for code in (code_s, code_m):
            self.assertEqual(code, cli.EXIT_NO_USABLE_MAPPING, "must not be an input refusal")
        for stats in (search_stats, model_stats):
            self.assertEqual(stats["status"], "infeasible")
            self.assertEqual(stats["reason"], "initial_capacity_exceeded")

        # Only the model path can explain itself in bytes, because the search
        # never got as far as producing an evaluation to explain.
        self.assertIn("initial layout needs 16 bytes", model_stats["error"]["message"])
        self.assertIsNone(model_stats["error"]["action_index"])
        self.assertEqual(model_stats["error"]["code"], "initial_capacity_exceeded")
        self.assertEqual(model_stats["action_count"], 0)
        # The mapping file declared ten actions even though none could run.
        self.assertEqual(model_stats["declared_action_count"], 10)
        self.assertIn("initial_capacity_exceeded", err_s)
        self.assertIn("initial_capacity_exceeded", err_m)

    def _search(
        self, scenario_path: Path, out: Path
    ) -> tuple[int, dict[str, Any], str]:
        code, _, err = run_cli(
            "search", "--scenario", str(scenario_path), "--out", str(out)
        )
        stats = read_artifact(out, cli.STATS_FILE)
        return code, stats, err


# ---------------------------------------------------------------------------
# Input refusals: exit 1, and no artefacts at all
# ---------------------------------------------------------------------------


class RefusalTests(ArtifactCase):
    """Exit 1 means no run happened, so no artefact may exist to suggest one."""

    def assert_refused(self, code: int, err: str, *argv: str, target: Path | None = None) -> None:
        self.assertEqual(code, cli.EXIT_INPUT_REFUSED)
        self.assertTrue(err.strip(), "a refusal must say something on stderr")
        if target is not None:
            self.assertFalse(target.exists(), f"{target} must not be created")

    def test_usage_errors_exit_one_not_two(self) -> None:
        """``argparse`` exits 2 on a usage error, which would collide with `unknown`."""
        for argv in (
            (),
            ("search", "--scenario", str(EXAMPLES / "chain-cap160.json")),
            ("frobnicate", "--out", str(self.out())),
            ("model", "--scenario", str(EXAMPLES / "chain-cap160.json")),
        ):
            with self.subTest(argv=argv):
                code, _, err = run_cli(*argv)
                self.assert_refused(code, err, *argv, target=self.out())

    def test_negative_budget_is_refused(self) -> None:
        code, _, err = run_cli(
            "search",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--out",
            str(self.out()),
            "--max-expanded-states",
            "-3",
        )
        self.assert_refused(code, err, target=self.out())

    def test_unreadable_and_malformed_scenarios(self) -> None:
        missing = self.tmp / "nope.json"
        code, _, err = run_cli("search", "--scenario", str(missing), "--out", str(self.out()))
        self.assert_refused(code, err, target=self.out())
        self.assertIn("cannot read", err)

        broken = self.tmp / "broken.json"
        broken.write_text("not json{", encoding="utf-8", newline="\n")
        code, _, err = run_cli("search", "--scenario", str(broken), "--out", str(self.out()))
        self.assert_refused(code, err, target=self.out())
        self.assertIn("not valid JSON", err)

    def test_unknown_key_is_an_input_refusal(self) -> None:
        with support.documents(
            scenario=support.example_document("scenario", "chain-cap160")
        ) as scenario_path:
            document = json.loads(scenario_path.read_text(encoding="utf-8"))
            document["bogus"] = 1
            sabotaged = write_json(scenario_path.parent / "bogus.json", document)
            code, _, err = run_cli(
                "search", "--scenario", str(sabotaged), "--out", str(self.out())
            )
        self.assert_refused(code, err, target=self.out())
        self.assertIn("invalid_input", err)

    def test_unsupported_semantics_is_not_a_generic_input_error(self) -> None:
        """DESIGN.md §8:270 keeps `unsupported` distinct from a malformed file."""
        workload = support.example_document("workload", "chain")
        workload["tensors"][0]["dtype"] = "GGML_TYPE_F16"
        with support.documents(
            workload=workload, scenario=support.example_document("scenario", "chain-cap160")
        ) as scenario_path:
            code, _, err = run_cli(
                "search", "--scenario", str(scenario_path), "--out", str(self.out())
            )
        self.assert_refused(code, err, target=self.out())
        self.assertIn("unsupported", err)

    def test_mapping_of_another_scenario_is_refused(self) -> None:
        """The mapping may be legal and simply belong elsewhere (engine §6)."""
        mapping = example_mapping()
        mapping["scenario_id"] = "some-other-scenario"
        mapping_path = write_json(self.tmp / "elsewhere.json", mapping)
        code, _, err = run_cli(
            "model",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--mapping",
            str(mapping_path),
            "--out",
            str(self.out()),
        )
        self.assert_refused(code, err, target=self.out())
        self.assertIn("identity_mismatch", err)

    def test_declared_fingerprint_that_does_not_match_is_refused(self) -> None:
        """DESIGN.md §4.3: a wrong digest must stop the replay, not warn about it."""
        found = self.out("search")
        self.search_example(EXAMPLES / "chain-cap160.json", found)
        mapping = read_artifact(found, cli.MAPPING_FILE)
        mapping["workload_fingerprint"] = "0" * 64
        tampered = write_json(self.tmp / "tampered.json", mapping)

        code, _, err = run_cli(
            "model",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--mapping",
            str(tampered),
            "--out",
            str(self.out("replay")),
        )
        self.assert_refused(code, err, target=self.out("replay"))
        self.assertIn("workload_fingerprint", err)

    def test_out_pointing_at_a_file_is_refused(self) -> None:
        blocker = self.tmp / "not-a-dir"
        blocker.write_text("", encoding="utf-8")
        code, _, err = run_cli(
            "search",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--out",
            str(blocker),
        )
        self.assertEqual(code, cli.EXIT_INPUT_REFUSED)
        self.assertIn("not a directory", err)


# ---------------------------------------------------------------------------
# Artefact hygiene
# ---------------------------------------------------------------------------


class ArtifactHygieneTests(ArtifactCase):
    """Determinism, line endings, and the assumptions block."""

    def test_reruns_are_byte_identical(self) -> None:
        """DESIGN.md §9 M0.4: a re-run from a clean directory must reproduce.

        ``stats.json`` is compared with ``search.wall_time_s`` masked, in the
        parsed structure rather than by string surgery, so a genuinely new field
        or a changed number still fails. That one field is a measured duration and
        cannot be reproducible; ``DESIGN.md`` §8:268 requires it anyway.
        """
        for name in ("a", "b"):
            code, _, err = run_cli(
                "search",
                "--scenario",
                str(EXAMPLES / "chain-cap160.json"),
                "--out",
                str(self.out(name)),
            )
            self.assertEqual(code, cli.EXIT_OK, err)

        for name in (cli.EVENTS_FILE, cli.STATES_FILE, cli.MAPPING_FILE):
            self.assertEqual(
                (self.out("a") / name).read_bytes(),
                (self.out("b") / name).read_bytes(),
                f"{name} must be reproducible byte for byte",
            )

        left = read_artifact(self.out("a"), cli.STATS_FILE)
        right = read_artifact(self.out("b"), cli.STATS_FILE)
        self.assertIn("search", left)
        left["search"]["wall_time_s"] = None
        right["search"]["wall_time_s"] = None
        self.assertEqual(left, right)

    def test_no_artifact_contains_a_carriage_return(self) -> None:
        """``.gitattributes`` says ``eol=lf``; text mode would silently write CRLF."""
        code, _, err = run_cli(
            "model",
            "--scenario",
            str(EXAMPLES / "chain-cap160.json"),
            "--mapping",
            str(EXAMPLES / "chain.mapping.json"),
            "--out",
            str(self.out("model")),
        )
        self.assertEqual(code, cli.EXIT_OK, err)
        self.search_example(EXAMPLES / "chain-cap160.json", self.out("search"))

        for directory in (self.out("model"), self.out("search")):
            for path in sorted(directory.glob("*.json")):
                raw = path.read_bytes()
                with self.subTest(path=path.name):
                    self.assertNotIn(b"\r", raw)
                    self.assertTrue(raw.endswith(b"\n"), "artefacts end with one newline")

    def test_every_artifact_identifies_its_scenario(self) -> None:
        """An artefact copied out on its own must still say what it is of."""
        self.search_example(EXAMPLES / "chain-cap160.json", self.out())
        stats = read_artifact(self.out(), cli.STATS_FILE)
        for name in (cli.EVENTS_FILE, cli.STATES_FILE, cli.MAPPING_FILE):
            document = read_artifact(self.out(), name)
            with self.subTest(name=name):
                self.assertEqual(document["schema_version"], "0.1")
                self.assertEqual(document["scenario_id"], stats["scenario_id"])
                if name == cli.MAPPING_FILE:
                    continue
                self.assertEqual(
                    document["scenario_fingerprint"], stats["scenario_fingerprint"]
                )
                self.assertEqual(
                    document["workload_fingerprint"], stats["workload_fingerprint"]
                )

    def test_assumptions_track_the_scenario(self) -> None:
        """Each caveat is emitted only while it is true, so it cannot go stale."""
        from tensor_mapping.artifacts import assumptions

        fixture = load_and_validate(EXAMPLES / "chain-cap160.json")
        exported = load_and_validate(EXAMPLES / "ggml" / "chain-cap160.json")

        fixture_notes = assumptions(fixture)
        exported_notes = assumptions(exported)
        # The two sentences differ only in their opening, and the fixture's
        # literally contains the words "real GGML export" inside "not a real GGML
        # export" -- so match on the opening, not on a substring of the claim.
        self.assertTrue(any(note.startswith("the graph is 'synthetic_fixture'") for note in fixture_notes))
        self.assertFalse(any(note.startswith("the graph is a real GGML export") for note in fixture_notes))
        self.assertTrue(any(note.startswith("the graph is a real GGML export") for note in exported_notes))
        self.assertTrue(any("hand-written" in note for note in fixture_notes))
        # The cost caveat survives either way: a real graph still has made-up
        # durations, and that is the mistake this line exists to prevent.
        for notes in (fixture_notes, exported_notes):
            self.assertTrue(any("synthetic_fixed" in note for note in notes))


class ModuleEntryPointTests(ArtifactCase):
    """One real subprocess, because that is the thing the contract spells out."""

    def test_python_dash_m_runs_and_reports_its_exit_code(self) -> None:
        """DESIGN.md §8:258 -- literally ``python -m tensor_mapping``."""
        out = self.out()
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tensor_mapping",
                "model",
                "--scenario",
                str(EXAMPLES / "chain-cap160.json"),
                "--mapping",
                str(EXAMPLES / "chain.mapping.json"),
                "--out",
                str(out),
            ],
            # The package is imported from the source tree, not installed, so the
            # working directory decides which `tensor_mapping` is found.
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("model: valid", completed.stdout)
        self.assertTrue((out / cli.STATS_FILE).is_file())

    def test_exit_code_two_reaches_the_shell(self) -> None:
        mapping = example_mapping()
        mapping["actions"] = mapping["actions"][:5]
        mapping_path = write_json(self.tmp / "short.json", mapping)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tensor_mapping",
                "model",
                "--scenario",
                str(EXAMPLES / "chain-cap160.json"),
                "--mapping",
                str(mapping_path),
                "--out",
                str(self.out()),
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, cli.EXIT_NO_USABLE_MAPPING)
        self.assertIn("incomplete_mapping", completed.stderr)


if __name__ == "__main__":
    unittest.main()
