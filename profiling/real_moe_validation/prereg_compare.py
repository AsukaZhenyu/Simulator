"""Compare the control and treatment arms of one model column, condition by condition.

    python -m profiling.real_moe_validation.prereg_compare <control/model/formal> <treatment/model/formal>

G2 and G3 of PREREG-decode-depth-windows-v2.md are decided here; G1, G4 and G5 are
read by prereg_check.py from each arm's own artifacts, and G3a (prefill is not
reachable by the rule) by replaying both rules on one cohort with
diag_history_split. Only published numbers are copied, never recomputed from
timings.

The v2 G3 is a decode claim: the treatment's decode mean and decode worst are
both lower than the matched control's. Prefill is printed beside it and is not
a test, because the two rules produce identical prefill keys; v1's stronger
form (no phase mean may rise) is still reported as ``g3_v1_all_phases`` so the
change of form is visible in the output rather than in prose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PHASES = ("total", "prefill", "decode_with_sampling")


def load(cohort):
    return json.loads((Path(cohort) / "CONDITION_MEAN_EVALUATION.json").read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    args = parser.parse_args(argv)
    control, treatment = load(args.control), load(args.treatment)
    left = {(c["placement"], c["request_id"]): c for c in control["conditions"]}
    right = {(c["placement"], c["request_id"]): c for c in treatment["conditions"]}

    failures = []
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            failures.append("%s present in only one arm" % (name,))
            continue
        if a["status"] != "evaluated" or b["status"] != "evaluated":
            if a["status"] != b["status"]:
                failures.append("%s status %s -> %s" % (name, a["status"], b["status"]))
            continue
        change = ["%s %+6.2f pts" % (phase, 100 * (b["APE"][phase] - a["APE"][phase])) for phase in PHASES]
        print("%-22s %-26s  %s  %s -> %s" % (
            name[0], name[1], "  ".join(change),
            "PASS" if a["passed"] else "FAIL", "PASS" if b["passed"] else "FAIL"))
        if not b["passed"]:
            failures.append("%s fails G2 at %s" % (name, {p: round(100 * b["APE"][p], 2) for p in PHASES}))

    rows = []
    for phase in PHASES:
        c_mean, t_mean = control["mean_APE"][phase], treatment["mean_APE"][phase]
        c_worst, t_worst = control["worst_APE"][phase], treatment["worst_APE"][phase]
        worsens = t_mean > c_mean
        rows.append(dict(phase=phase, control_mean=round(1000 * c_mean, 2),
                         treatment_mean=round(1000 * t_mean, 2),
                         control_worst=round(1000 * c_worst, 2),
                         treatment_worst=round(1000 * t_worst, 2),
                         mean_worsens_G3=worsens))
        if worsens:
            failures.append("G3(v1 form): %s mean rises %.3f %% -> %.3f %%" % (phase, 100 * c_mean, 100 * t_mean))
    decode = next(r for r in rows if r["phase"] == "decode_with_sampling")
    decode_claim = (treatment["mean_APE"]["decode_with_sampling"] < control["mean_APE"]["decode_with_sampling"]
                    and treatment["worst_APE"]["decode_with_sampling"] < control["worst_APE"]["decode_with_sampling"])
    if not decode_claim:
        failures.append("G3: decode mean/worst not both lower (%.3f/%.3f -> %.3f/%.3f %%)" % (
            100 * control["mean_APE"]["decode_with_sampling"], 100 * control["worst_APE"]["decode_with_sampling"],
            100 * treatment["mean_APE"]["decode_with_sampling"], 100 * treatment["worst_APE"]["decode_with_sampling"]))
    result = dict(control=str(args.control), treatment=str(args.treatment),
                  control_passed="%d/%d" % (control["passed_conditions"], control["complete_endpoint_conditions"]),
                  treatment_passed="%d/%d" % (treatment["passed_conditions"], treatment["complete_endpoint_conditions"]),
                  per_mille=rows, g3_decode_claim=decode_claim,
                  g3_v1_all_phases=not any(r["mean_worsens_G3"] for r in rows),
                  failures=failures,
                  g2_and_g3_hold=not failures)
    print(json.dumps(result, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
