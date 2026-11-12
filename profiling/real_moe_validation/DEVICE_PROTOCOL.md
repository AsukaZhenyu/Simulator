# Frozen device protocol for the model column

Every machine that contributes a number to the paper runs this procedure. The
procedure and the code it names are frozen; a machine adapts the toolchain, not
the method. A number produced any other way is not a paper number.

## 1. What is frozen

- **Code.** `origin/main` at the commit that also contains this file. A machine
  records `git rev-parse HEAD` beside every cohort; a cohort whose tree differs
  from that commit is not comparable.
- **Instrument.** The observer driver is composed from the repository alone:
  `tap_native/real_moe_tap_probe.cpp` -> `tap_v5.native_v5()` ->
  `runtime_windows.patched_native()`. Its sha256 is pinned by
  `test_runtime_windows.test_the_observer_driver_comes_from_the_repository_alone`
  and must be `54dceda5221e2dd74ab8b9d870e63c184b7a9caec12933ec4ae35e812f636f32`
  unless a new revision is registered and every machine moves together. The
  compiler/link lines are the recorded fixtures in
  `runtime_windows_native/build_records/`; a machine supplies its own toolchain
  and SDK paths, nothing else.
- **Requests.** Calibration and held-out requests come from the frozen suite and
  are selected by domain/length stratum, never by any latency. `--heldout-index`
  names the stratum; the published laptop and desktop rows use index 1.

## 2. What a machine may change

Only the adapters: `platform.hpp`, `multimodel/devices.py`, `telemetry.py`,
`host.py`, `preflight.py`, `ssd_measure.py`, the `run_*` collectors, and the
toolchain/runtime directory arguments. No platform may introduce a branch in
`runtime_window_analysis.py`, `window_work.py`, `modeling/llm_infer_model/` or
`mapping/`. If a device needs a change inside the frozen core, that is a new
registered revision for all machines, not a local edit.

## 3. Procedure, identical on every machine

1. **Green first.** `python tools/run_tests.py` passes, and the instrument's
   `--self-test` passes with its recorded checks.
2. **Campaign, N independent repetitions.** The same command N times, each into
   a fresh root, one native process per job, machine quiet, nothing else
   measuring:

   ```text
   python tools/run_accuracy_campaign.py --columns model \
     --root build/accuracy-<device>-<date>-rep<NN> --heldout-index 1 \
     --model-id <registry id> --model <gguf> --runtime-build <instrument build>
   ```

   Linux machines use their own interpreter and `--runtime-build`; nothing else
   differs. N = 10 unless the device protocol states otherwise.
3. **Per-run record.** Cohort mean and worst total APE, all 12 condition phase
   APEs, and for every bare held-out request the prefill first-forward duration
   from `forward_inventory[0]` (ticks are 100 ns).
4. **State declaration.** The first-forward durations are bimodal on some
   machines, and the mode is not a function of the request. Each machine
   freezes its band boundary once, from its first cohort's sorted durations, at
   the largest ratio gap between adjacent values; every later run on that
   machine uses that boundary. Both the occupancy and the durations are
   reported per run.
5. **The paper number.** The ensemble: the mean of the N cohort means, its
   between-run standard deviation, the per-run pass count, and the state
   occupancy. A single run is reported only as one of the N.

## 4. Refusals

A cohort is excluded, and its root kept as the refusal record, if preflight
fails, the software-test phase is not green, any job reports a status other
than completed, or the evaluation reads fewer than 12 conditions. The excluded
count and the reason are reported with the ensemble.

## 5. Why the repetitions are part of the method

On the laptop, two runs of the identical configuration gave cohort means of
11.57% and 3.83% (6/12 and 8/12 conditions). The cause is a per-request
first-forward state of about 2x on the hybrid placement's runtime bridge, seen
in 18 of 60 bare held-out requests, absent from the request, the token count,
the observer mode and the graph decisions, and not predictable from the
preceding request. The whole GPU forward of a model without routed experts
lands in one bridge window, so the state cannot be averaged away inside a
single run; only repetitions average it out, and only the occupancy explains
how far a single run can move.

## 6. Revision 2: the device state is an input, and the driver holds it

Repetitions measure the spread of an uncontrolled state; they do not control
it. Revision 2 adds the control, and with it a new entry point. A cohort run by
`tools/run_accuracy_campaign.py` on its own is one cohort of an ensemble, not a
paper number.

1. **Precondition.** One discarded campaign, run while the device is read on a
   cadence (in-process NVML, one reading every 2 s; no extra process). The
   machine is settled when, over the last 60 s, the temperature moved at most
   1.5 °C and the SM clock at most the profile's fraction. A machine that does
   not settle before the campaign ends is refused, and its trace is kept: the
   workload is the campaign's own, so a burn that settles a different thermal
   path would warm the wrong machine.
2. **Band.** Every reading of a counted cohort must be inside the declared
   band, and the state must not drift outside its declared stability limits
   within a cohort. A cohort that entered and left the band is excluded exactly
   as one that was never in it. The band and both tolerances are set from the
   precondition trace, which contains no accuracy number; they are calibrated to
   the machine, never to the result.
3. **Lock, where the machine allows it.** `nvidia-smi` clock pinning is
   attempted only when the profile asks for it, and is reported as held only
   after a read back shows the requested clock. A refused lock is recorded with
   the tool's own words and the cohort falls back to the band.
4. **Exclusions.** N counted cohorts, at most N+3 attempts. Every attempt is
   reported with its status and, for an excluded one, the reading that refused
   it; the excluded root stays on disk.
5. **The number.** The mean of the N cohort means with its between-run standard
   deviation, the per-run pass counts, the pooled per-condition phase APEs, and
   the state occupancy at the boundary frozen for that machine (section 3.4).
6. **The driver.** `tools/run_device_ensemble.py --profile <profile>`; one
   profile per device and model under `profiling/real_moe_validation/devices/`.
   A new machine runs `--dry-run` first and then `--precondition-only`, which
   measures one discarded campaign and reports the state trace it would gate on;
   the profile's band is set from that trace and not from any campaign result.

What revision 2 does not control, and therefore does not claim: contention from
other processes on the same machine, CPU frequency scaling, and which core a
thread lands on. Those are refused or recorded only where a profile says so.

## 7. Revision 3: the state is judged by drift, not by jitter

Revision 2's predicate reads the last 60 s of the discarded campaign. On both
machines that campaign is bursty by construction: the accelerator works for a
few seconds at a time while the host does its part, heats during a burst and
cools between bursts. Neither machine passed the predicate:

| machine | last 60 s temperature span | whole-campaign block means |
|---|---|---|
| RTX 5090 | 5.0 °C from cold, 4.0 °C warm | 1.13 °C |
| RTX 4070 Laptop | 12.0 °C from cold, 10.0 °C warm | 0.84 °C |

The block-mean column is the warm-start trace, which is the one both machines
kept as their refusal record.

The extremes describe the workload; the means describe the machine. A price is
a mean over the campaign, so a gate on the extremes refuses machines whose mean
state never moved, and a gate on the means measures the quantity the price is
sensitive to. Revision 3 changes the estimator and nothing else:

1. **Plateau.** The analysed span (the whole discarded campaign, `window_s:
   null`) is split into `blocks` equal spans of time; the machine is settled when
   the largest difference between the block means of the temperature is at most
   the declared tolerance. The same estimator judges a cohort's stability under
   section 6, item 2. Every verdict reports the drift it measured.
2. **No clock bound without a lock.** With clocks unlocked the SM clock follows
   the kernel mix the campaign happens to schedule — 180 to 2685 MHz inside one
   campaign on the same machine — so a bound on it measures the workload rather
   than the state. A profile that does not lock clocks declares `clock_fraction:
   null`; the clock is then recorded and reported in every cohort summary, not
   gated. A profile that locks keeps the bound, and the lock's read back is what
   fixes the clock.
3. **No tolerance was retuned.** The declared tolerances (1.5 °C plateau, 4 °C
   and 6 °C cohort stability) are the ones revision 2 declared before either
   trace existed; the estimator changed under them, and every verdict prints the
   drift it measured so a reader can check it against the trace.

What revision 3 still does not control: the machine's mean state can differ
between sessions (the band, and the ensemble's between-run standard deviation,
are the record of that), and a cohort whose mean holds while its workload
changes shape is admitted — the accuracy dispersion, not the state gate, is what
would show it.
