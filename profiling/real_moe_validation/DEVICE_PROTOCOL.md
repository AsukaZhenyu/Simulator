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
