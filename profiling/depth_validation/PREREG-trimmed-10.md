# PREREG — ten timed blocks, and an estimate that drops the extremes

Written before the first cohort measured under it. It changes the estimator of the
dense column (`profiling.depth_validation`) and nothing else.

## Why

The per-process time of this column was the plain mean of sixteen timed blocks. The
blocks carry a CV of 0.02–2.5 % on the two machines measured so far, so the mean is
not noisy in the ordinary sense — but a single disturbed block (a scheduler tick, a
clock ramp, one page fault) still enters it with full weight. Dropping exactly one
block at each end removes one such disturbance on either side and leaves the body of
the distribution intact.

The change was asked for after the 2026-10-26, 2026-11-07 and 2026-11-10 cohorts had
been read; it is therefore declared here before any cohort is measured under it, and
both machines are re-measured under it. No cohort of the old estimator is mixed with
a cohort of the new one in a table.

## The rule

* Ten timed blocks per process, eight replays per block.
* The warmup is unchanged and remains enforced by the probe: twenty iterations and
  at least 500 ms pass before the first block is timed.
* The process's measured time is the mean of the eight middle block times after the
  highest and the lowest of the ten blocks are dropped.
* In the fit, one block is one observation for an action: for each program, action
  and replay, the highest and the lowest of the ten block intervals are dropped and
  the remaining eight are averaged; a price for a coordinate is the mean of those
  per-program, per-replay values. A static coordinate is shared by the eight
  replays, so their estimates are averaged once more.

## What does not change

The price key (semantic coordinate plus the declared H2D history bucket), both
providers (`context_static`, `history_conditional`), the gates (10 % on each process
mean, 20 % on the p90 block APE), the eighteen cases, the profile/held-out split, the
frozen barrier, the no-retry rule and the numeric tolerances.

## What this cannot fix

The coordinate spread between placements. On the 2026-11-07 laptop cohort one
coordinate — a 2 048-byte F32 `dram -> vram` copy — measures 20–23 µs in twelve
programs and 489.7 / 547.7 µs in the two hybrid decode programs, and the frozen table
pools them into one price. Trimming a program's own blocks cannot separate two
programs that produce the same coordinate; that is a property of the key, and it is
reported by `fit-consistency.json` rather than changed here.

## What would count as a negative result

* A case whose pass/fail differs from the plain-mean cohort on the same machine:
  both estimates are reported for that case, with the dropped blocks.
* A fit that loses a coordinate that the plain mean priced (too few blocks to trim):
  reported with the coordinate named, and the run stops rather than falling back.
