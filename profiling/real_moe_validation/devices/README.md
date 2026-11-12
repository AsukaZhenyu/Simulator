# Device profiles for `tools/run_device_ensemble.py`

One file per machine and model. A profile is an adapter: it names this machine's
interpreter, its instrument build, its model file, and the device state it is
allowed to be measured in. The procedure the driver runs is the same everywhere
(`DEVICE_PROTOCOL.md` section 6); nothing in a profile may change it.

## Bringing up a new machine

```bash
python tools/run_device_ensemble.py --profile <profile>.json --dry-run
python tools/run_device_ensemble.py --profile <profile>.json --precondition-only
python tools/run_device_ensemble.py --profile <profile>.json
```

1. **`--dry-run`** resolves every path and prints the exact campaign command.
   It refuses placeholder values (`REPLACE_ME`) and a profile written for a
   different platform or device, so a copy carried to the wrong machine fails by
   name instead of resolving `~` to the wrong home directory.
2. **`--precondition-only`** runs one discarded campaign and writes
   `build/ensemble-<device>-<stamp>/precondition/device-state.jsonl`. That trace
   is what the band and the tolerances are set from: it contains timings and
   temperatures, never an accuracy number, so calibrating to it does not
   calibrate to the result.
3. **The full run** precondition, then N counted cohorts, then the ensemble.

## What to fill in

- `campaign.python`, `campaign.runner` — relative to the repository root.
- `campaign.arguments` — the frozen campaign's own arguments; only the
  interpreter, the instrument build and the model differ between machines.
- `state.band` — a closed interval per field. The laptop gates temperature
  only: its clocks legitimately swing by an order of magnitude between idle and
  loaded, so a clock interval would describe the workload rather than the state.
- `state.stability` — how far a field may move *within* a cohort; `*_fraction`
  names a field judged against its own median.
- `state.lock` — clock pinning, off unless the machine allows it. A lock is
  reported only after a read back shows the requested clock.
- `ensemble.state_boundary_ms` — the machine's first-forward band boundary,
  frozen from its first cohort and never moved (protocol section 3.4). Leave it
  `null` on a machine that has not run one yet; the driver declares it from the
  first ensemble and writes the rule it used into `ENSEMBLE.json`.

## Files

| profile | machine | model |
|---|---|---|
| `laptop-4070-qwen.json` | RTX 4070 Laptop, Windows | Qwen3-0.6B Q4_K_M |
| `server-5090-qwen.json` | RTX 5090, Linux | Qwen3-0.6B Q4_K_M |
| `server-5090-olmo.json` | RTX 5090, Linux | OLMoE-1B-7B Q4_K_M |
| `desktop-template.json` | desktop | fill in before use |
