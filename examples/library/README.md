# Standalone library examples

Install from the repository root with `python -m pip install .`, then run:

```bash
python examples/library/quickstart.py --model dense --out example-dense
python examples/library/quickstart.py --model moe --out example-moe
python examples/library/phase_and_state.py --out example-phase-state
```

The scripts can be copied outside the repository: they import the installed
library and never read fixtures or generated profile directories. Inputs are
fictional hardware, declared geometry, analytical costs, and explicitly labeled
synthetic MoE routes. These demonstrate semantics, not measured speedups.

Each output contains a persistent `project/` and detailed `result/`. Use new
directories. See the [API guide](../../docs/LIBRARY_GUIDE.md) to replace the
example model, hardware or workload with your own declarations.
