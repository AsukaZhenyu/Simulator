# Using the independently validated H2D coordinates

The retained v11 independent data have first-COPY mean absolute percentage error 2.9621% and worst coordinate error 11.9585%. Complete-window mean error is 1.5473%, with worst error 9.3926%; all 84 window coordinates meet the declared 10% condition-mean target. First-COPY results meet that target at 39 of 42 coordinates. These counts do not use raw CV as a prediction gate, and every sample remains retained.

Three first-COPY coordinates remain outside the 10% target:

| Host source | Bytes | Declared idle | Prediction | Measurement mean | Relative error |
|---|---:|---:|---:|---:|---:|
| Registered | 64 KiB | 0 ms | 30.792 us | 34.756 us | 11.4053% low |
| Registered | 64 KiB | 2 ms | 60.130 us | 53.707 us | 11.9585% high |
| Pageable | 4 KiB | 2 ms | 52.888 us | 59.551 us | 11.1886% low |

The original `INDEPENDENT_EVALUATION.json` and `validated-lifecycle-catalog.json` remain unchanged, including `first_qualified=false` for the complete grid. A separate `scoped-lifecycle-catalog.json` adds only eligibility metadata for the 39 individually validated coordinates. It does not change a price or fit a new parameter. The original modeling formula/source stays frozen; the added `ScopedTransferLifecycle` wrapper checks coordinate eligibility before delegating to that same formula.

Export after independent validation, outside another CPU/GPU measurement window:

```powershell
python -m profiling.dynamic_validation.h2d_impulse_catalog --out build/dynamic-diagnosis-20261028/h2d-v11-impulse-v1
```

Load one entry in modeling:

```python
import json
from llm_infer_model.performance.transfer_lifecycle_catalog import ScopedTransferLifecycle

catalog = json.load(open("scoped-lifecycle-catalog.json", encoding="utf-8"))
profile = next(p for p in catalog["models"] if p["memory_mode"] == "registered")
response = ScopedTransferLifecycle.from_profile(profile)
scope = dict(context_sha256=profile["context_sha256"],
             preparation_sha256=profile["preparation_sha256"])
ns = response.first_response_ns(1048576, 2000000, **scope)
block_ns = response.window_ns(1048576, 2000000, 64, **scope)
```

The adapter refuses the three failed first-COPY coordinates, unseen byte/idle coordinates, an unmeasured preparation/runtime, and unqualified window lengths. A first-COPY rejection does not revoke an independently successful complete-window coordinate. Conversely, a successful window never authorizes an inaccurate single COPY. `response.first_qualified` intentionally remains the original complete-grid flag; inspect `qualified_first_coordinates` for the finite allowed subset.

Qualification remains specific to the seven measured byte coordinates, idle values 0/2/10 ms, observed host category and original full-graph-prime context. No new device, arbitrary idle function, physical PCIe bandwidth scaling or whole-MoE timing accuracy is established by this export.
