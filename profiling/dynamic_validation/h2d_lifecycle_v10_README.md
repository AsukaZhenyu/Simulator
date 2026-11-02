# H2D lifecycle calibration v10

This experiment keeps the existing inclusive `Executor.action(COPY)` boundary and the existing response equations in `modeling.performance.transfer_recovery`. It adds independent transfer-window lengths so the first response and sustained completed responses can be assessed separately. It does not modify the v9 data or its failed support flags.

The source is a separately owned 8 MiB CPU arena; the target is a separately owned 8 MiB GPU arena. The source bytes repeat the original profile or held-out program's independently generated tensor values. Range poisoning verifies every declared view and checks that the target outside the view remains unchanged. The original complete Dense graph runs before each transfer window; its original arena remains pageable. Only the separate source arena is explicitly registered in the registered mode. Registration, allocation, full preparation, numerical checks and JSON serialization are outside the timed COPY window.

| Coordinate | Profiling | Independent validation |
|---|---|---|
| Bytes | 1 KiB, 128 KiB, 1 MiB, 8 MiB | 4 KiB, 64 KiB, 512 KiB, 2 MiB, 4 MiB |
| Requested idle | 0, 1, 5, 20 ms | 2, 10 ms |
| Completed copies per window | 1, 16, 128 | 8, 64 |
| Host memory | Observed pageable, explicitly registered | Same two categories |
| Fresh processes | Three per category | Three per category |
| Retained trials per condition/process | 12 | 12 |

The fitting code compares a static affine response, zero-initial-state recovery and preparation-conditioned recovery using leave-one-profiling-process-out error. It prefers fewer parameters when validation scores differ by at most one percentage point. Parameters and all independent predictions are frozen before held-out collection. The response-state constants are empirical completed-response parameters; they do not identify a unique physical clock, queue, operating-system or cache mechanism. Fitting uses first-COPY and summed COPY responses, while final window validation includes outer-loop gaps without adding an observed correction.

Primary acceptance evaluates **both** first-COPY condition means and complete-window condition means at a 10% relative-error target. A condition mean retains all 36 trials over three fresh processes. The two qualification results are reported separately. Individual process errors and raw CV remain visible as additional diagnostics; high measured CV is not by itself a prediction-accuracy failure. Passing a window condition does not authorize an inaccurate first-COPY price.

The frozen scope binds the runtime DLLs, collector binary, preparation semantics, memory category and arena boundary. An independently successful response applies to that stated scope; it does not silently replace arbitrary LLM COPY contexts or identify physical PCIe bandwidth scaling. The mathematical formulas remain in modeling; this module measures and estimates their parameters.

Run from the `Simulator` directory with `PYTHONPATH` containing `modeling` and the project root:

```powershell
python -m profiling.dynamic_validation.h2d_lifecycle_v10 prepare --out NEW_DIRECTORY --binary COMPILED_EXE --build-receipt BUILD_RECEIPT_JSON
# The coordinating agent authorizes the exact manifest SHA in authorization.json.
python -m profiling.dynamic_validation.h2d_lifecycle_v10 profile --out NEW_DIRECTORY
python -m profiling.dynamic_validation.h2d_lifecycle_v10 fit --out NEW_DIRECTORY
python -m profiling.dynamic_validation.h2d_lifecycle_v10 heldout --out NEW_DIRECTORY
python -m profiling.dynamic_validation.h2d_lifecycle_v10 evaluate --out NEW_DIRECTORY
```

CPU/GPU measurements must run without another experiment or compilation competing for resources. A failed or partial acquisition is preserved in its own directory rather than overwritten or silently retried.
