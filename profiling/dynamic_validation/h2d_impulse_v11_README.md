# H2D first-response impulse calibration

Version 11 follows the negative result from the retained v10 experiment. The v10 fitted active recovery constant collapsed to an effectively one-shot response, and a simple monotone idle law did not reliably interpolate first-COPY means. This experiment therefore measures a sustained completed response and an explicit first-response difference without claiming an identified power, queue, scheduler or physical DMA mechanism.

The equation belongs to `modeling.llm_infer_model.performance.transfer_lifecycle`:

`T(N, bytes, idle) = b(bytes) + p(bytes, idle, preparation) + (N − 1) b(bytes)`

`b` is a continuous piecewise-linear byte curve estimated from nonfirst individual COPY responses. `p` is a signed first-response difference at exact measured idle coordinates. Future window length does not enter the first-response interface. The model refuses unmeasured idle/preparation/runtime coordinates. Byte interpolation is available for explicitly unqualified analytical evaluation; independent qualification is limited to the retained measured byte coordinates.

Both calibration and validation use the finite byte set 4 KiB, 64 KiB, 512 KiB, 1 MiB, 2 MiB, 4 MiB and 8 MiB, and the exact requested idle values 0, 2 and 10 ms. Validation uses new processes and a distinct profile/held-out numerical program; these program seeds were previously used and are not claimed to be novel. No unseen-coordinate generalization is claimed.

Each category has three fresh calibration and three fresh validation processes. The host source is observed pageable or explicitly registered. Every process retains 24 trials at each byte/idle/window condition, with windows of 1, 16 and 64 copies. All six permutations of the three window lengths recur evenly, and byte/idle group orders use distinct calibration/validation random plans. First-response means pool all future window lengths: three processes × 24 trials × three window lengths = 216 samples per byte/idle/category. Full-window means for N=16 and N=64 retain 72 samples per coordinate. N=1 contributes first-response evidence rather than a second independent window gate for the same first response.

Preparation remains the original full-graph prime followed by prescribed idle. **There is no additional COPY primer or artificial warm sequence.** Allocation, source registration, numerical verification and JSON serialization are outside the measured COPY window. The original Executor COPY implementation and completion boundary are unchanged.

The parameter estimator uses individual first and nonfirst completed COPY labels. It does not fit complete Dense/MoE times or add a correction derived from held-out window duration. Parameters, scope and every prediction are frozen before held-out collection. First-response and full-window condition means have separate 10% error targets. Raw CV and individual-process errors remain visible diagnostics and do not automatically invalidate an accurate mean prediction.

Use a new directory, then run `prepare`, `profile`, `fit`, `heldout`, `evaluate` through `python -m profiling.dynamic_validation.h2d_impulse_v11`. The exact manifest must be authorized by the coordinating agent before collection. Keep GPU/CPU benchmark windows exclusive. Failed or partial campaigns remain separate; do not modify the v10 raw data, model parameters, predictions or acceptance flags.

The native `--self-test` only exercises its metadata entry path. It does not initialize backends, perform a real COPY or validate hardware timing. Actual byte-range/payload and numerical checks run within the separately authorized native acquisition.
