# Expanded analytical strategy comparison results

All results are uncalibrated analytical simulation, not real-device speedups. No performance measurement, fit or modeling mechanism changed.

Completed 352 of 352 declared runs, with 0 failures and 0 missing cells. Sixteen scenario IDs contain 14 distinct input behaviors plus 2 equivalence controls. All 48 matching control comparisons agree.

| Scenario | Best completed candidates, exact ties retained | Simulated ms | Reference speedup |
|---|---|---:|---:|
| dense-decode_heavy | e1-s0, e1-s1, e1-s2, e1-s3 | 0.224106 | 3.550191 |
| dense-long_prefill | e3-s0, e3-s2 | 0.255528 | 1.397240 |
| dense-mixed | e2-s0, e2-s2 | 0.532891 | 1.772058 |
| dense-short_prefill | e1-s0, e1-s1, e1-s2, e1-s3 | 0.029270 | 3.219884 |
| moe-decode_heavy-hot | e1-s0, e1-s1, e1-s2, e1-s3 | 0.395434 | 2.921706 |
| moe-decode_heavy-phase_shift | e1-s0, e1-s1, e1-s2, e1-s3 | 0.395434 | 2.921706 |
| moe-decode_heavy-rotating | e1-s0, e1-s1, e1-s2, e1-s3 | 0.426930 | 2.921894 |
| moe-long_prefill-hot | e3-s0, e3-s2 | 0.308428 | 1.235232 |
| moe-long_prefill-phase_shift | e3-s0, e3-s2 | 0.308428 | 1.235232 |
| moe-long_prefill-rotating | e3-s0, e3-s2 | 0.397564 | 1.153797 |
| moe-mixed-hot | e1-s0, e1-s1, e1-s2, e1-s3 | 0.931440 | 1.333313 |
| moe-mixed-phase_shift | e1-s0, e1-s1, e1-s2, e1-s3 | 0.931440 | 1.333313 |
| moe-mixed-rotating | e1-s0, e1-s1, e1-s2, e1-s3 | 0.933440 | 1.429742 |
| moe-short_prefill-hot | e1-s0, e1-s1, e1-s2, e1-s3 | 0.053042 | 2.459560 |
| moe-short_prefill-phase_shift | e1-s0, e1-s1, e1-s2, e1-s3 | 0.053042 | 2.459560 |
| moe-short_prefill-rotating | e1-s0, e1-s1, e1-s2, e1-s3 | 0.084538 | 2.632686 |

## Main comparisons excluding equivalence controls

Prefetch: 152 complete pairs, 30 faster, 32 tied and 90 slower. Cache-family change: 0 faster, 149 tied and 3 slower out of 152 complete pairs.
Strict execution-order reversal appears in 1 distinct input scenarios; 4 overlapping four-cell comparisons reverse. These are not independent statistical samples.
The predefined one-shot independent selector has positive finite-catalogue regret on 0 of 14 complete inputs; maximum 0.000000%.

## Important contract-dependent mechanism

Dense Prefill128: e0-s0 takes 357034 ns; e3-s0 takes 255528 ns. The all-GPU case performs 256 token-granular KV copies totaling 65536 B and 264192 ns. Of that, 256000 ns is the declared per-copy fixed latency. CPU Attention avoids those final-DRAM copies. This is a granularity/terminal-goal effect, not evidence that CPU Attention is physically faster.

## Evidence and reproduction

- Every candidate retains generated prices, compiled IR, exact events, final catalogues, decisions and policy audit; per-case file hashes are indexed.
- Execution source unchanged: 188d4456abf72c7efb9441e9ff783db8a73f4eb68c12f63b7ccde9db9cd80b00; protocol: 3ecaf16e876ee5ab0b04688ccdb2237482fa8cc815adb20bb2975e46e279c211; 9 frozen accuracy files unchanged.
- Enumeration covers this declared candidate matrix only. No numerical generation, new search method, multi-GPU, physical UMA or real-device accuracy claim follows.
- H64 geometry, zero declared scratch and token-granular transfer costs limit extrapolation. Near ties lack a performance-error confidence bound.
- Canonical campaign: `build/strategy-comparison-20261023/full-v1` under the repository root of the
  machine that ran it. Campaign outputs are regenerable and are not redistributed with this
  repository; rerun the campaign to obtain them.
- Readable Chinese report: produced outside this repository and not redistributed.
- Reproduce using experiments/strategy_comparison_campaign.py and the companion protocol document.

## Representative resource costs

Each pair uses the same input and terminal goals. The lowest numeric candidate ID represents exact latency ties; all tied candidates remain in the data.

| Scenario / candidate | Simulated ns | H2D B | D2H B | Peak DRAM B | Peak VRAM B | Weight reload B | Compute/COPY overlap ns |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense-long_prefill / e0-s0 | 357034 | 428032 | 98304 | 492544 | 353024 | 0 | 36626 |
| dense-long_prefill / e3-s0 | 255528 | 460800 | 196608 | 591360 | 287744 | 0 | 0 |
| dense-decode_heavy / e0-s0 | 795619 | 3552384 | 12288 | 402688 | 203040 | 3153920 | 30685 |
| dense-decode_heavy / e1-s0 | 224106 | 0 | 0 | 408672 | 0 | 0 | 0 |
| moe-decode_heavy-rotating / e0-s0 | 1247444 | 6805504 | 12288 | 1288448 | 206848 | 5521408 | 32126 |
| moe-decode_heavy-rotating / e1-s0 | 426930 | 0 | 0 | 1296448 | 0 | 0 | 0 |
| moe-mixed-phase_shift / e0-s0 | 1241901 | 6246400 | 104448 | 1349888 | 410624 | 4931584 | 256203 |
| moe-mixed-phase_shift / e1-s0 | 931440 | 0 | 0 | 1543168 | 0 | 0 | 0 |

## Adaptive expert rules and unexercised mechanisms

- e4 versus fixed CPU-expert e2, excluding equivalence controls: 40 of 40 pairs have identical latency; 40 have identical event hashes. This is not evidence of adaptive-policy benefit or universal policy equivalence.
- e5 versus fixed CPU-expert e2, excluding equivalence controls: 40 of 40 pairs have identical latency; 40 have identical event hashes. This is not evidence of adaptive-policy benefit or universal policy equivalence.
- CPU/GPU compute overlap is zero in every run. This catalogue exercises placement and compute/transfer overlap, not concurrent tensor partitioning.
- SSD traffic is zero in every run; no SSD-offload benefit is evaluated.
- An independent stdlib-only audit replays all 352 cases, including exact costs, dependencies, allocation peaks and tensor goals. It checks internal consistency, not hardware prediction accuracy.
