# Public expert-routing input

These optional input samples demonstrate importing observed MoE expert routes.
They are not profiles, calibrated costs, or evidence of inference speed. The core
Python library and its synthetic examples do not require them.

## Source and license

Creator: **aswinkumar99**. Dataset:
[qwen3.8-flash-next-expert-traces](https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces),
revision [fee2af4ea0da212d853d82943dad3897653066a2](https://huggingface.co/datasets/aswinkumar99/qwen3.8-flash-next-expert-traces/tree/fee2af4ea0da212d853d82943dad3897653066a2).
Original file: `diagnostic/smoke-batched-8way/trace.bin`.

The redistributed trace bytes, publisher metadata, and derived route values are
provided under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/),
as declared by the source dataset. They are excluded from the repository's MIT
code license. Preserve this attribution when redistributing them. No endorsement
by the dataset creator is implied.

Changes: three bounded byte ranges were extracted without changing their bytes;
the adapter selects UID 0, one layer and a position span, preserves source expert
IDs and rounded f16 gate weights, and expresses routes in the simulator's JSON
format. Local source paths were relocated to this directory. The upstream
metadata in [public-source-metadata.json](public-source-metadata.json) is retained
unchanged. The source model is `RadixArk/Qwen3.8-Flash-Next-NVFP4`; these inputs do
not include model weights or establish numerical equivalence to that model.

## Included ranges

| File | Offset in original trace | Bytes | SHA256 |
|---|---:|---:|---|
| `public-header.bin` | 0 | 5,672 | `f4247721bcd1c714c5b858687b4634f27f36ba4c7a3718a01f28a0c0b9c6b190` |
| `public-first256.bin` | 5,672 | 2,625,536 | `c372c445118fc2213812b07c1e286e8cb0814a0222c79ad4e23d98fd8a97f8f3` |
| `public-transition33.bin` | 14,856,360 | 338,448 | `dd11d00eabd4784d0a4fa5da13e243635507f7907cb2fc394cfe31e2ad7c0e18` |

Only these ranges were acquired. The publisher's full-file SHA256 is retained as
publisher metadata and was not independently verified. UID 0 contains 180 prompt
tokens and two observed decode forwards, mapped to absolute positions 180 and
181. An observed forward is not necessarily a completed HTTP output token.

## Rebuild and use

Run from the repository root. Choose an output directory that does not already
contain artifacts:

```powershell
python profiling/moe_routes.py convert-freetoken --header profiling/data/public-routes/public-header.bin --range 5672=profiling/data/public-routes/public-first256.bin --range 14856360=profiling/data/public-routes/public-transition33.bin --request-uid 0 --request-id freetoken-uid0 --source-metadata profiling/data/public-routes/public-source-metadata.json --output mapping/outputs/public-routes/bundle.json
python profiling/moe_routes.py select --bundle mapping/outputs/public-routes/bundle.json --request-id freetoken-uid0 --source-layer 3 --m2-layer-id public-qwen3.8-layer3 --phase decode --position-start 180 --tokens 1 --information-mode causal_replay --output mapping/outputs/public-routes/decode180.fragment.json
```

Ready-to-inspect route fragments are in `profiling/cases/`:
`moe-routes-public-prefill180.fragment.json`,
`moe-routes-public-decode180.fragment.json`, and
`moe-routes-public-decode181.fragment.json`. Their raw-source paths resolve from
the repository root on Windows and POSIX systems. Their historical
`source_bundle_sha256` records the original conversion artifact, including its
then-current path metadata; that large generated bundle is not distributed.
Rebuilding with new paths changes the bundle hash while preserving selected
route records. The original record and byte hashes are retained in the fragments.

The caller still supplies layer geometry, initial KV state, hardware and explicit
costs. Decode position 180 must not be evaluated with an empty history. Causal
policies see each route only at the model's disclosure boundary. Publisher hit
masks describe an all-resident source run, not an independently evaluated cache
policy in this simulator.
