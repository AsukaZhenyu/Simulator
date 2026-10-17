# Composable Attention fixtures

These fixtures specify synthetic graph/storage semantics and independent stage
durations. They are not measured costs or a claim of calibrated performance.

`dense-decode.stages.json` and `dense-prefill.stages.json` refine the existing
`mapping/examples/m2/hybrid-interface-synthetic-{decode-1,prefill-3}.json`
invocations. Their frontend already materializes post-RoPE Q/K/V and the optional
gate, so the transform preserves projection and FFN costs.

`moe-prefill.json` is a separate explicit tiny workload. Its projection weight
`W_proj` is 192 bytes: contiguous F32 Q (4x4), K (4x2), V (4x2), and O (4x4).
`moe-prefill.stages.json` declares a QKV linear projection with adjacent-pair
normal RoPE at each absolute token position. This frontend excludes normalization
and attention gates. Both projection and output read the whole packed allocation
at their chosen device. It does not pretend the original 32-byte opaque MoE
projection already contains sufficient matrices, and does not modify that fixture.

The transform materializes QK scores and probabilities, each of shape
`(history + tokens, tokens, query_heads, 1)`, and context. QK reads Q/K and every
historical KV block; QK includes inverse-square-root scaling and softmax applies
the absolute causal mask; PV reads K/V and
history, computes context, and is the sole producer of newly appended KV blocks.
Output projection adds the residual once. Each stage requires its own exact
duration and nonnegative workspace. New transfers also require exact byte prices.

The pure Attention core variant can also be refined, retaining its existing
output stage instead of creating a second projection/residual.

Run the semantic/graph tests from the repository root:

```powershell
$env:PYTHONPATH = 'Simulator/modeling'
python -m unittest discover -s Simulator/modeling/tests -p test_attention_stages.py -v
```

The tiny numerical oracles verify the stated causal/GQA and packed-frontend
mathematics. The event runtime tracks dependency, capacity, timing, and residency;
it does not load these weight values or numerically execute the attention kernel.
