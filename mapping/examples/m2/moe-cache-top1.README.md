# Synthetic MoE cache sequence

This episode exercises five calls on one declared MoE layer: one-token Prefill,
then four one-token Decode calls. Top-k is one and the fixed expert accesses
are `0, 1, 2, 0, 1`. Each expert weighs 64 bytes. The expert cache subbudget is
128 bytes in each memory domain; physical DRAM and VRAM capacities are 4096
bytes and still include all tensors, persistent KV, alignment and scratch.

All execution and transfer prices are explicitly synthetic. CPU calls each
cost 10,000,000 ns without an expert load; one SSD-to-DRAM expert load costs
3,000,000 ns. Under a two-expert DRAM cache, LRU loads five experts and a
future-informed eviction policy loads four, giving 65,000,000 ns and
62,000,000 ns respectively. These values demonstrate cache semantics; they
are not measurements or model performance claims.

The first invocation declares the actual initial state. Later invocation files
declare valid baseline history templates. During an episode, each subsequent
call must receive the preceding candidate's actual `state_out` instead: an
unbounded baseline's cached weights are never substituted for a constrained
policy's cache. KV positions advance from zero to five, request versions from
one to six, and weight versions remain one. Every immutable SSD source survives
DRAM or VRAM cache eviction.

`moe-cache-top1.episode.json` compares the explicitly named baselines and a
bounded offline cache search. Both CPU and GPU choices are declared in the
individual invocations; this episode fixes every macro to the first declared
device, CPU, so the cache comparison uses the same compute placement. The
CLI requires candidate and expansion budgets explicitly.
