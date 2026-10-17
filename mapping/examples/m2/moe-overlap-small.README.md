# Synthetic MoE prefetch and overlap episode

The episode runs a one-token Prefill followed by a one-token Decode on the
same declared layer. Each token selects experts 0, 1 and 2, with 64-byte
immutable SSD weights. Each expert group costs 5 ms on CPU; an SSD weight
read costs 3 ms. Projection, Attention, Router and Combine together cost
5 ms. All compute, workspace and transfer prices are synthetic.

For the first CPU call, serial execution takes 29 ms. A valid concurrent
schedule takes 23 ms: the Router completes at 4 ms, expert 0's weight arrives
at 7 ms, and its computation overlaps later experts' SSD reads. The three
reads share one SSD resource and remain serial. No timing discount is applied;
the tensor engine's actual start and end times account for the 6 ms reduction.

The second call receives the first call's actual cached weights and KV state.
With a 192-byte DRAM expert-cache quota, its three expert groups need no new
SSD read and take 20 ms. The complete CPU episode is therefore 49 ms under
serial execution and 43 ms with valid compute/transfer overlap. The cache
quota is additional to the whole-domain capacity ledger, which also counts
KV, activations, alignment, reserved copies and running workspace.

`moe-overlap-small.episode.json` fixes compute placement to the first declared
device, CPU. GPU costs are present only for separately declared synthetic
device-search variants. Expert preparation is restricted to the next macro
and two future macros; every selective copy still waits for the relevant
Router to finish. Cross-call lookahead in an offline search does not grant
permission to load an inactive expert or cross a future Router gate.

This is a functional schedule example, not measured MoE performance or an
original-model kernel reproduction. The independent F32 MoE measurement
contract is not used to price these changing schedules.
