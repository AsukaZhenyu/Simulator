"""注意力与 FFN 的族语义。

``llm_infer_model.components`` 存在的理由是把「这层是什么」和「展开怎么做」分开。
这些测试钉四件事：

* **词汇表只有一份**：注册表的键必须**逐字等于** ``m2.spec`` 的
  ``ATTENTION_KINDS`` / ``FFN_KINDS``。少一个键，那种类型就永远查不到；多一个键，
  就是格式里凭空多了一种没人会解析的类型。
* **未注册就拒绝**：绝不静默退回 ``full_causal`` / ``dense``——静默退回会让一个
  SWA 层被当成全局因果算完，而所有断言照样通过。
* **三种注意力真的不是一个东西**：SWA 的执行峰值大于最终占用，递归状态与上下文
  长度无关，全局因果两者相等。这三条是 ``M2_LAYER_DESIGN.md`` §4 三节各自的
  要点，所以各有一条断言。
* **MoE 的揭示门是结构**：拿不到本轮路由结果时 ``plan`` 直接拒绝，而不是拿均匀
  分布顶替（§5）。
* **路由是确定性的**：随机 top-k 只是 router 的替身，同一份负载必须每次路由到同一
  份结果，否则成本表与回放全对不上。

最后两条子进程测试钉两件事：**依赖方向**（``components`` 可以单独 import，不需要
先 import ``m2``——反向 import 一旦写到模块级，「先 import 谁」就会决定成败），以及
**种子跨进程稳定**（用两个不同的 ``PYTHONHASHSEED`` 跑同一段路由，结果必须逐字节
相同——这条是 ``hash()`` 的唯一克星，单进程里两种写法都绿）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from llm_infer_model.components import (
    ATTENTION_MODELS,
    FFN_MODELS,
    AttentionState,
    DenseFFN,
    FFNState,
    FullCausal,
    MoEFFN,
    RandomTopK,
    RecurrentLinear,
    SlidingWindow,
    attention_model,
    ffn_model,
)
from llm_infer_model.m2.spec import ATTENTION_KINDS, FFN_KINDS
from llm_infer_model.tensor import InvalidInput, Unsupported

MODELING_ROOT = Path(__file__).resolve().parents[1]

# Qwen3.5-4B 第 3 层（全注意力）的真实几何：kv = 2(K,V) × 4(kv_heads) × 256(head_dim)
# × 2 B(fp16)。三个测试用同一个数，是为了让「哪个族在增长」一眼可比。
LAYER3_KV_BYTES_PER_TOKEN = 4096
LAYER3_FFN_BYTES = 45_895_680


class RegistryTests(unittest.TestCase):
    """词汇表只有一份，且每一种都真的能造出来。"""

    def test_attention_registry_matches_the_format_vocabulary(self) -> None:
        self.assertEqual(sorted(ATTENTION_MODELS), sorted(ATTENTION_KINDS))

    def test_ffn_registry_matches_the_format_vocabulary(self) -> None:
        self.assertEqual(sorted(FFN_MODELS), sorted(FFN_KINDS))

    def test_every_declared_attention_kind_is_constructible(self) -> None:
        params = {
            "full_causal": {},
            "sliding_window": {"window": 4},
            "recurrent_linear": {"variant": "gated_delta", "components": [{"name": "s", "bytes": 8}]},
        }
        for kind in ATTENTION_KINDS:
            with self.subTest(kind=kind):
                model = attention_model(kind, kv_bytes_per_token=64, params=params[kind])
                self.assertEqual(model.kind, kind)

    def test_every_declared_ffn_kind_is_constructible(self) -> None:
        dense = ffn_model("dense", weight_bytes=64)
        self.assertEqual(dense.kind, "dense")
        moe = ffn_model("moe", params={"experts": 4, "top_k": 2, "expert_weight_bytes": 8})
        self.assertEqual(moe.kind, "moe")

    def test_unknown_attention_kind_is_refused_not_fallen_back(self) -> None:
        with self.assertRaises(Unsupported) as caught:
            attention_model("linear_attention", kv_bytes_per_token=64)
        self.assertIn("未注册", str(caught.exception))

    def test_unknown_ffn_kind_is_refused_not_fallen_back(self) -> None:
        with self.assertRaises(Unsupported) as caught:
            ffn_model("mixture_of_depths", params={})
        self.assertIn("未注册", str(caught.exception))

    def test_a_param_from_another_kind_is_a_document_error(self) -> None:
        # full_causal 写 window 是笔误，不是「被忽略的额外信息」。
        with self.assertRaises(InvalidInput):
            attention_model("full_causal", kv_bytes_per_token=64, params={"window": 4})

    def test_swa_without_a_window_is_refused(self) -> None:
        with self.assertRaises(InvalidInput):
            attention_model("sliding_window", kv_bytes_per_token=64, params={})

    def test_a_window_of_zero_is_refused(self) -> None:
        with self.assertRaises(InvalidInput):
            SlidingWindow(kv_bytes_per_token=64, window=0)


class FullCausalStateTests(unittest.TestCase):
    """§4 第 1 条：位置 p 只能读 <= p；总量随 token 增长。"""

    def test_reads_and_keeps_the_whole_history(self) -> None:
        state = attention_model("full_causal", kv_bytes_per_token=LAYER3_KV_BYTES_PER_TOKEN)
        got = state.state(history=3, tokens=1)
        self.assertEqual(got.read_positions, 4)
        self.assertEqual(got.kept_positions, 4)
        self.assertEqual(got.state_bytes, 4 * LAYER3_KV_BYTES_PER_TOKEN)

    def test_execution_peak_equals_the_final_state_because_blocks_are_not_recopied(self) -> None:
        state = FullCausal(kv_bytes_per_token=LAYER3_KV_BYTES_PER_TOKEN)
        got = state.state(history=11, tokens=5)
        self.assertEqual(got.peak_state_bytes, got.state_bytes)

    def test_prefill_from_empty_history_is_just_the_chunk(self) -> None:
        got = FullCausal(kv_bytes_per_token=LAYER3_KV_BYTES_PER_TOKEN).state(history=0, tokens=3)
        self.assertEqual(got.kept_positions, 3)
        self.assertEqual(got.state_bytes, 3 * LAYER3_KV_BYTES_PER_TOKEN)

    def test_negative_history_is_refused(self) -> None:
        with self.assertRaises(InvalidInput):
            FullCausal(kv_bytes_per_token=64).state(history=-1, tokens=1)


class SlidingWindowStateTests(unittest.TestCase):
    """§4 第 2 条：窗口包含当前 token；执行峰值不等于最终占用。"""

    def test_the_window_includes_the_current_token(self) -> None:
        # W=1 表示只看当前 token：一个位置，不是零个。
        got = SlidingWindow(kv_bytes_per_token=64, window=1).state(history=9, tokens=1)
        self.assertEqual(got.kept_positions, 1)

    def test_history_longer_than_the_window_stops_growing(self) -> None:
        short = SlidingWindow(kv_bytes_per_token=4096, window=8).state(history=7, tokens=1)
        long = SlidingWindow(kv_bytes_per_token=4096, window=8).state(history=1000, tokens=1)
        self.assertEqual(short.state_bytes, long.state_bytes)
        self.assertEqual(long.state_bytes, 8 * 4096)

    def test_execution_peak_exceeds_the_final_state_when_the_window_slides(self) -> None:
        # §4 第 2 条明说不许拿最终 W × KV/token 冒充执行峰值：旧块与新 chunk 同时在
        # 场。W=4、历史 8、本 chunk 3 个 token → 峰值 7 个位置，最终只有 4 个。
        got = SlidingWindow(kv_bytes_per_token=4096, window=4).state(history=8, tokens=3)
        self.assertEqual(got.kept_positions, 4)
        self.assertEqual(got.peak_state_bytes, 7 * 4096)
        self.assertGreater(got.peak_state_bytes, got.state_bytes)

    def test_prefill_reads_positions_that_are_evicted_before_the_chunk_finishes(self) -> None:
        # Queries 8,9,10 see [5,8], [6,9], [7,10]: union [5,10].
        got = SlidingWindow(kv_bytes_per_token=64, window=4).state(history=8, tokens=3)
        self.assertEqual(got.read_positions, 6)
        self.assertEqual(got.kept_positions, 4)

    def test_chunk_read_union_matches_explicit_causal_windows(self) -> None:
        # Compare against position sets, including W=1, long chunks and empty calls.
        for window in (1, 2, 4):
            for history in (0, 1, 3, 8):
                for tokens in (0, 1, 3, 7):
                    with self.subTest(window=window, history=history, tokens=tokens):
                        positions = set()
                        for query in range(history, history + tokens):
                            positions.update(range(max(0, query-window+1), query+1))
                        got = SlidingWindow(kv_bytes_per_token=64, window=window).state(
                            history=history, tokens=tokens
                        )
                        self.assertEqual(got.read_positions, len(positions))
                        self.assertEqual(got.kept_positions, min(window, history + tokens))

    def test_a_prefill_chunk_that_fits_the_window_has_no_extra_peak(self) -> None:
        got = SlidingWindow(kv_bytes_per_token=4096, window=4).state(history=0, tokens=3)
        self.assertEqual(got.kept_positions, 3)
        self.assertEqual(got.peak_state_bytes, got.state_bytes)


class RecurrentLinearStateTests(unittest.TestCase):
    """§4 第 3 条：固定大小状态，不生成随上下文增长的 KV。"""

    COMPONENTS = (("conv_state", 98_304), ("recurrent_state", 1_048_576))

    def _model(self) -> RecurrentLinear:
        return RecurrentLinear(variant="gated_delta", components=self.COMPONENTS)

    def test_the_state_is_constant_in_the_context_length(self) -> None:
        short = self._model().state(history=0, tokens=1)
        long = self._model().state(history=131_072, tokens=1)
        self.assertEqual(short.state_bytes, long.state_bytes)
        self.assertEqual(long.state_bytes, 98_304 + 1_048_576)

    def test_there_are_no_positions_to_read_or_keep(self) -> None:
        got = self._model().state(history=500, tokens=4)
        self.assertEqual(got.read_positions, 0)
        self.assertEqual(got.kept_positions, 0)

    def test_the_update_peak_double_counts_the_old_and_new_state(self) -> None:
        got = self._model().state(history=3, tokens=1)
        self.assertEqual(got.peak_state_bytes, 2 * got.state_bytes)

    def test_a_state_without_declared_components_cannot_be_built(self) -> None:
        with self.assertRaises(InvalidInput):
            RecurrentLinear(variant="unknown_variant", components=())

    def test_a_duplicated_component_name_is_refused(self) -> None:
        with self.assertRaises(InvalidInput):
            RecurrentLinear(variant="v", components=(("s", 8), ("s", 8)))


class FFNPlanTests(unittest.TestCase):
    """稠密没有可省的部分；MoE 只驻留本轮命中的专家。"""

    def test_dense_active_equals_total(self) -> None:
        got = DenseFFN(weight_bytes=LAYER3_FFN_BYTES).plan(tokens=3)
        self.assertEqual(got.active_weight_bytes, LAYER3_FFN_BYTES)
        self.assertEqual(got.total_weight_bytes, LAYER3_FFN_BYTES)
        self.assertEqual(got.active_weight_groups, (0,))

    def test_dense_needs_the_weight_bytes_from_the_layer(self) -> None:
        with self.assertRaises(InvalidInput) as caught:
            ffn_model("dense", params={})
        self.assertIn("weight_bytes", str(caught.exception))

    def test_moe_counts_each_expert_once_however_many_tokens_it_serves(self) -> None:
        # §5：同一专家服务多个 token 时，权重副本只占一次空间。
        model = MoEFFN(experts=8, top_k=2, expert_weight_bytes=1000, shared_expert_bytes=500)
        one = model.plan(tokens=1, expert_tokens={0: 1, 5: 1})
        many = model.plan(tokens=4, expert_tokens={0: 4, 5: 4})
        self.assertEqual(one.active_weight_bytes, many.active_weight_bytes)
        self.assertEqual(one.active_weight_bytes, 2 * 1000 + 500)

    def test_moe_total_covers_every_expert_plus_the_shared_one(self) -> None:
        model = MoEFFN(experts=8, top_k=2, expert_weight_bytes=1000, shared_expert_bytes=500)
        self.assertEqual(model.total_weight_bytes, 8 * 1000 + 500)

    def test_the_shared_expert_counts_even_with_no_routed_tokens(self) -> None:
        model = MoEFFN(experts=4, top_k=1, expert_weight_bytes=100, shared_expert_bytes=7)
        got = model.plan(tokens=1, expert_tokens={2: 1})
        self.assertEqual(got.active_weight_bytes, 107)

    def test_the_active_groups_carry_the_identities_not_just_the_count(self) -> None:
        # 策略要决定「哪些专家留在 VRAM」，靠的是名字。压成个数就选不了了。
        model = MoEFFN(experts=8, top_k=2, expert_weight_bytes=1000)
        got = model.plan(tokens=4, expert_tokens={5: 4, 0: 4})
        self.assertEqual(got.active_weight_groups, (0, 5))

    def test_the_order_the_routing_was_written_in_does_not_change_the_state(self) -> None:
        # 同一份路由写成什么顺序都是同一份路由，状态对象必须相等。
        model = MoEFFN(experts=8, top_k=2, expert_weight_bytes=1000)
        self.assertEqual(
            model.plan(tokens=4, expert_tokens={5: 4, 0: 4}),
            model.plan(tokens=4, expert_tokens={0: 4, 5: 4}),
        )

    def test_the_active_groups_must_be_a_canonical_ascending_tuple(self) -> None:
        with self.assertRaises(InvalidInput):
            FFNState(active_weight_bytes=1, total_weight_bytes=2, active_weight_groups=(5, 0))

    def test_the_same_group_twice_is_refused(self) -> None:
        with self.assertRaises(InvalidInput):
            FFNState(active_weight_bytes=1, total_weight_bytes=2, active_weight_groups=(0, 0))

    def test_the_reveal_gate_refuses_a_plan_without_routing(self) -> None:
        # §5：top-k 结果在 router 完成后才对因果回放策略可见。没有路由就没有账。
        model = MoEFFN(experts=256, top_k=8, expert_weight_bytes=1000)
        with self.assertRaises(Unsupported) as caught:
            model.plan(tokens=4)
        self.assertIn("router", str(caught.exception))

    def test_the_reveal_gate_refuses_an_empty_routing_vector(self) -> None:
        model = MoEFFN(experts=4, top_k=1, expert_weight_bytes=100)
        with self.assertRaises(InvalidInput):
            model.plan(tokens=2, expert_tokens={})

    def test_an_expert_outside_the_range_is_refused(self) -> None:
        model = MoEFFN(experts=4, top_k=1, expert_weight_bytes=100)
        with self.assertRaises(InvalidInput):
            model.plan(tokens=2, expert_tokens={4: 1})

    def test_more_expert_dispatches_than_tokens_times_top_k_is_refused(self) -> None:
        model = MoEFFN(experts=8, top_k=2, expert_weight_bytes=100)
        # 2 个 token × top_k 2 = 4 次分派；这里给了 5 次。
        with self.assertRaises(InvalidInput):
            model.plan(tokens=2, expert_tokens={0: 2, 1: 2, 2: 1})

    def test_missing_top_k_dispatches_are_refused(self) -> None:
        model = MoEFFN(experts=4, top_k=2, expert_weight_bytes=100)
        for counts in ({0: 1}, {0: 2, 1: 1}):
            with self.subTest(counts=counts), self.assertRaisesRegex(InvalidInput, "token dropping"):
                model.plan(tokens=2, expert_tokens=counts)

    def test_exact_dispatches_may_be_distributed_unevenly(self) -> None:
        model = MoEFFN(experts=4, top_k=2, expert_weight_bytes=100)
        got = model.plan(tokens=2, expert_tokens={0: 2, 1: 1, 2: 1})
        self.assertEqual(got.active_weight_groups, (0, 1, 2))
        self.assertEqual(got.active_weight_bytes, 300)

    def test_a_top_k_larger_than_the_expert_count_is_refused(self) -> None:
        with self.assertRaises(InvalidInput):
            MoEFFN(experts=4, top_k=5, expert_weight_bytes=100)


class RoutingTests(unittest.TestCase):
    """随机 top-k 是 router 的替身：确定性、可复现、来源标明是合成的。"""

    def _router(self) -> RandomTopK:
        return RandomTopK(experts=8, top_k=2, seed=1)

    def test_the_same_salt_always_gives_the_same_routing(self) -> None:
        self.assertEqual(
            self._router().route(tokens=6, salt="a"),
            self._router().route(tokens=6, salt="a"),
        )

    def test_a_pinned_routing_for_a_fixed_salt(self) -> None:
        # 写死一份答案：算法或种子推导一变，这条立刻红，而不是悄悄换掉一条 trace。
        got = self._router().route(tokens=6, salt="layer3:req0:0")
        self.assertEqual(got, _PINNED_ROUTING)

    def test_every_token_picks_exactly_top_k_experts(self) -> None:
        counts = self._router().route(tokens=6, salt="a")
        self.assertEqual(sum(counts.values()), 6 * 2)
        for expert, count in counts.items():
            self.assertTrue(0 <= expert < 8)
            self.assertGreaterEqual(count, 1)

    def test_all_experts_wanting_work_do_not_collapse_into_one(self) -> None:
        counts = RandomTopK(experts=128, top_k=8, seed=0).route(tokens=8, salt="layer3:req0:0")
        self.assertEqual(sum(counts.values()), 8 * 8)
        self.assertGreater(len(counts), 8)

    def test_a_small_pool_is_still_bounded(self) -> None:
        counts = RandomTopK(experts=4, top_k=2, seed=3).route(tokens=8, salt="x")
        self.assertEqual(sum(counts.values()), 16)
        self.assertLessEqual(len(counts), 4)

    def test_the_seed_is_stable_across_processes(self) -> None:
        """``hash()`` 在这里是错的——CPython 对字符串的 hash 每个进程都加盐。

        这条必须开子进程：同一个进程里跑两遍，``hash`` 和 ``crc32`` 都是绿的。
        """
        outputs = []
        for hash_seed in ("0", "1"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = hash_seed
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                str(MODELING_ROOT)
                if not existing
                else str(MODELING_ROOT) + os.pathsep + existing
            )
            completed = subprocess.run(
                [sys.executable, "-c", _ROUTE_IN_FRESH_PROCESS],
                cwd=MODELING_ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, f"stderr={completed.stderr!r}")
            outputs.append(completed.stdout.strip())
        self.assertTrue(outputs[0], "子进程什么都没打印")
        self.assertEqual(outputs[0], outputs[1])

    def test_a_top_k_larger_than_the_pool_is_refused(self) -> None:
        with self.assertRaises(InvalidInput):
            RandomTopK(experts=4, top_k=5)

    def test_the_source_is_marked_synthetic(self) -> None:
        # 随机路由不能报成实测，和成本表的 provenance 是同一条纪律。
        self.assertEqual(RandomTopK.source, "synthetic_random")

    def test_the_routing_feeds_the_ffn_plan_directly(self) -> None:
        router = RandomTopK(experts=128, top_k=8, seed=0)
        moe = MoEFFN(experts=128, top_k=8, expert_weight_bytes=1_000)
        counts = router.route(tokens=4, salt="layer3:req0:0")
        plan = moe.plan(tokens=4, expert_tokens=counts)
        self.assertEqual(plan.active_weight_bytes, len(counts) * 1_000)
        self.assertEqual(plan.active_weight_groups, tuple(sorted(counts)))
        self.assertEqual(plan.total_weight_bytes, 128_000)


class FamilyInvariantTests(unittest.TestCase):
    """稠密 FFN 是「1 选 1」的 MoE——两族只有一个轴。"""

    def test_dense_equals_a_one_expert_moe(self) -> None:
        dense = DenseFFN(weight_bytes=LAYER3_FFN_BYTES).plan(tokens=3)
        moe = MoEFFN(
            experts=1, top_k=1, expert_weight_bytes=LAYER3_FFN_BYTES
        ).plan(tokens=3, expert_tokens={0: 3})
        self.assertEqual(dense, moe)

    def test_the_ffn_state_refuses_an_active_larger_than_the_total(self) -> None:
        with self.assertRaises(InvalidInput):
            FFNState(active_weight_bytes=2, total_weight_bytes=1)

    def test_the_attention_state_refuses_a_peak_below_the_final(self) -> None:
        with self.assertRaises(InvalidInput):
            AttentionState(0, 0, state_bytes=100, peak_state_bytes=99)

    def test_a_boolean_is_not_a_width(self) -> None:
        with self.assertRaises(InvalidInput):
            FullCausal(kv_bytes_per_token=True)


_PINNED_ROUTING = {5: 2, 0: 2, 2: 1, 3: 3, 6: 1, 4: 3}

_ROUTE_IN_FRESH_PROCESS = textwrap.dedent(
    """
    # 同一个 salt 在两个不同的 PYTHONHASHSEED 下必须路由到同一份结果。
    from llm_infer_model.components import RandomTopK

    counts = RandomTopK(experts=8, top_k=2, seed=1).route(tokens=6, salt="layer3:req0:0")
    print(" ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    """
)


_ISOLATED_COMPONENTS = textwrap.dedent(
    """
    # 第一行就是被测的东西：components 先于 m2 被 import。
    from llm_infer_model import components

    from llm_infer_model.m2.spec import ATTENTION_KINDS, FFN_KINDS

    assert sorted(components.ATTENTION_MODELS) == sorted(ATTENTION_KINDS)
    assert sorted(components.FFN_MODELS) == sorted(FFN_KINDS)
    a = components.attention_model("full_causal", kv_bytes_per_token=4096).state(history=3, tokens=1)
    f = components.ffn_model("dense", weight_bytes=45895680).plan(tokens=3)
    print("ok", a.state_bytes, f.active_weight_bytes)
    """
)


class ImportDirectionTests(unittest.TestCase):
    """``components`` 先 import 也必须成立——反向 import 一旦写到模块级就会碎。"""

    def test_components_import_first_in_a_fresh_interpreter(self) -> None:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(MODELING_ROOT) if not existing else str(MODELING_ROOT) + os.pathsep + existing
        )
        completed = subprocess.run(
            [sys.executable, "-c", _ISOLATED_COMPONENTS],
            cwd=MODELING_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
        )
        self.assertEqual(completed.stdout.strip(), "ok 16384 45895680")


if __name__ == "__main__":
    unittest.main()
