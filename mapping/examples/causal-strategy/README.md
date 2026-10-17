# 可运行的 MoE 策略配置

这是合成小图的策略表达入口，没有真实硬件校准或论文复现结论。原始图、成本表与旧命令保持可用。

先在 Simulator 根目录安装两个包：

```powershell
python -m pip install -e ./modeling -e ./mapping
```

运行一个配置；输出目录必须不存在或为空：

```powershell
python -m tensor_mapping.causal_strategy mapping/examples/causal-strategy/rows-mixed.strategy.json --out mapping/outputs/my-rows-mixed
```

| strategy | 可以探索什么 |
| --- | --- |
| `whole-mixed` | 保留旧整体专家包装，GPU Attention + CPU 专家 |
| `rows-cpu` / `rows-gpu` | selected rows 的全部 CPU / GPU 固定执行 |
| `rows-mixed` | GPU gather、CPU 专家、GPU scatter；只跨设备传选中行 |
| `staged-hybrid` | GPU gate/up、CPU SwiGLU/down、GPU gather/scatter |
| `resident` | 专家权重驻留优先；初始 E0 已在 VRAM |
| `current-cost` | 按当前公开计算与搬运价格选择专家设备 |
| `history` | 由过去频次猜专家，错误的 E3 预测也发生真实 SSD 读取和 H2D |
| `fluctuating` | 6 ms 时 CPU 暂停并新增 3000 B VRAM 占用，8 ms 恢复 |
| `rows-retain` | 对照 `rows-cpu`：保留中间值与 `last_use` 释放 |
| `warm-cache-pressure` | 紧容量下驱逐继承的专家 VRAM 缓存，保留 SSD 源 |

策略文件必须给出全部八个字段：

```json
{
  "format_version": "causal-strategy-0.1",
  "id": "my-strategy",
  "invocation_file": "../m2/moe-interface-synthetic-prefill-2.json",
  "environment_file": "../causal/no-disturbance.environment.json",
  "execution_file": "rows.execution.json",
  "policy_file": "mixed.policy.json",
  "history_file": null,
  "max_decisions": 250
}
```

想改设备分工就复制一个 policy 改 `placements`；想用动态专家规则改 `experts.device_rule`；想拆开 FFN，换成 `staged.execution.json` 并指定四个专家阶段的设备。`execution_file:null` 明确保留原宏任务。缺成本、未知字段、缺阶段绑定都报错，不自动替换。

`release.rule=last_use` 只在本次图完全揭示后释放不再使用的中间值。`cache.rule` 管理专家权重，`prefetch.cache_budget_bytes` 只约束投机 VRAM 预取；三者不是同一个开关。禁用预取时 `max_count` 与 `cache_budget_bytes` 必须为零。

`summary.json` 给出时间、搬运量、DRAM/VRAM 峰值与来源；`policy_audit.json` 给出当前成本、容量、选择原因和预测命中/漏失；`decisions.json` 将每步观察、决定与编译动作连起来。`compiled_graph.json` 是完整私有图的调试导出，禁止回传给 Policy 作决策。

退出码：0 表示完成并合法；3 表示运行失败或预算内未完成；2 表示输入、编译或文件错误。合法不表示最优，合成时间差也不表示真实设备加速。

接口与组合方式见[策略指南](../../docs/COMPOSABLE_IR_GUIDE.md)。Policy 的 Python 开销未计入模拟延迟；此 CLI 每次创建新 Session，历史可以导入，但时钟和连续环境不跨调用续接；持续多轮使用上层 unified/Python 入口。
