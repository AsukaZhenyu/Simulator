# 一个入口执行当前策略与真实 M2 调用图

从 Simulator 目录运行 `python -m tensor_mapping.unified_strategy <strategy> --out <new-directory>`。

| 输入 | 验证的内容 |
| --- | --- |
| `fragment.strategy.json` | 单层 MoE selected rows、gate/up GPU、down CPU、独立子权重、CE/SM、projection/router 固定槽重用 |
| `dense-global.strategy.json` | 两层 Dense P3→D1；hidden 层间接线、连续 KV、外部内存占用和 CPU 暂停/恢复 |
| `moe-global.strategy.json` | 两层 MoE P2→D1；router 因果揭示、专家权重跨轮保留、同一服务时钟 |
| `moe-fragments-global.strategy.json` | 把子权重、固定槽、CE/SM、混合计算、两层与两轮放入同一条执行链 |

这些都是合成小图，用来检查表达和运行语义。运行时间不是真实模型性能，不用于论文之间比较。被引用的 layer/invocation/execution fixture 在 `modeling/tests/fixtures/`；完整图与未来路由由 modeling 私有保存。

顶层文件把模型与策略分开。`model_file` 指定图和输入；`policy_file` 绑定任务设备及专家规则；`services_file` 绑定搬运与暂存槽；`environment_file` 指定 modeling 私有的未来设备事件。`transport` 明确指定四类张量使用的 CE/SM、固定槽重用、已关闭作用域的释放和以张量为单位的缓存预算。没有服务合同时，各 engine 必须显式为 null。

跨轮输入 gate 的成本在全局模型里显式声明，表示 trace 输入就绪，不冒充真实 sampling 或 embedding。阶段成本必须存在精确形状，没有插值和自动拆分整块宏任务价格。

```powershell
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/unified/moe-fragments-global.strategy.json --out mapping/outputs/moe-fragments-global
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/unified/moe-fragments-global.strategy.json --max-decisions 3 --out mapping/outputs/global-partial
.\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/unified/moe-fragments-global.strategy.json --resume mapping/outputs/global-partial/checkpoint.json --out mapping/outputs/global-resumed
```

`--max-decisions` 是本次新增命令预算。恢复会重建原命令前缀，验证观察、决定、库存、时钟和在途服务完全一致，再继续；来源或代码发生变化时拒绝。主机重建开销不计入模拟服务时间。

先看 `summary.json` 和 `policy_audit.json`。`decisions.json` 只含公开观察；`compiled_graph.json` 和完整 `states.json` 属于私有调试证据。单独的 `persistent_state.json` 不具备 checkpoint 语义。
