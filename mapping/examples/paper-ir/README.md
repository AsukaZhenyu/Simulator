# 六项论文机制的统一 IR

先读 [快速对照](../../docs/PAPER_IR_QUICK_GUIDE.md)。本目录七份 `.strategy.json` 使用当前 `unified-strategy-0.2`；模型、布局、精确成本和环境均明确给出。支持的是论文机制子集，不是原系统复现。所有时间是合成价格，不能拿这些完成时间比较论文优劣。

| 完整 IR | 重点看什么 |
| --- | --- |
| [ktransformers.strategy.json](ktransformers.strategy.json) | 路由专家 CPU，共享专家及常用部分 GPU；不含 Expert Deferral |
| [neo.strategy.json](neo.strategy.json) | 已选 CPU 请求的 decode Attention 核心，GPU 投影/输出/FFN；不是双子批流水 |
| [fiddler.strategy.json](fiddler.strategy.json) | 驻留优先、CPU/GPU 完整路径；完整计费是项目扩展 |
| [hybrimoe.strategy.json](hybrimoe.strategy.json) | 三队列优先级、MRS；不含原整层计划/预测预取 |
| [flexgen-layer-major.strategy.json](flexgen-layer-major.strategy.json) | 三层/两批/两代，按层复用、SSD 读写 |
| [flexgen-batch-major.strategy.json](flexgen-batch-major.strategy.json) | 同样条件的批优先对照；order 仍在引用模型 |
| [atsinfer.strategy.json](atsinfer.strategy.json) | 人工分阶段计划、CE/SM、共享 PCIe、跨层槽；不含原 DP |

在 `Simulator` 下运行一个例子，输出路径必须不存在或为空：

```powershell
& .\modeling\.venv\Scripts\python.exe -m tensor_mapping.unified_strategy mapping/examples/paper-ir/fiddler.strategy.json --out mapping/outputs/paper-ir-fiddler-new
```

只要替换文件名，就能运行其余例子。上述 Python 应能导入 `llm_infer_model` 和 `tensor_mapping`；源码环境也可显式设置 `$env:PYTHONPATH = 'mapping;modeling'`。这不要求运行原论文仓库。

相关策略机制的回归测试：

```powershell
python -B -m unittest discover -s mapping/tests -t mapping -p test_paper_integration.py -v
```

测试检查既有机制的执行规则，不测硬件，也不搜索。审阅每份完整 IR 时，分别运行上方 CLI 并检查下列产物；不再依赖旧验收目录中的脚本。

每次产物可这样读：

| 文件 | 回答的问题 |
| --- | --- |
| `strategy.json` | 提交的整份方案是什么？ |
| `policy_audit.json` | 哪条规则作了这个选择，依据是什么？权威声明位于 `composition`。 |
| `execution.json` / `events.json` | Compiler 下发了什么，模拟何时开始/结束？ |
| `decisions.json` | 每次决策前，策略当时究竟知道什么？ |
| `summary.json` | 是否合法完成，内存与搬运字节是多少？ |
| `compiled_graph.json` / `states.json` | Modeling 私有调试图/状态；不要把它们当作策略可见输入。 |

注意两个接缝：`fiddler_paths` / `hybrimoe_queues` 是内置模块；block 的 order/home 仍在 model。当前 `.2` 尚未把论文所有决策维度都变成独立的通用表达式。
