# 多层、多轮连续执行的小例子

这组输入把真正经过 M2 校验的调用展开成一个计算图，再交给同一个运行时。
它不是单层时间乘层数：计算、搬运、内存、缓存副本、KV、设备环境和时钟都连续。
价格仍是明确的合成价格，不证明真实模型精度或实际加速。

| 入口 | 内容 |
|---|---|
| `dense.global.json` | Dense 物理 `.5` 前端，两层，prefill 3 token 后 decode 1 token |
| `moe.global.json` | MoE `.6` 前端，两层，prefill 2 token 后 decode 1 token，保留整个专家权重 |
| `moe-fragments.global.json` | 两层 MoE，三块 F32 子权重，prefill 2 token 后 decode 1 token |
| `moe-fragments.services.json` | 固定 128 B 暂存槽、CE/SM 两种搬运与共享链路合同 |

## 怎样连接

每轮层的顺序、请求顺序、token 数、绝对位置和硬件合同必须一致。每次调用仍独立
提供自己的层、成本和历史，先按 M2 原有规则校验；缺少精确 token/history 成本会
直接拒绝，不插值。

第一层的 hidden 输入由 trace 提供，后续层的 hidden 输入接到前一层的实际输出。
层间接口检查 dtype、物理形状、连续步长及存储字节。各次调用的操作和临时张量
使用调用作用域前缀，权重及 KV 的身份保持稳定。

下一轮的 `trace_supplied_next_token` gate 依赖前一轮最后一层的 hidden。gate 的
device、duration、workspace 必须明确提供；这些 fixture 声明 CPU 控制成本 7 ns。
它只表示 trace 中下一 token 输入可用的时刻，不执行数值采样、tokenizer 或 embedding。

后续调用的历史必须精确接上前轮的 KV 身份、位置、大小与状态版本。独立 decode
文件中声明的历史 READY 位置，只用于独立调用校验，不会被重新灌入全局库存。
例如 prefill 只在 DRAM 产生的 KV，不会因 decode 文件写了 VRAM 而凭空多出副本。

## 策略看得到什么

所有层的不可变权重目录一开始就可见，可提前加载并承担对应成本。后续调用的
任务在其依赖调用结束后才可见；MoE 的专家任务等到本调用 Router 完成后才公开。
前轮路由不会因下一轮的真实路由而改变策略观察。公开的每个操作有 layer_id 与
invocation_id，可区别各层、各轮。

`release_candidates` 给出已关闭调用里可释放的 DRAM/VRAM 副本。KV、权重、最终
目标与仍要用于下一层或下一 token gate 的 hidden 不会被列入。候选只是建议，
具体释放仍要通过内核的读锁、在途搬运和资源校验。

本版采用保守的调用边界：一个调用的全部操作结束后开放后续调用；不是跨请求
的完整流水/融合 batch，也不支持动态新请求到达。Dense pure-attention execution
sidecar 尚未接入这个全局入口，会明确拒绝。

## 运行与证据

Python API：

```python
from llm_infer_model.m2.global_execution import (
    compile_global_execution_file, GlobalM2Session,
)
compiled = compile_global_execution_file("dense.global.json")
session = GlobalM2Session(compiled)
observation = session.observe()  # 只把这个交给策略
```

在仓库根目录可使用统一映射入口：

```text
python -m tensor_mapping.unified_strategy mapping/examples/unified/dense-global.strategy.json --out mapping/outputs/dense-global
python -m tensor_mapping.unified_strategy mapping/examples/unified/moe-fragments-global.strategy.json --out mapping/outputs/moe-fragments-global
```

`test_global_execution.py` 验证单时钟、缓存复用、KV 接缝、隐藏未来路由、真实 gate
依赖、释放边界、源文件 SHA 以及在途库存。简单串行 CPU 参考在 Dense 得到 44007 ns，
在整个专家权重 MoE 得到 98000007 ns；这些是合成成本的可核算结果，不是两类模型
的性能对比。

`persistent_state_document()` 导出实际副本状态、运行中任务、服务账本、已关闭
调用和明确声明的权重版本。它是运行库存说明，不是独立可恢复的 checkpoint；
统一映射入口另行提供绑定源文件与运行时版本的动作回放 checkpoint。
