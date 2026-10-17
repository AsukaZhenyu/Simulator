# 多轮组合验收夹具

本目录服务于 `../../execution_combined_demo.py` 与
`mapping/tests/test_execution_phase_composition.py`。

它引用既有 `modeling/tests/fixtures/global_execution` 的真实物化 Dense `.5`
模型结构与两层 Prefill(3)→Decode(1) 输入。"真实物化"指 norm/QKV/FFN
矩阵、布局和状态接口完整，不表示使用了真实模型权重或硬件实测成本。

每个 scope 先展开 AttentionStages，再展开 FFN 中间轴分片。CPU `[0,8)`、
GPU `[8,32)` 固定协作；`merge_input_count=2` 明确求和输入数。
projection/Attention 按 phase 从 GPU 切换到 CPU，FFN 分片绑定保持不变。
两层权重叶子分别用于 startup 与 spare 准备，同一份 KV 和权重贯穿两轮。

`generate.py` 内的 Prefill、Decode 阶段时间是分别书写的独立合成常数，
没有按 token 数或 decode 时长比例推导。workspace 也为独立合成声明。
改变形状或分片数，需要新的匹配合同，不能复用这些成本冒充测量。

传输表保留既有 route/bytes 键值，仅为新物化的真实字节数补充独立常数：

- 144 bytes：Prefill 的 3 keys × 3 queries × 4 heads × 4 bytes 分数/概率。
- 288 bytes：24 个 FFN 列 × 3 tokens × 4 bytes 的中间结果。
- 1536 bytes：16 hidden × 24 FFN 列 × 4 bytes 的一片矩阵。

这些合成常数只检验容量、动作、依赖与组合行为，不产生真实性能或策略收益结论。

从项目根目录再生成本目录的 JSON：

```powershell
python Simulator/mapping/examples/python/fixtures/execution-phase-transforms/generate.py
```

所有 JSON 明确以 UTF-8/LF 写出。模拟实际运行时直接读取 JSON，源码绑定
仍涵盖它们引用的既有模型、成本、状态与新变换文件。
