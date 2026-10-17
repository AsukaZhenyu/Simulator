# Dense / MoE 共享 FFN 执行变换验收

`ffn-execution-0.1` 是独立的执行变换声明，不能作为原层成本表加载。
它把 SwiGLU 的中间维度切成连续、不重叠且完整覆盖的区间。
每个区间同时配套 gate/up 的输出列与 down 的输入行，产生完整 hidden
宽度的部分结果，最后通过 `ffn_sum` 求和。所有分片都可以绑定 CPU/GPU。

Dense 使用已有完整 decoder 契约：RMSNorm 只做一次，FFN residual 只加一次。
MoE 使用 `selected_rows + swiglu_staged + moe-execution-0.2` 的明确三矩阵，
复用同一 FFN 变换，不改变 gather、路由、scatter、weighted combine 与 KV。

## 已保存输入

- `dense.execution.json`：H=16、I=32，CPU 区间 `[0,8)`、GPU 区间 `[8,32)`；
  源输入是 `mapping/examples/m2/hybrid-interface-synthetic-decode-1.json`。
- `moe.execution.json`：H=4、I=8，CPU 区间 `[0,2)`、GPU 区间 `[2,8)`；
  先加载相邻 `../subweights/invocation.json`，并应用
  `../subweights/execution.json`，再应用本声明。

所有数字均为**独立声明的合成成本**；没有按原宏任务时间比例分摊。
成本键包含 kind、device、phase、batch、tokens、visible_history、hidden_size、
intermediate_size。分片阶段的 intermediate_size 是局部宽度；norm、sum、
residual 使用完整宽度。workspace 必须为已知非负整数。新字节形状的每条
开放传输路线都必须有精确价格，不插值。

外层必填 `merge_input_count` 必须等于分片数量，合并成本与 workspace 明确属于
这个输入数的合同。只修改 `shards` 从两片改成一片或三片会报错，不能静默复用
旧合并价格。改变数量需要显式声明新的 `merge_input_count` 与对应成本；未来真实
profile 的物理签名也必须包含这个输入数。`ffn_sum` 的图参数和阶段元数据均记录
实际 input count。这里的合成价格不构成对其他合并数量的测量证据。

## 存储语义

原矩阵被独立、连续 F32 分片替代，父分配删除，每个子分配分别按设备粒度
计账。gate/up 分片为 `[H, width]`，down 分片为 `[width, H]`（GGML ne 顺序）。
它们是**初始表示**，不是对已驻留父矩阵免费做运行时切片或零成本重排。
所有未激活专家也保留完整叶子权重清单，供未来需求/预取使用。
实际权重数值尚未加载，不支持量化矩阵比例切分或真实 profile 直接继承。

## 验收命令

在 `Simulator/modeling` 下运行：

```powershell
python -m unittest discover -s tests -p test_ffn_execution.py -v
```

验收包含非均匀分片和单分片、Dense/MoE 独立数值参考、多行专家、完整动作回放、
精确 VRAM 峰值以及少 1 B 拒绝、双向搬运、缺失成本/未知 workspace 拒绝、
Attention/KV 原样保留，以及 Attention 拆分后接 FFN 拆分的组合。

数值参考在测试中用随机非精确二进制输入解释实际的阶段依赖与分片范围，并与
完整 SwiGLU 独立计算比较。另有逐阶段 F32 物化舍入参考，要求分片归约误差
非零且小于 `2e-6`，避免精确二进制玩具数据掩盖数值差异。它证明此变换的小例语义，不代表生产内核实现或
硬件时间准确度，也不会将产物的 `numeric_execution_verified` 置为真。
