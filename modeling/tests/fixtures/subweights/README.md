# 明确的专家子权重

这是 `moe-execution-0.2` 的合成输入，不是真实模型性能数据。

`invocation.json` 引用 `layer.json` 与 `costs.json`。把 `execution.json` 通过
`parse_execution_spec` 解析后交给 `transform_moe_execution` 或 `CausalM2Session`。

隐藏维度为 4，FFN 中间维度为 8。每个专家是连续 F32 的三块矩阵：

| 子矩阵 | GGML physical_shape | 父文件字节范围 | 独立存储 |
|---|---|---|---|
| gate | `[4, 8]` | `[0, 128)` | 128 B |
| up | `[4, 8]` | `[128, 256)` | 128 B |
| down | `[8, 4]` | `[256, 384)` | 128 B |

第一维元素相邻，权重不可变，子身份为 `w:moe-small:E0:v1:gate` 等。
父 384 B 张量从运行图移除，三个子张量继承其初始存储位置。每份 DRAM/VRAM
副本独立计入容量和对齐开销；搬运只按对应矩阵的 128 B 计入负载。此变换在
初始输入阶段发生，不表示执行过程中可以免费拆解已驻留的父张量。

清单的 `source_evidence.sha256` 是规范化布局声明的 SHA-256，可通过
`partition_digest` 复核。它不是权重值的哈希，不证明真实模型文件或内核性能。
目前拒绝量化切片、父子别名、未核验的真实权重声明和成本插值。

`test_subweights.py` 用随机非二进制精确 F32 输入核对独立 packed 参考与拆分
SwiGLU 计算。此数值检查是 Python 算术验证，不宣称 CUDA/CPU 原生内核已验证。
对 gate/up 放 GPU、其余阶段放 CPU 的串行运行，手算期望 H2D 832 B、D2H 256 B、
SSD 读取 1152 B、VRAM 峰值 1092 B；1091 B 容量必须失败。

旧 `moe-execution-0.1` 保持原有“阶段各自读取整个专家权重”的语义，不自动升级。
