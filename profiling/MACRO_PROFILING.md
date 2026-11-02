# Attention / FFN 宏任务测量

本工具直接测量已声明的 CPU 子图，不将孤立 `MUL_MAT` 相加冒充宏任务。

## 1. 当前可以测什么

CPU、B=1，GGML 公共 API，权重与输入生成、分配及量化在计时外。每个样本直接执行一次完整的已声明子图，保留全部墙钟样本。

| kind | 计时内的操作 | 输入与输出 |
|---|---|---|
| `dense_ffn` | gate/up matmul → SwiGLU split → down matmul → 最终输出复制 | hidden 输入 `[H,q]`，输出 `[H,q]`；三个独立权重矩阵 |
| `attention` | 追加新 K/V → QKᵀ / sqrt(D) → causal mask → softmax → PV → 可选 sigmoid 门控 → O projection → 最终输出复制 | Q `[D,q,Hq]`，新 K/V `[D,q,Hkv]`，已有 KV 与追加槽位；输出 `[H,q]` |

Attention 支持 MHA/GQA/MQA，要求 `heads % kv_heads == 0`；`hidden` 可以不等于 `heads*head_dim`。Q/K 作为已完成位置变换的输入提供，首版不执行 QK norm 或 RoPE。

FFN 权重可统一 F32/F16/Q4_K/Q6_K，也可分别声明 gate/up/down 的 dtype。量化约束检查对应矩阵的 reduction dimension，不检查 token 数是否为 256 的倍数。

**这是部分 decoder block 子图的测量。** 当前不包含完整 projection、norm、RoPE、residual，不支持 GPU、批量 B>1、Flash Attention、F16 KV 或并发预测。

## 2. 文件与运行

- `native/macro_probe.cpp`：原生子图、独立 double 参考、KV 内容校验与分配账本。
- `macro_contract.py`：执行参数、物理签名、支持范围的严格校验。
- `run_macro_measurements.py`：复用公共记录/统计协议，绑定 native 返回值并保存原始证据。
- `cases/macro_cpu_smoke.json`：5 条基础 case。
- `cases/macro_cpu_validation.json`：15 条精度、GQA、mask、history 和门控验证 case。
- `cases/qwen35_4b_layer3_macro_cores.json`：4 条真实层尺寸的子图配置。

按[构建说明](README.md)在新目录配置已有 GGML 依赖，再构建目标。不要覆盖已由实验 SHA 绑定的程序：

```powershell
cmake --build "<new-CPU-build-directory>" --config Release --target ggml-macro-probe

python profiling/run_macro_measurements.py --dry-run `
  --suite profiling/cases/macro_cpu_validation.json --threads 2

python profiling/run_macro_measurements.py `
  --suite profiling/cases/macro_cpu_validation.json `
  --native-bin "<macro-probe-directory>" `
  --runtime-dir "<matching GGML shared library directory>" `
  --threads 2 --warmup 5 --samples 20 `
  --output profiling/results/new-macro-run
```

Windows 多配置程序通常位于新构建目录的 `Release/`；Linux 单配置不加该层。Python 只使用标准库，构建不修改或重新构建 llama.cpp 源树。

输出目录必须为空，与原有 runner 共用 `.measurement.lock`，防止两个测量脚本同时采集。失败保留 stdout/stderr、命令与 case；默认停止，`--keep-going` 可继续余下 case。JSON/CSV 使用 LF。

## 3. 配置例子

混合精度 FFN：

```json
{
  "id": "ffn_decode",
  "kind": "dense_ffn",
  "phase": "decode",
  "batch": 1,
  "tokens": 1,
  "visible_history": 3,
  "hidden": 2560,
  "intermediate": 9216,
  "weight_dtypes": {"gate": "q4_k", "up": "q4_k", "down": "q6_k"}
}
```

统一精度时改为一个 `weight_dtype` 字段，二者不可同时出现。Attention 需显式提供 `heads`、`kv_heads`、`head_dim` 和 `attention_gate: none|sigmoid`，`weight_dtype` 指 O projection 权重。

FFN 不读取 `visible_history`，但记录并绑定请求的历史键。这是该任务定义的历史无关性质，不是测量了 attention 的历史成本。Attention 真正构造 `history+q` 长度的 cache；第 i 个 query 只能读取 history 加本次前 i+1 个位置。

未知执行字段被拒绝。写入 `kv_dtype=f16`、`rope=true` 或参数拼错不会被静默忽略。解释性信息放在明确的 metadata 字段，不能用它改变执行路径。

## 4. 数值与 KV 验证

输入生成器使用 `mt19937_uniform24_v2`。参考由独立 double 实现构造，实际存储权重先反量化，再执行完整 FFN 或 causal attention 数学流程。

输出元素数 ≤512 时全部核对；较大输出最多检查 512 个分散位置、覆盖最多三个 token 位置，记录实际覆盖范围。所有输出都检查有限性。参考阈值随 quantized 与浮点路径区分；量化内核的 activation 转换可以产生非零执行误差，这不评估原 F32 模型的量化质量。

Attention 另对 cache 全部元素验证：历史前缀保持原值，新增槽位逐元素等于新 K/V。预热与样本重复覆盖同一批新增位置，不随样本数增长历史。

当前 KV 读写顺序由 b8705 CPU 的逐节点图执行保证。该保证不能直接用于 GPU、异步执行或图重排；扩展时需重新建立依赖和验证。

## 5. 时间与 workspace

每个样本包含完整 CPU graph dispatch、完成同步与最终 output copy。分配、权重生成/量化和主机参考在计时外。额外一次 `setup_compute_repetitions=1` 先准备 CPU scratch，再执行请求的 warmup 次数；`--warmup 0` 也不是冷启动测量。

Resident buffer 放输入、权重、最终输出和 KV。内部节点由独立 gallocr 管理。当前：

`workspace_bytes = internal_allocated_buffer_bytes + cpu_cplan_work_size_bytes`

这是含 allocator alignment 的内部可复用容量加 CPU scratch，有明确的 conservative allocated-capacity 口径。不能再把同一批内部节点逐个加入模拟器，否则会双重计费。新 KV 输入与 cache 追加槽位是两份独立存储，分别记录。

账本列出 resident logical/allocated 字节、内部逻辑张量 sum、内部实际分配与 scratch。Host 参考副本、线程栈、backend context 等运行开销不包含在该 workspace 中。它不直接证明整层或进程峰值；M2 仍须核对端口、持久 reserve 和同时存活的状态。

## 6. 查询与模型接入

现有查询入口可以查询宏任务记录：

```powershell
python profiling/query_profile.py `
  --profile "<completed-macro-profile>" `
  --case-id "<exact-case-id>"
```

返回时间、workspace、包含矩阵 dtype/实现变体的签名、统计与出处。仍然 `capacity_verified=false`，不会自动生成 M2 成本表。

模型侧通过真实端口、权重组、B=1 和 scratch 定义绑定该子图。`export_m2_macros.py` 可显式导出匹配的部分 block，见[物理合同](../modeling/llm_infer_model/m2/PHYSICAL_CONTRACT.md)。本入口未执行完整 projection，不能将这里的两类子图价格补进符号三任务链就声称整层已校准；完整声明 block 有[独立入口](DENSE_BLOCK_MEASUREMENT.md)。

Qwen 的事实与边界证据在 `cases/qwen35_4b_layer3_boundary.json`。FFN case 使用真实矩阵尺寸和 Q4_K/Q4_K/Q6_K 标签，但权重内容由程序生成。Attention case 只有相同几何/门控/O projection；原图为 Flash Attention、F16 KV、预分配 padded cache，当前为显式 Attention、F32 KV、最小 `history+q` cache，两者不能共用成本或物理容量结论。

## 7. 验证

合同测试位于 `profiling/tests/test_macros.py` 与 `test_m2_export.py`。换机器或参数后重新采集，保存明确的 warmup、样本数、线程和质量门；短样本 smoke 只检查执行入口，不作为可信策略收益。组合成本的独立核验步骤见[执行变换校准工具](execution_calibration/README.md)，当前完整性能质量门尚未通过，本机历史结果不随仓库分发，不能跨不同物化合同复用价格。
