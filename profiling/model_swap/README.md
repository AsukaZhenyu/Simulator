# 换一个模型：机械步骤

本目录只做一件事：把一个模型的**结构**变成 M2 的 layer/geometry 骨架。
它不测量、不推理、不加载权重，也不发明任何 `duration_ns`。

入口：

```text
python profiling/model_swap/build_model.py --gguf <model.gguf> --name <id> --out <new-dir>
python profiling/model_swap/build_model.py --metadata <metadata.json> --name <id> --out <new-dir>
```

`--gguf` 只用 `GGUFReader` 读文件头和 tensor 目录（不读 tensor 数据，不做推理）；
需要可选依赖 `pip install -e ".[gguf]"`。`--metadata` 读同一组规范化字段，走完全
相同的构造代码，用于本机没有 GGUF 的情况。输出文件不存在时才写入；默认拒绝
覆盖已有文件（`--force` 显式覆盖）。`GGUFReader` 会解析整个 metadata KV 表
（含 tokenizer 数组），本机这两个 Qwen3.5 文件约 20 s；它只读 header 与 tensor
目录，不加载 tensor 数据。

输出三个文件（都小于 200 KB 量级，可直接随模型描述一起提交）：

| 文件 | 内容 | 能否直接给 M2 用 |
|---|---|---|
| `<id>.geometry.json` | 模型级结构：层数、hidden、FFN/专家、heads/KV heads、每层 attention 类型、权重字节、必须补测清单 | 是描述，不是可执行文档 |
| `<id>.layer.json` | 一个代表层的 `m2-layer-0.1`（dense）或 `m2-layer-0.6`（MoE）描述符 | 能过 `parse_layer_document`；成本仍需另给 |
| `<id>.costs.template.json` | 每个宏任务 × prefill/decode 的空条目 | **故意不能**过 `parse_cost_table`，直到填数 |

## 1. 一个模型的输入由哪几个文件构成

M2 的单位是「一个 decoder layer 的一次调用」。一个模型要跑起来需要三份文档，
外加硬件容量（容量写在 invocation 里，不在 layer 里）：

### `*.layer.json` — 结构（可复用）

- 必填：`format_version`、`id`、`hidden_size`、`hidden_bytes_per_token`、
  `kv_bytes_per_token`、`attention`、`ffn`、`tasks`。
- `provenance` 建议显式写；不写时解析器默认 `synthetic_layer`，但那会让
  「这是测出来的还是合成的」变成猜的。
- `weights` 可空；一旦给，`storage_bytes` 必填且 >= 1。带 `physical` 的版本
  （0.2/0.3/0.4/0.5）才允许/要求 `dtype` 与 `shape`。
- `attention_params`：`sliding_window` 需要 `window`；`recurrent_linear` 需要
  `variant` 与 `components`；`full_causal` 不许有参数。
- `ffn_params`：`moe` 需要 `experts`、`top_k`、`expert_weight_bytes`；
  `m2-layer-0.6` 还要求 `expert_weights` 逐个列出，且每个专家的
  `storage_bytes == expert_weight_bytes`。
- 任务图必须是 DAG。可执行切片要求 `projection -> attention -> dense_ffn`
  （`.6` 要求 `projection -> attention -> router -> expert_ffn -> combine`）。
- 带 `physical` 的文档必须同时给 physical 合同；生成器**不**输出 physical，
  因为合同、线程数、activation/KV dtype 都不是模型元数据。

### `*.costs.json` — 逐形状成本（必须实测）

- 必填：`source`、非空 `entries`。每条 entry 必填 `kind`、`device`、`phase`、
  `batch`、`tokens`、`visible_history`、`provenance`、`workspace_bytes`，以及
  `duration_ns` / `duration_s` **恰好一个**。
- `measured_macro` 还要求显式 `validation_only`、`accounting`、
  `workspace_scope="macro_internal_scratch"`、`workspace_source`、
  `physical_signature` + 对应 `contract_hash`。
- `m2-layer-0.6` 只接受 `provenance="synthetic_layer"`；实测 MoE 成本目前没有
  可装载的边界。MoE 的实测数字要走 `profiling/MOE_MEASUREMENT.md` 的独立工具，
  不能直接塞进 `.6` 表。
- 未知 workspace 只能写 `null`（且会被执行拒绝），不能写 0。

### `*-prefill-3.json` / `*-decode-1.json` — 一次调用

- 必填：`id`、`layer`、`phase`、`requests`、`hardware`、`state_in`，以及
  `costs_file` 与内嵌 `costs` **恰好一个**。
- `layer_file` 与 `costs_file` 相对**调用文件**解析。
- `devices` 默认 `["cpu"]`；hybrid/MoE 必须显式。
- `hardware`：`dram_capacity_bytes`、`vram_capacity_bytes` 容量必填；
  `memory_mode` 默认 `separate`，`shared` 时必须给 `shared_capacity_bytes`。
- `state_in` 必须显式，零历史也要写出 `{"next_position": 0, "blocks": []}`；
  `.6` 还必须给 `weight_copies`、`route_source`、`route_trace`。
- 现成模板：`mapping/examples/m2/fcd-prefill-3.json` 与 `fcd-decode-1.json`。

### 入口

`mapping/tensor_mapping/strategy_inspection.py` 与 `strategy_compiler.py` 都先
`parse_invocation_document`，再按 `layer_file` 读 layer，最后
`validate_invocation(layer, invocation)`。所以验收时直接跑这条链路，不要只看
生成器自己的断言。

## 2. 换模型步骤

1. **生成结构**
   ```text
   python profiling/model_swap/build_model.py \
     --gguf <path>/<model>.gguf --name <id> --out profiling/model_swap/out/<id>
   ```
   没有 GGUF 时，手写 metadata（见 §3 的字段表）并改用 `--metadata`。
   只想看结构、不生成代表层时加 `--structure-only`。
2. **人工检查 geometry**
   - `attention.layer_kind_counts` 是否与模型族一致（Qwen3.5 是 3/4 线性 +
     1/4 full attention）。
   - `weight_layout.q_gate_interleaved_layers` 是否非空；若 Q 与 gate 打包在
     一个 tensor 里，M2 dense 合同要求先拆分。
   - `ffn.kind`、`experts`、`top_k` 是否正确。
   - 代表层默认是第一个 `full_causal` 层；`--layer N` 可覆盖。
3. **选测量切片并补成本**
   - dense：`profiling/cases/dense_layer_qwen35_geometry.json` 的 8 条
     （projection/attention/dense_ffn/block × prefill/decode）；
   - CPU 完整 block：`profiling/DENSE_BLOCK_MEASUREMENT.md`；
   - CUDA：`profiling/DENSE_CUDA_MEASUREMENT.md`；
   - MoE：`profiling/MOE_MEASUREMENT.md`（独立子图，不能直接当 `.6` 整层价）；
   - 把 `<id>.costs.template.json` 的 `null` 逐条换成实测值、签名和来源。
     新形状重新测；不插值、不把 joint 时间乘系数分给每个任务。
4. **构造 invocation**
   复制 `fcd-prefill-3.json` / `fcd-decode-1.json`，改 `id`、`layer`、
   `layer_file`、`costs_file`、`requests`、`hardware`、`state_in`。
   容量按目标机器给，不要沿用 fcd 的 4096 B。
5. **验收**（见 §4）后再把文件放进 `mapping/examples/m2/`。

## 3. 哪些量能解析、哪些必须测

| 量 | 来源 | 说明 |
|---|---|---|
| `block_count`、`hidden_size`、`intermediate_size` | GGUF metadata | 直接读 |
| `heads`、`kv_heads`、`head_dim` | GGUF metadata | `key_length`/`value_length` 不一致时拒绝（M2 只有一个 head_dim） |
| `rope_dims`、`rope_base`、`norm_epsilon` | GGUF metadata | 只记录；dense 合同本身只支持 normal RoPE |
| `experts`、`top_k`、专家/共享 FFN 宽度 | GGUF metadata | `expert_count`、`expert_used_count`、`expert_feed_forward_length`、`expert_shared_feed_forward_length` |
| 每层 attention 类型 | GGUF tensor 名 | 含 `ssm_*` → `recurrent_linear`；含 `attn_q/k/v` → `full_causal` |
| 每层 FFN 类型 | GGUF tensor 名 | 含 `ffn_gate_exps`/`ffn_gate_inp` → `moe` |
| weight shape/dtype/字节 | GGUF tensor 目录 | 精确 payload 字节，不含容器 padding |
| `vocab_size`、global/total 权重字节 | GGUF tensor 目录 | 只用于容量换算 |
| `hidden_bytes_per_token`、`kv_bytes_per_token` | **合同推导**，不是实测 | 生成器按 M2 的 F32 activation + F32 KV 算；换 dtype 必须同步改合同 |
| `duration_ns`、`workspace_bytes` | **必须实测** | 没有任何元数据能给出 |
| `physical_signature`、`contract_hash` | **必须实测后生成** | 绑定执行边界 |
| threads、线程池行为、batch>1、搬运价格、CV | **必须实测** | 不是模型结构 |

手写 metadata 的最小字段（`--metadata`）：

```json
{
  "schema_version": "m2-model-metadata-0.1",
  "architecture": "toy",
  "model_name": "Toy-4B",
  "block_count": 8,
  "hidden_size": 2560,
  "intermediate_size": 9216,
  "heads": 16,
  "kv_heads": 4,
  "head_dim": 256,
  "rope_dims": 64,
  "rope_base": 10000000.0,
  "norm_epsilon": 1e-06,
  "full_attention_interval": 4
}
```

`full_attention_interval=4` 表示第 4、8、… 层是 full attention，其余记为
`recurrent_linear`（与 Qwen3.5 一致）。也接受 `layer_types` 列表显式指定每层
类型，或 `layers` 列表带 `tensors`（每个 tensor 给 `name`/`shape`/`dtype`/
`storage_bytes`）。MoE 在没有 tensor 目录时必须额外给
`expert_weight_bytes`、`router_weight_bytes`、`attention_weight_bytes`
（可选 `shared_expert_bytes`）。

## 4. 验收判据

1. `parse_layer_document(<id>.layer.json)` 通过；MoE 还要通过
   `validate_moe_layer`（`build_model` 生成 `.6` 时已按五段链和逐专家字节构造）。
2. 用真实 invocation 跑 `parse_invocation_document` + `validate_invocation`：
   `layer.id` 与 invocation 的 `layer` 一致；`layer_file`/`costs_file` 能解析；
   `state_in` 的记录与 `requests` 一一对应。
3. 成本表里每个需要的 `(kind, device, phase, batch, tokens, visible_history)`
   都有条目；没有 `null`；实测条目带完整签名与来源；CV 闸门默认 <= 0.1
   （`--allow-unstable` 只能产出 validation-only）。
4. 独立联合图与分阶段之和的误差在事先声明的阈值内；否则只能写
   `calibration_claimed=false`，不能把「能跑」说成「已校准」。
5. 未知 workspace 不得当 0；不得跨形状复用价格；不得把整层时间分给每个宏任务。
6. 生成器本身：`profiling/tests/test_model_swap.py` 覆盖 GGUF 与 metadata 两种
   入口；GGUF 文件不存在时该用例跳过。

## 5. 每模型估算工作量

以下**不是实测**，只是按 `profiling/` 脚本的默认协议和 suite 条数估的
order-of-magnitude；本仓库没有为这张表跑过 GPU/benchmark。默认协议：
dense/macro/native 为 warmup 5 + samples 20（每 case 25 次完成），MoE 为
warmup 16 + samples 31（每 case 47 次）。

| 模型类别 | 需要新增的 microbench | 完成次数 | 估计机器时间 |
|---|---|---|---|
| Dense、全层 `full_causal`、<=4B、CPU 单层 | `dense_layer_qwen35_geometry.json` 8 条 + `qwen35_4b_layer3_native.json` 18 条 + `qwen35_4b_layer3_macro_cores.json` 4 条 | 8×25 + 18×25 + 4×25 = 750 | 0.5–1.5 h（含 native 构建与 export；首次构建另计） |
| 同上，加单 GPU | 再加 `dense_layer_cuda_qwen35_geometry.json` 8 条 + CUDA memory 8 条 | 再加 8×25 + 8×4 = 232 | 再加 0.5–1 h |
| MoE（如 35B-A3B 的 full-attention 层） | dense attention/projection 约 6 条 + `moe_public_ffn_cpu.json` 2 条 + `moe_public_ffn_cpu_shared.json` 2 条 | 约 150 + 2×47 + 2×47 = 338 | 2–4 h（含路由 trace 导入、`moe_rows_reference` 验证；不含专家重排实验） |
| Qwen3.5 这类 hybrid 的**整栈** | 只有每第 4 层是 `full_causal`；SSM 层没有 M2 执行入口 | 0（不可测） | 阻塞：只能校准代表层，不能声称整模型 |

读脚本规模的依据：`run_dense_layer_measurements.py` 291 行、`run_dense_cuda_measurements.py`
393 行、`run_moe_measurements.py` 203 行、`run_measurements.py` 328 行、
`run_macro_measurements.py` 263 行；suite 条数来自 `profiling/cases/*.json`。
新增一个**形状**（例如 prefill 16 token）只需在该形状上重跑对应 case，成本近似
线性增加；新增一个**模型**在结构上不增加条数，只换几何和权重字节。

## 6. 当前模型族的坑（生成器会记录，但不会替你修）

- **Qwen3.5 是 hybrid**：`full_attention_interval=4`，其余层是 `ssm_*`
  （`recurrent_linear`）。M2 当前只实现 `full_causal` 执行；生成器默认只给第一个
  full-attention 层做 layer 骨架，geometry 里保留全部层类型。
- **Q/gate 打包**：Qwen3.5 的 `attn_q` 输出宽度是 `2*heads*head_dim`，gate 交错
  在里面。M2 dense 合同声明独立的 Q 与 `gate_proj`；生成器把它标为
  `W_q_gate_packed` 并写进 `weight_layout`，做 physical 层前必须拆分。
- **RoPE**：GGUF 的 `rope.dimension_sections` 是 M-RoPE 分区；dense 合同只支持
  normal adjacent-pair RoPE。生成器只记录 `rope_dims`/`rope_base`，不声称等价。
- **KV dtype**：GGUF 可能记录 F16 KV；M2 合同用 F32 KV。`kv_bytes_per_token`
  是按合同推的，不是 GGUF 的原始值。
- **MoE `.6`**：只接受 `synthetic_layer` 成本；实测 MoE FFN 是独立子图，不能
  直接填进整层表。

## 7. 第二个模型该选哪一类

**选全层 `full_causal` 的稠密小模型**（≤4B，未打包 Q/gate 更好），不要选另一个
hybrid 或另一个 MoE：

- 稠密小模型生成出的 layer 直接落在 M2 当前实现的 `full_causal + dense` 切片上，
  8 条 CPU 测量就能闭环，验收链路最短。
- Qwen3.5 hybrid 的 3/4 层是 `recurrent_linear`，M2 没有执行入口；换它只能再校准
  一个代表层，不能覆盖整模型。
- MoE 的 `.6` 只收合成成本，实测数字进不了表；选 MoE 会把「测完就能用」变成
  「测完还要改边界」。
