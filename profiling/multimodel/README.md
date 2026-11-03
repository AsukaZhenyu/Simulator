# 多模型、多设备验证

这个目录回答一个问题：**同一个成本模型，换模型、换机器之后还准不准。**

它不重复实现测量协议。测量仍然是 `profiling/` 里已有的套件，这里只做三件事：
固化"哪几个模型"、"每台机器是什么"、"数据与判据是什么"，并在测量之前把
**字节身份**和**结构身份**钉死。三件都通过，才允许开始测。

## 目录

| 文件 | 作用 | 会不会随设备变 |
|---|---|---|
| `models.json` | 五个模型的仓库、文件名、commit、字节数、SHA-256、从 GGUF 头读出的几何 | 不变 |
| `fetch_models.py` | 按 manifest 下载并逐字节校验 SHA-256，可断点续传 | 不变 |
| `run.py` | `survey` / `fetch` / `structure` / `status` 三个门加一个查看 | 不变 |
| `devices.py` | 只靠观察写设备描述（核、内存、GPU、OS），观测不到就写 `null` | 不变 |
| `STANDARD.md` | 验收判据与预注册的预期，**测量之前写定** | 不变 |
| `data/shapes.json` | 成本表要填的形状矩阵（batch、prefill tokens、decode history） | 不变 |
| `data/prompts.json` | 输入文本的来源；token 数按各模型分词器实测，不按字符估 | 不变 |
| `devices/<id>.json` | 这台机器的实际描述 | **每个设备分支不同** |
| `campaigns/<date>/` | 这次战役的结构报告与小体积汇总 | **每个设备分支不同** |

权重不进仓库：4-bit GGUF 是 0.4–9.5 GB，GitHub 单文件 100 MB、单资产 2 GB、
免费 LFS 1 GB 都放不下。所以仓库里放 manifest，字节由每台机器自己拉并用
SHA-256 证明是同一份。

## 三道门

```text
PYTHONPATH=. python -m profiling.multimodel.run survey --out devices/<id>.json
PYTHONPATH=. python -m profiling.multimodel.run fetch
PYTHONPATH=. python -m profiling.multimodel.run structure --out campaigns/<date>/structure
```

1. **`survey`**：写 `devices/<id>.json`。只写能观测到的：CPU 型号、物理核、
   逻辑处理器、内存字节、`nvidia-smi` 能答的 GPU。答不上来的字段是 `null`，
   不推断（`unknown_fields_are_null_not_inferred`）。它**不含任何价格**——
   价格属于产生它的那次测量，把它搬进另一台机器的报告就是 R3 要防的事。
2. **`fetch`**：下载到 `LLM_INFER_MODELS_DIR`（默认
   `~/.cache/llm-infer-simulator/models`），按 manifest 的 commit 解析，
   逐字节校验 SHA-256。已存在但不是这份字节时报错退出，**不静默覆盖**。
3. **`structure`**：用 `profiling/model_swap/build_model.py` 重新读本机这份文件
   的头与张量目录，把几何和 `models.json` 逐字段比对。manifest 里写 `null`
   的字段表示**不断言**，不是"必须为空"；写了值的字段必须完全相等（浮点用
   1e-5 相对容差）。

`status` 只列哪些模型在位、字节是否对。

三道门全绿之前测出来的东西不进报告。

## 为什么是这五个模型

三个 dense 是同一族（qwen3）、同一分词器、同一量化档，只差规模，所以误差曲线
的规模效应不会被"换了个族"污染。0.6B / 1.7B 是 3:1 GQA，4B 是 4:1，顺带跨过
kv-heads 这一档。

两个 MoE 在**两条轴上都不同**：Granite 没有共享专家、40 选 8；Qwen1.5-MoE 有共享
专家、60 选 4。如果只测一个，任何误差都能被"路由太稀疏"或"有共享专家"解释掉。

**没有选进来的**（理由不是跑不动，是成本表没有那一项）：

| 候选 | 实测证据 | 缺什么 |
|---|---|---|
| Qwen3.5 全系（本地五个文件） | `full_attention_interval: 4`，含 `ssm.*` | 36/40 层是 SSM，`spec.py` 只实现 `full_causal` |
| Granite-4.0-H-Tiny | `granitehybrid`，`head_count_kv` 只有第 5/15/25/35 层非零 | 36/40 层是 Mamba-2 |
| Trinity-Nano-Preview | `afmoe`，`attention.sliding_window: 2048` | 滑窗注意力的 KV 曲线与 full-causal 不同 |

llama.cpp 支持这些架构（`llama-arch.cpp` 里有 `GRANITE_HYBRID` / `AFMOE`），
所以它们不是"跑不起来"，是"跑起来也没有可装载的成本表"。见 `STANDARD.md` §1。

## 每台设备一条分支

`main` 上只有代码、manifest、判据和空目录。设备分支只改两个地方：

- `devices/<id>.json`（这台机器的事实）
- `campaigns/<date>/`（小体积报告：结构报告、汇总、预测与实测的对照）

原始轨迹、逐试次样本、模型文件都**不进仓库**（量级 10 GB 起，与
`profiling/**/results/` 的既有做法一致：原始证据留在本地归档）。

因为设备分支不碰 `main` 上的任何其他文件，合回来时不会有冲突。合并顺序无关。

## 三道门之后

测量与预测**不在这个目录里重写**，用已有的入口：

| 环节 | 入口 |
|---|---|
| 逐形状 CPU 成本（`projection` / `attention` / `dense_ffn` / `block`） | `profiling/run_dense_layer_measurements.py` + `profiling/export_dense_layer.py` |
| 宏任务成本与物理签名校验 | `profiling/run_macro_measurements.py` + `profiling/export_m2_macros.py` |
| MoE FFN 子图 | `profiling/run_moe_measurements.py`（**不能**填进 `m2-layer-0.6`，见 R9） |
| 整模型请求级对照（唯一的"真实模型 → 预测 vs 墙钟"闭环） | `profiling/real_moe_validation/runtime_window_campaign.py` + `runtime_window_predict.py` |

**已知缺口，别当成已完成**：上面最后一条目前只对固定的 OLMoE 成立，而且是
窗口价格目录、不是 `m2-layer` 成本表。**dense 一侧还没有"GGUF → 成本表 →
预测 vs 实测墙钟"的闭环**，`dac_validation/e2a.py` 用的是合成程序（`build_case`，
hidden 硬编码 256/512）。所以本轮 dense 能拿到的是"逐形状成本在真实几何上
是否仍然满足验收判据"，不是整模型端到端误差。这一点必须写进结果，不能含混成
"在真实模型上验证了预测"。

## 测试

```text
PYTHONPATH=. python -m unittest profiling.tests.test_multimodel
```

不联网、不需要模型、不需要 GPU。需要用真实 GGUF 的用例在文件不存在时跳过。
