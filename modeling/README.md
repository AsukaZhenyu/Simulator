# LLM-Infer 数学模型与模拟器

本目录把研究问题从旧推理引擎中解耦。当前 v0.8 先回答一个问题：

> 单请求顺序 decode 中，当层权重不能全部驻留显存或策略选择流式加载时，计算时间、权重传输时间、窗口容量和可重叠程度如何共同决定每 token 时延？

## v0.8 范围

已建模：

- 顺序执行的层，包括从 Qwen3.5 GGUF 识别出的 SSM/全注意力混合层；
- 单请求、单 token decode；
- 一个 GPU 计算引擎；
- 一个 DRAM→VRAM/H2D 传输引擎；
- 固定大小的层窗口；
- 传输与计算并发；
- VRAM 容量约束；
- 原生 CPU/GPU 静态卸载的简化对照；
- 窗口和带宽参数扫描。
- 从 GGUF 张量目录提取逐层精确存储字节数、张量数和层类型；
- 用 `2 × rank>=2张量元素数` 生成逐层 matvec FLOPs 代理量；
- 可选的逐层实测计算时间和传输时间覆盖值。
- 可复跑的 `llama-bench` resident/pipeline 校准命令；
- 可复跑的 CUDA pinned/pageable H2D 带宽测试；
- 可与权重流式服务重叠的一次/token流水线控制路径；
- 从多模型端到端测量拟合有效带宽、加载延迟和控制路径开销；
- 从 Nsight SQLite 自动提取稳态 graph、CUDA API、kernel 和 memcpy 分解；
- 把直接测得的 embedding D2H 从合成控制开销中拆为可调反事实参数；
- 对小窗口下暴露到关键路径的 host staging 服务建立有效修正；
- 从 GGUF 元数据计算 SSM 固定状态和随上下文线性增长的 Attention KV；
- 用 cached-depth `llama-bench` 分离 prefill 与真实上下文后的 decode；
- 把 SSM、Attention KV PCIe 和同步 backing-file I/O 拆成独立服务；
- 让 Attention KV 的驻留/驱逐由层窗口决定，并计算时延/容量 break-even；
- 校准误差与窗口外推失效范围报告。

暂未完整建模：长上下文 Attention 计算量、prefill、MoE路由、多请求、batch、DRAM/文件缓存竞争和数值正确性。v0.8 已显式加入2B小窗口的两级 KV backing-file 有效服务，但还不是 worker、FIFO、copy engine、staging ring、页缓存和SSD队列的细粒度模型。graph fragmentation 和同步仍由实测残差表示；4B/9B存储参数尚未校准。

## 数学模型

对层 `i`，输入：

```text
W_i：该层权重字节数
F_i：该层每 token FLOPs
P_g：GPU有效算力
B_h：H2D有效带宽
L_h：单次H2D固定延迟
B_s：host pageable→pinned staging有效带宽
K_o：host staging能够完全被隐藏的窗口阈值，当前标定为4
B_r：状态往返的有效带宽
B_c：KV backing-file 命中文件缓存时的有效带宽
B_u：超过文件缓存 knee 后的有效带宽
M_c：单向 KV 文件缓存工作集 knee
S_t：流水线路径一次/token固定控制开销
E_t：S_t 中由 Nsight 直接测得的 embedding D2H 服务时间
R_t：S_t 中由 Nsight 直接测得的状态/缓存CUDA copy服务时间
α_e：embedding fallback比例，当前实现为1，完全消除为0
S_i：可选的遗留逐层调度开销，默认0
K：窗口层数
c：上下文token数
c_i：可选的逐层实测计算时间
d_i：可选的逐层实测传输时间
```

每层服务时间为：

```text
C_i = c_i（存在实测值时），否则 F_i / P_g
D_i = d_i（存在实测值时），否则 W_i / B_h + L_h
```

Qwen3.5 的循环层状态使用 `llama.cpp` 中的 FP32 存储；全注意力层 KV 默认按 F16 建模：

```text
M_ssm,layer = [(d_conv-1) × (d_inner + 2×n_group×d_state)
               + d_state×d_inner] × 4
M_kv,layer(c) = c × n_kv_head × (d_key+d_value) × 2
M_ssm = ΣM_ssm,layer
M_kv(c) = ΣM_kv,layer(c)
T_ssm = 2 × M_ssm / B_r
```

当 `K < N_layers` 时，稳态下每个 token 需要重新流式加载全部层权重，并在层驱逐/重载时让 Attention KV 经过 GPU、pinned DRAM 和 backing file。`K=N_layers` 不驱逐层，因此 Attention KV 跨 token 留在GPU；当前实现仍对 SSM 状态做固定往返：

```text
P_control  = S_t - E_t - R_t + α_e × E_t
T_control  = ΣC_i + P_control + T_ssm + N_layers × S_i
H_stage    = ΣW_i / B_s
φ(K)       = clip((K_o - K) / (K_o - 2), 0, 1)
T_kv,pcie  = 2×M_kv(c)/B_r                     （仅K<N）
T_storage  = 2×[min(M_kv,M_c)/B_c
                + max(M_kv-M_c,0)/B_u]          （仅K<N）
T_transfer = ΣD_i + φ(K)×H_stage + T_kv,pcie + T_storage

理想重叠下界：T_lower = max(T_control, T_transfer)
无重叠上界：  T_upper = T_control + T_transfer
```

当 `state_placement=resident` 时，SSM 与 Attention KV 都不往返，但容量必须为全部状态预留。两级存储参数是本机 Windows 文件缓存、同步 `_read/_write` 与SSD的合成有效值，不是NVMe标称带宽。

`φ(K)` 是从当前实现机制得到的低维有效模型：调度器在完成第 `i` 层后释放 `i-1`，可用于隐藏 host staging 的前视距离约为 `K-2`。因此 `K=2` 没有有效前视，2B 实测暴露完整的 65.48 ms host staging 服务；`K>=4` 视为已隐藏。`K=3` 的线性插值和这一参数向 4B/9B 的迁移都还只是待验证假设。`K=1` 存在更严重的病态执行路径，不由该公式解释。

容量约束也随状态策略变化。`roundtrip` 只要求当前窗口的权重和状态同时可用；`resident` 要求全部状态加当前权重窗口同时常驻：

```text
M_required,roundtrip = max_window Σ(W_i + M_state,i)
M_required,resident  = max_window ΣW_i + M_state(c)
```

解析器还直接求两个边界：`T_control(c)=T_transfer` 对应的状态时延 break-even，以及 `M_required,resident<=M_available` 对应的最大常驻上下文。

当所有层能够常驻时，稳态权重流式传输量为0，但流水线路径的 `S_t` 仍然存在。窗口容量采用连续 `K` 层权重和的最大值，并从VRAM中预先扣除全局张量、KV和计算工作区。这里的“窗口大小”既是容量参数也是策略参数：即使显存足够，选择 `K < N_layers` 仍表示主动采用流式加载。

离散事件模拟器不直接假设完全重叠，而是显式调度：

```text
LOAD_WEIGHT(layer) on H2D
COMPUTE(layer)     on GPU
PIPELINE_CONTROL   on CONTROL
STATE_ROUNDTRIP    on CONTROL when state_placement=roundtrip
ATTENTION_KV_PCIE_EXPOSED on STATE_IO when K < N_layers
KV_STORAGE_IO_EXPOSED on STORAGE when K < N_layers
EVICT(layer)       after compute
HOST_STAGING_EXPOSED on aggregate critical path when K < K_o
```

因此模拟结果应位于解析下界和无重叠上界之间。

## 张量级内核（`llm_infer_model/tensor/`）

上面那套是**逐层、浮点秒**的 v0.8 模型。另有一个**张量级、整数纳秒**的内核，服务 `mapping/` 的映射与搜索：

| 模块 | 职责 |
|---|---|
| `tensor/spec.py` | 不可变契约对象、输入校验与指纹；M0 不建模的东西直接拒收，不做近似 |
| `tensor/engine.py` | 状态、四个动作（`COPY_H2D` / `COMPUTE` / `EVICT` / `ADVANCE`）、固定计划回放 |
| `tensor/layer_costs.py` | 把上面的逐层成本方法桥接成内核要的 `Costs` |

三条边界值得记住：

- **单位**：内核一律整数纳秒，本目录其余部分一律浮点秒。转换只发生在 `layer_costs` 里，按 `round(seconds * 1_000_000_000)`；旧 API 的秒单位不变。
- **方向**：`tensor_mapping → llm_infer_model` 单向。本包不导入 `mapping`，所以它能在没装 `mapping` 的条件下独立导入并回放一份计划。
- **`layer_costs` 不是 `ModelConfig` → GGML 图的转换器**：真实一层含多个算子，把整层成本摊给其中一个小矩阵乘会得到看起来合理但没有物理含义的数，所以「哪个算子就是这个计算节点、哪份权重就是这份权重张量」必须由调用方显式给出，它只负责核对（ID、覆盖范围、权重字节数）而不猜。旧模型独有的 KV 分层存储、SSM 状态往返、每 token 控制开销、host staging 等本轮**未迁移**，遇到非零值逐项点名并拒绝。

测试在 `tests/test_tensor_layer_costs.py`（19 项），其中一项把整个场景在**屏蔽 `tensor_mapping`** 的子进程里重建，用来证明内核确实不依赖 `mapping`。

## 安装

只需要 Python ≥3.10 和 pip，不需要 conda 或 uv。

核心模型（`model` / `analytical` / `simulator` / `calibration` / `nsys_analysis` / `cli`）
**只用标准库**；只有 GGUF 结构提取需要可选的 `gguf` 依赖。

```powershell
cd modeling
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[gguf]"
```

`.[gguf]` 拉入 `gguf` 及其传递依赖（numpy、PyYAML、requests、tqdm）。只跑核心模型
可以改用 `pip install -e .`，此时 GGUF 相关的测试会带原因跳过、相关命令会给出安装
提示，不会静默降级。

下面所有命令里的 `python` 都指 `.venv\Scripts\python`（或先激活 `.venv`）。

### 和 `mapping/` 一起装

`mapping/` 依赖本包的张量内核 `llm_infer_model.tensor`（`spec` / `engine` /
`layer_costs`：图与资源的校验、状态与合法动作、以及把本目录的闭式层成本换算成整数
纳秒）。**两个包要在同一次 pip 调用里一起装**，在 Simulator 根目录执行：

```powershell
python -m pip install -e ./modeling -e ./mapping
```

单装 `./mapping` 会让 pip 去 PyPI 找 `llm-infer-model` 然后失败。这是本仓库的本地包
依赖，不是第三方依赖：`modeling/pyproject.toml` 的 `dependencies` 仍为空，张量内核
也**只用标准库**。整套张量执行规则只有一份实现，就在本包里，`mapping/` 侧不保留副本
（`tensor_mapping/spec.py` / `engine.py` 只剩转出名字的兼容层）。

跑测试：

```powershell
.venv\Scripts\python -m unittest discover -s tests    # 57 项
```

不需要 GPU、不需要 1.28 GB 的模型文件，也不需要预先生成 `outputs/`。装了 `.[gguf]`
时 57 项全跑；只装核心包时 GGUF 相关项带原因跳过。

## 运行

在本目录执行：

```powershell
python -m llm_infer_model analyze `
  --config configs\historical_9b_approx.json

python -m llm_infer_model simulate `
  --config configs\historical_9b_approx.json `
  --trace outputs\historical_9b_trace.csv

python -m llm_infer_model sweep `
  --config configs\historical_9b_approx.json `
  --windows 1,2,4,8,16,32 `
  --bandwidth-gbps 3,6,12,24,48 `
  --output outputs\historical_9b_sweep.csv
```

### 一键重建 outputs/

`outputs/` 不入库，克隆后用一条命令重建全部**免 GPU** 产物：

```powershell
python -m llm_infer_model regenerate --outputs outputs
```

它做三件事：从 measurements 重新拟合 → 重建 `configs/calibrated_rtx4070/` 与拟合报告；
对 `generated` 和 `calibrated_rtx4070` 两套配置各跑 analyze / simulate / trace / sweep；
再跑 embedding 反事实和状态驻留两个对照实验。

默认使用仓库自带的示例测量数据 `tests/data/measurements_rtx4070_20260821.json`
（一次真实 RTX 4070 测量）。要换成自己机器的数据，先用 `benchmark-llama`、
`benchmark-h2d`、`analyze-nsys` 采集，再用 `--measurements` 传进来。

`regenerate` 不重跑需要 GPU 的原始测量，也不重建 `outputs/gguf/`（需要 1.28 GB 模型）
和 `outputs/nsys/`（需要 Nsight trace）；这两者要用 `extract-gguf` 和 `analyze-nsys`。

### 从本地 GGUF 建立真实层结构

GGUF 解析用 PyPI 的 `gguf` 包（`pip install -e ".[gguf]"`）。要改用一个 llama.cpp
源码树里的 `gguf-py`，用 `--gguf-python-path` 指向它的目录。

```powershell
python -m llm_infer_model extract-gguf `
  --model ..\design\Qwen-Series-GGUF-Q4_K_M\Qwen3.5-9B-Q4_K_M.gguf `
  --output outputs\gguf\qwen35_9b_q4_k_m_manifest.json `
  --layer-csv outputs\gguf\qwen35_9b_q4_k_m_layers.csv
```

然后以 9B 总层计算时间 48 ms/token 作为暂定参考，生成结构配置：

```powershell
python -m llm_infer_model config-from-gguf `
  --manifest outputs\gguf\qwen35_9b_q4_k_m_manifest.json `
  --template configs\historical_9b_approx.json `
  --reference-manifest outputs\gguf\qwen35_9b_q4_k_m_manifest.json `
  --reference-compute-ms 48 `
  --output configs\generated\qwen35_9b_q4_k_m.json
```

### 校准当前机器

下面两个命令分别采集原生全显存 decode 和 `K=4` 流水线。`benchmark-llama` 会保留命令、原始样本和标准错误输出：

```powershell
python -m llm_infer_model benchmark-llama `
  --executable ..\design\release-b8705\llama.cpp-b8705-layer\build\bin\Release\llama-bench.exe `
  --model ..\design\Qwen-Series-GGUF-Q4_K_M\Qwen3.5-9B-Q4_K_M.gguf `
  --generation-tokens 128 --repetitions 5 `
  --output outputs\calibration\resident_9b.json

python -m llm_infer_model benchmark-llama `
  --executable ..\design\release-b8705\llama.cpp-b8705-layer\build\bin\Release\llama-bench.exe `
  --model ..\design\Qwen-Series-GGUF-Q4_K_M\Qwen3.5-9B-Q4_K_M.gguf `
  --window 4 --generation-tokens 8 --repetitions 3 `
  --output outputs\calibration\pipeline_9b_k4.json

# 在8192-token cached context之后只测decode
python -m llm_infer_model benchmark-llama `
  --executable ..\design\release-b8705\llama.cpp-b8705-layer\build\bin\Release\llama-bench.exe `
  --model ..\design\Qwen-Series-GGUF-Q4_K_M\Qwen3.5-2B-Q4_K_M.gguf `
  --window 4 --prompt-tokens 8192 --generation-tokens 16 --repetitions 2 `
  --output outputs\context_validation\2b_k4_ctx8192_g16_r2.json
```

CUDA Toolkit 自带的带宽测试可通过同一入口运行：

```powershell
python -m llm_infer_model benchmark-h2d `
  --executable "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\demo_suite\bandwidthTest.exe" `
  --memory pinned --start-mib 32 --end-mib 160 --increment-mib 16 `
  --output outputs\calibration\h2d_pinned.json
```

整理好测量文件后，一条命令生成硬件校准配置和误差报告：

```powershell
python -m llm_infer_model calibrate-configs `
  --measurements outputs\calibration\measurements_rtx4070_20260821.json `
  --output-dir configs\calibrated_rtx4070 `
  --report outputs\calibration\calibration_fit_rtx4070.json
```

Nsight 报告导出为 SQLite 后，可重复提取稳态分解。下面命令会自动排除首 token 的 graph capture 和最后一个没有后继边界的 token：

```powershell
python -m llm_infer_model analyze-nsys `
  --sqlite outputs\nsys\llminfer_2b_k24.sqlite `
  --generation-tokens 64 `
  --layer-count 24 --attention-layers 6 --ssm-layers 18 `
  --embedding-bytes 417177600 `
  --output outputs\nsys\qwen35_2b_k24_decomposition.json
```

生成 embedding D2H 反事实窗口对照：

```powershell
python -m llm_infer_model compare-embedding-fallback `
  --config configs\calibrated_rtx4070\qwen35_2b_q4_k_m_rtx4070.json `
  --windows 1,2,4,8,16,24 `
  --output outputs\counterfactual_no_embedding\qwen35_2b.csv
```

生成状态往返/常驻的上下文与窗口对照：

```powershell
python -m llm_infer_model compare-state-residency `
  --config configs\calibrated_rtx4070\qwen35_2b_q4_k_m_rtx4070.json `
  --contexts 0,512,2048,8192,32768,131072,262144 `
  --windows 4,8,16,24 `
  --output outputs\state_residency\qwen35_2b.csv

python -m llm_infer_model validate-context `
  --config configs\calibrated_rtx4070\qwen35_2b_q4_k_m_rtx4070.json `
  --specification outputs\context_validation\qwen35_2b_context_validation_spec.json `
  --report outputs\context_validation\qwen35_2b_context_validation_report.json `
  --csv outputs\context_validation\qwen35_2b_context_validation.csv
```

生成配置中的某一层可加入下面两个字段。存在时，解析模型和模拟器会直接采用实测时间：

```json
{
  "name": "blk.0",
  "kind": "ssm",
  "weight_bytes": 142394112,
  "flops": 436797440,
  "measured_compute_seconds": 0.00142,
  "measured_transfer_seconds": 0.02471
}
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

## 配置的定位

- `toy_dense_decode.json`：完全合成，用于检查公式、事件顺序和容量约束。
- `historical_9b_approx.json`：用毕设叙述中的 `3.81 GB/token`、`5.66 GB/s` 和 `1—2 ms/layer` 构造的近似边界案例。它不是复现实验，也不能作为校准后的真实模型。
- `configs/generated/qwen35_*_q4_k_m.json`：层数、层类型、逐层权重字节、全局张量字节和状态几何来自真实 GGUF；计算时间由 9B 的 48 ms 假设进行统一标尺校准；带宽、workspace 和VRAM仍来自模板假设。它们是“结构真实、性能半校准”的 v0.3 配置，不是硬件测量结果。
- `configs/calibrated_rtx4070/*.json`：使用本机 RTX 4070 Laptop 的原生 resident decode、流水线 K=2/K=4/K=N、CUDA H2D 和 Nsight 分解生成。它们是当前 v0.8 的主要配置，并包含 embedding D2H、SSM往返、小窗口host staging，以及2B的窗口感知 Attention KV/两级 backing-file 分量。

该近似配置的首轮自洽检查为：

```text
解析传输时间       0.6738 s/token
解析理想吞吐上限   1.484 tokens/s
事件模拟时延       0.6753 s/token
事件模拟吞吐       1.481 tokens/s
简化静态卸载吞吐   3.781 tokens/s
追平静态卸载带宽   约 14.4 GB/s
```

这些数值是由近似参数推导出的 sanity check，不是新的实验结果。它们说明第一版模型已经能表达一个关键边界：在权重流量不变时，窗口只负责增加重叠，不能消除传输下限；进入强传输受限区后，继续增大窗口的收益会迅速消失。

## 本地 Qwen3.5 的结构结果

下表中的权重字节、层数和层类型来自 GGUF；计算时间是以 9B=48 ms/token 得到的统一有效吞吐标尺；传输与吞吐使用 5.66 GB/s、20 us/层和 `K=4`。因此吞吐是模型输出，不是实测值。

| 模型 | 层结构 | 流式层权重 | 全局张量 | 计算代理 | 传输服务时间 | K=4模拟吞吐 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B Q4_K_M | 18 SSM + 6 Attention | 0.853 GB | 0.417 GB | 9.526 ms | 151.132 ms | 6.601 token/s |
| Qwen3.5-4B Q4_K_M | 24 SSM + 8 Attention | 2.208 GB | 0.521 GB | 24.765 ms | 390.832 ms | 2.554 token/s |
| Qwen3.5-9B Q4_K_M | 24 SSM + 8 Attention | 4.263 GB | 1.407 GB | 48.000 ms | 753.830 ms | 1.324 token/s |

由此可以先得到一个不依赖旧实验好坏的判断：若 9B 每 token 确实重新搬运 4.263 GB 层权重，在 5.66 GB/s 下仅传输就需要约 0.754 s；优化窗口只能把计算藏到传输后面，无法突破约 1.33 token/s 的带宽上限。真正值得研究的方向应当减少每 token 搬运字节、增加跨 token/请求复用，或提升有效传输带宽，而不是只继续增大双缓冲窗口。

上面的 `K=N_layers` 纯计算值只属于未校准的结构模型，没有计入输出词表投影、kernel launch、KV/SSM扫描和流水线切图开销，不能当成真实预测。

## RTX 4070 Laptop 的 v0.4 校准与 Trace 分解

原始测量位于 `outputs/calibration/measurements_rtx4070_20260821.json`。CUDA H2D 中位数为：

```text
pinned host memory   12.112 GB/s
pageable host memory  5.660 GB/s
```

有意思的是，旧实验使用的 5.66 GB/s 几乎正好等于本机 pageable H2D，而当前流水线已经使用 pinned staging，端到端性能仍显著低于 12.1 GB/s。v0.4 把权重流式服务和一次/token控制路径分开，多模型联合拟合得到：

```text
流水线有效带宽 B_effective       5.076 GB/s
权重加载固定延迟 L_effective     1.484 ms/layer
三模型 K=4 拟合相对 RMSE         2.47%
K=N额外路径 P_token             54.45 / 73.60 / 73.88 ms/token
```

| 模型 | 原生全显存实测 | K=4流水线实测 | K=4模拟 | 模拟时延误差 | K=N流水线实测 |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 124.398 t/s | 4.725 t/s | 4.898 t/s | -3.65% | 16.002 t/s |
| Qwen3.5-4B | 62.809 t/s | 2.112 t/s | 2.070 t/s | +2.04% | 11.170 t/s |
| Qwen3.5-9B | 37.501 t/s | 1.123 t/s | 1.126 t/s | -0.26% | 9.946 t/s |

`K=N` 不再流式搬运层权重，却仍比原生全显存慢 3.8—7.8 倍。v0.3 曾把差值除以层数，得到看似稳定的 2.269、2.300、2.309 ms/layer；Nsight 证明这个解释不成立。差值里包含一次/token的大块 embedding 回传、每个 SSM 层的状态往返、graph fragmentation 和大量同步，不能作为可泛化的逐层常数。

稳态 Nsight 结果如下。D2H/H2D 均为每 token 实际 GPU memcpy 字节；embedding 列与 GGUF 中 `token_embd.weight` 的字节数逐字节相等，并且每 token 恰好出现一次。

| 模型 | CUDA graphs/token | 公式检查 | embedding D2H | 总D2H/token | 总H2D/token | stream sync/token |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 79 | `3×24+6+1` | 397.85 MiB | 418.49 MiB | 19.27 MiB | 323 |
| 4B | 105 | `3×32+8+1` | 497.31 MiB | 549.26 MiB | 50.26 MiB | 425 |
| 9B | 105 | `3×32+8+1` | 545.62 MiB | 597.57 MiB | 50.27 MiB | 425 |

2B 的 node-level 对照进一步排除了“pipeline kernel 计算变慢”：

| 2B路径 | 稳态时延 | kernel GPU时间 | kernel/memcpy活跃 | 空闲或未跟踪 | graphs/token |
|---|---:|---:|---:|---:|---:|
| 原生 resident | 8.52 ms | 7.48 ms | 7.62 ms | 0.90 ms | 1 |
| pipeline K=24 | 62.94 ms | 7.46 ms | 43.33 ms | 19.61 ms | 79 |

两条路径的 kernel GPU 时间几乎相同；额外时延主要是约 35.9 ms/token 的 D2H/H2D 服务以及约 18.7 ms/token 的新增空隙/同步。node-level tracing 本身使绝对时延略有扰动，因此吞吐校准仍使用未插桩 `llama-bench`，Trace 只用于组成解释。

v0.4 因而使用两个可重叠服务路径：

```text
P_token(model) = T_pipeline,K=N - T_native_resident
T_control      = T_native_resident + P_token
T_stream       = layer_weight_bytes / B_effective + N_layers × L_effective
T_decode       >= max(T_control, T_stream)
```

完整报告和可重复分析命令见 `outputs/nsys/README.md`。

### v0.5：去除 embedding D2H 的保守反事实

反事实只减去 Nsight 直接测到的 `token_embd.weight` D2H GPU 时间，保留所有状态往返、graph split和同步空隙：

| 模型 | 移除时间 | 当前K=N | 反事实K=N | 加速 |
|---|---:|---:|---:|---:|
| 2B | 33.91 ms/token | 16.00 t/s | 34.99 t/s | 2.19× |
| 4B | 40.38 ms/token | 11.17 t/s | 20.35 t/s | 1.82× |
| 9B | 45.06 ms/token | 9.95 t/s | 18.02 t/s | 1.81× |

但在 `K=4/8/16`，权重流式路径仍然远慢于控制路径，预测吞吐基本不变。这意味着修复 embedding 回传值得做，但只解决全驻留/大窗口路径；低显存研究仍必须减少每 token 的层权重搬运量。完整表见 `outputs/counterfactual_no_embedding/README.md`。

### v0.6：小窗口 host staging 暴露

2B 的 `K=2` 与 `K=4` graph-level trace 具有相同的图数量、传输字节和同步次数，但空闲/未跟踪时间显著不同：

| 2B Trace | 时延 | graphs/token | H2D/token | D2H/token | sync/token | GPU复制活跃 | 空闲/未跟踪 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=2 | 286.51 ms | 79 | 835.60 MiB | 421.63 MiB | 407 | 155.41 ms | 131.10 ms |
| K=4 | 208.98 ms | 79 | 835.60 MiB | 421.63 MiB | 407 | 160.16 ms | 48.81 ms |

这排除了“`K=2` 搬了更多数据”这一解释。代码中的 `AsyncLayerLoader` 只有一个 worker、一个 FIFO work queue 和四个 staging buffer；调度器在计算第 `i` 层后释放 `i-1`，使有用预取距离约为 `K-2`。`K=2` 因而没有可隐藏 host staging 的前视距离，额外等待表现为空隙而不是额外 PCIe 字节。

未插桩 `llama-bench` 的 `K=2-K=4` 时延差是 65.48 ms。用完整层权重字节除以该差值得到有效 host-staging 带宽：

```text
B_host_stage = 0.852688 GB / 0.065481 s = 13.02 GB/s
```

v0.6 在 `K=2` 暴露完整 host-staging 服务，在 `K>=4` 将其视为被隐藏，并把 `K=3` 作为待验证的线性插值。更新后的验证为：

| K | 实测 | 模拟 | 结论 |
|---:|---:|---:|---|
| 1 | 超过150秒未完成，已中断 | 3.608 t/s | 当前实现存在病态路径，模型无效 |
| 2 | 3.608 t/s | 3.712 t/s | host-staging修正后，时延误差-2.79% |
| 4 | 4.725 t/s | 4.904 t/s | 流式服务拟合点，时延误差-3.65% |
| 8 | 4.942 t/s | 4.904 t/s | 外推时延误差+0.77% |
| 24 | 16.002 t/s | 16.002 t/s | 使用该模型的K=N控制路径测量校准 |

`K=2` 已从约 26% 的时延低估收敛到 2.79%，但这只是用同一个 2B 测量点标定后的闭环检查，不是独立验证。`K=3`、4B/9B 小窗口和 `K=1` 均不能据此宣称已经预测准确。

### v0.7：上下文相关状态字节（已被v0.8修正执行路径）

GGUF 元数据和 `llama.cpp` 状态分配公式给出的固定 SSM 状态为：

| 模型 | SSM层 | 固定SSM状态 | Trace非embedding复制服务 | 有效状态带宽 |
|---|---:|---:|---:|---:|
| 2B | 18 | 19.27 MiB | 3.266 ms/token | 12.373 GB/s |
| 4B | 24 | 50.25 MiB | 8.366 ms/token | 12.596 GB/s |
| 9B | 24 | 50.25 MiB | 8.260 ms/token | 12.758 GB/s |

2B 的单层语义状态由1,048,576字节 recurrent matrix 和73,728字节 convolution state组成；18层合计20,201,472字节，几乎逐字节等于 K=N Trace 的 H2D 状态流量。4B/9B 的52,690,944字节也与50.26 MiB/token H2D吻合。因此状态结构来自模型语义，只有传输服务率是 Trace 校准参数。

v0.7 曾假设任何流水线窗口都会逐 token 加载/保存全部 SSM 与 Attention KV，并据此得到下表。它保留为研究演进记录，但其中时延 break-even 已失效：

| 模型 | 状态追上权重流式服务 | K=4状态常驻最大上下文 | K=N、context=0驻留加速 | K=N状态常驻最大上下文 |
|---|---:|---:|---:|---:|
| 2B | 71,045 tokens | 559,536 tokens | 1.055× | 502,245 tokens |
| 4B | 75,549 tokens | 201,385 tokens | 1.103× | 142,791 tokens |
| 9B | 153,184 tokens | 166,114 tokens | 1.090× | 53,082 tokens |

容量上界仍可作为“全部状态常驻”的反事实参考，但时延列忽略了小窗口的同步文件读写，并错误地让 `K=N` 搬运 Attention KV，不能继续作为当前结论。

### v0.8：窗口感知 KV 驱逐与 backing-file I/O

cached-depth `llama-bench` 与成对 Nsight trace 否定了v0.7的统一状态假设：

| 路径 | context | 时延 | H2D | D2H | GPU活跃 | 空隙/未跟踪 |
|---|---:|---:|---:|---:|---:|---:|
| K=4 | 0 | 156.10 ms | 835.57 MiB | 421.60 MiB | 112.89 ms | 43.21 ms |
| K=4 | 8K | 225.78 ms | 931.60 MiB | 517.60 MiB | 125.54 ms | 100.24 ms |
| K=24 | 0 | 60.43 ms | 19.27 MiB | 418.49 MiB | 35.22 ms | 25.21 ms |
| K=24 | 8K | 58.75 ms | 19.31 MiB | 418.49 MiB | 36.19 ms | 22.55 ms |

`K=4` 每个方向新增约96 MiB，精确等于6个Attention层的8K F16 KV；时延增加69.68 ms，其中只有12.64 ms是新增GPU活跃，57.04 ms表现为空隙。`K=24`没有新增KV复制或时延。源码也给出相同生命周期：`transition_layer()` 在 `K==N_layers` 时直接返回，小窗口则同步执行 `save_kv()`/`load_kv_async()`，其中 backing storage 最终调用文件 `_write/_read`。

2B `K=4` 的两级有效参数为：缓存段3.648 GB/s、非缓存段0.598 GB/s、单向 cache knee 145.12 MiB（约12.4K context）。新的状态时延 break-even 约13,152 tokens，不再是v0.7的71,045。4K/8K/16K/32K预测时延误差分别为13.7%、9.6%、5.6%、2.2%；这些点参与了存储拟合，是闭环检查。独立的 `K=N` 测量与trace验证的是“全窗口不驱逐Attention KV”这一结构结论。

完整原始样本、统一测量—预测表、Trace差分、拟合边界与重跑命令见 `outputs/context_validation/README.md`。262K、4B/9B和并发请求仍然是外推，不可当成已验证吞吐。

## 第一阶段研究输出

第一阶段不追求“模拟器像真实硬件一样复杂”，而是生成三张边界图：

1. `带宽 × 窗口 → tokens/s`；
2. `计算/传输比 → 瓶颈区域`；
3. `VRAM容量 × 模型权重 → 可行域`。

如果一个机制在理想下界下都没有收益，就不进入真实实现。如果模型显示存在显著空间，再加入Trace校准和最小C++/CUDA验证。

embedding、`K=2` host-staging、状态几何和2B多上下文KV存储路径已经完成。下一步应把decode拆分结果迁移到4B/9B，并为prefill建立独立事件模型；随后评估真正减少主瓶颈——每 token 层权重与KV文件流量——的跨 token/请求复用策略。
