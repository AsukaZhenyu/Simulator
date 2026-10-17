# 执行变换性能校准合同

日期：2026-10-17。本文为代码只读审查和实验协议，不包含本轮硬件实测结果。结果应链接独立的、带时间与哈希的实验目录。

## 1. 本轮要证明什么

目标是检验：在相同数学运算、张量布局、设备、执行边界和初始驻留条件下，性能器预测的完成时间与新进程实测是否一致，以及是否能正确排序不同策略。

需要分开三个结论：

1. **功能正确**：变换前后数值等价，KV、路由和依赖正确。
2. **延迟校准**：已声明形状与执行方式的 heldout 延迟误差通过预先固定的门。
3. **容量校准**：逻辑张量、对齐、额外缓冲和后端工作区均已知，并与真实分配或容量边界匹配。

通过第 1 项不代表通过第 2 项；GPU 工作区未知时可以报告延迟诊断，不能声称第 3 项通过。合成价格下的模拟完成不属于性能准确性证据。

## 2. 现有入口与可复用范围

| 入口 | 可直接复用的能力 | 不能据此声称的能力 |
|---|---|---|
| `strategy_calibration/calibrate.py` | 完整 Dense block，CPU 分段串行与 GPU 驻留联合图；profile/heldout 分离、环境和程序绑定、原始样本保存 | 新 FFN 分片、QK/softmax/PV、混合 CPU/GPU、预取、独立 P/D 系统 |
| `matched_mechanisms/experiment.py` | F32 专家 FFN 四阶段；Attention core/output；三种真实 FFN 混合路径；CE/SM 搬运；profile-only 冻结与独立进程验证 | 两个设备同时执行 FFN 中间维分片、完整 decoder FFN、分段 Attention、计算与预取重叠 |
| `matched_mechanisms/export_profiles.py` | 带精确端口、fixture SHA、形状、workspace、质量标记的测量侧表 | 自动将侧表填入 M2 或新执行变换 |
| `matched_mechanisms/replay_serial.py` | 已知 CPU 工作区的实测端口图，通过 Modeling 真正执行动作、依赖和工作区账本 | 新 Python 策略编译链已校准；GPU 容量已校准 |
| `native/matched_mechanisms/compute.cpp` | 实际 GGML F32 算子、完整路径 wall 时间、数值验证、显式 copy | 预训练模型内核、量化阶段、多线程性能规律、CUDA graphs |
| `native/matched_mechanisms/transfer.cu` | pinned CE、具体 UVA SM copy、CE H2D + SM D2H 双 stream、全部字节验证 | 通用共享链路公平分配规律、计算与搬运重叠规律、SSD 路径 |

可从 `Simulator` 目录调用已有入口，路径必须显式提供：

```powershell
python profiling/matched_mechanisms/experiment.py --suite <suite.json> --compute-binary <compute.exe> --transfer-binary <transfer.exe> --runtime-dir <matching-runtime-dir> --timeout-s 180 --output <new-directory>
python profiling/matched_mechanisms/verify.py --experiment <experiment-directory> --output <new-verification.json>
python profiling/matched_mechanisms/export_profiles.py --experiment <experiment-directory> --output <new-stage-port-profiles.json>
python profiling/matched_mechanisms/replay_serial.py --experiment <experiment-directory> --output <new-replay-directory>
```

不要用旧实验目录覆盖新实验。旧诊断与冻结资产应保持原样。上述命令是运行入口，不表示本轮已运行。

## 3. 与 10 月 17 日新变换的逐项对接

| 新 IR 动作 | 现有真实 probe | 对接判断与还缺的证据 |
|---|---|---|
| `ffn_gate/up/swiglu/down` | `gate/up/activation/down` | 数学运算候选对应，但必须重新匹配 H、分片宽度 I、q、端口布局、F32 和输出物化边界。不能把完整宽度延迟按比例切分 |
| `ffn_norm` | 完整 Dense probe 内部存在 norm | 没有同边界独立价格；需要单独测量 RMSNorm、epsilon 与 norm weight |
| `ffn_sum` | 无 | 需要真实地汇总 N 个 `[H,q]` 部分输出。包括 N=1 时仍存在的 IR 动作；不能自行当作零成本 |
| `ffn_residual` | 完整 Dense probe 内部存在 residual | 需要独立 residual add，且只能执行一次 |
| CPU/GPU 中间维分片 | 混合 probe 是整权重上 GPU、整行上 CPU、或 CPU 前三阶段到 GPU down | 这些是不同策略。必须实际运行两块分片、D2H/H2D 和 SUM，并验证完整 FFN 输出 |
| `attention_qk/softmax/pv` | 只有 `attention_core` | 无独立阶段价格，不能将 core 时间分摊成三份；需新阶段图及完整串行链 |
| `attention_output` | 独立 output 图 | 仅在 sigmoid gate、O projection、residual、尺寸及布局全部一致时可复用；MoE 无 gate frontend 不能复用有 gate 价格 |
| `attention_projection` | matched probe 从 post-RoPE Q/K/V 起步 | 未测。Dense 物理 frontend 含 RMSNorm/QKV/QKnorm/RoPE/可选 gate；MoE frontend 则无 norm/gate，二者需要分别绑定 |
| 启动预载 | 单次 H2D 和含拷贝 FFN 路径 | 能测独立搬运；仍需匹配整份策略的起止、初始驻留和真实依赖，并计入预载时间 |
| 空闲预取 | CE/SM 双向搬运 | 未包含计算竞争，不等于预取校准。需与当前计算同时运行且保存端到端时间和资源区间 |
| P/D 设备放置 | 独立 prefill/decode 和 KV 端口 | 未执行跨设备 P→D 链。需携带真实 KV，精确计入转移与后续 decode；当前不涵盖独立实例队列 |
| MoE 路由/缓存/驱逐 | probe 输入为已选出的 compact 行 | 可以校准专家计算，但没有测 router、gather/scatter/combine、路由揭示时点或缓存命中过程 |

### 3.1 特别注意 Attention 的 KV 物化边界

现有 native core 先把全长 `key_cache/value_cache` 复制到 appended 缓冲，再写入新增 K/V，随后计算 QK、mask、softmax、PV。`key_cache` 和 `key_appended` 是 head-major、容量为 history+q 的独立驻留缓冲。

新 `attention_stages.py` 的逻辑 KV 是按 position 的 F32 block；QK 可读历史与新增 K，PV 产生新增 KV block。二者数学意义可对应，但物理物化顺序、布局和分配不同。不能在未声明额外 layout/assembly 动作时，将旧 `attention_core` 的总时间当作新三阶段的价格或直接比较各阶段。

最小新 probe 应选择一种明确合同：

- 按新 block 布局执行真实 gather/assembly，并把成本归入声明的阶段；或者
- 采用连续 KV 的具体后端 lowering，显式保留 assembly 成本与存储，并说明它是该 lowering 的成本。

历史长度必须精确匹配。不能将 hist=0 的 Attention 数字复制到 hist=3。FFN 没有 KV 输入，可以声明 `history_axis=not_applicable_no_KV_input`，但这是语义等价声明，不能伪装成测过每个 history。

### 3.2 新变换的 provenance 仍为合成

`ffn_execution.py` 和 `attention_stages.py` 当前解析器仅接受 `provenance=synthetic_execution`，阶段 `workspace_bytes` 必须为非负整数。现有 GPU native 私有 scratch 为 `null`。因此现状下不能忠实地把 GPU 实测表自动填入新执行变换。

安全的第一步是保存真实测量侧表，建立相同端口/动作的独立延迟对照；CPU 已知 workspace 可以进入受控的 measured replay。后续若接入新变换，应新增明确的实测 provenance、binding 和未知容量状态，而非把测量写成 synthetic 或把未知 workspace 写成 0。

## 4. 每个比较必须绑定的合同

每个实验项至少保存下列身份。任意一项不匹配应拒绝“准确度比较”，最多保留为不同执行方式的诊断。

- **数学语义**：具体算子、SwiGLU 公式、norm epsilon、residual 次数、GQA 映射、QK scale、绝对因果 mask、RoPE、gate 与否、KV 写入规则。
- **几何**：B、q、history、H、I、heads、KV heads、head_dim；每个 FFN 分片的 start/end；SUM 的输入数量；每个专家的实际 token 行数。
- **数据**：dtype、量化格式、矩阵布局、stride、contiguous/materialized、张量字节数、seed 和输入/权重 fixture SHA。F32 证据不能覆盖 Q4_K/Q6_K。
- **后端**：GGML 源版本、二进制和 DLL SHA、CPU 线程与 affinity、GPU index/identity、CUDA graphs、MMQ/CUBLAS 与可见设备等实际继承环境。
- **执行边界**：融合或分阶段、每步是否 synchronize、host 调度、staging memcpy、allocation/init 是否计时、timer 类型与 completion wait。
- **初始/最终状态**：输入和权重在哪个设备、是否 warm-resident、KV 内容与长度、最终输出在哪个设备、允许保留哪些副本。
- **资源与容量**：设备/链路槽、并行许可、额外 staging、工作区 known/null/source、逻辑与物理容量结论的区分。
- **策略身份**：Python 源码、lowered IR、具体动作/依赖、编译器与性能器源码 SHA，以及 profile-only 模型 SHA。

分配和初始化通常可以从单任务稳态计时中排除；首次启动或预载策略若会真实产生这些动作，应单列或纳入端到端边界，不能仅因方便而隐去。

## 5. 因果路由与预取的特殊要求

路由由 Modeling/trace 提供，不得由 Mapping 在 router 完成前读取未来激活专家。静态知道模型有哪些专家是允许的，提前知道本次请求命中哪个专家是另一种 oracle 假设。

每个 MoE 策略对照应固定相同路由 trace 和权重/input fixture，记录：路由揭示时间、实际 compact 行、命中/未命中、每次加载/驱逐及字节、最终输出。可以分别评估 history-only 策略和显式 oracle 上界，但必须分栏报告。

启动预载的比较起点必须在第一笔预载提交前。若预载发生在计时起点之前，应命名为 warm-resident 策略，并另报准备成本和摊销轮数。空闲预取必须在真实依赖允许时发起；未到就绪状态不得开始使用。不能从“带预载的执行少做一次 copy”直接推出总时间更快。

## 6. 并发计时要求

串行任务相加与并发任务取 max 是两个不同模型。并发可能因共享 DRAM、PCIe、GPU SM 或 host 调度而降低单项速度。旧单项价格无法单独证明任意重叠可靠。

并发路径必须直接测端到端 wall：起点在必要工作提交之前，终点在所有输出和必要搬运完成之后。另保存 CPU 时间段、CUDA event 区间、依赖 event、stream 和每个 copy 的字节数。wall 对比 wall；不能用某个 CUDA kernel event 时间对比包含 host dispatch 的模拟总时间。

现有 transfer 的双 stream 是 CE H2D+具体 UVA SM D2H；它没有计算图，不能代表 GPU compute+CE prefetch。空 kernel/event 的观测值包含控制与等待，不应从每个任务中机械扣除。

## 7. 预先固定的验收与报告

新实验应在运行前保存 suite 和门，profile seed 与 heldout seed 不交叉，至少两次独立进程，各自保留全部逐轮样本。不能删除离群点或看完 heldout 后重设阈值。已有实验必须按其原门解读，不追溯改门。

建议本轮新 matched 实验沿用已有保守门：进程内 CV ≤ 0.15、进程中位数间 CV ≤ 0.20、heldout 绝对相对误差 ≤ 0.20。这是本轮工程接通门，不代表论文目标。结果若不通过，应列出失败项和误差来源，不将其导出为 accepted 成本。

每个原子阶段和端到端策略都报告：profile 预测、heldout 观测、带符号相对误差 `(pred-truth)/truth`、绝对相对误差、进程内与跨进程 CV、分层 bootstrap 区间及样本数。区间只能说明当前固定预测和机器条件下的不确定性，不能证明新形状泛化。

报告还应明确区分：

- **同形状独立进程验证**：检查运行稳定性与时间迁移。
- **不同输入 seed**：检查输入内容改变后的稳定性，不能代替新形状测试。
- **未测组合验证**：只用原子 profile 预测从未整体拟合过的分片/预取方案，最有助于检验调度模型。
- **新形状泛化**：如果查表拒绝插值，则标为未支持；不能用同形状 heldout 宣称已覆盖。

策略排序至少给出成对结果。若两策略 heldout 差异小于噪声区间，标为“无法区分”，不强行判胜负。对可区分的对照报告排序是否正确；选出预测最优策略后，报告真实 regret：`(真实选中时间 - 真实最佳时间) / 真实最佳时间`。本轮若有明确对照，可将可区分配对排序全正确、regret ≤ 10% 设为策略选择门；需在测量前写入 suite，不能事后补门。

缺失 probe、native 失败、环境/源码变动、未知工作区、数值错误、质量门失败都必须出现在完整报告中。“完整报告”可以得到否定结论，不意味着所有机制都已校准。

## 8. 最小新增 probe 顺序

1. **FFN 分片真实路径**：复用现有 F32 数据生成和四阶段，增加两个独立 materialized 分片、norm、SUM、residual；先固定串行执行以核对成本，再执行 CPU/GPU 同时分片，保留两种路径。测量分片宽度本身，而非缩放完整 FFN 时间。
2. **Attention 独立阶段**：同一 fixture 执行 QK→mask/softmax→PV→output 串行链；严格声明 KV assembly 的归属，分别覆盖 P3/h0 与 D1/h3，先单设备，再跨设备。
3. **预载与重叠**：同一份权重和输入对比 cold demand、startup preload、spare prefetch。所有路径以相同边界计时，保留 copy+compute 并发实测。
4. **P→D 持续状态**：先单层，真实生成并交接 KV，比较 Prefill GPU→Decode CPU 与同设备路径；之后扩展多层/多轮。
5. **MoE 外围**：固定真实 route trace，在已校准专家阶段外补 router、gather/scatter/combine 和 cache miss。不可仅测 compact FFN 就声称 MoE 全层校准。

今天可立即复用已有工具重测驻留 Dense 和旧 matched 边界，验证测量基础是否仍稳定；这些结果应明确列为基线。只有上述新增真实路径与新 IR 动作逐项匹配后，才能把“新执行变换的性能准确度已校准”作为结论。

## 9. 只读审查依据

- `strategy_calibration/README.md`、`calibrate.py`、`suite.json`。
- `matched_mechanisms/README.md`、`experiment.py`、`export_profiles.py`、`replay_serial.py`、`suite.json`。
- `native/matched_mechanisms/compute.cpp`、`transfer.cu`。
- `../modeling/llm_infer_model/m2/ffn_execution.py`、`attention_stages.py`。

本文件未执行硬件测量，未改写程序、旧实验、成本表或任何 fixture。

## 10. 本轮能力状态矩阵

下表以最终 `transforms-context-01` 为主，保留 `transforms-01`、`transforms-dual-01` 和已完成基线的诊断。全部新增路径都未通过完整稳定性门，因此没有新增机制被标为“完整校准”。“完整”只针对表中限定的边界，不能外推到整个系统。

状态定义：

- **完整校准（限定范围）**：语义与执行边界对齐，数值/覆盖/稳定性/heldout 误差门均通过，模型在 heldout 前冻结，证据可重算。
- **局部诊断**：有真实测量，但质量门未通过、计时/端口边界仅部分对齐，或仍未执行对应真实组合。必须写出是哪一种。
- **未测**：没有对应路径的真实样本，或 probe 仍在实现/构建，不能用合成运行补足。

| 能力及限定范围 | 当前状态 | 获得完整校准还必须出现的证据 | 本轮结果引用 |
|---|---|---|---|
| 单设备驻留 Dense block（旧 probe 边界） | 局部诊断：已有历史工具与实测，本轮尚待复核 | 新 manifest、数值/KV、profile/heldout、预声明门；不能借此覆盖新拆分 | 待填 |
| 专家 compact-row F32 gate/up/SwiGLU/down | 局部诊断：新分片真实阶段与输出通过数值，质量未通过 | 每个分片形状的稳定性门与接受成本 | `transforms-01/raw`；`baseline-component-analysis-final/report.json` |
| F32 Dense FFN norm + 四阶段 + SUM + residual | 未测：新增合同 | norm 参数、SUM 输入数量、残差一次、完整数值等价、各组件和端到端误差 | 待填 |
| CPU/GPU 中间维 FFN 分片，固定输入/输出设备 | 局部诊断：同 worker 组件上下文已对齐，真实分片/SUM/DRAM 输出；时间与 CV 仍有缺口 | 同边界稳定延迟与性能器实测 provenance 回放 | `transforms-context-01/analysis/report.json`；第 15 节 |
| CPU/GPU 分片真实重叠 | 局部诊断：v1 0/20、dual 19/20、context 20/20 host 区间重叠，kernel 重叠仍未证实 | GPU event/time line，保证实际 lowering 与预测重叠假设相同 | 三轮 `raw` 中 `diagnostic_completed_work_brackets` |
| 新 Attention QK→softmax→PV 分段 | 局部诊断：具体连续 KV lowering 数值正确，最新 4 项误差≤20%，CV 未通过 | 质量门与新 IR block 布局的逐动作绑定 | `transforms-context-01/analysis/report.json` |
| Attention output（sigmoid gate + O + residual） | 局部诊断：真实具体阶段已测 | 几何/layout 精确一致，稳定性通过；无 gate 变体另算 | `transforms-context-01/raw` |
| 启动预载总成本 | 局部诊断：每次三份 GPU 权重重新加载已计入 FFN 路径；完整 startup 策略仍未测 | Python 策略起点与原生路径绑定，稳定性门通过 | `transforms-context-01/analysis/report.json` 的 `hybrid/preload` |
| warm-resident 执行（准备在计时外） | 局部诊断：旧工具有此边界 | 明确命名，准备成本另列；不能当作 cold/preload 总成本的公平赢家 | 待填 |
| 空闲预取 + 计算重叠 | 未测 | 真实依赖、copy + compute 同时运行、端到端 wall、竞争和最终输出 | 待填 |
| 单层 Prefill→Decode 跨设备 KV 交接 | 未测 | 连续真实 KV 内容、跨设备搬运、相同 request 和最终输出位置 | 待填 |
| MoE router/gather/scatter/combine 与 cache miss | 未测 | 固定路由 trace、正确揭示时间、实际 compact 行与每次搬运、完整输出 | 待填 |
| GPU 真实容量边界 | 未测：私有 workspace 未知 | 后端完整额外分配与 scratch 可观察或有明确可证明上界；不得填 0 | 待填 |
| 多层、多轮、设备波动下端到端准确度 | 未测 | 对应完整真实执行样本；功能测试和单层阶段价格均不能替代 | 待填 |

报告发布时只依据最终原始产物更新“当前状态”和“本轮结果引用”。若所有组件和策略误差通过、但 GPU workspace 仍未知，可将具体延迟行标“完整校准（限定延迟）”，同时保留容量行为“未测”；不应合并成一个系统级 PASS。

## 11. 新 probe 的独立语义复核（结果前）

只读审阅对象：`native/execution_transforms/compute.cpp`、`execution_calibration/analysis.py`、`execution_calibration/analyze_matched.py`。源文件还在实现中，本节不是最终字节哈希绑定，也不替代运行后 verifier。

### 11.1 FFN 公平比较边界

新 probe 的 all-GPU、CPU/GPU 分片和分片加预载均以 **DRAM 输出**为终点，所有需要的不可变输入已在 CPU/GPU 双驻留。all-GPU 也包含完整输出 D2H，因此没有拿 GPU 原地输出与混合 DRAM 输出直接排名。

分片沿 gate/up 输出列、down 输入行配对，分片权重在计时前独立物化。CPU/GPU 各自执行四阶段，GPU partial D2H 后通过 CPU GGML SUM 合并。此次数学边界仍为 **compact selected-row FFN，没有 norm、router、gather/scatter、residual**。它可验证专家内部分片机制，不能将通过状态扩展为 Dense 完整 FFN 或 MoE 整层通过。

启动式搬运策略每轮先通过 pinned staging 复制三份 GPU 分片权重，再执行分片路径；这三次复制在端到端计时内。初始显存分配仍已存在，因此应称“显式重新加载权重”，不宣称 cold allocator/cache，也不能和“在计时外完成预载”的 warm-resident 口径混称。

### 11.2 并发与模型独立性

持久 worker 提交 GPU 分支，主线程执行 CPU 分支，然后等待 GPU 完成。代码允许真实 overlap；是否真的出现 CPU/GPU 执行区间交叠，应由独立 trace/event 支持。worker 控制区间与实际 kernel 区间不是同一个东西。小形状即使提交了并发，也可能因为调度与派发开销而没有可观的重叠。

主预测应使用独立阶段价格构成 `max(sum(CPU stages), sum(GPU stages)) + partial D2H + CPU SUM` 等预声明表达式，而不是用实测 `parallel_branches` 总时间拟合 resident 总时间。后者可另存为诊断：它能帮助定位并发模型误差，但会绕过本来要验证的调度预测。

`dispatch_barrier` 是空 worker 握手观测，含线程唤醒/等待，不能称纯 launch。若模型选择加它，需预先声明表达式及归属，并避免和各 GGML 同步阶段的 host 控制成本重复计入。若简单 `max` 预测失败，应保留失败，而非用同次策略总时间拟合一个补偿常数。

### 11.3 Attention 的限定 lowering

新增阶段使用 post-RoPE Q/K/V，分别运行 QK、causal mask/softmax、PV、sigmoid gate/O/residual。全 GPU 与 CPU QK+softmax→probabilities H2D→GPU PV/output 都含最终 D2H。

其 QK 包含连续 key materialization，PV 再物化 K/V 来产生 append 输出，原 cache 保持不变。这是明确可执行的 lowering 选择，但仍不是新 IR position-major KV block 的直接物理实现；多出的 assembly/storage 必须披露。可以校准这个具体阶段图，不应把它无条件变成所有 Attention lowering 的通用价格。

### 11.4 分析防泄漏复核

`freeze_model` 只选择 `split=profile`，拒绝任意 declared strategy total 充当组件；主组件价采用独立阶段中位数。heldout 总时间用于验证而非拟合。runner 仍必须确认每行数值检查、profile/heldout seed 不交叉、预声明重复进程覆盖，以及模型在 heldout 之前保存并在结束后保持相同 SHA。

历史 `analyze_matched.py` 调用原 verifier，并逐项证明重建的组件价格与原先冻结模型一致。历史 CPU serial 的终点是 DRAM，不与旧 GPU/hybrid 的 VRAM 输出组跨边界排序。策略排序的 `latency_quality_accepted` 仅表示参与项通过延迟质量门，不自动表示排序正确或 regret 达标。

### 11.5 性能器闭环与组件公式需要分开

当前新增分析器用预声明 `sum/max` 表达式计算预测，不等于执行了 Python frontend→IR→Compiler→Modeling 的完整新策略回放。完整报告应分别标明：

| 证据层 | 能证明什么 | 仍需补什么 |
|---|---|---|
| 原生真实路径与数值检查 | 该 C++ lowering 的输出正确，并实际运行声明的运算/搬运 | 与策略 IR 的每个动作和状态逐项绑定 |
| profile-only 组件表达式 | 独立阶段预测未拟合组合的误差与局限 | 是否与性能器实际依赖、槽位、控制开销一致 |
| Modeling 动作回放 | 性能器采用相同实测动作后得到的时间、状态、容量判断 | 新变换需新的实测 provenance/端口合同，GPU 未知 workspace 仍限制容量验收 |

旧 `replay_serial.py` 提供第二层到第三层的 CPU 旧端口图例子，新分片和 Attention 图尚不能因此自动视为闭环。报告中的“完整校准”必须标清是哪一层及什么范围。

### 11.6 冻结源复核记录

在 runner 宣布冻结后再次只读核对：未发现阻断运行的问题。确认 runner 检查协议、端口字节、固定数值容差、CPU 已知/GPU 未知 workspace，并绑定新 C++、被 include 的旧 C++、二进制、运行库和 Python 源；每个 heldout raw 记录启动时 frozen model SHA。两个 FFN 模型（ideal 与加空 worker 握手的 dispatch model）均由运行前代码明确给出；不同 resident/preload 起点分组，不能事后择优隐藏失败。

本次审阅到的字节 SHA256：

| 文件 | SHA256 |
|---|---|
| `native/execution_transforms/compute.cpp` | `a5cb4aebfadfc372d321ec0f933261db30c725d156c69abb2dbc7cdd3cd70f24` |
| `execution_calibration/run_native.py` | `829030143ae1b171d1332d1eb5de83f66120f6f685d4fe594edea4d5025b66b4` |
| `execution_calibration/analysis.py` | `2dfe8cdf3481b6eb377aeef22ccf75bfc2cc63aef748f8204a3ba21739787ab7` |
| `execution_calibration/transform_suite.json` | `208cf49ca7811221d50c72026e9a33d0c7470e7168d6469df41067f6a00f392f` |

这些是审阅对象身份，真正运行的身份以实验 manifest/source snapshots 为准。后续若发生任何修订，应保留旧实验并重新绑定，不能将本节当作修订后文件已经被审过。

## 12. 精度发现与基线结果的独立审阅

### 12.1 已观察到的精度现象

证据目录为 `results/20261017-execution-calibration/transform-smoke-01` 与 `transform-smoke-02`。两个目录中 H512/I1024/q32/CPU-cut256 项使用相同几何、CLI 和 seed=101：

- 首次默认精度版本有 4 项 GPU gate/up 检查失败（完整矩阵与 GPU 分片各两项），最大绝对误差 `3.75160016119e-5`。
- 修订版本同时对所有 matmul 请求 `GGML_PREC_F32`，并向子进程明确设置 `NVIDIA_TF32_OVERRIDE=0`。相同项全部通过，所有检查中的最大绝对误差为 `2.60770320892e-8`。
- 修订版本的 4 个 smoke case 全部通过；固定容差没有放宽。smoke 的每项只有 3 个计时样本，作用是数值与协议入口检查，不是延迟质量验收。

可用报告表述：**“F32 张量存储类型不足以定义完整计算精度合同。默认配置在较大 q 的矩阵乘中出现数值检查失败；明确请求 F32 matmul 并禁用 TF32 override 后，同一输入恢复通过。因此正式测量绑定这个明确的计算精度配置，并保存最初失败。”**

需要区分观察与推断：源码中存在 TF32-enabled cuBLAS handle 和特定 F32 matmul 路径，支持“后端计算精度路径可能相关”的解释；但本轮同时改变两个开关并重新编译，没有做两开关的 2×2 消融。首次 raw 也未记录 `NVIDIA_TF32_OVERRIDE`。因此不能说已经证明 TF32 是唯一原因，不能说两个开关各自必要或各自充分，也不能声称直接观测到具体硬件 kernel 使用了 TF32。修订是为正式实验建立明确合同，不是已完成因果归因。

第 11.6 节保留的是修订前审阅身份；正式 `transforms-01/manifest.json` 绑定的修订后对象已再次只读核对：

| 文件 | 修订后 SHA256 |
|---|---|
| `native/execution_transforms/compute.cpp` | `de455ca3f5963bc9427496cf560c6cc95044eb920bfecc1f575fa221123b1b92` |
| `execution_calibration/run_native.py` | `5d463adce711b53fca3dedece0f6faf3a9c23d599488c40ccc30f2e8395d578d` |
| `execution_calibration/analysis.py` | `2dfe8cdf3481b6eb377aeef22ccf75bfc2cc63aef748f8204a3ba21739787ab7` |
| `execution_calibration/transform_suite.json` | `208cf49ca7811221d50c72026e9a33d0c7470e7168d6469df41067f6a00f392f` |

### 12.2 已完成基线的稳定性结论

只读核对 `baseline-01/measurements.json`、`baseline-01/report.json`、`baseline-component-analysis-final/report.json` 和 `baseline-modeling-replay/report.json`：

- 原生进程记录 `96/96` 成功完成并通过数值检查。
- 旧协议共 99 个时间指标，9 项通过其全部质量门。它包括直接测同一执行边界的时间稳定性验证，不能混同独立组件预测。
- 独立组件预测共 19 个端到端项，其中 18 项的 heldout 中位数相对误差不超过 20%；唯一超过的是小形状 FFN GPU serial，误差约 28.83%。
- 19 项都存在参与组件或目标的进程内 CV 超门，3 项还存在跨进程 CV 超门，因此 **0/19 通过完整组件模型验收**。最终分析已禁止跨不同初始驻留合同的排名。
- 旧 CPU Modeling 端口图能合法回放，但相关阶段质量未通过，仍为诊断，不能导出 accepted 成本。

可用报告表述：**“多数端到端中位数预测已接近实测，但当前样本的波动未达到预先声明的稳定性门，暂不能作为已接受的成本表。”**

不能只挑 18/19 的点估计报“校准成功”，也不能因 CV 失败就断言中位数完全无用或整个性能器无效。后台应用、操作系统调度、频率、微秒级计时与控制路径等都是可能解释；本轮未进行隔离它们的受控消融，不把任何一种解释写成已经证实的原因。

## 13. `transforms-01` 最终结果的独立复核

本节仅做读取、SHA、计数和组件表达式重算，没有运行硬件、删改样本或改变冻结模型。目录：`results/20261017-execution-calibration/transforms-01`。

### 13.1 证据完整性

- 28 条原生进程记录均通过，全部 412 项数值检查通过，保留 25,908 条计时样本。
- 原始 record SHA 与 `measurements.json` 一致，manifest 的前后资产一致，当前被绑定资产的 SHA 也一致。
- 每个 heldout raw 的启动模型 SHA 与 profile 冻结 SHA 一致；精度字段与继承环境均为正式声明的 F32 请求和 TF32 override=0。
- 独立重算每个组件的 profile-only pooled median 与模型一致，34 个预声明表达式与报告预测一致。

这里 **34 是 19 个主模型预测加 15 个 dispatch 备选模型预测**，不是 34 个独立真实策略。两模型复用了同一份真实 heldout 策略时间，不能当作样本或策略数量翻倍。

### 13.2 准确度和稳定性

| 模型 | 预测项 | 点误差≤20% | 全质量门通过 | 进程内 CV 失败 | 跨进程 CV 失败 |
|---|---:|---:|---:|---:|---:|
| 主模型（串行相加、分支理想 max） | 19 | 13 | 0 | 19 | 4 |
| FFN dispatch 备选模型 | 15 | 10 | 0 | 15 | 4 |

主模型中，5 个 all-GPU resident FFN 的点误差为 2.21%～10.55%，4 个 Attention 路径的点误差为 0.37%～10.72%。FFN 分片 resident 路径的点误差为 14.28%～36.51%，均为低估。空 worker 握手修正没有使完整质量门通过。

不能将误差门的点估计与完整验收混为一谈。所有策略均未通过进程内 CV 门，所以此次应交付 **真实数值/执行证据、误差诊断与待修假设**，不能发布 accepted 的新成本表。

### 13.3 排序反例

H256/I512/q3/CPU-cut128 的 resident 对照：

| 路径 | 主模型预测（μs） | heldout 实测中位数（μs） |
|---|---:|---:|
| all-GPU，最终 DRAM 输出 | 472.25 | 461.75 |
| CPU/GPU 分片，CPU SUM，最终 DRAM 输出 | 415.80 | 566.80 |

主模型选择了分片，实际额外耗时约 22.75%。这是有价值的模型诊断反例，但由于稳定性门未通过，不应写成已经统计确认的普遍规律或论文收益。

### 13.4 关键限定：没有观测到实际分支重叠

独立读取 20 个 FFN raw 的 `diagnostic_completed_work_brackets`，**20/20 的 `completed_work_bracket_overlap_ns=0`，且全部 `kernel_overlap_proved=false`**。这些是每个进程额外的一次 untimed 诊断，不是所有计时样本的完整 GPU timeline。

因此此次代码虽然通过 worker 提交两分支，但诊断未观察到 CPU 分支和 GPU host completed-work 区间重叠。不能写“已验证 CPU/GPU 真正并行”，也不能直接将理想 max 的误差归结为性能器的一般并发模型错误。单逻辑处理器 affinity 可能使 GPU worker 的提交延后，与现象相容，但尚未做唯一因果归因。

同一 profile 内，真实 `parallel_branches` 也显著长于独立阶段 max。例如 H512/I1024/q32 的阶段 max 预测为 1325.25 μs，真实分支路径 profile 中位数为 2015.45 μs，heldout 为 2166.95 μs；空 worker 握手仅约 9.1 μs。这说明差距不能只靠加一个空握手常数解释，但仍需要先使原生执行边界与预测并发假设匹配，再研究共享资源竞争。

下一轮应保留本次结果，新建协议：明确 GPU 提交与 CPU 开始的先后条件，用真实 GPU event 或可解释的时间线观察重叠，单独控制 GPU dispatch 的 CPU 资源。只有确认执行产生了目标重叠，才用重叠模型的预测误差讨论资源竞争；不得直接用本次组合总时间回填一个成本来消除反例。

## 14. 双逻辑处理器变体的最终独立审阅

目录：`results/20261017-execution-calibration/transforms-dual-01`。这是保留 v1 源码/程序/结果后，新建的独立协议和原生程序。独立逐 hunk 对比确认数学图、数据生成、路径、计时次序和 oracle 没有改变；差异限 affinity 设置/读回、worker 参数、metadata 与注释。v1 仍是单逻辑处理器提交诊断，不能将它覆写为后来观察到并发的结果。

### 14.1 新合同与证据完整性

- Windows 处理器组 0 中，进程 mask 实测为 5，主线程 mask=1（CPU 0），GPU dispatch worker mask=4（CPU 2）。这里只证明两个不同的逻辑处理器，未据此证明两个独立物理核心。Attention 不创建 worker，其 worker mask/group 为 null。
- profile seed=41，heldout seed=53，每个 case 各有两个新的独立进程，28 个 PID 互不相同；未复用 v1 的成本模型。
- 28 个原生进程、412 项数值检查全部通过，保留 25,908 个计时样本。F32 精度请求、TF32 override=0、dispatch CPU 环境与 actual affinity 读回一致。
- 独立核对全部 raw SHA、manifest 前后及当前资产 SHA、heldout 启动 frozen model SHA；CPU 工作区保持已知，GPU 私有工作区始终 null。
- 独立从 profile 原始样本重算每个组件中位数及 34 个预测，均与冻结模型/报告一致。

本次审阅的关键源身份：

| 文件 | SHA256 |
|---|---|
| `native/execution_transforms_dual/compute.cpp` | `fd5659470a8c27280bfb9e06f6bcad4bb61d2332314a6e6aa482a9da48ed6834` |
| `execution_calibration/run_dual.py` | `1154a2131b48ababde670986d054db6c6b29d20f2ba6df9cdc1cc2d1292eb983` |

### 14.2 重叠证据的准确措辞

20 个 FFN raw 的额外单次诊断中，19 次 host completed-work 区间有交叠，交叠范围为 0～370.2 μs。它表明当前线程调度合同允许两分支的 host 区间交叠，纠正了 v1 中没有观察到交叠的问题。所有 raw 仍明确 `kernel_overlap_proved=false`，不能把这些 host brackets 写成 GPU kernel trace，也不能推断每个计时样本都达到相同重叠程度。

v1 和 dual 同时存在 seed 与执行时段差别，因而两轮之间是调度合同敏感性诊断，不是只改变 affinity 的严格配对因果实验。

### 14.3 点误差、稳定性与排序

| 模型 | 预测项 | 点误差≤20% | 全质量门通过 | 进程内 CV 失败 | 跨进程 CV 失败 |
|---|---:|---:|---:|---:|---:|
| 主模型 | 19 | 11 | 0 | 19 | 2 |
| dispatch 备选模型 | 15 | 7 | 0 | 15 | 8 |

这些结果不支持“新性能器全部预测不准”的结论：

- Attention 的 4 个主预测误差为 3.18%～13.21%。
- H512/I1024/q32 的 mixed resident 与 reload 路径误差分别为 12.94% 和 13.32%。
- 四个小 FFN mixed resident 的点误差为 67.30%～76.13%，均为高估。这是需要定位的突出问题，方向与 v1 的低估不同。
- 主模型共 7 个公平 resident 配对，4 个中位数排序正确、3 个错误。最差点估计 regret 约 85.53%。区间不重叠仍不代替全部质量门，因此排序均保持诊断状态。

### 14.4 新发现：孤立组件与组合路径的 host 提交合同不完全相同

代码中所有孤立 GPU 阶段 profile 都由 **主线程、CPU 0** 调用；真实 FFN 双分支路径的 GPU 阶段则由 **worker、CPU 2** 调用。数学形状与 GPU 相同，但 wall 时间包含的 host 控制线程和 affinity 不同。现有组件键没有显式区分这个条件。

这是确定的代码边界差异，不是已经证实的全部误差原因。不能把小 FFN 的高估单独归因于数学 max 规则、GPU 性能变化或某一操作系统行为。下一步应在同一个 worker/affinity 上测独立 GPU 阶段，或显式建模 host dispatch 资源；另保留 kernel/event 区间以区分设备计算与 host 同步。不能借此次组合实测总时间反向填价，掩盖组件合同没有对齐的问题。

因此本轮最终状态仍是：**真实阶段与分片执行、明确精度和 affinity、独立成本预测与反例已经形成可复核证据；稳定性和执行边界校准仍未闭环。**

## 15. 最终同 worker 上下文实验的独立审阅

本轮根报告以 `results/20261017-execution-calibration/transforms-context-01` 为主。旧基线、v1、dual、数值失败 smoke 全部保留。最终轮没有修改公式、容差或质量门，也没有以组合真实总时间拟合组件。

### 15.1 已修复的具体合同差距

新 `native/execution_transforms_dual/compute_context.cpp` 仅将 GPU **分片**的独立阶段测量移入同一个 GPU dispatch worker，并记录该测量线程的实际 mask/group。计时在 worker 内部完成，排除外层 wake/wait；这些控制成本仍由预声明理想与 dispatch 两种模型检验。all-GPU 完整阶段与 all-GPU 策略均留在主线程；数学图、数值 oracle、实际策略执行路径、精度与容差不变。

独立读取全部 raw：20 个 FFN 进程都记录 `profile_gpu_shard_thread=dispatch_worker_same_as_parallel_strategy`、profile mask=4/group=0，与实际策略 worker 的读回值相同。8 个 Attention 进程正确记录 no-shard/N/A 和 null。已修复第 14.4 节指出的明确线程上下文错配，不能将这个问题继续列为当前轮的既存缺陷。

源文件 SHA256：`37ed5525074d2653050daa433a7b938a6f7f751a1afdeaac17a8c21cc2173cd4`。runner 沿用已冻结 `run_dual.py`，SHA256 为 `1154a2131b48ababde670986d054db6c6b29d20f2ba6df9cdc1cc2d1292eb983`。新 C++ 与其包含的旧 helper 都由本轮 source snapshot/inventory 绑定。

### 15.2 原始证据与冻结核查

- profile seed=67、heldout seed=79，28 个互异 PID；每个 seed/case 各两次独立进程。实际预测模型由本轮 profile 独立冻结，不复用旧轮成本。
- 28/28 进程、412/412 数值检查通过，25,908 条逐样本时间保留。
- 所有 raw SHA、source/runtime/程序前后及当前 SHA、heldout 启动模型 SHA 均一致。
- 实际读回 process mask=5、main mask=1、worker mask=4；F32 请求与 TF32 override=0、dispatch CPU 环境一致。GPU workspace 始终 null，CPU 工作区来自明确 allocator/cplan，未填假零。
- 独立从 profile 原始数据重算每个组件的中位数与 34 个表达式，全部与冻结模型及最终报告一致。
- 20/20 个 FFN 单次额外诊断观察到 host completed-work 区间交叠，范围 19.3～253.1 μs；全部仍为 `kernel_overlap_proved=false`。

### 15.3 最终误差与质量门

| 模型 | 预测项 | 点误差≤20% | 全质量门通过 | 进程内 CV 失败 | 跨进程 CV 失败 |
|---|---:|---:|---:|---:|---:|
| 主模型 | 19 | 11 | 0 | 19 | 4 |
| dispatch 备选模型 | 15 | 9 | 0 | 15 | 8 |

19 个主预测中：

- 5 个 all-GPU FFN 的误差为 0.93%～14.27%。
- 4 个 Attention 路径的误差为 3.77%～11.15%。
- 4 个小 FFN mixed resident 低估 31.65%～44.60%。较大 FFN mixed resident 低估约 20.63%。
- 权重重新加载路径部分接近、部分仍超 20% 门；完整质量门也全部未通过。

不能将 `0/19` 完整通过解释为“19 个点预测全部不准”：它同时检查点误差与稳定性，而此次所有项都有进程内 CV 门失败。也不能只引用 11/19 或单设备阶段误差来声称整个策略模型已校准。

### 15.4 最终排序与结论边界

主模型 7 个同 resident 初始/最终合同的配对排序全部与 heldout 中位数一致，按预测选择的点估计 regret 均为 0。5 个 FFN dispatch 备选对照也方向一致。由于参与项的完整延迟质量门没有通过，这些都是 **排序方向的诊断证据**，不是已通过稳定性验收的策略选择器。

本轮移除了已知的组件提交上下文错配，但并发路径仍可能受实际派发控制、线程唤醒/同步、共享资源、计时边界和样本波动影响。现有结果未隔离这些原因，不能把剩余低估唯一归因于其中某一项。下一轮应把控制/依赖和真实事件区间对齐，继续使用独立 heldout，不能回填组合总时间或放宽本轮质量门来制造通过。

三轮不同 seed、执行时段与协议逐步修订的结果，可以说明完善测量合同的重要性；不是严格只改变一个变量的收益消融。最终可交付结论为：**同上下文真实执行与数值证据已经齐备，单设备与 Attention 点误差较小，当前小 FFN 分片时间仍明显低估；排序方向已提供初步证据，性能准确性和稳定性尚未完成验收。**
