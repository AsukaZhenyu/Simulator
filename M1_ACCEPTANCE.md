# M1 验收规则：混合计算、有限内存与 SSD 读写

日期：2026-09-26。状态：设计预期，实施结果待验收；以下数字来自固定成本与手算，不是硬件测量。规则见 [M1_DESIGN.md](M1_DESIGN.md)。本次新增 G/H 的 SSD 溢写、恢复和输出交付验收，A～F 禁用写入且原有数字不变。

## 1. 共同条件与验收方法

- 固定 F32、连续、独立 allocation，禁止原地与重计算。W 为 4×4、64 B；x/h/a/b/y 为长度 4 向量、16 B；shape 遵循 GGML。
- 一个 CPU、一个 GPU 计算槽，一个共享 PCIe 槽，一个共享 SSD 读写槽（`ssd_io_slots=1`）；默认允许不同资源重叠。alignment=1 B、workspace=0、两域静态预留=0。
- 未特别说明时 DRAM/VRAM 各 256 B；不存在的 SSD 源不可凭空使用。
- 明确列出各场景允许设备和四条路线；未列出的路线不允许，使用空集合。每条允许路线的各张量成本都必须给出，其他项不参与搜索。A～F 的 `dram_to_ssd` 为空。
- 先写至少一份人工 mapping 回放，再无限预算搜索最优值/无解证明。表中时序是人工参照，不强制搜索返回相同路径。
- 峰值、搬运量要对照指定人工计划检查；搜索只优化 makespan，等时间解的峰值/额外搬运可能不同，不能据此否定合法最优解。
- 所有 ms 换成整数 ns 保存。任何用例的预算截止都不能冒充 infeasible。

## 2. A：单算子的 CPU/GPU 选择包含搬运

图 `y=W*x`，操作 c1；W 和 x 初始仅在 DRAM，目标 y 在 DRAM。GPU 时间 2 ms，H2D W/x 分别 3/1 ms，D2H y 为 1 ms。允许设备 cpu/gpu，允许路线仅 H2D W、x 和 D2H y。

| CPU 时间 | 最优时间 | 一个最优方案 |
|---|---:|---|
| 6 ms | 6 ms | CPU 0–6，y 已在目标域 |
| 9 ms | 7 ms | W H2D 0–3，x H2D 3–4，GPU 4–6，y D2H 6–7 |

纯 GPU 下界是必要 H2D 总共 4 ms + 计算 2 ms + 输出 D2H 1 ms = 7。不能报告“GPU 只需 2 ms，所以总选 GPU”。人工 CPU 方案不搬运；人工 GPU 方案 H2D=80 B、D2H=16 B、SSD=0。

将 allowed_devices 限制为 cpu / gpu 后，分别验证 6 或 9 ms / 7 ms；纯 CPU 对照同时禁用无关搬运。修改目标为 VRAM 时，重新计入 CPU 输出的 H2D（须显式允许并给成本），不可只改终止标签。

## 3. B：真正有收益的混合链

图 `h=relu(x); y=relu(h)`，x 初始 DRAM；目标 y 在 DRAM。c1/c2 均允许 cpu/gpu，成本如下：

| 操作/搬运 | 时间 |
|---|---:|
| c1 CPU / GPU | 1 / 6 ms |
| c2 CPU / GPU | 8 / 1 ms |
| H2D x / h | 2 / 2 ms |
| D2H h / y | 1 / 1 ms |

仅上述四个搬运对象被允许。四种放置的最短时间：

| 放置 c1→c2 | 时间 | 必要路径 |
|---|---:|---|
| CPU→CPU | 9 ms | 1+8 |
| GPU→GPU | 10 ms | H2D x 2 + 6 + 1 + D2H y 1 |
| CPU→GPU | **5 ms** | CPU 0–1，H2D h 1–3，GPU 3–4，D2H y 4–5 |
| GPU→CPU | 17 ms | 2+6+1+8 |

联合搜索必须得到 5 ms。用限制 allowed_devices 的场景验证四行，不能靠 GUI 上的设备标签冒充执行约束。混合人工计划 H2D=16 B、D2H=16 B；4 ms 时全部算子 DONE 但 y 仍在 VRAM，不能判 goal，删除最后 D2H/ADVANCE 后回放应 incomplete_mapping。

## 4. C：CPU/GPU 并行与复制资源

图 `a=relu(x); b=relu(x); y=a+b`。x 初始同时在 DRAM、VRAM，两域各计 16 B。a 节点仅 CPU，3 ms；b 节点仅 GPU，4 ms；add 仅 CPU，1 ms；仅允许 D2H b，1 ms；目标 y 在 DRAM。

- 默认最优 **6 ms**：CPU a 0–3、GPU b 0–4，D2H b 4–5，add 5–6。GPU 的 b 未复制前不能直接供 CPU add 使用。
- `allow_cpu_gpu_overlap=false` 时最优 **8 ms**：先 GPU b 0–4，再 CPU a 4–7，与 D2H b 4–5 重叠，add 7–8。先 CPU 再 GPU 会得到 9 ms，搜索不应被固定派发顺序限制。
- 同一计算槽启动第二个任务应 RESOURCE_BUSY；另一设备合法独立任务可在同一 t 启动。

另设 `u` 初始 DRAM、`v` 初始 VRAM；`a=relu(u)` 仅 GPU、`b=relu(v)` 仅 CPU，计算均 1 ms；H2D u 3 ms、D2H v 2 ms，目标 a 在 VRAM、b 在 DRAM。最优 **6 ms**；两次 PCIe 搬运不得重叠。启动 H2D 后同刻启动 D2H 必须 RESOURCE_BUSY。此例用两个输出，验证逐输出目标域。

## 5. D：有限 DRAM 改变 SSD 预取收益

CPU-only 图 `h=W1*x; y=W2*h`。x、W1 初始仅 DRAM，W2 初始仅 SSD；CPU c1/c2 各 2 ms，READ_SSD W2 为 3 ms。仅允许读取 W2，不允许 H2D/D2H，目标 y 在 DRAM。

| DRAM 容量 | 最优 | 说明 |
|---|---:|---|
| 160 B | **5 ms** | c1 0–2 与 READ W2 0–3 重叠，c2 3–5 |
| 96 B | **7 ms** | c1 0–2，释放 W1/x 后 READ W2 2–5，c2 5–7 |
| 95 B | **infeasible** | c1 同时需要 x16+W1 64+h输出16=96 |

160 B 人工计划在 t=0 同时预留 h16、W2 64，连同 x16/W1 64 共160；t=2 显式释放 W1/x，保留 h；t=3 c2 分配 y，总占用96。其峰值160。96 B 人工计划峰值96。SSD 读取量均64 B。

容量96时，c1 运行中尝试 READ W2 必须 CAPACITY_EXCEEDED，错误指明 DRAM；不能隐式排队至 t=2。容量160且 `allow_copy_compute_overlap=false` 时最优 **7 ms**，验证 SSD 读取也服从该开关。

## 6. E：SSD→DRAM→VRAM 两段流水与源锁

GPU-only 图 `h=W1*x; y=W2*h`。x 初始仅 VRAM，W1/W2 初始仅 SSD。每份权重 READ 4 ms、H2D 3 ms，GPU c1/c2 各2 ms。仅允许 READ/H2D W1、W2；目标 y 在 VRAM。

| DRAM / VRAM | 最优 | 一个达到最优的时序 |
|---|---:|---|
| 128 / 160 B | **13 ms** | READ W1 0–4，H2D W1 4–7；READ W2 4–8，H2D W2 8–11；c1 7–9，c2 11–13 |
| 64 / 160 B | **16 ms** | READ W1 0–4，H2D W1 4–7；释放 DRAM W1 后 READ W2 7–11，H2D W2 11–14；c1 7–9，c2 14–16 |
| 63 / 160 B | **infeasible** | 连一份64 B读取目标都放不下 |
| 128 / 96 B | **14 ms** | c1占满96，W2 H2D须等到9；H2D W2 9–12，c2 12–14 |

每份 SSD 读取之后才能 H2D，禁止直接 SSD→VRAM；第一行的下界是连续两次 SSD 读取至少8 ms，最后必要 H2D 3 ms，再执行至少2 ms。反序加载无法改善该界。

DRAM=64 的关键：W1 在 H2D 4–7 期间仍是锁定的 DRAM 源，不能提前释放后让 W2 偷用同一64 B。t=7 后才允许释放，决定16 ms的下界。两份读取和H2D总量各128 B；没有D2H。

人工第一行应显式在 t=7 释放 DRAM W1，在 c1 完成后释放 VRAM W1/x；峰值 DRAM128、VRAM160。64 B行峰值 DRAM64；对齐前后搬运量不得变化。

## 7. F：DRAM 容量使 CPU/GPU 最优选择反转

图 `y=W*x`，W 初始仅 SSD、x 初始仅 DRAM，目标 y 在 DRAM。CPU/GPU c1 均2 ms；READ W4、H2D W3、H2D x1、D2H y1 ms。允许上述路线；VRAM96 B。

| DRAM | 最优 | 关键原因 |
|---|---:|---|
| 96 B | **6 ms** | READ W 0–4，CPU 4–6；W+x+y=96 |
| 80 B | **10 ms** | CPU输出分配需要96，改用GPU；x H2D可与SSD读重叠 |

80 B 手写方案：READ W 0–4、H2D x 0–1；H2D W 4–7；GPU 7–9；D2H y 9–10。W 的 DRAM 源必须锁到7；7时 GPU 就绪后可释放 DRAM W，以给 y 腾空间。无需释放 DRAM x 也能完成，此时峰值 DRAM80、VRAM96，SSD64/H2D80/D2H16 B。

GPU下界 READ4+H2D W3+GPU2+D2H y1=10；H2D x可以藏在读取期间。80 B 下CPU执行始终要求输入80加输出16，不能先释放锁定输入来执行。再把 VRAM 改95 B，CPU/GPU都放不下，必须证明 infeasible。

此例必须同时搜索放置和数据准备，不能先按计算耗时固定选 CPU，再把容量错误报成整个场景不可行。

## 8. G：内存不足时必须溢写中间张量并读回

CPU-only 图 `h=relu(x); b=W*z; y=h+b`。x、z 初始仅 DRAM，各16 B；W 初始仅 SSD，64 B；h、b、y 各16 B。目标 y 在 DRAM。c1/c2/c3 分别是上述三个操作，CPU 时间分别1/2/1 ms。

只允许 WRITE_SSD h（2 ms）、READ_SSD W（3 ms）、READ_SSD h（1 ms）；没有 PCIe 路线，也不能写 x/z/W。h 的 SSD 副本初始不存在。此图用于检验长寿命中间值的保存与恢复，不把 RELU 中间值宣称为实际 KV 计算。

| DRAM 容量 | 最优 | 说明 |
|---|---:|---|
| 112 B | **6 ms** | READ W 0–3 与 c1 0–1 重叠，c2 3–5，c3 5–6；无需写 h |
| 96 B | **10 ms** | c1 0–1，WRITE h 1–3，READ W 3–6，c2 6–8，READ h 8–9，c3 9–10 |
| 95 B | **infeasible** | c2 的 W64+z16+b输出16 至少需要96 |

96 B 的人工计划要明确释放：c1 完成后释放 x；WRITE h 完成后释放 DRAM h；c2 完成后释放 W/z，再读回 h。峰值 DRAM96 B，VRAM0；SSD 写16 B、读80 B（W64+h16），H2D/D2H 均0。各时刻的预留、源锁和恢复必须可由回放检查。

96 B 下的下界：c2 运行时自身占满96，h 无法同时留在 DRAM，而 c3 仍需 h；所以 h 必须先生成、写完并释放，c2 完成后才能恢复。提前读取 W 会挡住 c1 的输出分配；无用的提前读取/驱逐不能改善结果。共享 SSD 槽使 WRITE h 与 READ W 串行；恢复 h 的预留也会阻止 c2 启动。这给出1+2+3+2+1+1=10 ms。112 B 下 READ W、c2、c3 的必要链给出3+2+1=6 ms，c1 可重叠。

容量96时，禁用 WRITE h 或删掉 READ h 路线均应证明 infeasible；有 SSD 备份但无法读回不能冒充可恢复。WRITE h 1–3 期间 EVICT DRAM h 必须拒绝，READ h 也不能读取尚未完成的 SSD 目标。112 B 参照计划的 SSD 写量为0、读量64 B；这些流量是指定人工计划的预期，不额外要求所有等时间最优路径都无冗余搬运。

## 9. H：输出写完 SSD 才算交付

单算子图 `y=relu(x)`，计算时间1 ms，WRITE y 为2 ms，目标 y 在 SSD。写入前 SSD y 不存在；只允许必要路线，无 SSD 读取。

| 变体 | 允许路线 | 最优时间与人工计划 |
|---|---|---|
| CPU-only，x 初始 DRAM | WRITE y | **3 ms**：CPU 0–1，WRITE y 1–3 |
| GPU-only，x 初始 VRAM | D2H y（1 ms）、WRITE y | **4 ms**：GPU 0–1，D2H y 1–2，WRITE y 2–4 |

CPU 计划峰值 DRAM32 B（x+y）、VRAM0，SSD 写16 B。GPU 计划峰值 VRAM32 B（x+y）、DRAM16 B，D2H16 B、SSD 写16 B。两者 READ 字节为0。只计算完 y 或只完成 D2H，都还不是 goal；SSD y 为 RESERVED_COPY 时不得提前成功。写入中的 DRAM y 不能释放；写完后源副本仍在内存，不要求为达到 goal 额外释放。此例到达 goal 后不追加动作；若更大图还有其他任务，已交付 SSD 输出的非目标内存副本可按规则释放。

GPU 变体将 DRAM 改为15 B 应 infeasible，因为 D2H 目标必须分配16 B，不能用隐藏缓冲绕过容量。删除 WRITE 路线应 infeasible；不能把目标标签为 SSD 当成自动存储。

## 10. 必须覆盖的语义与边界

| 用例 | 预期 |
|---|---|
| CPU输入仅在VRAM，或GPU输入仅在DRAM | INPUT_NOT_READY，不自动搬运 |
| COPY/READ/WRITE的源不存在或未就绪 | 拒绝，无目标分配 |
| 对READY/RESERVED目标重复复制 | 拒绝，不重复计费 |
| 两个H2D、H2D+D2H、两个SSD读、SSD读+写、两个SSD写 | 同资源第二任务RESOURCE_BUSY；测试保证其他前置满足 |
| SSD与PCIe、CPU与GPU同刻启动 | 条件满足时合法，启动不推进t |
| 计算或复制期间EVICT被读源 | TENSOR_IN_USE；包括CPU输入、PCIe源、WRITE的DRAM源；SSD读目标预留也不可释放 |
| 释放RESERVED_OUTPUT/RESERVED_COPY | 拒绝，不回收在途空间 |
| 另一域READY副本存在 | 可释放未锁域副本，但不得丢失后续可达性 |
| 另一域只有RESERVED备份 | 不能据此删除唯一READY活值 |
| 唯一中间值仍有消费者 | LIVE_VALUE_LOSS |
| 所有计算DONE、y尚未送达目标域 | 不算goal；不能删除唯一y |
| 目标域y已经READY | 禁止EVICT该目标副本 |
| 非目标副本已无用途 | 可显式释放；不能凭空释放ABSENT |
| CPU/GPU同时尝试同一算子 | 第二次拒绝，即使设备不同 |
| 同刻结束CPU/GPU/复制任务 | 一次ADVANCE全部完成，再作决策 |
| CPU/GPU workspace=16 B | 分别在DRAM/VRAM预留，完成释放，不双记 |
| 域reserve=16 B | 从一开始计费，峰值包含reserve |
| alignment=32 B的16 B向量 | 占32，搬运仍16；两域不同alignment分别应用 |
| 两域初始副本 | 两边各计一次；初始超容量报告域及required/capacity |
| COPY SSD↔VRAM、EVICT SSD | unsupported，不用隐藏DRAM缓冲实现 |
| 初始SSD叶子input/weight | 可读；没有生产者，具有明确张量身份 |
| 有生产者的中间值预置为初始SSD/DRAM/VRAM副本 | invalid_input，不能跳过计算 |
| 无SSD源或SSD写入未完成时READ | 拒绝；写完的中间值可按允许路线读回 |
| SSD目标已经READY/RESERVED时WRITE | 拒绝，不覆盖同名快照、不重复计费 |
| 写入完成后释放DRAM唯一内存副本 | 仅在剩余READY SSD副本可经允许路线满足后续用途时合法 |
| 写入目标在途，尚无READY备份 | 不能据此释放唯一READY活值 |
| WRITE任务启动/结束 | 无额外DRAM副本；源占用锁到结束，SSD目标完成后才READY |
| 允许的设备/路线缺成本、未知ID、目标不完整 | invalid_input；不能补默认时长 |
| 0/负/非整数持续时间、非有限推导值 | 拒绝 |
| view/in-place/非F32/重计算/多槽 | unsupported，范围不因CPU扩展而偷偷变大 |
| 无任务ADVANCE、完成后多余动作 | 拒绝；计划中途结束为incomplete_mapping |

为副本恢复增加两个小用例：共享权重有后续消费者，释放DRAM缓存后从SSD重读并恢复；中间h先D2H保存，再EVICT VRAM，随后按允许路线H2D恢复供下一GPU消费者。逐次计时计字节，无重计算。若删除恢复路线，释放动作应被拒绝或相应固定计划明确失败，不得得到假可行结果。

## 11. 搜索、兼容与 GUI

- A～H 的无限预算搜索返回上述最优值/无解证明；允许多个等价最优方案。每份结果再经 model 回放，数字和事件一致。
- 增加DRAM/VRAM容量、扩大合法设备/路线集合，不得使可证明最优值变差；禁用重叠不能更快。比较时保持初始数据、目标域和成本不变。
- 同一场景内，相同GPU状态但不同DRAM驻留、运行时SSD副本（不存在/写入中/已就绪）、目标预留或任务方向/设备/剩余时间，不得错误合并；插入顺序不改变key。初始SSD副本是静态场景内容，但运行时写出的副本必须进入key；跨场景缓存身份还必须包含场景指纹。
- 小预算无goal是unknown，有可行解但未证最优是feasible；两者 optimality_proven=false。一个死路或窗口失败不能得到搜索infeasible。
- 搜索加载/释放循环应去重终止，不增加任意步数剪枝后声称已穷尽。
- 无过程记录与带GUI记录的同预算搜索结果一致；记录被截断只标“展示不完整”，不改变求解结果。GUI不再重走第二份Dijkstra。
- 修改设备集合、路线、DRAM容量、SSD成本、目标域或初始位置后，旧指纹计划拒绝；0.1旧指纹保持兼容。
- 原M0链式160/96/95、残差、分叉、旧窗口对照及旧 modeling 测试不退化。不能改旧预期数字来适配M1；实际跳过原因单独记录。
- GUI至少展示B的CPU→GPU→DRAM交付、E的两段权重流水、F的容量导致放置反转、G的写完→释放→读回、H的SSD输出交付。SSD单泳道区分读/写；各域副本状态、内存容量、锁定原因、操作设备和“输出是否已交付”可见。
- 默认GUI搜索有限预算；巨大树不必全部渲染。CPU/GPU本地访存成本不能又被前端额外算一次。

## 12. 交付与完成标准

1. 0.2加载、统一状态引擎、固定计划回放、精确搜索；0.1兼容。
2. A～H workload/scenario、至少一份每类人工计划和测试。容量/设备变体优先参数化，避免堆大量近似JSON。
3. 输出0.2的双域峰值、四方向字节（含SSD读与写）、资源事件、运行时SSD状态和目标域，保存可回放mapping。失败明确定位动作、域、资源或张量。
4. GUI接入同一求值与搜索路径，保留旧查看器功能。
5. 简短验证记录：实际命令、基线/新增测试结果、各场景结果、环境限制与尚未实测项。明确合成成本，不能用“截图正常”代替回放。

M1完成意味着上述限定模型的语义与搜索通过验收，不意味着完成真实LLM、物理内存分配器或SSD预测校准。实测成本样例另标是否实际采集，注明来源与适用范围。

开发安装继续在 Simulator 根目录使用 `python -m pip install -e ./modeling -e ./mapping`。之后在 modeling 目录跑原测试，在 mapping 目录跑 `python -m unittest discover -s tests -t .`，并保留现有GUI自检命令。新schema沿用 `python -m tensor_mapping model/search`，README给出至少一个M1复跑示例。
