# 2026-10-12 验收：工具可复用，时间数据没有通过校准

这是开发者本机实验的历史说明。以下 `results/` 与构建记录只在原工作区保留，不随库或 Git 仓库分发，也不是安装和使用前提。新环境请从[工具说明](README.md)生成自己的证据，并运行 `python -B -m unittest discover -s profiling/reconfiguration/tests` 检查无硬件合同。

这轮完成了真实的权重驻留/重新加载测量入口，以及独立 CUDA 分配/释放观察。**数值和证据核验通过，性能质量未通过；没有生成可接受的真实成本表。**

## 证据位置

- 最终实验：`results/20261012-desktop-02/report.json`。
- 独立重算核验：`results/20261012-desktop-02/verification.json`。
- 执行环境、源代码/二进制 hash：`results/20261012-desktop-02/manifest.json`、`results/20261012-desktop-02/ffn/manifest.json`。
- 原始 FFN stdout、数值检查、逐轮样本：`results/20261012-desktop-02/ffn/raw/`，实际 F32 fixture 在相邻 `fixtures/`。
- allocator 原始 stdout 与逐轮内存检查点：`results/20261012-desktop-02/allocator-measurements.json`、`results/20261012-desktop-02/raw/`。
- 18 项无硬件合同测试：`results/contract-tests.json`、`results/contract-tests.log`。
- 本机编译命令和成功输出：`allocator-build.json`。构建目录没有进入可复用默认配置。

当前可复用统计源代码与最终实验的执行快照一致。重算器重新验证原始样本、命令、fixture SHA/FNV、端口与形状、数值检查、未知 workspace、profile-only 模型和 heldout 门。源文件或二进制在两端 inventory 中没有改变。旧的原生 probe 文件/二进制也由已有 FFN manifest 保护；没有修改冻结 CPU 诊断或旧成本数据。

首轮 `20261012-desktop-01` 保留为工具开发期间的初步实验，当时 3/44 个 FFN 指标通过、0/2 个路径比较可接受。之后只收紧了 JSON 布尔/整数合同，并明确共享桌面运行上下文；最终源代码冻结后重新独立执行了一次。没有删掉首轮样本，也没有依据失败结果改变质量门或选择更好的一轮作为成本表。

## 完成与未通过的分别是什么

| 检查 | 最终结果 | 意思 |
|---|---:|---|
| FFN native 进程 | 8/8 成功 | profile/heldout、两种形状、各两次独立进程 |
| allocator native 进程 | 4/4 成功 | 两种字节量，各两次独立进程 |
| FFN 全部端口数值检查 | 通过 | CPU/GPU/混合路径与独立参考一致，在声明 atol/rtol 内 |
| 权重/输入非退化检查 | 通过 | 实际 F32 有足够不同值，避免整数/2048 输入的虚假精确性 |
| allocator 全字节校验 | 通过 | 每次 H2D 后 D2H 比较全部字节，比较在计时外 |
| FFN 单指标完整质量门 | 0/44 | 不能当作 accepted latency profile |
| resident/reload 两路径共同通过 | 0/2 | 不能用本机数字声称重配置收益 |
| allocator API 重复稳定性 | 0/8 | 仅为诊断，且本来没有 heldout 预测验证 |
| 无硬件合同测试 | 18/18 | 关键签名、未知项、退化输入和非法数据严格拒绝 |
| 原始证据重算 | 通过 | 证明报告诚实地对应产物；不等于时间数据可靠 |

## 时间诊断，不能当成 GPU 的固有速度

这里的时间都是关闭 CUDA graphs、单次完成同步的 host wall time。GPU四阶段链按阶段同步，重新加载还包含 pinned staging memcpy 与 H2D 等待。同进程同 fixture 的指标是按组先后测量，没有交替随机配对；不能把两个中位数之差当成独立识别的纯搬运成本。

| 明确 FFN 子集 | resident heldout 中位数 | reload heldout 中位数 | resident 最大进程内 CV | reload 最大进程内 CV |
|---|---:|---:|---:|---:|
| H=64, I=128, q=1，权重 96 KiB | 2.6422 ms | 3.0544 ms | 0.416 | 0.319 |
| H=256, I=512, q=1，权重 1.5 MiB | 2.37695 ms | 3.2055 ms | 0.674 | 0.269 |

预先声明的 CV 门为 0.15，两组均失败。较大形状的 resident profile→heldout 误差为 38.7%，也超过 20% 门；profile 的独立进程中位数 CV 达 0.558。不能用较大形状更快的表面现象得出硬件规律。

测量在 RTX 4070 Laptop GPU 的正常共享桌面上执行；开始前 NVIDIA 观察显示约 6.5/8 GiB 已占用。工具不关闭其他应用、不控制功耗/频率，并明确记录其他 agent 编码/测试可能同时发生。当前没有足够证据把波动唯一归因于 CUDA、操作系统、其他应用、GPU 时钟或本 probe；这里保留未知原因。

## 分配释放的实际边界

新增 allocator 程序测的是直接 `cudaMalloc/cudaFree`，不是 GGML allocator/pool；每次释放前数据搬运已完成。它保存了前、已分配、释放后的 `cudaMemGetInfo`，没有强制它们与请求字节数相等。

最终共享设备观察中，1.5 MiB 案例的“释放后 free 减释放前 free”在部分样本为负，在另一些样本为正，范围约 −15.8 MiB 到 +15.7 MiB。这不是本次权重释放字节的可靠计量，不能用于证明 GGML 或模拟器的逻辑 `EVICT` 把显存交还给其他应用。准确值见逐样本 checkpoints。

GPU `workspace_bytes=null`，`logical_evict_cost_ns=null`，`mapper_control_cost_ns=null` 保持未知。idle synchronize 虽是实际 API 观察，也不能代替策略解析、图修改、检查点协调、驻留转换或编译器的控制开销。整个 CPU 占用率变化规律与真实 Agent 干扰也没有被这套子集校准。

## 这轮对主项目的贡献

1. 获得可以跨机器重跑、严格保留物理边界的测量入口，免于以后拿合成搬运时间冒充校准。
2. 给研究 runner 一个明确门槛：只有 exact shape/device/thread/variant 匹配且两条路径质量通过，才能有限使用相应 latency 观察；未知 GPU 工作区仍不能用于真实容量证明。
3. 留下可重复的反面证据：正常共享桌面的时间波动目前足以淹没小任务的重配置比较。模拟实验可以继续研究方法，但不能声称这些真实数据已经验证了性能收益。

下一轮应先固定真实执行后端与变更入口，测量完整的“安全点→释放/保留→图复用或重建→重载→继续执行”路径；随后在受控条件下使用交替配对与独立进程 heldout。质量门继续保留，不能通过填写 GPU workspace=0 或删离群点补齐。
