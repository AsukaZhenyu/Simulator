# 执行变换性能校准

本目录给出可复用的测量、分析和原始证据核验工具。它验证具体物化后的执行边界，不把小型 FFN/Attention 核心实验称为完整 Dense/MoE、多层或多轮推理校准。

## 测什么

新 native probe 为 `../native/execution_transforms/compute.cpp`，通过源码包含复用原 matched probe 的 GGML 会话和数学 oracle；两份源文件均进入实验哈希清单。

- FFN：生成 F32 gate/up/down 权重，沿中间维分成 CPU 和 GPU 两部分；两个分支真实提交执行，搬回 GPU 部分输出后在 CPU 求和。输入与权重预先驻留，所有比较路径最终都把输出交付 DRAM。没有 norm、residual、MoE 路由或 gather/scatter。
- Attention：post-RoPE QKV，真实执行 QK、因果 mask/softmax、PV、sigmoid gate/O 投影/residual。比较全 GPU 与 CPU QK/softmax→概率 H2D→GPU PV/O，两者最终输出 DRAM。KV 具体物化由 native `boundary` 声明，不冒称覆盖所有 IR attention 变体。
- 预载：FFN 每次先把 GPU 分片的三份权重从 DRAM 搬到 GPU，再执行混合分片。它是明确计费的重载路径，不是冷缓存实验，也没有验证预取与计算重叠。

CPU 计算线程数固定为显式 1。首版单进程固定到所选逻辑处理器，GPU dispatch worker 继承该 affinity；后续版本分别验证主线程与 worker 的实际 affinity，见下方最新入口。其系统软件竞争是本实验条件的一部分，不能推导任意 CPU 多线程/资源竞争规律。

新增 probe 对矩阵乘法显式使用 `GGML_PREC_F32`；runner 同时为子进程设置并记录 `NVIDIA_TF32_OVERRIDE=0`，固定原有数值验收容差，不通过放宽误差门掩盖 TF32 的舍入误差。旧 matched baseline 使用原始精度合同，因此新旧 probe 的耗时差异不能被解释为策略收益。

## 如何预测

只用 profile 进程的独立阶段时间拟合精确形状表。在任何 heldout 子进程启动前，保存 `prediction-model.json` 并冻结 SHA256。profile/heldout 的 seed 分离，每个 repeat 都是新 native 进程；不会拿同进程的不同计时样本当独立进程验证。

FFN 预先声明两种预测模型：

```text
GPU = sum(GPU full gate/up/activation/down) + output D2H
Mixed ideal = max(sum(CPU shard stages), sum(GPU shard stages)) + output D2H + CPU sum
Mixed dispatch model = separately measured empty-worker handshake + Mixed ideal
Preload = separately measured three-weight H2D + corresponding Mixed prediction
```

空 worker 握手只是一个预先声明的机制假设，不是从 heldout 拟合的截距。它可能与计算重叠或重复计算某些开销，因此也必须接受独立验证。`hybrid/parallel_branches` 实测总时间只作诊断，禁止回填成预测成本。

Attention 按真实顺序求和：全 GPU 的 QK/softmax/PV/output/D2H；混合链的 CPU QK/softmax、概率 H2D、GPU PV/output/D2H。

不同初始条件不混排：resident 策略只与 resident 比较；preload 单列。CPU 输出与 GPU 输出位置不同的旧 matched 路径也分组比较。数值中位数排序和置信区间不重叠的排序分开，质量失败保留为诊断。

## 运行

由一个进程串行运行硬件。工具使用 profiling 共享 `.measurement.lock`，不移除别人持有的锁；不会关闭用户应用、调时钟或改电源计划。

```powershell
python profiling/execution_calibration/run_native.py `
  --suite profiling/execution_calibration/transform_suite.json `
  --binary profiling/native/execution_transforms/build-local/execution-transform-probe.exe `
  --runtime-dir <matching-GGML-runtime-directory> `
  --runtime-dir <CUDA-runtime-directory> `
  --timeout-s 240 --output <new-experiment-directory>
```

正式 suite 预先声明 7 个案例、profile seed 17、heldout seed 29、各 repeat 2，共 28 个独立进程。每个指标 51 样本，warmup 20 且最低 200 ms。质量门沿用 15% 进程内 CV、20% 进程间 CV、20% heldout 误差；失败不删除、不改门。

加入 `--dry-run` 只输出计划，不执行硬件。通过 `--preserve-file` 可额外保护既有二进制或诊断资产的哈希。

输出包括命令、子进程 PID、启动/结束时间、完整 stdout/stderr、实际继承运行标志、原始样本、数值检查、输入/权重 digest、源代码快照、binary/runtime 前后哈希、冻结模型，以及 `analysis/REPORT.md`、JSON、CSV 和 SVG。

所有计时样本保留。CV 超门、进程失败、缺阶段或未知项会原样显示；小误差本身不足以接受。GPU 私有 workspace 为 `null`，不会被替换成 0，也不导出已验收的 GPU 容量成本表。

## 离线核验

```powershell
python profiling/execution_calibration/verify_native.py `
  --experiment <completed-experiment-directory> --output <new-verification.json>
```

核验 raw SHA/命令/数值与 workspace，重建 profile-only 模型，检查每个 heldout 对冻结模型的绑定，重算误差、CV、bootstrap 和排序。当前分析实现必须与实验实际执行源码哈希一致；分析公式修改后应进行新实验，不改写旧证据。

旧 matched native 的完整实验可用以下命令重算“组件预测真实组合链”的误差，而不是把 joint 重测当作组合模型准确度：

```powershell
python profiling/execution_calibration/analyze_matched.py `
  --experiment <completed-matched-experiment> --output <new-analysis-directory>
```

该命令先调用旧核验器。旧 Attention 仍只有 core/output，旧 FFN hybrid split 仍是 CPU gate/up/activation→GPU down；不能据此宣称新分片与 QK/softmax/PV 已被校准。

无硬件协议测试：

```powershell
python -B -m unittest discover -s profiling/execution_calibration/tests -v
```

## 推荐的双逻辑处理器实验

首次 `transforms-01` 的 20 个 FFN 进程中，host bracket overlap 全为 0。该版把 CPU 计算线程和 GPU dispatch worker 绑定到同一逻辑处理器；它是完整保留的诊断实验，不能据此声称 CPU/GPU 并发重叠已被校准。

后续使用独立的 `run_dual.py` 和 `../native/execution_transforms_dual/`，保持首次运行的源码、二进制、模型、样本和报告不变。主线程绑定逻辑处理器 0，FFN 的 GPU dispatch worker 绑定逻辑处理器 2；native 读取并返回实际 process/thread affinity 掩码，runner 检查实际值与显式请求一致。Attention 本身是分段串行链，不会创建并行 worker，不能把配置双核解释为该路径在并行执行。

```powershell
$env:GGML_CALIBRATION_GPU_DISPATCH_CPU_ID='2'
python profiling/execution_calibration/run_dual.py `
  --suite profiling/execution_calibration/transform_dual_suite.json `
  --binary profiling/native/execution_transforms_dual/build-local/execution-transform-dual-probe.exe `
  --runtime-dir <matching-GGML-runtime-directory> `
  --runtime-dir <CUDA-runtime-directory> `
  --timeout-s 240 --output <new-dual-experiment-directory>
```

该后续协议继续使用原先质量门和预测公式，profile seed 为 41、heldout seed 为 53，仍为 7 案例、28 个新进程。变更包括明确记录的 affinity 条件、采样种子和运行时间，未用首次 heldout 数据拟合新价格或放宽验收门；它并非配对单因素消融。host bracket 只是一次未计时诊断调用的完成区间；即使观察到重叠，也不是逐 kernel 的 GPU event trace。

双核实验须使用对应核验器，额外核验实际 affinity 与 child environment 绑定：

```powershell
python profiling/execution_calibration/verify_dual.py `
  --experiment <completed-dual-experiment-directory> --output <new-verification.json>
```

首次实验的 `audit_results.py` 包含单逻辑处理器的说明，不应直接套作双核报告。所有诊断残差只用于分析失配原因，不替换正式冻结成本；GPU 私有 workspace 和真实多层/多轮 LLM 性能验证仍未完成。

## 最新推荐：组件采样与策略使用同一 worker

双逻辑处理器版中，独立 GPU shard 的 wall 成本在主线程测量，而混合策略在 worker 执行。最新 `execution-transform-context-probe` 把 GPU shard 独立阶段的采样放到同一个持久 worker；阶段计时不含外部唤醒/等待，空握手继续单独测量。完整 GPU 策略和其组件都在主线程执行。构建命令见 [native README](../native/execution_transforms_dual/README.md)。

```powershell
$env:GGML_CALIBRATION_GPU_DISPATCH_CPU_ID='2'
python profiling/execution_calibration/run_dual.py `
  --suite profiling/execution_calibration/transform_context_suite.json `
  --binary profiling/native/execution_transforms_dual/build-local/execution-transform-context-probe.exe `
  --runtime-dir <matching-GGML-runtime-directory> `
  --runtime-dir <CUDA-runtime-directory> `
  --timeout-s 240 --output <new-context-experiment-directory>

python profiling/execution_calibration/verify_context.py `
  --experiment <completed-context-experiment-directory> --output <new-verification.json>
```

最终协议使用新的 profile seed67、heldout seed79，仍为28个独立进程。核验器额外检查采样线程角色、实际 mask/group、精度环境和context源码快照。2026-10-17的本机最终轮7组resident中位数排序正确，但19条主预测均未通过全部稳定性/误差门；不能把该结果描述成可靠选优保证。历史原始证据与报告只保留在开发者本机，不随仓库分发；当前用户应按上述 `run_dual.py` / `verify_context.py` 命令在目标硬件重新采集、核验。

已完成 context 实验后，可以对新生成的 `analysis/report.json` 绘图；绘图额外需要 matplotlib 和 numpy，它们不是核心库依赖：

```powershell
python profiling/execution_calibration/plot_report.py `
  --experiment <completed-context-experiment-directory> `
  --output <new-figure-directory>
```

`summarize_final_report.py` 是本机多轮 campaign 的历史汇总辅助程序，要求 baseline、原始/dual/context 等完整目录结构，不是单次新实验的通用入口。原先进入测量清单的 `summarize_experiment.py` 保留原字节和源码快照。预测、门槛和统计仍来自冻结的 `analysis.py`，未因报告编写而调整。
