# 独立的驻留执行变体校准

此工具测量当前机器的明确驻留执行变体，不修改被测程序、成本表或模拟器。它使用已有 native probe，为每次新运行保存原始样本和失败输出。线程、精度、历史长度、阈值和工作区按声明合同绑定。

当前真实执行范围是单设备、B1、完整声明的 Dense decoder block。CPU probe 的三个阶段与联合图共用驻留张量和内部分配，验证端口、KV 与输入内容完全一致；可以验证三阶段串行计划与联合计划。GPU probe 实际执行完整驻留联合图与三个独立驻留宏任务，验证 CUDA 设备、无 CPU fallback、数值和 KV 内容。它没有执行 GPU 分段串行链，也没有混合 CPU/GPU 链路。

主误差比较使用同一种执行方式：CPU 从 profile 的分段成本预测新进程中的分段串行总时间；GPU 从 profile 的联合图成本预测新进程中的联合图时间。CPU 另保存仅用 profile 拟合的截距：它是串行总时间中位数与各阶段中位数之和的差，包含外层计时与中位数非可加性，不能解释成单独测得的派发开销。分段成本和联合图之间的差异单列为执行方式差异；不能用这种差异证明混合调度、缓存、预取或重叠已校准。

`suite.json` 预先声明小形状功能实验以及 H2560/I9216 的 Qwen3.5 形状、P3/h0 和 D1/h3。大形状的矩阵权重使用明确的 Q4_K/Q6_K、norm 权重为 F32。生成权重用于执行与数值检查，不代表预训练模型的内容或原模型 kernel。profile seed=17、heldout seed=29，各自运行两个独立进程；pilot 不进入拟合。CPU 使用一个明确绑定的 OS 逻辑处理器，最低预热时长 500 ms；GPU CUDA graphs 显式关闭。预热、样本数和阈值在看见结果之前已写进配置。不会删掉离群点或反复调参直到误差门通过。

从 `Simulator` 目录运行；将程序和依赖路径改为当前机器的实际路径：

```powershell
python profiling/strategy_calibration/calibrate.py `
  --suite profiling/strategy_calibration/suite.json `
  --cpu-binary <existing-paired-CPU-probe.exe> `
  --gpu-binary <existing-Dense-CUDA-probe.exe> `
  --runtime-dir <matching-GGML-runtime-directory> `
  --runtime-dir <CUDA-runtime-directory> `
  --preserve-file <additional-file-to-preserve> `
  --timeout-s 240 --output <new-empty-experiment-directory>
```

程序先取得全测量工具共用的 `.measurement.lock`，顺序运行 CPU/GPU，保持其他应用、电源设置和 GPU 时钟原样。不会移除别人持有的锁。加入 `--dry-run` 可检查完整计划而不调用硬件。只运行 CPU 时显式将 `backends` 改为 `["cpu"]`；需要在换机器时明确修改 CPU mask。

`--preserve-file` 可重复，未额外指定文件时可省略。若调用方另有明确的文件哈希清单，可选传入 `--freeze-manifest <manifest.json>`；格式为 `{"files": {"<file-path>": "<SHA256>"}}`。不要引用已删除的旧实验目录，也不要覆盖被其他实验绑定的程序。

产物包括：

- `manifest.json`：CPU/GPU 环境、程序与运行库哈希、原有冻结文件前后哈希、实验协议。
- `plan.json`、`raw/`、`measurements.json`：每一个进程的命令、完整 stdout/stderr、逐轮样本、数值/KV 检查和 CV 标记。
- `prediction-model.json`：仅 profile 拟合的精确形状模型，开始 heldout 之前保存并固定 SHA256。
- `report.json`：带符号/绝对误差、预先声明的质量门，以及按进程和样本分层 bootstrap 的验证中位数区间。区间以已经冻结的预测值为条件，不能解释成新形状的泛化保证。

模型只能在关联 manifest 的机器、二进制、线程和执行边界内解读；工具不提供跨机器复用已有模型或自动 M2 导出。报告保留所有质量失败。任何 native 失败导致 `calibrated=false`，原始失败仍保存。CV 超过门或验证误差超过门时也不声称校准。

GPU 私有 scratch 始终为未知，延迟结果与容量验证分开。CPU 原始记录包含已验证的各图工作缓冲和完整驻留分配，但此工具同样不导出容量结论。真实混合链路、cache、prefetch 和 overlap 需要后续可执行策略 runner 验证。

不使用硬件的测试：

```powershell
python -m unittest discover -s profiling/strategy_calibration/tests -v
```

重新核对已完成实验，不调用硬件：

```powershell
python profiling/strategy_calibration/verify.py `
  --experiment <completed-experiment-directory> `
  --suite profiling/strategy_calibration/suite.json `
  --executed-source <experiment/source/calibrate.executed.py> --output <new-verification.json>
```

核验器重新校验每份原始 native 输出的精确绑定、数值/KV/设备与未知工作区，检查源代码、运行库、旧程序和冻结资产哈希，只用 profile 重建模型，并重算验证误差、bootstrap 区间和质量门。

当前接受门只检查各进程内 CV 和 pooled median 的点估计误差，没有把跨进程漂移纳入接受条件。因此 `calibrated=true` 只能解读为这两个条件通过。可另行生成进程级审查：

```powershell
python profiling/strategy_calibration/process_audit.py --experiment <completed-experiment-directory> --output <new-process-audit.json>
```

该审查列出进程中位数、误差和跨进程 CV，不改写原门。实验后的环境观察不能替代子进程启动时的证据。

runner 在 child 启动前记录实际继承 flags，拒绝上下文、源码、运行库或 suite 在实验中变化，并保存 `source/calibrate.executed.py`。核验指定的执行快照时必须匹配原 manifest，且统计重建函数须与快照一致。工具合同不同于[最新执行变换校准](../execution_calibration/README.md)，不能用此驻留实验覆盖混合或分片策略。
