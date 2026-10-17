# Dense 分段与联合图的配对诊断

本工具区分三个问题：独立 resident 阶段成本与联合图的差异、共享 resident 下两种执行方式的差异，以及阶段价格能否预测下一次独立执行。它们不能互相替代为“模拟器准确率”。

## 执行与计时边界

`native/dense_layer_probe.cpp --stage paired` 将同一已分配完整图按节点区间分成 projection/attention/FFN。分段路径与联合图共用 resident、内部地址、backend 和计算线程池；下游读取实际的上游输出，没有额外外部 copy。每轮按 AB/BA 交替比较：A 为三段连续执行，B 为一次联合执行。

- `graph`：每次 backend graph compute，包含计划和提交。
- `plan`：计时前创建并复用四份执行计划，仍包含计算和同步；四份独立 cplan scratch 全部记账，不以联合 scratch 替代。

分段成本与联合成本可能因计划、派发、节点同步、缓存工作集和 allocator 布局不同而变化。不能乘一个经验比例，把一种执行方式的价格改称另一种执行方式。

逐轮保存 projection、attention、FFN、阶段和、外层 serial 与 joint 时间。全部端口/KV 在计时外逐字节比较，独立数值参考另检查；摘要不替代数值正确性证据。

热身同时满足显式次数和最短持续时间。`--cpu-mask` 按当前拓扑为每个计算线程指定可用逻辑处理器；Windows 合同限定 group 0，并核对实际 OS mask。未绑定则保留 unbound/unknown，不猜默认 CPU。

## 复用入口

从 `Simulator/` 运行，提供匹配的 GGML runtime 和具有 paired 支持的程序。下列路径占位符必须替换；需要构建时使用新目录，避免覆盖冻结程序。

```powershell
& "<paired-probe.exe>" --cpu-topology
python profiling/run_dense_paired_measurements.py --suite profiling/cases/dense_error_grid.json --native-bin "<paired-probe.exe>" --runtime-dir "<matching GGML runtime directory>" --compute-mode graph --threads 2 --warmup 10 --min-warmup-ms 500 --samples 40 --seed 17 --repeats 2 --cpu-mask "<observed logical IDs>" --split development --output profiling/results/my-paired-development-graph
python profiling/analyze_dense_paired.py --measurements profiling/results/my-paired-development-graph/measurements.json --output profiling/results/my-paired-report.json
```

拓扑查询同样需要可解析的运行库路径。另一执行方式显式使用 `--compute-mode plan`；留出形状选择 `--split held_out`。`--dry-run` 只展示计划。输出目录须为空，测量共享 `.measurement.lock`，失败及所有样本保留。

`cases/dense_error_grid.json` 声明 12 个形状，包含小 F32 和较大 Q4_K/Q6_K 几何。development 与 held_out 按物理形状划分；不能仅改 phase、mode 或 mask 就将同形状改称留出。换线程、精度或采样方式也要重新绑定合同。

## 分析和质量解释

`analyze_dense_paired.py` 核对配对记录的 raw/stdout、命令、manifest、统计、配置与文件身份；`analyze_dense_error.py` 读取独立四阶段格式。两者服务不同数据合同，不能互换输入。

配对分析分别给出：

1. 阶段和与外层 serial：检查同批计时账本，不作为预测验证。
2. repeat 0 的阶段中位数之和预测独立进程 repeat 1 的 serial：检验同一执行方式的迁移。
3. segmented 与 joint 的差异：比较执行方式，不自动解释成模拟误差。
4. AB/BA 顺序、每阶段 CV、跨 run/mode 和最差形状。

质量目标应在测量前确定，误差小不能抵消稳定性失败。若改用批次计时，需重复完整 projection→attention→FFN 序列，保存总时间、次数和原始样本；不能重排阶段、删离群点或事后改门制造通过。

这些是共享 resident 的分段图诊断，不可直接填进独立 resident M2 成本或 workspace。测量不逐线程复刻操作系统调度，也不从后台应用数量推断可用算力比例；外部负载未测时保留未知。

## 与当前校准的关系

本工具只验证其 CPU 执行变体；GPU、分片、搬运或重叠必须分别定义边界。[执行变换校准工具](execution_calibration/README.md)提供独立组件与组合路径的 profile/heldout 流程；当前完整性能质量门尚未通过，用户应在目标硬件重新采集、核验，本机历史结果不随仓库分发。合同测试位于 `profiling/tests/test_dense_paired.py`、`test_dense_paired_analysis.py` 和 `test_dense_error.py`，不替代性能验收。
