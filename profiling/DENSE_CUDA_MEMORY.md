# GPU 内存证据与显式预算

本工具提供可复用的 GPU 内存观测和容量输入绑定。它不输出性能成本，也不以设备内存快照冒充完整 workspace 或进程峰值。

## 两种证据各自说明什么

| 字段 | 含义 | 可以支持的结论 |
|---|---|---|
| resident_logical_bytes | 声明的权重、端口、KV逻辑字节 | 张量账目是否完整 |
| resident_allocated_bytes | GGML resident buffer公开分配容量 | resident分配与逻辑账目之间的padding |
| internal_allocated_buffer_bytes | gallocr公开图内buffer容量 | 已知图内分配组件 |
| cudaMemGetInfo free/total/used | 指定CUDA设备在同步checkpoint的可用内存查询 | 那一时刻的设备范围观察 |
| workspace_bytes / process_peak_bytes | 当前为null | 仍未知，不能转成0或上界 |

GGML 私有 CUDA pool、类型转换临时区和 cuBLAS 分配不完全包含在 gallocr 内。NVIDIA 文档说明，free查询可能与其他分配发生竞争，也不保证所有报告为free的字节都能分配。[CUDA内存API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)。查询中的total与device属性totalGlobalMem分别保留，不强制将两种口径相等。

最大used是七个checkpoint中的最大观察值，不是连续峰值、进程独占量、宏任务scratch或容量证明。relative delta允许为负，不截成0；不根据设备free推断外部预留。

## 独立原生程序与观测runner

`native/dense_layer_cuda_memory_probe.cpp` 复用 Dense CUDA 图实现，只在独立编译变体中启用观察 hook；默认性能构建的 hook 为空操作。CMake 目标为 `ggml-dense-layer-cuda-memory-probe`。沿用[CUDA 构建说明](DENSE_CUDA_MEASUREMENT.md)，在新的构建目录选择该目标，避免覆盖已冻结程序。

每条case记录七个同步checkpoint：context初始化、上游准备完成、stage分配、setup执行、warmup完成、验证重复完成、stage释放。非block任务先运行完整block准备真实上游输入，prep对象随后释放；后续观测仍经历了该准备过程和运行时初始化。报告不能把baseline差值归因于一个独立stage。block记录相同label但上游准备次数为0。

~~~powershell
python profiling/run_dense_cuda_memory.py --suite profiling/cases/dense_layer_cuda_smoke.json --native-bin "<memory-probe-directory>" --runtime-dir "<GGML DLL目录>" --runtime-dir "<CUDA Toolkit bin>" --cuda-device 0 --cuda-graphs disabled --warmup 1 --samples 3 --seed 17 --output profiling/results/my-cuda-memory
~~~

samples仅为验证执行次数。嵌套的dense_validation保留数值、KV、端口与设备驻留检查；其中wall_ns不是本阶段成本。顶层purpose=memory_diagnostics_only、timing_usable_as_cost=false，独立schema不能作为现有成本导出输入。

产物为manifest.json、plan.json、raw/、memory_report.json、summary.csv；失败和超时原始输出保留，JSON/CSV使用LF，与其他profiling程序共享测量锁。cuda_memory_contract.load_memory_profile核对完整计划、raw/hash/命令/签名/设备/逐阶段观测后供下游使用；读取可移植证据不要求当时的runtime仍安装在本机。

## 绑定明确的容量预算

M2已具有vram_capacity_bytes、vram_runtime_reserved_bytes、vram_alignment_bytes。新bind_cuda_budget.py只写这三项，参数必须全部显式给出，包括预留来源和设备UUID。相对layer/costs引用转换为原文件的绝对路径，保存原文件hash，避免输出位置变化后引用漂移。源invocation、成本和观测资料不覆盖。

~~~powershell
python profiling/bind_cuda_budget.py --invocation mapping/examples/m2/cuda-interface-synthetic-prefill-3.json --memory-profile profiling/results/my-cuda-memory --capacity-bytes <声明总预算> --runtime-reserved-bytes <声明外部或运行时预留> --alignment-bytes <声明分配对齐> --reserve-source "<预留的出处与范围>" --device-uuid "<profile中的GPU UUID>" --output profiling/results/my-budget/invocation.json
~~~

显式0预留是调用者声明，工具不会默认生成0。该工具不运行模拟、不生成workspace、不修改成本；预算sidecar分别记录model_capacity_under_declared_budget=null、workspace_bound_known=false、physical_capacity_verified=false。后续模型可以验证给定预算下的合成已知scratch计划，但这不等于证明真实GPU容量。即使预留很大，真实观察成本的workspace=None仍会在lookup处被拒绝。

## 验证与限制

`profiling/tests/test_cuda_memory.py` 与 `test_cuda_budget.py` 检查证据和预算合同。读取本次产生的 `memory_report.json` 时，以其中的原始观测、签名和未知项为准；单元测试不代替真实硬件容量验收。

完整 scratch 仍需要额外 allocator/runtime 证据或可审计上界。容量预算与性能来源分别保留声明和质量标记，不以有限次 snapshot 冒充硬上界。当前完整性能质量门尚未通过；[执行变换校准工具](execution_calibration/README.md)继续将未知 GPU workspace 保留为未知，本机历史结果不随仓库分发。
