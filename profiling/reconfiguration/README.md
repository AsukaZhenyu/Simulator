# 权重驻留与重新加载：可复用实测子集

这个入口回答一个小问题：同一份 F32 FFN 数据，权重已经在 GPU 时需要多久，重新搬入权重再执行同一条计算链需要多久。它复用 `matched_mechanisms` 的 GGML CPU/GPU 程序，不另写假计算，也不修改已冻结的 CPU 诊断和旧数据。

## 实际测了什么

| 项目 | 真正执行的内容 | 不能据此声称什么 |
|---|---|---|
| resident FFN | GPU gate、up、SwiGLU、down，四阶段各同步一次 | 完整 decoder layer、真实模型吞吐 |
| weight reload FFN | CPU 权重 memcpy 到 pinned staging，三份权重 H2D，各等待完成，再执行相同四阶段 | 冷启动、容量回收、异步完美重叠 |
| CPU FFN、混合路径 | 相同输入和权重，CPU 一线程与明确 CPU→GPU stage 路径 | 操作系统 CPU 占用率规律、多线程可用性规律 |
| allocator APIs | 独立 CUDA 程序执行 cudaMalloc、pinned H2D、idle stream synchronize、cudaFree | GGML 分配器把显存返还给系统、策略控制器开销 |

FFN 内没有 router、gather/scatter、RMSNorm 或残差。输入是已经选好的 compact token 行。`history=0` 是没有 KV 端口的显式占位，不能外推为 attention 的历史长度实验。

重新加载路径是**覆盖已经分配好的 GPU 权重 buffer**。FFN 的分配、初始化和数值检查均在计时外。单独的 allocator probe 记录分配/释放 API 的墙钟及内存检查点；它没有假装做完“释放权重→重建图→重载→继续真实 LLM”的全链。逻辑 `EVICT`、CUDA `cudaFree` 和其他应用真正拿到显存，是三件不同的事。

## 配置与复用

`suite.json` 明确给出形状、F32、逻辑 GPU index、CPU thread=1、CPU id、seed、预热、样本数、profile/heldout 和预先声明的质量门。首个套件只用 96 KiB、1.5 MiB 权重，避免在共享桌面上造成持续压力。换机器请显式修改 CPU id 和 device，不默认猜测硬件配置。

需要已有 GGML CPU/CUDA 与 `matched-compute-probe`。构建方法见 [matched_mechanisms 文档](../matched_mechanisms/README.md)。新增 allocator probe 是独立 CMake 项目，不下载或重建 GGML：

```powershell
cmake -S profiling/reconfiguration/native -B <new-allocator-build> `
  -DCMAKE_CUDA_ARCHITECTURES=<your GPU architecture>
cmake --build <new-allocator-build> --config Release
```

Windows 需在已配置 MSVC/CUDA 的开发环境中构建。若 CUDA Toolkit 与新 MSVC 的版本检查不兼容，应按本机已验证的构建方式明确处理；不能在工具中静默忽略版本检查。本机编译命令与实际输出单独保存在 `allocator-build.json`，其中的路径是本机证据，不是可复用默认值。

先检查计划，完全不调用硬件：

```powershell
python profiling/reconfiguration/run.py --suite profiling/reconfiguration/suite.json --dry-run
```

实际测量（输出必须是新目录）：

```powershell
python profiling/reconfiguration/run.py `
  --suite profiling/reconfiguration/suite.json `
  --compute-binary <matched-compute-probe executable> `
  --allocator-binary <reconfiguration-allocator-probe executable> `
  --runtime-dir <matching GGML runtime directory> `
  --runtime-dir <CUDA runtime directory> `
  --timeout-s 180 --output <new experiment directory>
```

所有测量使用已有 `.measurement.lock` 串行执行，不关闭用户应用，不改变 CPU/GPU 时钟或系统服务。锁不能阻止其他应用占用机器。suite 中的 `run_context` 和 manifest 的前后环境快照记录干扰范围，测量并不保证独占设备。

核验与测试无需硬件：

```powershell
python profiling/reconfiguration/verify.py `
  --experiment <completed experiment> --output <new verification.json>
python -m unittest discover -s profiling/reconfiguration/tests -v
```

## 产物和质量门

`ffn/` 保留已有工具的原始 stdout、逐样本纳秒、数值检查、实际 F32 little-endian fixture、端口/形状、CPU 工作区来源、GPU 未知工作区、执行源代码和依赖 hash。profile-only 模型在 heldout 开始前冻结。

外层 `raw/`、`allocator-measurements.json` 保存 allocator 程序的全部输出与内存检查点。`report.json` 同时保存 resident/reload 两条路径的质量结果、同进程同 fixture 的中位数比较、非退化输入检查和 API 稳定性。它们按指标分组计时，**不是交替顺序的 paired benchmark**；不能通过两中位数相减，称为已独立测得的纯搬运或派发开销。

FFN 使用原有门：进程内 CV ≤ 0.15、同 split 独立进程中位数 CV ≤ 0.20、heldout 相对误差 ≤ 0.20。只有两条路径同时通过，`both_latency_gates_passed` 才为真。全部离群点保留，不在失败后放宽门槛。

allocator APIs 使用同样的 CV 门检查重复稳定性，但没有独立 heldout 预测验收，所以只能称 `stable_api_observation`，不能称为已校准模型。

GPU `workspace_bytes`、`logical_evict_cost_ns`、`mapper_control_cost_ns` 均为 `null`。即使 CUDA malloc/free 观察很稳定，也不能自动填入控制器的迁移成本或 GPU 容量模型。本工具不会写 M2 成本表或模拟器配置。

## 下一步的准确校准边界

正式研究首先应选定真实 runtime 的权重 buffer 生命周期、图重建/复用和安全点，执行一次完整变更，再测它与继续驻留的配对代价。控制器计时必须覆盖同一控制入口，GPU 私有 scratch 必须另取可信上界，CPU 干扰必须实际跑受控负载。这里只提供这些工作所需的最小物理证据和可重跑入口。
