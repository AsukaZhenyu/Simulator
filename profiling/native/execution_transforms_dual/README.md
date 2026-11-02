# 双逻辑 CPU 的执行变换校准变体

这是独立实验变体，不修改 `execution_transforms/` 的源、程序或正式 v1 结果。`compute.cpp` 从 v1 的固定 SHA256 复制，图构建、权重分片、数值参考、容差、三条 FFN 策略、Attention 策略和计时边界保持相同；只改变显式线程亲和性及其观测字段。来源与构建身份见 `source-provenance.json` 和 `build-local/compute-build.json`。

v1 的全部 FFN 未计时完成区间没有重叠；因此本变体研究是否是单逻辑 CPU 上的派发争用阻止了两分支重叠，而不是直接宣称原 `max(CPU,GPU)` 模型不成立。

所有 CLI 参数沿用 v1。运行器还必须显式传入环境变量：

```text
GGML_CALIBRATION_GPU_DISPATCH_CPU_ID=<与 --cpu-id 不同的逻辑 CPU 索引>
NVIDIA_TF32_OVERRIDE=0
```

没有 GPU 派发 CPU 的默认值。旧 helper 完成参数解析后，将 Windows 进程亲和性设置为两个逻辑 CPU 的并集，主线程绑定 `--cpu-id`，持久 GPU worker 绑定显式派发 CPU。设置失败或回读不一致直接失败；仅支持 Windows processor group 0，其他系统拒绝，不能静默退回单 CPU。不同逻辑 CPU 不自动保证不同物理核。

输出保留 `execution-transform-native-0.1` schema 与所有 v1 metrics，增加：

```json
{
  "dispatch_cpu_id": 2,
  "affinity_variant": "dual_logical_cpu_explicit",
  "observed_affinity": {
    "process_mask": 5,
    "main_thread_mask": 1,
    "main_thread_group": 0,
    "dispatch_worker_created": true,
    "dispatch_thread_mask": 4,
    "dispatch_thread_group": 0
  }
}
```

上例仅解释 CPU 0 和 CPU 2 的掩码含义，不能当作换机器时的隐含默认值。掩码来自 `GetProcessAffinityMask` 与 `GetThreadGroupAffinity` 的实际回读。FFN 才创建持久 worker；Attention 的 CPU→GPU 链本身串行，`dispatch_worker_created=false`，dispatch mask/group 为 `null`。

CPU/GPU 的额外未计时完成区间仍保留；即使回读亲和性正确、host 区间开始重叠，也不能将其解释为已证明的 CUDA kernel 重叠比例。是否改善预测误差应由新 seed、新进程的独立实验检验；不能删除 v1 的失败或修改其验收门。

## 推荐构建与运行

推荐新实验使用 `execution-transform-context-probe`。它把 GPU shard 的独立阶段测量放在混合策略实际使用的 GPU worker 内部，排除外层唤醒计时；fullGPU 仍在主线程测量。原 `execution-transform-dual-probe` 目标保留，用于复核上一轮对照。两者共用同一套显式 GGML/CUDA 构建依赖。

```powershell
cmake -S profiling/native/execution_transforms_dual -B <new-build> `
  -DGGML_SOURCE_DIR=<existing-ggml-root> -DGGML_BUILD_DIR=<existing-build-root>
cmake --build <new-build> --config Release --target execution-transform-context-probe
```

CLI 参数与 v1 相同，所有参数仍需显式提供；从上一节所列实际掩码选择 `--cpu-id` 和 `GGML_CALIBRATION_GPU_DISPATCH_CPU_ID`，并由实验运行器显式对子进程设置 `NVIDIA_TF32_OVERRIDE=0`。生成的程序需要与输入 build 匹配的 GGML/CUDA DLL；不将本机库路径或 CPU 索引作为默认值。也可以用同一 CMake 工程构建原 `execution-transform-dual-probe` 目标。

context raw 增加 `profile_gpu_shard_thread=dispatch_worker_same_as_parallel_strategy` 与实际回读的 `profile_gpu_shard_affinity_mask/group`。Attention 没有 FFN shard，这些字段明确标为 not applicable / `null`。来源及精确改动保存在 `source-context-provenance.json`、`source-context.diff`；本机构建记录在 `build-local/compute-context-build.json`。本目录的构建本身没有执行硬件测量。
