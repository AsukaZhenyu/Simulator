# 与拆分策略匹配的可复用测量

这套工具测量真实执行，不补合成价格。其端口和计时边界独立于完整 Dense block，价格不能混用。

## 实际执行范围

`native/matched_mechanisms/compute.cpp` 使用已有 GGML 库，F32 权重由显式 seed 生成并保存为 little-endian 文件。它不加载预训练权重，不预测专家路由。

- 专家 FFN：分别执行 gate、up、SwiGLU、down；完整串行链与完整联合图都执行并计时。输入是已经选出的 compact token 行。这里没有 router、gather、scatter、RMSNorm 或残差，不能把价格直接填成 Dense decoder FFN。
- Attention：输入为已完成 norm/RoPE 的 Q 和新增 K/V；纯 core 执行完整历史 KV 的物化拼接、因果 GQA、softmax、PV。输出处理单独执行 sigmoid gate、O projection 和 residual。完整 core→output 链及联合图都计时。KV 拼接/物化是这一具体实现的成本边界，不能无条件用于采用另一种 KV 布局的实现。
- FFN 混合路径：权重 H2D→GPU 四阶段；选中行 D2H→CPU 四阶段→输出 H2D；CPU gate/up/SwiGLU→中间结果 H2D→GPU down。权重和数据都通过显式 CUDA copy 真实搬运，分配、初始化与数值检查不计时；host pinned staging 的 CPU memcpy 计入完整路径。

每个 GGML 张量的 `ne`、逻辑字节和输入 fixture 摘要都有产物。gate/up 权重为 GGML `ne=[H,I,1]`，down 为 `[I,H,1]`；`ne0` 连续，矩阵按输出行展开。选中 token 数据为 `[H,q,1]`，gate/up/activated 为 `[I,q,1]`，输出为 `[H,q,1]`。Q/K/V 是 `[head_dim,q,heads/kv_heads]`，与 token-major FFN 排列不同。

权重和原始输入的文件是 native 初始化后保持不变的实际 F32 值，可以做物理 SHA 绑定；gate/up/activated/context 等文件是初始独立 oracle/reference，执行中会重新计算，实际值可能相差若干 ULP。侧表标记 `fixture_identity_scope`，没有声称捕获计时瞬间的中间张量位级身份。实际中间/最终端口由 native 数值检查覆盖。

`transfer.cu` 真正执行以下动作：

- CE H2D / D2H：pinned host buffer 与 `cudaMemcpyAsync`。
- SM H2D / D2H：CUDA kernel 直接读写映射的 pinned host memory（UVA）；这是一种具体 SM 实现，不能替代其他系统的 SM copy 实现。
- CE H2D 与 SM D2H 在两个 stream 同时提交：保存完整 wall 时间和各自 CUDA event 区间，验证全部字节。
- empty kernel 与 empty events：独立记录启动/控制路径的观测时间。它们包含 enqueue、event 与等待，不能当成纯 launch overhead，也没有从计算或搬运时间中相减。

它没有由这些样本反推出通用 PCIe 公平分配规律，也不证明共享链路模型已校准。

## 构建与运行

新目录有独立 CMake 项目，显式提供已有 GGML 源码与 build，依赖没有被重编译。CPU-only 可设置 `MATCHED_ENABLE_CUDA=OFF`；请求 GPU 时不会偷偷使用 CPU。

```powershell
cmake -S profiling/native/matched_mechanisms -B <new-build> `
  -DGGML_SOURCE_DIR=<existing-ggml-source> `
  -DGGML_BUILD_DIR=<existing-ggml-build> -DMATCHED_ENABLE_CUDA=ON
cmake --build <new-build> --config Release

python profiling/matched_mechanisms/experiment.py `
  --suite profiling/matched_mechanisms/suite.json `
  --compute-binary <matched-compute-probe.exe> `
  --transfer-binary <matched-transfer-probe.exe> `
  --runtime-dir <matching-ggml-runtime-dir> `
  --runtime-dir <cuda-runtime-dir> `
  --timeout-s 180 --output <new-experiment-directory>
```

所有形状、精度、CPU 线程/逻辑处理器、GPU index、预热、样本数、seed、质量门都在 suite 显式声明。当前 Windows CPU 合同只支持一个明确绑定的逻辑处理器和一个线程；其他系统 native 没有实现 affinity，不能声称绑定生效。换机器需明确修改 CPU id 和路径。CUDA graphs 明确关闭。

程序取得所有测量工具共用的 `.measurement.lock`，逐个启动测量进程，不关闭用户应用，不调功耗或时钟。profile 完成后冻结模型，才启动 heldout。所有离群点、失败输出及不通过的门都保留。

## 校准的意思

`prediction-model.json` 只使用 profile 样本拟合精确形状中位数。`report.json` 比较新 seed、新进程的完整链/路径时间，还单列阶段加搬运的 component-sum 预测误差。

接受条件同时包括进程内 CV、同一阶段不同进程中位数的 CV、heldout 相对误差；不会只看 pooled median 把进程漂移藏起来。分层 bootstrap 区间保存不确定性，但不是新形状的泛化保证。

串行总时间与各阶段中位数之和的差不能被解释为独立测得的派发开销。完整图与拆分图采用不同执行边界，价格也分别保存。

CPU 工作区来自 graph gallocr buffer 与显式 CPU cplan scratch。GPU 只报告可观察的 gallocr 大小，私有 CUDA scratch 保持 `null`，因此没有真实 GPU 容量验收；不会把未知写成 0。

```powershell
python profiling/matched_mechanisms/verify.py `
  --experiment <completed-directory> --output <new-verification.json>
python profiling/matched_mechanisms/export_profiles.py `
  --experiment <completed-directory> --output <new-stage-port-profiles.json>
python profiling/matched_mechanisms/replay_serial.py `
  --experiment <completed-directory> --output <new-serial-replay-directory>
python -m unittest discover -s profiling/matched_mechanisms/tests -v
```

核验器重新检查原始 native 输出、命令、数值、端口、fixture 文件及 SHA、profile-only 模型和 heldout 门。当前统计代码必须与执行快照一致。

导出 `matched-stage-port-profiles-0.1` 侧表，而不是自动填 M2。每项包含 exact shape、device、输入输出端口、source fixture SHA、workspace、duration 与 `latency_quality_accepted`。消费者必须匹配完整语义和物化合同；质量不通过的条目只能用于诊断。GPU workspace 未知时仍不能生成需要已知 workspace 的成本条目。

FFN probe 的 history=0 是没有 KV 端口的显式占位；侧表声明 `history_axis=not_applicable_no_KV_input`，并非测量了历史长度 0 的特殊 FFN 算子。Attention 则必须精确匹配可见历史，不能复制价格到另一历史长度。

`replay_serial.py` 把侧表绑定到独立的 CPU tensor-port 图，真正通过 Modeling 的动作/依赖/工作区账本执行，并把预测总时间与 heldout 原生串行链比较。图使用现有 `synthetic_layer` 结构标签，实测物理语义和数值证据保存在外部侧表；这没有放宽 `moe-execution` 的 synthetic gate。质量失败时仍保存诊断，不产生 accepted 成本表。GPU capacity replay 明确拒绝，因为 workspace 未知。逻辑 DRAM 预算和 alignment 是显式 replay 条件，不代表真实进程内存已经校准。

## 明确未完成

量化阶段、CUDA graphs、真正跨设备 Attention 链、多线程利用率变化规律、请求/层间完整流水、任意 SM kernel 的泛化、真实预训练模型、通用并发链路模型和 GPU 私有工作区尚未校准。

本机历史同边界实测与组件诊断不随仓库分发。新环境应按本页命令先生成测量和侧表，再运行核验及 `replay_serial.py`；[执行变换校准工具](../execution_calibration/README.md)另提供组件到组合路径的独立验证。当前完整性能质量门尚未通过，不能据此声称完整 Dense/MoE 或论文系统的性能复现。
