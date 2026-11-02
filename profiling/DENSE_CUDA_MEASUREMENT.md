# Dense CUDA 测量与 M2 接口

本工具测量明确的单设备 Dense block，并导出保留未知 workspace 的观察表。其他后端或组合策略的功能通过，不能替代本合同的性能校准。

## 当前边界

独立原生程序 ggml-dense-layer-cuda-probe 在一个显式指定的 CUDA 设备上测量 projection、attention、dense_ffn、block。前三项对应 m2-layer-0.4 的三个可调度任务，block 仅供同边界组合验证，不导出第四个成本。

数学边界沿用完整 Dense 0.3：hidden/I32 绝对位置、RMSNorm、独立 QKV/可选 gate projection、可选 Q/K norm、normal RoPE、显式 causal Attention、O projection、SwiGLU 与两条 residual。B=1，激活和 KV 为 F32，norm 权重为 F32；矩阵权重可显式选 F32/F16/Q4_K/Q6_K。F32 端口不代表 CUDA 矩阵内部运算全部使用 F32；签名记录 matmul_precision=ggml_cuda_backend_default。

每个 GGML 图节点先检查指定 CUDA backend 是否支持，再检查 resident、节点及其源张量的实际 buffer 属于所选 CUDA 设备。没有混合 scheduler，不允许 CPU fallback。设备身份由 CUDA API 查询，不依赖 NVML；nvidia-smi 失败不能自动推出没有 GPU。

当前只支持 cuda_graphs=disabled，runner 显式设置 GGML_CUDA_DISABLE_GRAPHS=1。enabled/auto、CPU threads、paired stage、非 normal RoPE 等不受支持的输入会失败。该版本不声称复现原 Qwen 的交错 projection、MRoPE/NeoX、Flash Attention/F16 KV。

## 构建与运行

从匹配的 GGML 源码/已编译库和 CUDA Toolkit 构建独立目标。使用新构建目录，不覆盖已由实验 SHA 绑定的程序：

~~~powershell
cmake -S profiling/native -B "<new-CUDA-build-directory>" -DGGML_SOURCE_DIR="<GGML 源码根目录>" -DGGML_BUILD_DIR="<已有 GGML 构建目录>" -DMEASURE_ENABLE_CUDA=ON
cmake --build "<new-CUDA-build-directory>" --config Release --target ggml-dense-layer-cuda-probe
~~~

公共头、链接库和运行库必须版本匹配；更换 GGML、编译器或机器后重新构建并验证支持范围。

~~~powershell
python profiling/run_dense_cuda_measurements.py --suite profiling/cases/dense_layer_cuda_smoke.json --native-bin "<CUDA-probe-directory>" --runtime-dir "<GGML DLL 目录>" --runtime-dir "<CUDA Toolkit bin>" --cuda-device 0 --cuda-graphs disabled --warmup 5 --samples 20 --seed 17 --output profiling/results/my-cuda-smoke
~~~

cuda_device 必须显式给出。三个 suite 分别验证小 F32 图、真实层几何 Q4_K/Q6_K、F16/无可选 gate 或 QK norm/已有 history：dense_layer_cuda_smoke.json、dense_layer_cuda_qwen35_geometry.json、dense_layer_cuda_f16_validation.json。真实几何仅采用模型维度和权重类型；内容仍为确定性生成值。

输出 manifest、plan、raw stdout/stderr、measurements 和 summary.csv。共享 .measurement.lock 与其他 profiling runner 互斥，case 串行运行。失败和超时输出保留；不删除离群样本。JSON/CSV 写入 LF。

## 计时、数值与内存

初始化、分配、量化、外部 H2D/D2H、参考计算和上游输入准备均在计时外。每个 sample 计一次 ggml_backend_graph_compute_async 加一次显式 backend_synchronize 的墙钟时间，包含 host 派发和完成等待。声明的物化阶段输出复制在图内，计入时间。resident 权重/输入反复复用，不等于冷访存、纯 kernel 时间、CUDA graph 时间或并发成本。

数值检查使用实际存储/解量化权重的参考结果。小图逐元素检查，大图使用声明的确定性输出点采样；全输出有限性、KV 追加内容与旧 history 保留不抽样。导出还要求四阶段的端口摘要和上游输入来源完全一致。通过检查不等于验证预训练模型精度或完整生成序列。

已知的 resident logical/allocated bytes、gallocr allocated capacity 和内部张量 logical bytes 分别记录。GGML 私有 CUDA pool 与运行时临时开销没有完整公开统计接口，所以总 workspace_bytes、backend_scratch_bytes、runtime_overhead_bytes 保持 null，workspace_bound=unknown。不得用内部张量总和、cudaMemGetInfo 差值或 0 代替完整 workspace；不得因此宣称真实显存容量可行。

## 导出和模型侧对接

~~~powershell
python profiling/export_dense_cuda.py --measurements profiling/results/my-cuda-smoke/measurements.json --layer mapping/examples/m2/cuda-interface-synthetic.layer.json --output profiling/results/my-cuda-export --allow-unstable
~~~

仅在层描述的形状、数学路径、dtype、设备序号、graph 模式与测量完全一致时导出。默认拒绝 CV>0.1 的数据；--allow-unstable 是显式诊断豁免，不能证明校准。完整四阶段组必须齐全，block 与分段端口必须匹配；不插值、不复制 history、不替换 fixture。

GPU 导出表使用 m2-layer-0.4、device=gpu，保留 workspace_bytes=null；证据报告明确 validation_only=true、capacity_verified=false、execution_ready=false、calibration_claimed=false。这是可追溯观察表。模型可以解析并保存未知，但即使显式允许 validation_only 成本，执行 lookup 仍拒绝未知 workspace。要实际回放实测表，需要具有出处且覆盖同计时边界的 workspace 上界。

模型的接口 fixture 使用明确标注的合成时间与合成 scratch，仅验收单 GPU compute slot、VRAM 中权重/端口/KV、prefill→decode 状态与精确容量检查；不证明实测速度或物理容量。CPU 与 GPU 使用不同物理合同和格式版本，没有伪造 GPU threads。本 0.4 单设备合同不包含跨域 KV 迁移、CPU/GPU 混合执行或搬运重叠；其他策略入口有各自的合同。

## 验证与限制

该入口的接口和导出回归在 `profiling/tests/test_dense_cuda.py`。性能校准必须使用匹配边界的独立进程和预声明质量门；测试通过不能替代真实性能或容量证明。CUDA graphs、batch、跨层及真实 MoE 需要分别建立合同。

分片和异构路径的采集与独立核验入口见[执行变换校准工具](execution_calibration/README.md)，当前完整延迟质量门尚未通过。其 F32 明确精度合同不等同本入口的 backend-default 精度合同，不能直接互换时间。本机历史结果不随仓库分发，应在目标硬件重新采集。
