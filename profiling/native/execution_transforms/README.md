# 执行变换的真实校准 probe

独立 GGML F32 程序，复用 `../matched_mechanisms/compute.cpp` 的后端、固定形状图和独立数值参考。不修改旧程序或旧实验。构建与运行清单必须同时绑定这两个 C++ 源文件的 SHA256。

## 构建

```powershell
cmake -S profiling/native/execution_transforms -B <new-build> `
  -DGGML_SOURCE_DIR=<existing-ggml-root> -DGGML_BUILD_DIR=<existing-build-root>
cmake --build <new-build> --config Release
```

这是显式 CUDA 项目；没有 CPU fallback。主程序与 GGML/CUDA 的匹配 DLL 由调用方提供。`build-local/compute-build.json` 保存本机构建命令和编译器输出，不能作为另一机器的路径默认值。

## 参数

所有参数都必须显式给出：

```text
--family ffn|attention --backend hybrid
--hidden H --intermediate I --cpu-intermediate Ic
--tokens Q --history K --heads N --kv-heads Nk --head-dim D
--threads 1 --cpu-id C --device G --warmup W --min-warmup-ms M
--samples S --seed R --dtype f32 --cuda-graphs disabled
```

FFN 要求 `1 <= Ic < I`，且 history/heads/kv-heads/head-dim 全为 0。Attention 要求 intermediate/cpu-intermediate 均为 0、`H=N*D`、`N % Nk=0`。Windows 的单个 CPU affinity 也约束 GPU 派发线程；它不是多核 CPU 性能实验。其他系统由已有 helper 明确不应用 affinity，不能声称已绑定。

### 存储精度与计算精度

所有新图的 `MUL_MAT` 节点显式设置 `GGML_PREC_F32`。运行器还必须对子进程显式设置并记录 `NVIDIA_TF32_OVERRIDE=0`，native 在 payload 中同时记录 `matmul_precision=GGML_PREC_F32_explicit` 和实际继承的 TF32 flag。F32 张量端口本身不能证明 F32 数学：源绑定 GGML b8705 的 `ggml-cuda/common.cuh:1407` 创建 TF32-enabled cuBLAS handle，`ggml-cuda/ggml-cuda.cu` 的 F32 `cublasSgemm` 路径可能使用此模式。计算精度应进入成本条目的绑定键。

大形状首次 smoke 的数值失败应原样保留；新精度合同采用相同 `atol=1e-5, rtol=1e-4`，没有放宽误差阈值，也没有改动旧 matched 程序或旧基线。

## 实际执行与公平边界

FFN 输入为 compact token 行。计算是 gate/up→SwiGLU→down，不含 RMSNorm、router、gather/scatter、残差。CPU shard 为中间维 `[0,Ic)`，GPU shard 为 `[Ic,I)`；gate/up 的输出列与 down 的输入行配对。权重预先物化为独立分片。三个实测策略最终输出都交付 DRAM：

- `gpu/resident`：GPU 完整四阶段串行图链，加最终输出 D2H。
- `hybrid/resident`：持久 GPU 派发线程执行 GPU shard 四阶段，主线程同时执行 CPU shard 四阶段；等待两者完成，GPU 部分结果 D2H，再执行一次 GGML CPU SUM。
- `hybrid/preload`：先把三个 GPU shard 权重从 DRAM 经 pinned staging 真实 H2D，再执行同一混合链。每次迭代都支付复制，不声称恢复冷缓存或量化 SSD 读取。

三种策略的不可变输入均已在 CPU/GPU 双驻留；初始化和初始分配不计时。不能把它们解释为从只有一份 CPU 输入开始的完整请求时间。D2H/H2D 计时包括 pinned staging 的 CPU memcpy 和完成等待。图派发、同步、线程握手都留在真实策略时间内，没有从时间中扣除。

`hybrid/dispatch_barrier` 独立测量空 worker 握手，不能事后当作任意截距拟合。`diagnostic_completed_work_brackets` 是额外一次未计时执行的 host 完成区间，不是 CUDA kernel trace，不证明 kernel 重叠比例。

Attention 输入为 post-RoPE Q/K/V 与原始 KV；执行因果 GQA、sigmoid gate、输出投影和残差：

- `gpu/resident`：GPU QK→softmax→PV→输出处理→最终 D2H。
- `hybrid/resident`：CPU QK→softmax→概率 H2D→GPU PV→输出处理→最终 D2H。

QK 产生缩放后的完整 scores，softmax 负责绝对位置因果遮罩。QK 物化私有 K scratch；PV 再完成外部可见 K/V append 输出。原始 cache 输入保持不变。此物化合同与位置分块 IR 的实现边界存在差别，不能直接宣称该 IR lowering 已获校准。

## 输出与预测

输出一个 `execution-transform-native-0.1` JSON，包含所有原始 `wall_ns` 样本、精确形状、完整请求协议、端口字节与输入 fixture 摘要、数值误差和工作区来源。非二进制整点的 F32 数据由显式 seed 生成，最终结果和中间端口对独立参考逐元素检查。阈值固定为 `atol=1e-5, rtol=1e-4`；失败返回 2，调用方必须保留原始输出。

只用 profile 的独立 components 预测 heldout 的真实策略：

```text
FFN GPU = sum(gpu/full_{gate,up,activation,down}) + gpu/output_d2h
FFN mixed = max(sum(cpu/shard_*), sum(gpu/shard_*))
            + hybrid/output_d2h + cpu/sum
FFN preload = hybrid/preload_weights + mixed
Attention GPU = gpu/qk + gpu/softmax + gpu/pv + gpu/output + gpu/output_d2h
Attention mixed = cpu/qk + cpu/softmax + hybrid/probabilities_h2d
                  + gpu/pv + gpu/output + gpu/output_d2h
```

这些是待验证的预测模型。尤其 FFN 的 `max` 假设可能因 CPU affinity、派发和资源竞争失败；失败应进入报告，不能修改公式直到通过。策略的完整实测时间不会用于拟合其自身预测。

CPU workspace 来自实际 gallocr 加显式 cplan scratch；GPU 私有 scratch 为 `null`，即使可观察 gallocr 非零也不构造容量保证。本程序没有校准量化、CUDA graphs、多请求、多层 P/D 队列、预取与计算并发、SSD 路径或任意设备状态波动。
