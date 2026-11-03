# 精确尺寸的 SSD 读写测量（绕过页缓存）

`ssd_measure.py` 每次只计时一次精确字节数的读或写，保留全部原始纳秒样本。它不做插值，
不导出 bandwidth 标量，也不把 flush 时间混进写耗时。`transfer_table.py` 把测量结果导出成
模拟器可直接加载的 `m2-layer-0.7` 成本表（见下）。

## 测的是什么

- 一个样本 = 一个对齐的 `read` 或 `write`，长度恰好是 `--sizes-bytes` 中的一个值。
- 文件句柄复用；每个样本之前重新定位到 4096 对齐的随机偏移（`--random-offset`，
  默认开），避免跨样本的顺序预取；偏移始终落在文件内。
- 每轮先读后写，读/写各计一个样本，避免只测一侧造成偏热/偏缓存。
- `--warmup` 轮不计时；`--samples` 轮的每个样本都进入 `samples_ns`。
- 写样本在写系统调用返回时停表；`fsync` / `FlushFileBuffers` / `F_FULLFSYNC` 另行计时，写入
  `flush_samples_ns` 和 `flush_ns`。**flush 从不加到 `duration_ns` 上。**
- 绕过机制（`cache_bypass`）：

| 平台 | 机制 | `cache_bypass` |
|---|---|---|
| Windows | `CreateFileW(FILE_FLAG_NO_BUFFERING \| FILE_FLAG_WRITE_THROUGH)`，`VirtualAlloc` 对齐缓冲，`msvcrt.open_osfhandle` | `unbuffered` |
| Linux | `O_DIRECT`；不可用/打开失败时回退 `posix_fadvise(..., POSIX_FADV_DONTNEED)`，每个样本前调用 | `unbuffered` / `fadvise_dontneed` |
| macOS / Darwin | 每个句柄 `fcntl(fd, F_NOCACHE, 1)`；写后屏障用 `fcntl(fd, F_FULLFSYNC)` 而不是 `fsync` | `f_nocache` |
| 测试 | 普通缓冲 I/O，只用于 `--cache-bypass off` | `buffered` |

Darwin 没有 `O_DIRECT`，`FADV_DONTNEED` 在那里也不生效，所以**不允许**在 macOS 上回退到
`fadvise_dontneed`：那会声称一次从未发生的绕过。走不通就报错，不会降级成假标签。

开启绕过时，size 必须是 4096 的倍数；不满足会直接报错，不会悄悄改成向上取整后的长度。
对齐缓冲分配失败也直接报错，不会退化成缓冲 I/O。

## 没有测什么

- **页缓存**：只有 `cache_bypass` 为 `unbuffered`/`fadvise_dontneed`/`f_nocache` 时
  `physical_ssd_bandwidth_verified` 才为 `true`。这是对“实际使用的机制”的声明，不是对
  盘控制器缓存的独立核验；`fadvise_dontneed` 本身是 best-effort，不保证数据不在盘内缓存，
  Darwin 的 `F_NOCACHE` 同样只对本次句柄生效。
- **队列深度**：本工具一次只有一个同步 I/O 在途，不是并发 IOPS 曲线。
- **写放大/控制器行为**：没有 FTL、OP 空间或垃圾回收观测。
- **写缓存**：本工具不主动把盘写满，所以**它测到的是盘当时所在的那一档**。连续写同一尺寸
  直到累计几十 GiB 会看到速度从 2.5 GB/s 掉到 0.46 GB/s（见下面「第二档写价」）；要拿到
  第二档的数，就得单独跑一次已经写满的 campaign。
- **一次性大段顺序写**：每个 campaign 只为该尺寸建一个 `size + 64 MiB` 的 backing file，
  全部样本都落在这同一段 LBA 上（偏移随机、互相覆盖）。所以「累计写了 47 GiB」实际是
  **在 1.1 GiB 的 LBA 区间里反复重写**，不是往空盘上顺序追加 47 GiB。两者的写放大不同，
  这些数字不能直接当成「一次写 47 GiB 的耗时」用。要测后者得另写一个一次创建、只追加、
  写完删的 campaign。
- **持久性**：`flush_ns` 只说明本次 flush 的耗时，没有断电/掉电验证。
- **插值与带宽标量**：不产出。summary 里的
  `duration_ns = latency_ns + ceil(size_bytes / bytes_per_ns)` 是最小二乘拟合，
  仅作描述（`fit_is_descriptive_not_used_for_lookup: true`），并给出每个实测点的相对残差；
  不要拿它做查表。
- **设备身份**：`storage_device` 只是便宜的探测（Windows `Get-PhysicalDisk`、Linux
  `/sys/block/*/device/model`、macOS `diskutil info -plist /`），失败为 `null`，且没有
  绑定到 `--directory` 实际所在的物理设备。

## JSON 与 `transfer_entries` 的对应

顶层：`schema_version: "ssd_storage_measurements_v1"`、`started_at`/`finished_at`
（ISO-8601）、`command_line`、`platform`、`storage_device`、`protocol`、
`measurements`、`summary`。每条测量记录含 `size_bytes`、`route`、`cache_bypass`、
`physical_ssd_bandwidth_verified`、`samples_ns`、`statistics`、`duration_ns`
（= `ceil(median_ns)`）、`flush_ns` 等。

`transfer_table.py` 把这份 JSON 导出成 **`m2-layer-0.7` 成本表**
（`provenance: "measured_transfer"`）。它是一张完整可加载的表，不是片段：M2 要求
计算条目 `entries` 与 `transfer_entries` 并存，所以必须给一张 `.5` 或 `.7` 的底表，
导出的表沿用它的计算条目并把版本升到 `.7`。

```powershell
python profiling\storage\transfer_table.py --measurements D:\ssd-measure\laptop-nvme.json --base mapping\examples\m2\storage-interface-synthetic.costs.json --output D:\ssd-measure\laptop-nvme.costs.json
```

逐字段对应：

| 记录字段 | `transfer_entries` 字段 |
|---|---|
| `READ_FILE` / `route {source: "ssd", destination: "dram"}` | `route: "ssd_to_dram"` |
| `WRITE_FILE` / `route {source: "dram", destination: "ssd"}` | `route: "dram_to_ssd"` |
| `size_bytes` | `storage_bytes`（精确查表，不插值） |
| `duration_ns` | `duration_ns`（见下面 flush 与 timing_scope） |
| `cache_bypass` | `cache_policy` |
| 出处 | `provenance: "measured_transfer"` |

导出器守住的四条：

- **一个记录换一个键。** 没测过的字节数不会出现在表里，也不会拿邻居顶替；内核在编译时
  会拒绝那个尺寸，而不是悄悄插值。权重张量的大小通常不是 2 的幂，所以**要搬多大就测
  多大**：把张量的 storage bytes 直接写进 `--sizes-bytes`。可用
  `--require-sizes 1572864,3145728` 让导出在缺尺寸时直接失败。
- **flush 要么不含、要么明说。** 默认 `--flush exclude`：写耗时就是写调用返回时的耗时，
  `timing_scope` 记为 `single_aligned_write_wall_excluding_flush`，持久化代价不在里面。
  `--flush include` 把两个中位数相加，`timing_scope` 记为
  `single_aligned_write_plus_flush_medians`——那是**两个中位数之和**，不是和的分布。
  读没有 flush，`timing_scope` 恒为 `single_aligned_io_wall`。
- **不绕过页缓存的测量只能作 validation_only 表。** `cache_policy` 不是
  `unbuffered`/`fadvise_dontneed`/`f_nocache` 时（即 `--cache-bypass off`），导出表被
  强制标成 `validation_only: true`，M2 会拒绝用它支撑结论。
- **重复定价不许静默覆盖。** 底表已经给某个 `(route, size)` 定过价时导出失败，除非显式
  `--replace`。`.6`（MoE 专家组）底表不能被改名成 `.7` 稠密混合。

导出表里仍然**没有**队列深度、写放大和控制器行为；`comment` 字段里写下的是这份表来自
哪次测量（平台、设备、样本数、机制），供人核对，内核不读它。

## 第二档写价（写缓存耗尽之后）

同一条写命令在同一个盘上会有两个速度：缓存吃得下时约 2.5 GB/s，吃完之后约 0.46 GB/s，
差 5.4 倍，而且**掉下去之后在一次连续写入里不会自己回来**。所以写有第二档价，导出时必须
说明门限：

```powershell
python profiling\storage\transfer_table.py --measurements D:\ssd-measure\laptop-nvme.json `
  --sustained-measurements D:\ssd-measure\laptop-nvme-filled.json `
  --sustained-after-bytes 42949672960 `
  --base mapping\examples\m2\storage-interface-synthetic.costs.json `
  --output D:\ssd-measure\laptop-nvme.costs.json
```

- `--sustained-measurements` 是**第二份 campaign**：先把盘写到它掉档，再在这个状态下重测
  同样的尺寸。导出器只取这份文件里的**写**记录；GC 压力下读也会变慢（本机 3.75 →
  3.06 GB/s），但那是另一个机制，不拿它顶替读价格。
- `--sustained-after-bytes` 是那份 campaign 开始时盘上已经累计写了多少字节，也就是
  `storage_write_regime.after_cumulative_bytes`。它**必须由测量方给**：单条测量记录里没有
  这个量，导出器不会去猜。实测同一条命令在不同时候给出过 12 / 35 / 47 GiB（见下），取
  测量值时要留裕度。
- 三个方向都拒绝：第二份缺尺寸、第二份多出尺寸、第二档比第一档还快（两列写反了）。
- 只给 `--measurements` 而不给第二份，导出的就是单档表；表里没有
  `storage_write_regime`，内核行为与这个功能出现之前完全一样。反过来，底表已经带门限时
  不允许只拿一份新 campaign 去覆盖它。

**门限不是器件常数，掉下去之后也会回来。** 本机同一条命令（`--sizes-bytes 1073741824`、
`--cache-bypass required`）在同一天跑出过三个门限：80 样本那次在第 35 GiB 掉档，60 样本那次
在**第 12 GiB** 掉档（中间还有一次 47 GiB），都是同一种盘、同一个进程参数。掉档之后停手，
盘会自己恢复：第 12 GiB 掉档那次结束约 2.5 分钟后重跑，16 GiB 全是满速；再隔 5 s、再隔
30 s 各追 16 GiB，仍然全是满速——也就是**掉档后 3.6 分钟内盘又吃下了 48 GiB 满速写**。
恢复得这么快，说明「60 s 才恢复」那类固定数字没有意义，门限与恢复都得按盘当时的状态取。

这两件事合起来看：门限 ≈ 盘上当时可用的空闲块预算，按满速写（2.5 GB/s）减去后台回收
（约 0.46 GB/s）的净速率消耗；停手时回收照常跑，按 0.46 GB/s 左右把预算还回来。服务循环
里没有这段空闲，所以内核「场景内不回退」是对的；离线换模型、换专家集这类有间隔的负载
则会自己回到第一档，不该拿一次掉档去给整个会话定价。

**什么时候需要它**：权重是不变的，Dense 卸载的流量几乎全是读，写完
`EvaluationResult.ssd_write_bytes` 通常远小于门限，单档价就是准确的。会真的越过去的
是 MoE 专家换入换出（为了腾地方把一组专家写出去）和把 KV 落到盘上的长会话。

## 三个平台的命令

笔记本（Windows + NVMe）与台式机（Windows + NVMe）只在目录和标签上不同：

```powershell
python profiling\storage\ssd_measure.py --sizes-bytes 4096,16384,65536,262144,1048576,4194304,16777216 --directory D:\ssd-measure --output D:\ssd-measure\laptop-nvme.json --platform laptop-windows-nvme --samples 15 --warmup 3

python profiling\storage\ssd_measure.py --sizes-bytes 4096,16384,65536,262144,1048576,4194304,16777216 --directory E:\ssd-measure --output E:\ssd-measure\desktop-nvme.json --platform desktop-windows-nvme --samples 15 --warmup 3
```

上面这串 2 的幂是给 `summary.directions.*.fit` 描述曲线形状用的。**要让模拟器查表，尺寸必须
逐字等于要搬的张量 storage bytes**：查表不插值，少一个字节就是 `Unsupported`。真实权重张量
（例如 4096×4096 的 fp16 = 33554432）不是 2 的幂，把它也加进 `--sizes-bytes`，并用
`--require-sizes` 固定下来：

```powershell
python profiling\storage\ssd_measure.py --sizes-bytes 4096,16384,65536,262144,1048576,4194304,16777216,33554432 --directory D:\ssd-measure --output D:\ssd-measure\laptop-nvme.json --platform laptop-windows-nvme --samples 15 --warmup 3
python profiling\storage\transfer_table.py --measurements D:\ssd-measure\laptop-nvme.json --base mapping\examples\m2\storage-interface-synthetic.costs.json --output D:\ssd-measure\laptop-nvme.costs.json --require-sizes 33554432
```

台式机用完全相同的两条命令，只换 `--directory`/`--output`/`--platform` 标签；代码路径与
笔记本逐字节相同。

**iPhone 不能从 Windows 产出**，也不能直接跑本脚本。Darwin 上用的机制与上表 macOS 一行
**完全相同**（`F_NOCACHE` + `F_FULLFSYNC`），所以 [device/](device/README.md) 里放了一份同
协议的 Swift 实现：设备只产出**原始观测**（`samples_ns` 与它实际用过的偏移），统计量与
`duration_ns` 由 `import_device_measurements.py` 在导入时用**同一份** `build_record` /
`validate_record` 算出来——设备不自己算派生值，因为在另一台机器上算出来的数没人能核验。

```powershell
python profiling\storage\import_device_measurements.py --observations mac.json --output mac.measurements.json
python profiling\storage\transfer_table.py --measurements mac.measurements.json --base mapping\examples\m2\storage-interface-synthetic.costs.json --output mac.costs.json
```

注意：**租一台远程 MacBook 只能拿到 Mac 那份数据，拿不到 iPhone 的**——iPhone 需要一台
本地 Mac 加真机才能接上。如果目标是「第三台设备」，Apple Silicon Mac 本身就是统一内存的
Darwin 设备，比 iPhone 更好比。

## 在第二台机器上跑（一次完整 campaign）

目标是让「同一份代码、同一份表结构」在两台机器上各产出一张可并排比对的表。按顺序跑四组，
中间不要开别的程序（尤其不要有别的写入），全部落在**同一个分区**上（工具不校验 `--directory`
背后是哪块物理盘，跨盘的数字不能混着用）：

```powershell
$D = "D:\ssd-measure"     # 换成第二台机器上那块盘的目录
$S = "4096,16384,65536,262144,1048576,1474560,4194304,5898240,13271040,16777216,19353600,33554432"

# 0) 先问这个目录是不是一块盘（下面「租来的 UMA 机器」讲了它拒绝什么）
#    --expected-commit 填**笔记本上** `git rev-parse --short HEAD` 的输出；
#    短 SHA 与全长 SHA 都能比（前缀 ≥ 7 位）。
python profiling\storage\preflight.py --directory $D --expected-commit <笔记本的 commit>

# 1) 干净状态下的主尺寸表
python profiling\storage\ssd_measure.py --sizes-bytes $S --directory $D --output $D\desktop-clean.json --platform desktop-windows-nvme --samples 24 --warmup 2 --cache-bypass required

# 2) 掉档 campaign：同一尺寸写 80 次，找出这台盘这一次的门限
python profiling\storage\ssd_measure.py --sizes-bytes 1073741824 --directory $D --output $D\desktop-cliff.json --platform desktop-windows-nvme --samples 80 --warmup 2 --cache-bypass required

# 3) 紧接着（盘已经掉档）重跑主尺寸表 —— 这份才是第二档价的来源
python profiling\storage\ssd_measure.py --sizes-bytes $S --directory $D --output $D\desktop-sustained.json --platform desktop-windows-nvme --samples 24 --warmup 2 --cache-bypass required

# 4) 大块：32 MiB – 1 GiB
python profiling\storage\ssd_measure.py --sizes-bytes 33554432,67108864,100663296,134217728,268435456,536870912,1073741824 --directory $D --output $D\desktop-large.json --platform desktop-windows-nvme --samples 14 --warmup 2 --cache-bypass required

# 5) 导出单档表（必做），以及可选的第二档表
python profiling\storage\transfer_table.py --measurements $D\desktop-clean.json --base mapping\examples\m2\storage-interface-synthetic.costs.json --output $D\desktop-nvme.costs.json --require-sizes 1474560,5898240,13271040,19353600,33554432
python profiling\storage\transfer_table.py --measurements $D\desktop-clean.json --sustained-measurements $D\desktop-sustained.json --sustained-after-bytes <步骤 2 读出的 GiB 数 × 2^30> --base mapping\examples\m2\storage-interface-synthetic.costs.json --output $D\desktop-nvme-twotier.costs.json
```

四件事要在记录里对得上，否则两台机器的数字不能并列：

- **同一个 commit**：两台机器各跑 `git rev-parse HEAD`，必须一致。
- **同一份 `--sizes-bytes`**：少一个字节，导出就会拒绝（不插值）。
- **门限由步骤 2 读出来再填**：`desktop-cliff.json` 里第一个 `samples_ns` 慢样本的下标就是
  门限所在的 GiB 数；它不是器件常数，**不能照抄笔记本的 12/35/47**。
- **`--sustained-after-bytes` 必须由人给**，导出器不会猜；步骤 3 与步骤 2 之间的空档越短越好。

跑完把整个 `$D` 目录拷回来，不要只拷导出表——原始 `samples_ns` 是唯一能重算的东西。

## 租来的 UMA 机器（DGX Spark / RTX Spark）

两台机器名字接近，能做的事差别很大：

| | RTX Spark（N1X） | DGX Spark（GB10） |
|---|---|---|
| 系统 | **Windows 11 on ARM64** | Linux aarch64（DGX OS） |
| 内存 | 128 GB LPDDR5X 统一（Grace CPU + Blackwell GPU，NVLink-C2C） | 128 GB 统一，~273 GB/s |
| 绕过路径 | 本仓库的 **Windows 分支原样可用**（`ctypes` 在 ARM64 Windows 上仍是 64 位，`HANDLE` 是 `c_void_p`） | Linux **`O_DIRECT`** 分支 |
| 怎么拿到 | 买（约 $2,500 起），一般不是租 | **可按小时租**，约 $0.75–0.79/h，给 root SSH + Docker |
| 对论文的价值 | 真 UMA 硬件块，且**代码路径与笔记本逐字节相同** | 真 UMA 硬件块 + Linux/ARM64 路径 + 真 NVMe 数字 |

**先跑 `preflight.py`，不要直接开测。** 租来的机器通常给的是容器，工作目录落在
`overlay` 上——在那里测出来的数字不属于任何一块盘，而 `ssd_measure.py` 自己不会检查这一点
（它只如实记录用到了哪个 API）。`preflight.py` 会读出 `findmnt` 的 `FSTYPE`/`SOURCE`，
overlay / tmpfs / nfs / cifs / `fuse.*` 一律 **REFUSE**；同时它也做一次 4 KiB 的真实绕过
试探，回答「这次 campaign 到底会拿到 `unbuffered` 还是退成 `fadvise_dontneed`」。两种
CUDA 路径上的租机都要先确认 `$D` 在本地块设备上（`findmnt -T $D` 的 `SOURCE` 应是
`/dev/...`）。

这能拿到什么、不能拿到什么，要分开说：

- **能拿到**：真实的 `memory_mode: "shared"` 硬件块（`shared_capacity_bytes` 是它自己的
  128 GB），以及在它自己的 NVMe 上测出的 `(route, storage_bytes)` 价格。UMA 的
  `storage-interface-synthetic-uma-prefill-3.json` 已经把「同一 layer + 同一成本表、只换
  `hardware`」这条通路跑通了，换成真机只是换文档。
- **不能直接拿到**：这台机器**自己的计算与搬运价格**。`.5`/`.7` 的 `entries` 来自
  llama.cpp + CUDA 的测量战役（`profiling/steady_costs` 等），租机上一次预算内的运行
  只能覆盖 SSD 这一层。把笔记本的表原样贴到 Spark 的 `hardware` 块上，模型照样能跑，但
  那是在用一台机器的价格描述另一台机器——文档里必须写明，或者先在这台机器上真正测一遍。

**Linux 机器（例如一台 Arch）**：命令完全相同，只把 PowerShell 的 `$D` 换成 shell 变量、
`--platform` 换标签。这台机器上会走 **`O_DIRECT`**（文件系统不支持时自动退到
`fadvise_dontneed` 并在记录里如实写，不会假装是 `O_DIRECT`）。这两条路径在此之前**只有单元
测试，没有真机证据**，所以一台旧 Linux 机器的价值不只是「第三台」：它补的是另一个操作系统
和另一种绕过机制。`O_DIRECT` 要求 4096 对齐，工具已经处理；8 GB 内存够用，最大尺寸 1 GiB
时缓冲区加 backing file 约 2 GiB。跑之前 `git rev-parse HEAD` 同样要对得上。

## 无硬件测试

```powershell
python -m unittest discover -s profiling/storage/tests -t profiling -p "test*.py" -v
```

测试只用 `--cache-bypass off`，不替代真实硬件验收；CI 里不假定存在 SSD。Darwin 的
`F_NOCACHE` 分支只在选择逻辑和记录校验上被测到（`DarwinMechanismTests`），**真机上的
绕过效果没有在 Windows 上验证过**。
