# Darwin 设备上的测量（macOS / iPhone）

`ssd_measure.py` 在 Windows 和 Linux 上跑；Darwin 上它也能跑，但要测 iPhone 就得在设备上跑。
这个目录放的是同一套协议的 Swift 实现：**设备只产出原始观测，不产出统计量**。

```
SSDMeasure.swift   协议本体（macOS / iOS 通用）
main.swift         macOS 命令行入口
```

设备上报的 JSON 回到仓库这边，用
[`import_device_measurements.py`](../import_device_measurements.py) 转成标准的
`ssd_storage_measurements_v1` 文档：

```powershell
python profiling\storage\import_device_measurements.py --observations mac.json --output mac.measurements.json
python profiling\storage\transfer_table.py --measurements mac.measurements.json --base mapping\examples\m2\storage-interface-synthetic.costs.json --output mac.costs.json
```

**为什么设备不算统计量**：`statistics`、`duration_ns`、`quality_flags` 都是派生值。设备在另一台
机器、另一种语言里算出它们，等于先做了一个本仓库无法核验的声明。所以 app 只交
`samples_ns` 和它实际用过的偏移，由 `build_record` / `validate_record` 在导入时算——
两台机器的校验器是同一份代码。

> **这份 Swift 代码还没有在 Mac 上编译过**（写它的机器是 Windows）。第一次在 Mac 上构建就是
> 真正的编译检查；编译不过以编译器为准，不要把它当成已验证的产物。

## 在 macOS 上跑

```bash
cd profiling/storage/device
swiftc -O SSDMeasure.swift main.swift -o ssdmeasure

./ssdmeasure --sizes 4096,16384,65536,262144,1048576,1474560,4194304,5898240,13271040,16777216,19353600,33554432 \
             --samples 15 --warmup 3 \
             --directory ~/ssd-measure --output ~/ssd-measure/mac-$(date +%Y%m%d).json \
             --platform mac-darwin
```

目录要先建好（`mkdir -p ~/ssd-measure`）。**这 12 个尺寸必须和 Windows 那份逐字相同**，否则导出
时 `--require-sizes` 会拒绝，两台机器的表也就不能并排比。

## 在 iPhone 上跑

iPhone 不能从 Windows 产出，也不能远程接进一台租来的 Mac——**租远程 MacBook 只能拿到 Mac 上
的那份数据，拿不到 iPhone 的**。iPhone 需要：一台本地 Mac + Xcode + 真机签名 + 那台手机。

1. Xcode 新建一个 iOS App（SwiftUI，最低 iOS 16）。
2. 把 `SSDMeasure.swift` 拖进 target（不要加 `main.swift`）。
3. 加一个按钮调用：

```swift
let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
let document = try SSDMeasure.run(
    sizes: [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 33554432],
    samples: 15, warmup: 3,
    directory: documents, platformLabel: "iphone-<model>")
let data = try JSONSerialization.data(withJSONObject: document,
                                      options: [.prettyPrinted, .sortedKeys])
try data.write(to: documents.appendingPathComponent("iphone.json"))
```

4. 让 app 显示「启用文件共享」（`UIFileSharingEnabled` / `LSSupportsOpeningDocumentsInPlace`），
   跑完从「文件」app 里把 `iphone.json` 拷出来；或者用 AirDrop / `xcrun devicectl` 取。

真机上有三件事必须记，否则数字是虚假精度：

- `thermal_state`（已自动记 `ProcessInfo.thermalState`）——iOS 会限频，`serious`/`critical` 下的
  样本不能和 `nominal` 混在一起；
- 电量（在 app 里读 `UIDevice.current.batteryLevel`，写进 `battery_level`），低电量模式会降频；
- 屏幕保持常亮、app 在前台——后台进程会被挂起，测量就断了。

## 导入器会拒绝什么

- `platform_api` 不是 `F_NOCACHE` / `F_GLOBAL_NOCACHE`（没有声明绕过就没法记 `f_nocache`）；
- 尺寸不是 4096 的倍数，或者 `protocol.sizes_bytes` 与 `observations` 对不上（缺一个、多一个）；
- 某个方向少于 3 个样本，或写的次数与 `F_FULLFSYNC` 的次数不相等；
- 偏移不是 4096 对齐，或 `offset + size` 超出声明的 backing file；
- 出现了导入器不认识的字段——没人读的字段就是没人检查的字段。

这些都由 `profiling/storage/tests/test_device_import.py` 逐条钉住。
