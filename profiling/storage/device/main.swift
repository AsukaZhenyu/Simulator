// macOS command-line entry point.
//
//   swiftc -O SSDMeasure.swift main.swift -o ssdmeasure
//   ./ssdmeasure --sizes 4096,16384,...,33554432 --samples 15 --warmup 3 \
//                --directory ~/ssd-measure --output ~/ssd-measure/mac.json
//
// Then, back on the Windows machine that owns the cost table:
//   python profiling/storage/import_device_measurements.py \
//     --observations mac.json --output mac.measurements.json

import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data(("ssdmeasure: " + message + "\n").utf8))
    exit(2)
}

func value(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let index = args.firstIndex(of: "--" + name), index + 1 < args.count else { return nil }
    return args[index + 1]
}

let sizesText = value("sizes") ?? fail("--sizes is required (comma-separated bytes)")
let sizes = sizesText.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
if sizes.isEmpty { fail("--sizes did not parse into any integers") }

let directory = URL(fileURLWithPath: (value("directory") ?? ".").expandingTildeInPath)
let output = URL(fileURLWithPath: (value("output") ?? "ssd-device.json").expandingTildeInPath)
let samples = Int(value("samples") ?? "15") ?? 15
let warmup = Int(value("warmup") ?? "3") ?? 3
let label = value("platform") ?? "mac-darwin"

do {
    let document = try SSDMeasure.run(
        sizes: sizes, samples: samples, warmup: warmup,
        directory: directory, platformLabel: label)
    let data = try JSONSerialization.data(
        withJSONObject: document, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: output)
    print("wrote \(output.path)")
} catch {
    fail(String(describing: error))
}
