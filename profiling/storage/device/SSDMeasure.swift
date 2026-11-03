// Exact-size, cache-bypassing SSD access measurement on Darwin.
//
// This is the same protocol as profiling/storage/ssd_measure.py, written for
// macOS and iOS because a phone cannot be measured from Windows. It reports
// **observations only**: raw nanosecond samples, the offsets it used, and the
// bypass API it actually called. It deliberately does not compute statistics,
// medians or durations -- profiling/storage/import_device_measurements.py runs
// the same validator the Windows apparatus uses, so the arithmetic has exactly
// one implementation no matter which machine produced the numbers.
//
// Protocol (must match the Python tool byte for byte in what it measures):
//   * backing file = roundUp(size) + min(max(size, 4096), 64 MiB), prefilled
//     with 0xA5 and fsynced before the clock starts;
//   * warmup rounds are untimed; every timed round does one read and one write;
//   * each sample seeks to a random 4096-aligned offset where offset + size
//     still fits inside the backing file;
//   * fcntl(fd, F_NOCACHE) on the handle -- the Darwin mechanism, the same one
//     an iPhone must use;
//   * a write's clock stops when pwrite returns; fcntl(fd, F_FULLFSYNC) is
//     timed separately and is never added to the write duration.
//
// Build: see README.md in this directory. This file has not been compiled on
// the machine that wrote it (Windows); treat the first Mac build as the real
// compile check.

import Foundation
#if canImport(Darwin)
import Darwin
#endif

public struct DeviceMeasurementError: Error, CustomStringConvertible {
    public let description: String
    init(_ message: String) { self.description = message }
}

public enum SSDMeasure {

    public static let alignment = 4096
    public static let maxExtraBytes = 64 << 20

    /// The backing file size for one transfer size, exactly as the Python tool
    /// computes it. A wider offset window makes cross-sample sequential
    /// prefetch less useful; the extra region is capped so large sizes do not
    /// double the disk use.
    public static func backingFileBytes(for size: Int) -> Int {
        let rounded = (size + alignment - 1) / alignment * alignment
        let extra = min(max(size, alignment), maxExtraBytes)
        return rounded + extra
    }

    /// Run the protocol and return a JSON-serializable dictionary in the
    /// `ssd_device_observations_v1` shape.
    public static func run(
        sizes: [Int],
        samples: Int = 15,
        warmup: Int = 3,
        directory: URL,
        platformLabel: String,
        platformAPI: String = "F_NOCACHE",
        randomOffset: Bool = true
    ) throws -> [String: Any] {
        guard !sizes.isEmpty else { throw DeviceMeasurementError("no sizes") }
        guard Set(sizes).count == sizes.count else {
            throw DeviceMeasurementError("sizes repeat")
        }
        guard samples >= 3 else { throw DeviceMeasurementError("samples must be >= 3") }
        guard warmup >= 0 else { throw DeviceMeasurementError("warmup must be >= 0") }
        for size in sizes where size % alignment != 0 {
            throw DeviceMeasurementError(
                "size \(size) is not a multiple of \(alignment); cache-bypassing I/O is aligned or it is not this measurement")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw DeviceMeasurementError("directory does not exist: \(directory.path)")
        }

        let startedAt = ISO8601DateFormatter().string(from: Date())
        var observations: [[String: Any]] = []
        for size in sizes {
            observations.append(try measureSize(
                size: size, samples: samples, warmup: warmup,
                directory: directory, randomOffset: randomOffset))
        }
        let finishedAt = ISO8601DateFormatter().string(from: Date())

        var document: [String: Any] = [
            "schema_version": "ssd_device_observations_v1",
            "started_at": startedAt,
            "finished_at": finishedAt,
            "platform_label": platformLabel,
            "platform_api": platformAPI,
            "directory": directory.path,
            "protocol": [
                "sizes_bytes": sizes,
                "samples": samples,
                "warmup": warmup,
                "alignment_bytes": alignment,
                "random_offset": randomOffset,
            ],
            "observations": observations,
        ]
        for (key, value) in deviceIdentity() { document[key] = value }
        return document
    }

    // MARK: - One size

    private static func measureSize(
        size: Int, samples: Int, warmup: Int,
        directory: URL, randomOffset: Bool
    ) throws -> [String: Any] {
        let backingBytes = backingFileBytes(for: size)
        let url = directory.appendingPathComponent("ssd_measure_\(getpid())_\(size).bin")
        try? FileManager.default.removeItem(at: url)

        let buffer = UnsafeMutableRawPointer.allocate(byteCount: size, alignment: alignment)
        defer { buffer.deallocate() }
        memset(buffer, 0xA5, size)

        let fd = open(url.path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        guard fd >= 0 else {
            throw DeviceMeasurementError("cannot create \(url.path): errno \(errno)")
        }
        defer { close(fd); try? FileManager.default.removeItem(at: url) }

        if fcntl(fd, F_NOCACHE, 1) != 0 {
            throw DeviceMeasurementError(
                "fcntl(F_NOCACHE) failed (errno \(errno)); without it the run cannot claim f_nocache")
        }
        try prefill(fd: fd, bytes: backingBytes)
        guard fsync(fd) == 0 else {
            throw DeviceMeasurementError("fsync before timing failed (errno \(errno))")
        }

        let blocks = (backingBytes - size) / alignment
        var readSamples: [UInt64] = []
        var writeSamples: [UInt64] = []
        var flushSamples: [UInt64] = []
        var readOffsets: [Int] = []
        var writeOffsets: [Int] = []

        for round in 0..<(warmup + samples) {
            let readOffset = offset(blocks: blocks, random: randomOffset)
            let writeOffset = offset(blocks: blocks, random: randomOffset)

            let readStart = nowNs()
            try fullPread(fd: fd, buffer: buffer, size: size, offset: readOffset)
            let readEnd = nowNs()

            let writeStart = nowNs()
            try fullPwrite(fd: fd, buffer: buffer, size: size, offset: writeOffset)
            let writeEnd = nowNs()

            let flushStart = nowNs()
            guard fcntl(fd, F_FULLFSYNC) == 0 else {
                throw DeviceMeasurementError("F_FULLFSYNC failed (errno \(errno))")
            }
            let flushEnd = nowNs()

            guard round >= warmup else { continue }
            readSamples.append(readEnd - readStart)
            writeSamples.append(writeEnd - writeStart)
            flushSamples.append(flushEnd - flushStart)
            readOffsets.append(readOffset)
            writeOffsets.append(writeOffset)
        }

        return [
            "size_bytes": size,
            "backing_file_bytes": backingBytes,
            "read_samples_ns": readSamples.map { Int($0) },
            "write_samples_ns": writeSamples.map { Int($0) },
            "write_flush_samples_ns": flushSamples.map { Int($0) },
            "read_offsets_bytes": readOffsets,
            "write_offsets_bytes": writeOffsets,
        ]
    }

    // MARK: - Primitives

    private static func offset(blocks: Int, random: Bool) -> Int {
        guard random, blocks > 0 else { return 0 }
        return Int.random(in: 0...blocks) * alignment
    }

    private static func prefill(fd: Int32, bytes: Int) throws {
        let chunk = 1 << 20
        let scratch = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: alignment)
        defer { scratch.deallocate() }
        memset(scratch, 0xA5, chunk)
        var written = 0
        while written < bytes {
            let amount = min(chunk, bytes - written)
            try fullPwrite(fd: fd, buffer: scratch, size: amount, offset: written)
            written += amount
        }
    }

    private static func fullPread(
        fd: Int32, buffer: UnsafeMutableRawPointer, size: Int, offset: Int
    ) throws {
        var done = 0
        while done < size {
            let n = pread(fd, buffer + done, size - done, off_t(offset + done))
            if n < 0 {
                if errno == EINTR { continue }
                throw DeviceMeasurementError("pread failed (errno \(errno))")
            }
            if n == 0 { throw DeviceMeasurementError("pread returned EOF inside the file") }
            done += n
        }
    }

    private static func fullPwrite(
        fd: Int32, buffer: UnsafeRawPointer, size: Int, offset: Int
    ) throws {
        var done = 0
        while done < size {
            let n = pwrite(fd, buffer + done, size - done, off_t(offset + done))
            if n < 0 {
                if errno == EINTR { continue }
                throw DeviceMeasurementError("pwrite failed (errno \(errno))")
            }
            done += n
        }
    }

    private static func nowNs() -> UInt64 {
        clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
    }

    // MARK: - Device identity

    /// Fields the importer accepts. Values are never invented: a probe that
    /// fails leaves the key out rather than writing a guess.
    private static func deviceIdentity() -> [String: Any] {
        var identity: [String: Any] = [:]
        #if os(macOS)
        identity["system"] = "macOS"
        #elseif os(iOS)
        identity["system"] = "iOS"
        #else
        identity["system"] = "Darwin"
        #endif
        let version = ProcessInfo.processInfo.operatingSystemVersion
        identity["release"] = "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
        #if os(iOS)
        if let machine = sysctlString("hw.machine") { identity["machine"] = machine }
        identity["thermal_state"] = thermalStateName(ProcessInfo.processInfo.thermalState)
        #else
        if let model = sysctlString("hw.model") { identity["machine"] = model }
        #endif
        return identity
    }

    private static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }

    private static func thermalStateName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
