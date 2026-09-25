// The idiomatic API: an actor that owns one core instance, so calls from concurrent tasks are
// serialized and the core's single-threaded state is never shared.

/// A request for ``CoreOp/runCLI``, encoded straight into guest memory by ``encode(into:)``.
///
/// Layout (little endian, offsets from the request start):
/// `u32 magic "CLI1", u32 argc, u32 nfiles, u32 0`, then `argc x {u32 off, u32 len}`,
/// then `nfiles x {u32 name_off, u32 name_len, u32 data_off, u32 data_len}`, then the bytes.
public struct CLIRequest: Sendable {
    public var arguments: [[UInt8]]
    public var files: [(name: [UInt8], data: [UInt8])]

    /// `argv[0]` is the program name, as in C. Only files the command line names are sent.
    public init(argv: [String], files: [String: [UInt8]]) {
        arguments = argv.map { Array($0.utf8) }
        self.files = files.keys.sorted().filter { argv.contains($0) }.map { (Array($0.utf8), files[$0]!) }
    }

    public var byteCount: Int {
        16 + 8 * arguments.count + 16 * files.count + arguments.reduce(0) { $0 + $1.count }
            + files.reduce(0) { $0 + $1.name.count + $1.data.count }
    }

    public func encode(into out: UnsafeMutableRawBufferPointer) {
        precondition(out.count == byteCount)
        out.storeLE32(CoreLayout.cliMagic, at: 0)
        out.storeLE32(UInt32(arguments.count), at: 4)
        out.storeLE32(UInt32(files.count), at: 8)
        out.storeLE32(0, at: 12)
        var at = 16 + 8 * arguments.count + 16 * files.count
        func put(_ bytes: [UInt8]) -> UInt32 {
            let start = at
            bytes.withUnsafeBytes { src in
                if !src.isEmpty { UnsafeMutableRawBufferPointer(rebasing: out[start..<(start + src.count)]).copyMemory(from: src) }
            }
            at += bytes.count
            return UInt32(start)
        }
        for (k, a) in arguments.enumerated() {
            out.storeLE32(put(a), at: 16 + 8 * k)
            out.storeLE32(UInt32(a.count), at: 20 + 8 * k)
        }
        for (k, f) in files.enumerated() {
            let o = 16 + 8 * arguments.count + 16 * k
            out.storeLE32(put(f.name), at: o)
            out.storeLE32(UInt32(f.name.count), at: o + 4)
            out.storeLE32(put(f.data), at: o + 8)
            out.storeLE32(UInt32(f.data.count), at: o + 12)
        }
    }
}

public actor MemCore {
    private let engine: MemCoreEngine

    public init(wasm: [UInt8]) throws {
        engine = try MemCoreEngine(wasm: wasm)
    }

    /// Runs `cleanroom-transformer ARGS...` exactly like the native binary. `argv[0]` is the
    /// program name; `files` are the files the command may open, by the names used in `argv`.
    public func run(_ argv: [String], files: [String: [UInt8]] = [:]) throws -> CLIResult {
        let request = CLIRequest(argv: argv, files: files)
        return try engine.execute(.runCLI, inputCount: request.byteCount, fill: request.encode(into:), outputCapacity: 1 << 16) { _, p in
            let code = p.loadLE32(at: 0), so = Int(p.loadLE32(at: 4)), se = Int(p.loadLE32(at: 8))
            return CLIResult(exitCode: code, stdout: Array(p[16..<(16 + so)]), stderr: Array(p[(16 + so)..<(16 + so + se)]))
        }
    }

    /// The commitment roots of a memory file, or why it was rejected.
    public func roots(of file: [UInt8]) -> Result<CoreRoots, MemCoreError> {
        Result {
            try engine.execute(.roots, inputCount: file.count, fill: { dst in
                file.withUnsafeBytes { dst.copyMemory(from: $0) }
            }, outputCapacity: 4 + CoreRoots.byteSize) { status, p -> Result<CoreRoots, MemCoreError> in
                status == .rejected ? .failure(.rejected(String(decoding: p, as: UTF8.self))) : .success(CoreRoots(payload: p))
            }
        }
        .mapError { $0 as? MemCoreError ?? .trap(String(describing: $0)) }
        .flatMap { $0 }
    }

    public func shake256(_ data: [UInt8], outputBytes: Int = 64) throws -> [UInt8] {
        precondition((1...1024).contains(outputBytes))
        return try engine.execute(.shake256, inputCount: 4 + data.count, fill: { dst in
            dst.storeLE32(UInt32(outputBytes), at: 0)
            data.withUnsafeBytes { src in
                if !src.isEmpty { UnsafeMutableRawBufferPointer(rebasing: dst[4...]).copyMemory(from: src) }
            }
        }, outputCapacity: 4 + outputBytes) { _, p in Array(p) }
    }

    public func stats() throws -> CoreStats {
        try engine.execute(.stats, inputCount: 0, fill: { _ in }, outputCapacity: 64) { _, p in CoreStats(payload: p) }
    }

    /// Formats doubles exactly as the engine's JSON writer does (Python repr rules).
    public func format(_ values: [Double]) throws -> [String] {
        try engine.execute(.formatF64, inputCount: 8 * values.count, fill: { dst in
            for (k, v) in values.enumerated() { dst.storeBytes(of: v.bitPattern.littleEndian, toByteOffset: 8 * k, as: UInt64.self) }
        }, outputCapacity: 32 * values.count + 8) { _, p in
            String(decoding: p, as: UTF8.self).split(separator: "\n", omittingEmptySubsequences: false).dropLast().map(String.init)
        }
    }

    /// Escape hatch for tests and tools: runs `body` with the engine inside the actor.
    public func withEngine<T: Sendable>(_ body: (MemCoreEngine) throws -> T) rethrows -> T {
        try body(engine)
    }
}
