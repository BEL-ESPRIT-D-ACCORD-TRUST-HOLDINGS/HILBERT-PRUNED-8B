// The memcore.wasm ABI, mirrored from ../../core/memcore.c (see ../../ARCHITECTURE.md).

/// Operations accepted by `execute_core_step(op, in_ptr, in_len, out_ptr, out_cap) -> status`.
/// Every output starts with a little-endian `u32` payload length, then the payload.
public enum CoreOp: UInt32, Sendable {
    /// in: `u32 out_len` + data. Payload: the SHAKE256 digest (1...1024 bytes).
    case shake256 = 1
    /// in: memory file bytes. Payload: ``CoreRoots`` (264 bytes), or the rejection message.
    case roots = 2
    /// in: ``CLIRequest``. Payload: `u32 exit, u32 stdout_len, u32 stderr_len, u32 0`, stdout, stderr.
    case runCLI = 3
    /// Payload: ``CoreStats`` (10 x u32).
    case stats = 4
    /// in: f64 values. Payload: each formatted like the engine's JSON writer, one per line.
    case formatF64 = 5
}

/// Status codes returned by `execute_core_step`.
public enum CoreStatus: UInt32, Sendable, Error {
    case ok = 0
    case badOp = 1
    /// An input or output range is outside one allocated heap block, misaligned, or overlapping.
    case badPointer = 2
    case badRequest = 3
    /// `out[0..<4]` holds the total size needed.
    case outTooSmall = 4
    case noMemory = 5
    /// The input is well-formed but invalid; the payload is the reason.
    case rejected = 6
}

/// Fixed layout facts of the core, checked against the running module in the tests.
public enum CoreLayout {
    public static let abiVersion: UInt32 = 1
    public static let pageSize = 65_536
    public static let stackSize: UInt32 = 262_144
    /// `--max-memory` of the link: 4096 pages.
    public static let maxMemoryBytes = 268_435_456
    /// Every payload pointer the core hands out is aligned to this.
    public static let alignment: UInt32 = 8
    /// The envelope in front of every output.
    public static let envelopeSize = 4
    public static let cliMagic: UInt32 = 0x3149_4C43  // "CLI1" little endian
}

/// `CoreRoots`: the commitment of a memory file (cleanroom-transformer SPEC.md 6.2).
public struct CoreRoots: Equatable, Sendable {
    public var idRoot: [UInt8]
    public var timeRoot: [UInt8]
    public var head: [UInt8]
    public var root: [UInt8]
    public var count: UInt64

    public static let byteSize = 4 * 64 + 8

    init(payload p: UnsafeRawBufferPointer) {
        precondition(p.count == Self.byteSize)
        idRoot = Array(p[0..<64])
        timeRoot = Array(p[64..<128])
        head = Array(p[128..<192])
        root = Array(p[192..<256])
        count = p.loadLE64(at: 256)
    }

    public var rootHex: String { root.hex }
}

/// `CoreStats`: the memory map as the running core sees it.
public struct CoreStats: Equatable, Sendable {
    public var abi, stackSize, dataEnd, heapLo, heapHi, memoryBytes, usedBytes, usedBlocks, freeBlocks, fault: UInt32

    init(payload p: UnsafeRawBufferPointer) {
        precondition(p.count == 40)
        let v = (0..<10).map { p.loadLE32(at: 4 * $0) }
        (abi, stackSize, dataEnd, heapLo, heapHi) = (v[0], v[1], v[2], v[3], v[4])
        (memoryBytes, usedBytes, usedBlocks, freeBlocks, fault) = (v[5], v[6], v[7], v[8], v[9])
    }
}

/// The result of one command line, as the native binary would produce it.
public struct CLIResult: Equatable, Sendable {
    public var exitCode: UInt32
    public var stdout: [UInt8]
    public var stderr: [UInt8]

    public var stdoutText: String { String(decoding: stdout, as: UTF8.self) }
    public var stderrText: String { String(decoding: stderr, as: UTF8.self) }
}

extension UnsafeRawBufferPointer {
    /// Little-endian loads that do not require alignment.
    func loadLE32(at offset: Int) -> UInt32 {
        UInt32(littleEndian: loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }

    func loadLE64(at offset: Int) -> UInt64 {
        UInt64(littleEndian: loadUnaligned(fromByteOffset: offset, as: UInt64.self))
    }
}

extension UnsafeMutableRawBufferPointer {
    func storeLE32(_ value: UInt32, at offset: Int) {
        storeBytes(of: value.littleEndian, toByteOffset: offset, as: UInt32.self)
    }
}

extension Sequence where Element == UInt8 {
    public var hex: String {
        let digits = Array("0123456789abcdef".utf8)
        var out = [UInt8]()
        for b in self {
            out.append(digits[Int(b >> 4)])
            out.append(digits[Int(b & 15)])
        }
        return String(decoding: out, as: UTF8.self)
    }
}
