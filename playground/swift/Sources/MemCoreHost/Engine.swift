// Low-level embedding of memcore.wasm in WasmKit: guest allocation, bounds-checked zero-copy
// views onto linear memory, and the execute_core_step protocol. Not thread-safe; ``MemCore``
// (an actor) is the isolated, idiomatic API on top.
import WasmKit

public enum MemCoreError: Error, Equatable, Sendable {
    case abiMismatch(UInt32)
    case missingExport(String)
    /// The module trapped; the core is written never to trap, so this is a bug report.
    case trap(String)
    /// A host access fell outside linear memory; caught before touching memory.
    case outOfBounds(offset: UInt32, count: Int, memoryBytes: Int)
    case misaligned(offset: UInt32)
    case allocationFailed(bytes: Int)
    case requestTooLarge(bytes: Int)
    case status(CoreStatus)
    /// The core rejected the input (``CoreOp/roots``); the message is the engine's error text.
    case rejected(String)
}

/// A block the host allocated in the core's heap with `alloc`; freed with ``MemCoreEngine/free(_:)``.
public struct GuestBuffer: Equatable, Sendable {
    public let offset: UInt32
    public let count: Int

    /// For raw protocol work; the core checks every range it is given and flags bad frees.
    public init(offset: UInt32, count: Int) {
        self.offset = offset
        self.count = count
    }
}

public final class MemCoreEngine {
    private let instance: Instance
    private let memory: Memory
    private let fnAlloc, fnDealloc, fnStep, fnMemoryBytes, fnFault: Function

    /// Instantiates the core from its bytes (no imports) and checks its ABI version.
    public init(wasm: [UInt8]) throws {
        let module = try parseWasm(bytes: wasm)
        let store = Store(engine: Engine())
        let inst = try module.instantiate(store: store)
        func export(_ name: String) throws -> Function {
            guard let f = inst.exports[function: name] else { throw MemCoreError.missingExport(name) }
            return f
        }
        guard let mem = inst.exports[memory: "memory"] else { throw MemCoreError.missingExport("memory") }
        instance = inst
        memory = mem
        fnAlloc = try export("alloc")
        fnDealloc = try export("dealloc")
        fnStep = try export("execute_core_step")
        fnMemoryBytes = try export("core_memory_bytes")
        fnFault = try export("core_fault")
        let abi = try Self.u32(try export("core_abi_version").invoke([]))
        guard abi == CoreLayout.abiVersion else { throw MemCoreError.abiMismatch(abi) }
    }

    // ------------------------------------------------------------------ raw exports
    private static func u32(_ results: [Value]) throws -> UInt32 {
        guard results.count == 1, case .i32(let v) = results[0] else { throw MemCoreError.trap("unexpected results \(results)") }
        return v
    }

    private func invoke(_ f: Function, _ args: [UInt32]) throws -> [Value] {
        do {
            return try f.invoke(args.map { .i32($0) })
        } catch let t as Trap {
            throw MemCoreError.trap(t.description)
        }
    }

    /// Current size of linear memory, from the core itself (no copy of memory is made).
    public func memoryBytes() throws -> Int { Int(try Self.u32(try invoke(fnMemoryBytes, []))) }

    /// Sticky fault flag: 1 after a free with a wrong size or of a block that is not allocated.
    public func fault() throws -> UInt32 { try Self.u32(try invoke(fnFault, [])) }

    public func alloc(_ count: Int) throws -> GuestBuffer {
        guard count > 0, count <= Int(UInt32.max) else { throw MemCoreError.allocationFailed(bytes: count) }
        let p = try Self.u32(try invoke(fnAlloc, [UInt32(count)]))
        guard p != 0 else { throw MemCoreError.allocationFailed(bytes: count) }
        guard p % CoreLayout.alignment == 0 else { throw MemCoreError.misaligned(offset: p) }
        return GuestBuffer(offset: p, count: count)
    }

    public func free(_ b: GuestBuffer) throws {
        _ = try invoke(fnDealloc, [b.offset, UInt32(b.count)])
    }

    /// Raw `execute_core_step`; the pointer protocol is enforced by the core.
    public func stepRaw(op: UInt32, inOffset: UInt32, inCount: UInt32, outOffset: UInt32, outCapacity: UInt32) throws -> UInt32 {
        try Self.u32(try invoke(fnStep, [op, inOffset, inCount, outOffset, outCapacity]))
    }

    // ------------------------------------------------------------------ zero-copy memory access
    /// Runs `body` on the guest bytes `[offset, offset + count)` in place. Throws, without touching
    /// memory, when the range is not inside linear memory.
    public func withGuestBytes<T>(offset: UInt32, count: Int, _ body: (UnsafeMutableRawBufferPointer) throws -> T) throws -> T {
        let size = try memoryBytes()
        guard count >= 0, UInt64(offset) + UInt64(count) <= UInt64(size) else {
            throw MemCoreError.outOfBounds(offset: offset, count: count, memoryBytes: size)
        }
        return try memory.withUnsafeMutableBufferPointer(offset: UInt(offset), count: count, body)
    }

    public func withGuestBytes<T>(_ b: GuestBuffer, _ body: (UnsafeMutableRawBufferPointer) throws -> T) throws -> T {
        try withGuestBytes(offset: b.offset, count: b.count, body)
    }

    /// A copy of the whole linear memory (for determinism checks).
    public func snapshot() throws -> [UInt8] {
        try withGuestBytes(offset: 0, count: try memoryBytes()) { Array($0) }
    }

    // ------------------------------------------------------------------ one step
    /// Allocates the input and output blocks, lets `fill` write the request straight into guest
    /// memory, runs `op`, and hands `read` the payload in place. Grows the output once when the core
    /// reports how much it needs. Both blocks are freed on every path.
    public func execute<T>(
        _ op: CoreOp,
        inputCount: Int,
        fill: (UnsafeMutableRawBufferPointer) throws -> Void,
        outputCapacity: Int = 4096,
        read: (CoreStatus, UnsafeRawBufferPointer) throws -> T
    ) throws -> T {
        guard inputCount >= 0, inputCount < Int(UInt32.max) else { throw MemCoreError.requestTooLarge(bytes: inputCount) }
        let input = inputCount > 0 ? try alloc(inputCount) : nil
        defer { if let input { try? free(input) } }
        if let input { try withGuestBytes(input, fill) }
        var capacity = max(outputCapacity, CoreLayout.envelopeSize)
        for attempt in 0..<2 {
            let output = try alloc(capacity)
            defer { try? free(output) }
            let raw = try stepRaw(op: op.rawValue, inOffset: input?.offset ?? 0, inCount: UInt32(inputCount),
                                  outOffset: output.offset, outCapacity: UInt32(capacity))
            guard let status = CoreStatus(rawValue: raw) else { throw MemCoreError.trap("unknown status \(raw)") }
            if status == .outTooSmall && attempt == 0 {
                capacity = Int(try withGuestBytes(offset: output.offset, count: 4) { UnsafeRawBufferPointer($0).loadLE32(at: 0) })
                continue
            }
            guard status == .ok || status == .rejected else { throw MemCoreError.status(status) }
            return try withGuestBytes(output) { out in
                let bytes = UnsafeRawBufferPointer(out)
                let length = Int(bytes.loadLE32(at: 0))
                guard length <= capacity - CoreLayout.envelopeSize else { throw MemCoreError.status(.badRequest) }
                return try read(status, UnsafeRawBufferPointer(rebasing: bytes[4..<(4 + length)]))
            }
        }
        throw MemCoreError.status(.outTooSmall)
    }
}
