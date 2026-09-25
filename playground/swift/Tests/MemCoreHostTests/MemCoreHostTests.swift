// End-to-end tests of the Swift host: initialization, byte encoding, execution dispatch, state
// retrieval, determinism, boundaries, concurrency, and byte-for-byte parity with the native
// cleanroom-transformer binary when it is built (../../cleanroom-transformer/build).
import Foundation
import XCTest
import MemCoreHost

// little-endian helpers (the library's own are internal; the tests use only the public API)
extension UnsafeRawBufferPointer {
    fileprivate func loadLE32(at offset: Int) -> UInt32 { UInt32(littleEndian: loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
}
extension UnsafeMutableRawBufferPointer {
    fileprivate func storeLE32(_ v: UInt32, at offset: Int) { storeBytes(of: v.littleEndian, toByteOffset: offset, as: UInt32.self) }
}

private let packageDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../..").standardized
private let wasm: [UInt8] = {
    let path = ProcessInfo.processInfo.environment["MEMCORE_WASM"] ?? packageDir.appendingPathComponent("../core/memcore.wasm").path
    return [UInt8](FileManager.default.contents(atPath: path)!)
}()
private let nativeBinary: String = ProcessInfo.processInfo.environment["CLEANROOM_BIN"]
    ?? packageDir.appendingPathComponent("../../cleanroom-transformer/build/cleanroom-transformer").standardized.path

/// Builds a memory file in the engine's line format, hashing with the core's own SHAKE256.
private func makeMemory(_ core: MemCore, _ rows: [(id: String, question: String, answer: String, time: UInt64)]) async throws -> [UInt8] {
    var prev = String(repeating: "0", count: 128)
    var file = [UInt8]()
    for (seq, r) in rows.enumerated() {
        let line = #"{"seq": \#(seq), "time_us": \#(r.time), "id": "\#(r.id)", "question": "\#(r.question)", "answer": "a", "#
            + #""answer_text": "\#(r.answer)", "option_ids": ["a", "b"], "probabilities": [0.5, 0.5], "#
            + #""prompt_sha256": "\#(String(repeating: "0", count: 64))", "prompt_version": "direct-options-v1", "#
            + #""revision": "r", "recalled": 0, "prev": "\#(prev)"}"#
        prev = try await core.shake256([0] + Array(line.utf8)).hex
        file += Array(line.utf8) + [0x0A]
    }
    return file
}

private let rows: [(id: String, question: String, answer: String, time: UInt64)] = [
    ("q1", "First?", "yes", 1_000), ("q2", "Second?", "no", 2_000), ("q1", "First again?", "maybe", 5_000), ("q3", "Ünïcode?", "ja", 9_000),
]

final class MemCoreHostTests: XCTestCase {
    func testInitializationAndMemoryMap() async throws {
        let core = try MemCore(wasm: wasm)
        let s = try await core.stats()
        XCTAssertEqual(s.abi, CoreLayout.abiVersion)
        XCTAssertEqual(s.stackSize, CoreLayout.stackSize)
        XCTAssertGreaterThanOrEqual(s.dataEnd, s.stackSize, "data follows the stack (--stack-first)")
        XCTAssertGreaterThanOrEqual(s.heapLo, s.dataEnd)
        XCTAssertEqual(s.heapLo % CoreLayout.alignment, 0)
        XCTAssertEqual(Int(s.memoryBytes) % CoreLayout.pageSize, 0)
        let bytes = try await core.withEngine { try $0.memoryBytes() }
        XCTAssertEqual(bytes, Int(s.memoryBytes))
        XCTAssertEqual(s.fault, 0)
    }

    func testShake256MatchesFIPS202() async throws {
        let core = try MemCore(wasm: wasm)
        let empty = try await core.shake256([], outputBytes: 32)
        XCTAssertEqual(empty.hex, "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f")
        let abc = try await core.shake256(Array("abc".utf8), outputBytes: 32)
        XCTAssertEqual(abc.hex, "483366601360a8771c6863080cc4114d8db44530f8f1e1ee4f94ea37e78b5739")
    }

    func testCLIRequestEncoding() {
        let r = CLIRequest(argv: ["p", "memory-root", "--memory", "m"], files: ["m": [1, 2, 3], "unused": [9]])
        XCTAssertEqual(r.files.count, 1, "only files named on the command line are sent")
        var buf = [UInt8](repeating: 0xEE, count: r.byteCount)
        buf.withUnsafeMutableBytes { r.encode(into: $0) }
        buf.withUnsafeBytes { b in
            XCTAssertEqual(b.loadLE32(at: 0), CoreLayout.cliMagic)
            XCTAssertEqual(b.loadLE32(at: 4), 4)
            XCTAssertEqual(b.loadLE32(at: 8), 1)
            let off = Int(b.loadLE32(at: 16 + 8 * 1)), len = Int(b.loadLE32(at: 20 + 8 * 1))
            XCTAssertEqual(String(decoding: b[off..<(off + len)], as: UTF8.self), "memory-root")
            let dataOff = Int(b.loadLE32(at: 16 + 8 * 4 + 8))
            XCTAssertEqual(Array(b[dataOff..<(dataOff + 3)]), [1, 2, 3])
        }
        XCTAssertFalse(buf.contains(0xEE), "every byte of the request is written")
    }

    func testProveVerifyAndRejectThroughTheCLI() async throws {
        let core = try MemCore(wasm: wasm)
        let memory = try await makeMemory(core, rows)
        let root = try await core.run(["cleanroom-transformer", "memory-root", "--memory", "m.jsonl"], files: ["m.jsonl": memory])
        XCTAssertEqual(root.exitCode, 0, root.stderrText)
        let rootHex = try XCTUnwrap(root.stdoutText.split(separator: "\"").dropLast(1).last).description
        XCTAssertEqual(rootHex.count, 128)
        guard case .success(let roots) = await core.roots(of: memory) else { return XCTFail("roots rejected") }
        XCTAssertEqual(roots.rootHex, rootHex)
        XCTAssertEqual(roots.count, 4)

        let proof = try await core.run(["cleanroom-transformer", "memory-prove", "--memory", "m.jsonl", "--id", "q1"], files: ["m.jsonl": memory])
        XCTAssertTrue(proof.stdoutText.hasPrefix(#"{"type": "id-membership", "id": "q1", "entry": "{\"seq\": 2"#))
        let ok = try await core.run(["cleanroom-transformer", "memory-verify", "--proof", "p.json", "--root", rootHex], files: ["p.json": proof.stdout])
        XCTAssertEqual(ok.stdoutText, "valid: id has this latest entry under root \(rootHex) (4 entries)\n")

        var tampered = memory
        tampered[tampered.firstIndex(of: UInt8(ascii: "?"))!] = UInt8(ascii: "!")
        let bad = try await core.run(["cleanroom-transformer", "memory-root", "--memory", "m.jsonl"], files: ["m.jsonl": tampered])
        XCTAssertEqual(bad.exitCode, 1)
        XCTAssertEqual(bad.stderrText, "cleanroom-transformer: m.jsonl: memory line 2: prev does not match the hash of the previous entry (history was changed)\n")
        guard case .failure(.rejected(let why)) = await core.roots(of: tampered) else { return XCTFail("tampering accepted") }
        XCTAssertTrue(why.hasSuffix("history was changed)"))
    }

    func testNativeParity() async throws {
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: nativeBinary), "native binary not built")
        let core = try MemCore(wasm: wasm)
        let memory = try await makeMemory(core, rows)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("memcore-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        var files = ["m.jsonl": memory]
        let proof = try await core.run(["x", "memory-prove", "--memory", "m.jsonl", "--from", "1001", "--to", "1999"], files: files)
        files["gap.json"] = proof.stdout
        for (name, data) in files { try Data(data).write(to: dir.appendingPathComponent(name)) }
        let cases: [[String]] = [
            ["memory-root", "--memory", "m.jsonl"], ["memory-recall", "--memory", "m.jsonl", "--id", "q1", "--last", "1"],
            ["memory-prove", "--memory", "m.jsonl", "--id", "q3"], ["memory-prove", "--memory", "m.jsonl", "--from", "1000", "--to", "1000"],
            ["memory-verify", "--proof", "gap.json"], ["memory-verify", "--proof", "gap.json", "--root", "00"],
            ["memory-similar", "--memory", "m.jsonl", "--id", "q1"], ["memory-root"], [], ["nope"],
        ]
        for args in cases {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: nativeBinary)
            p.arguments = args
            p.currentDirectoryURL = dir
            let out = Pipe(), err = Pipe()
            p.standardOutput = out
            p.standardError = err
            try p.run()
            let so = out.fileHandleForReading.readDataToEndOfFile(), se = err.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            let got = try await core.run(["cleanroom-transformer"] + args, files: files)
            XCTAssertEqual(got.exitCode, UInt32(p.terminationStatus), "\(args)")
            XCTAssertEqual(got.stdout, [UInt8](so), "\(args)")
            XCTAssertEqual(got.stderr, [UInt8](se), "\(args)")
        }
    }

    func testDeterminismBitIdenticalLinearMemory() async throws {
        func script(_ core: MemCore) async throws -> [UInt8] {
            let memory = try await makeMemory(core, rows)
            _ = try await core.run(["x", "memory-prove", "--memory", "m", "--id", "q2"], files: ["m": memory])
            _ = await core.roots(of: memory)
            _ = try await core.run(["x", "memory-verify", "--proof", "missing"], files: [:])
            return try await core.withEngine { try $0.snapshot() }
        }
        let a = try await script(try MemCore(wasm: wasm))
        let b = try await script(try MemCore(wasm: wasm))
        XCTAssertEqual(a.count, b.count)
        XCTAssertTrue(a == b, "linear memories differ")
    }

    func testBoundariesAreCheckedBeforeMemoryIsTouched() async throws {
        let core = try MemCore(wasm: wasm)
        try await core.withEngine { e in
            let size = try e.memoryBytes()
            XCTAssertThrowsError(try e.withGuestBytes(offset: UInt32(size - 4), count: 8) { _ in }) { error in
                XCTAssertEqual(error as? MemCoreError, .outOfBounds(offset: UInt32(size - 4), count: 8, memoryBytes: size))
            }
            XCTAssertThrowsError(try e.withGuestBytes(offset: UInt32.max, count: 1) { _ in })
            XCTAssertThrowsError(try e.alloc(0))
            let out = try e.alloc(64)
            XCTAssertEqual(try e.stepRaw(op: 4, inOffset: 0, inCount: 0, outOffset: 16, outCapacity: 64), CoreStatus.badPointer.rawValue, "stack region")
            XCTAssertEqual(try e.stepRaw(op: 4, inOffset: 0, inCount: 0, outOffset: out.offset, outCapacity: 65), CoreStatus.badPointer.rawValue, "past the block")
            XCTAssertEqual(try e.stepRaw(op: 4, inOffset: 0, inCount: 0, outOffset: out.offset, outCapacity: 8), CoreStatus.outTooSmall.rawValue)
            XCTAssertEqual(try e.stepRaw(op: 77, inOffset: 0, inCount: 0, outOffset: out.offset, outCapacity: 64), CoreStatus.badOp.rawValue)
            XCTAssertEqual(try e.fault(), 0)
            try e.free(GuestBuffer(offset: out.offset, count: 32))
            XCTAssertEqual(try e.fault(), 1, "a free with the wrong size is refused and flagged")
        }
        // a request whose argument table points past its end
        do {
            _ = try await core.withEngine { e in
                try e.execute(.runCLI, inputCount: 24, fill: { b in
                    b.storeLE32(CoreLayout.cliMagic, at: 0)
                    b.storeLE32(1, at: 4)
                    b.storeLE32(0, at: 8)
                    b.storeLE32(0, at: 12)
                    b.storeLE32(20, at: 16)
                    b.storeLE32(99, at: 20)
                }) { status, _ in status.rawValue }
            }
            XCTFail("a malformed request was accepted")
        } catch {
            XCTAssertEqual(error as? MemCoreError, .status(.badRequest))
        }
    }

    func testConcurrentCallsAreSerializedByTheActor() async throws {
        let core = try MemCore(wasm: wasm)
        let memory = try await makeMemory(core, rows)
        let baseline = try await core.stats()
        let outputs = try await withThrowingTaskGroup(of: CLIResult.self) { group in
            for k in 0..<32 {
                group.addTask {
                    try await core.run(["x", k % 2 == 0 ? "memory-root" : "memory-recall", "--memory", "m"], files: ["m": memory])
                }
            }
            return try await group.reduce(into: [CLIResult]()) { $0.append($1) }
        }
        XCTAssertEqual(Set(outputs.map(\.stdout)).count, 2)
        let after = try await core.stats()
        XCTAssertEqual(after.heapHi, baseline.heapHi, "every step returned the heap to its baseline")
        XCTAssertEqual(after.usedBytes, baseline.usedBytes)
    }

    func testDoubleFormattingMatchesTheEngineWriter() async throws {
        let core = try MemCore(wasm: wasm)
        let got = try await core.format([0.5, 1e22, 5e-324, 0.1 + 0.2, -0.0, 1e16, 123456.0, .infinity, 0.0001])
        XCTAssertEqual(got, ["0.5", "1e+22", "5e-324", "0.30000000000000004", "-0.0", "1e+16", "123456.0", "Infinity", "0.0001"])
    }
}
