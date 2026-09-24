// memcore-cli: `cleanroom-transformer memory-*` on the WebAssembly core, with real files.
//
//   swift run memcore-cli memory-root --memory memory.jsonl
//
// Arguments are passed through unchanged; every argument that names an existing file is loaded
// and handed to the core under that name, so errors and output match the native binary.
// The module is read from $MEMCORE_WASM, or from ../../core/memcore.wasm next to this package.
import Foundation
import MemCoreHost

func wasmPath() -> String {
    if let p = ProcessInfo.processInfo.environment["MEMCORE_WASM"] { return p }
    let source = URL(fileURLWithPath: #filePath)  // .../swift/Sources/memcore-cli/main.swift
    return source.deletingLastPathComponent().appendingPathComponent("../../../core/memcore.wasm").standardized.path
}

let argv = ["cleanroom-transformer"] + CommandLine.arguments.dropFirst()
var files: [String: [UInt8]] = [:]
var isDirectory: ObjCBool = false
for a in argv.dropFirst() where !a.hasPrefix("--") && FileManager.default.fileExists(atPath: a, isDirectory: &isDirectory) && !isDirectory.boolValue {
    if let d = FileManager.default.contents(atPath: a) { files[a] = [UInt8](d) }
}

do {
    guard let wasm = FileManager.default.contents(atPath: wasmPath()) else {
        FileHandle.standardError.write(Data("memcore-cli: cannot read \(wasmPath()) (set MEMCORE_WASM)\n".utf8))
        exit(70)
    }
    let core = try MemCore(wasm: [UInt8](wasm))
    let r = try await core.run(argv, files: files)
    FileHandle.standardOutput.write(Data(r.stdout))
    FileHandle.standardError.write(Data(r.stderr))
    exit(Int32(r.exitCode))
} catch {
    FileHandle.standardError.write(Data("memcore-cli: \(error)\n".utf8))
    exit(70)
}
