// swift-tools-version:6.0
// Swift host for memcore.wasm (../core): a WasmKit embedding with zero-copy, bounds-checked access
// to the core's linear memory, an actor API, and `memcore-cli`, which runs the cleanroom-transformer
// memory-* commands on real files. See ../ARCHITECTURE.md.
import PackageDescription

let package = Package(
    name: "MemCoreHost",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "MemCoreHost", targets: ["MemCoreHost"]),
        .executable(name: "memcore-cli", targets: ["memcore-cli"]),
    ],
    dependencies: [
        // 0.2.x builds with Swift 6.0 and later; 0.3 and newer need swift-tools 6.3.
        .package(url: "https://github.com/swiftwasm/WasmKit.git", .upToNextMinor(from: "0.2.1")),
        // WasmKit 0.2.x does not compile against swift-system 1.7+, which added its own `Stat` type.
        .package(url: "https://github.com/apple/swift-system", "1.5.0"..<"1.7.0"),
    ],
    targets: [
        .target(
            name: "MemCoreHost",
            dependencies: [.product(name: "WasmKit", package: "WasmKit"), .product(name: "SystemPackage", package: "swift-system")]
        ),
        .executableTarget(name: "memcore-cli", dependencies: ["MemCoreHost"]),
        .testTarget(name: "MemCoreHostTests", dependencies: ["MemCoreHost"]),
    ]
)
