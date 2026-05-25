// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "mlx-swift-ml-sharp",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(name: "MLXSharp", targets: ["MLXSharp"]),
        .executable(name: "sharp-bench", targets: ["SharpBench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3"))
    ],
    targets: [
        .target(
            name: "MLXSharp",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "SharpBench",
            dependencies: ["MLXSharp"]
        ),
        .testTarget(
            name: "MLXSharpTests",
            dependencies: ["MLXSharp"],
            resources: [
                .copy("Fixtures")
            ]
        ),
    ]
)
