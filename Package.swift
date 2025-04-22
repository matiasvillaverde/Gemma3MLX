// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Gemma3MLX",
    platforms: [
        .macOS(.v14),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "Gemma3MLX",
            targets: ["Gemma3MLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0"),
    ],
    targets: [
        .target(
            name: "Gemma3MLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]),
        .testTarget(
            name: "Gemma3MLXTests",
            dependencies: ["Gemma3MLX"],
            resources: [
                .copy("models"),
                .copy("images")
            ]),
    ]
)
