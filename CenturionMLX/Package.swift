// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "CenturionMLX",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "CenturionMLX", targets: ["CenturionMLX"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.3"),
    ],
    targets: [
        .target(
            name: "CenturionMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
            ],
            resources: [
                .copy("Resources"),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v5),
            ]
        ),
    ]
)
