// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "KoharuApple",
    platforms: [.iOS("26.0")],
    products: [
        .library(name: "KoharuApple", type: .static, targets: ["KoharuApple"]),
    ],
    targets: [
        .target(name: "KoharuApple", path: "Sources"),
    ]
)
