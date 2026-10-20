fn main() {
    println!("cargo:rerun-if-changed=swift");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios") {
        link();
    }
}

// Mirrors Tauri's Swift plugin linking: swift-rs builds the package as a static library for the
// iOS SDK, which only a macOS host provides.
#[cfg(target_os = "macos")]
fn link() {
    let sdk_root = std::env::var_os("SDKROOT");
    // swift-rs picks the iOS SDK itself; an inherited SDKROOT points it at the wrong one.
    unsafe {
        std::env::remove_var("SDKROOT");
    }
    let deployment = std::env::var("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or_else(|_| "26.0".into());
    swift_rs::SwiftLinker::new("11.0")
        .with_ios(&deployment)
        .with_package("KoharuApple", "swift")
        .link();
    if let Some(root) = sdk_root {
        unsafe {
            std::env::set_var("SDKROOT", root);
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn link() {
    println!("cargo:warning=building koharu-apple for iOS needs a macOS host to compile its Swift");
}
