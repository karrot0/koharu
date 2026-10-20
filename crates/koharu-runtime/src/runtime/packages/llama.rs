use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use strum::EnumProperty;

#[cfg(target_os = "ios")]
use crate::runtime::ios;
use crate::{
    Hardware,
    runtime::{
        DiscoverablePackage, Package, RuntimePackage, graph::Component, loader, packages::Cuda,
        sealed,
    },
};
#[cfg(not(target_os = "ios"))]
use crate::{Store, download, source::extract};

pub(crate) const RELEASE: &str = "b11642";

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, strum::Display, strum::EnumProperty)]
pub(crate) enum Llama {
    #[strum(
        serialize = "windows-cuda",
        props(
            asset = "x86_64-pc-windows-msvc-cuda.tar.gz",
            libraries = "llama.dll,mtmd.dll"
        )
    )]
    WindowsCuda,
    #[cfg_attr(
        all(target_os = "linux", target_arch = "aarch64"),
        strum(props(asset = "aarch64-unknown-linux-gnu-cuda.tar.gz"))
    )]
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "aarch64")),
        strum(props(asset = "x86_64-unknown-linux-gnu-cuda.tar.gz"))
    )]
    #[strum(serialize = "linux-cuda", props(libraries = "libllama.so,libmtmd.so"))]
    LinuxCuda,
    #[strum(
        serialize = "windows-hip",
        props(
            asset = "x86_64-pc-windows-msvc-hip.tar.gz",
            libraries = "llama.dll,mtmd.dll"
        )
    )]
    WindowsHip,
    #[strum(
        serialize = "linux-hip",
        props(
            asset = "x86_64-unknown-linux-gnu-hip.tar.gz",
            libraries = "libllama.so,libmtmd.so"
        )
    )]
    LinuxHip,
    #[strum(
        serialize = "windows-vulkan",
        props(
            asset = "x86_64-pc-windows-msvc-vulkan.tar.gz",
            libraries = "llama.dll,mtmd.dll"
        )
    )]
    WindowsVulkan,
    #[strum(
        serialize = "linux-vulkan",
        props(
            asset = "x86_64-unknown-linux-gnu-vulkan.tar.gz",
            libraries = "libllama.so,libmtmd.so"
        )
    )]
    LinuxVulkan,
    #[strum(
        serialize = "macos-metal",
        props(
            asset = "aarch64-apple-darwin-metal.tar.gz",
            libraries = "libllama.dylib,libmtmd.dylib"
        )
    )]
    MacosMetal,
    // Like upstream llama.cpp's XCFramework, one framework carries ggml, llama, and mtmd.
    #[strum(serialize = "ios-metal", props(libraries = "llama.framework/llama"))]
    IosMetal,
}

impl Llama {
    #[cfg(not(target_os = "ios"))]
    fn asset(self) -> &'static str {
        self.get_str("asset").expect("llama package has an asset")
    }

    fn libraries(self) -> impl Iterator<Item = &'static str> {
        self.get_str("libraries")
            .expect("llama package has libraries")
            .split(',')
    }

    fn complete(self, path: &Path) -> bool {
        self.libraries().all(|name| path.join(name).is_file())
    }
}

impl sealed::Sealed for Llama {}

impl Package for Llama {
    #[cfg(target_os = "ios")]
    async fn install(self) -> Result<PathBuf> {
        ios::frameworks(|root| self.complete(root))
    }

    #[cfg(not(target_os = "ios"))]
    async fn install(self) -> Result<PathBuf> {
        let target = Store::root()
            .join("llama")
            .join(RELEASE)
            .join(self.to_string());
        Store::directory(
            target,
            move |path| self.complete(path),
            move |stage| async move {
                let asset = self.asset();
                let url = format!(
                    "https://github.com/koharu-rs/llama/releases/download/{RELEASE}/{asset}"
                );
                let archive = tempfile::Builder::new().suffix(".tar.gz").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                extract(
                    archive.path(),
                    &stage,
                    &["**/*.dll", "**/*.dylib", "**/*.so", "**/*.so.*"],
                )
            },
        )
        .await
    }
}

impl DiscoverablePackage for Llama {
    fn discover(hardware: &Hardware) -> Option<Self> {
        if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
            if hardware.supports_cuda() {
                return Some(Self::WindowsCuda);
            }
            if hardware.supports_rocm() {
                return Some(Self::WindowsHip);
            }
            if hardware.supports_vulkan() {
                return Some(Self::WindowsVulkan);
            }
            None
        } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            if hardware.supports_cuda() {
                return Some(Self::LinuxCuda);
            }
            if hardware.supports_rocm() {
                return Some(Self::LinuxHip);
            }
            hardware.supports_vulkan().then_some(Self::LinuxVulkan)
        } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            hardware.supports_cuda().then_some(Self::LinuxCuda)
        } else if cfg!(target_os = "ios") {
            hardware.supports_metal().then_some(Self::IosMetal)
        } else if hardware.supports_metal() {
            Some(Self::MacosMetal)
        } else {
            None
        }
    }
}

impl RuntimePackage for Llama {
    const NAME: &'static str = "llama";

    fn dependencies(self, hardware: &Hardware) -> Result<Vec<Component>> {
        match self {
            Self::WindowsCuda | Self::LinuxCuda => Ok(vec![
                Component::Cuda(Cuda::Runtime13),
                Component::Cuda(Cuda::Blas13),
            ]),
            Self::WindowsHip | Self::LinuxHip => Ok(vec![Component::Rocm(hardware.rocm_target()?)]),
            Self::WindowsVulkan | Self::LinuxVulkan | Self::MacosMetal | Self::IosMetal => {
                Ok(Vec::new())
            }
        }
    }

    async fn activate(self) -> Result<()> {
        let root = self.install().await?;
        for library in self.libraries() {
            loader::load(root.join(library), false)
                .with_context(|| format!("failed to activate llama library {library}"))?;
        }
        Ok(())
    }
}
