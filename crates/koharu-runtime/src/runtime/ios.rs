//! iOS executes only code that is signed into the app bundle, so the llama.cpp and
//! stable-diffusion.cpp packages ship as XCFrameworks that the iOS build embeds, and the app
//! resolves them from the bundle instead of the download store. iOS runs detection through
//! Core ML, so it embeds no Torch runtime.

use std::path::Path;
#[cfg(target_os = "ios")]
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::{
    download,
    runtime::packages::{diffusion, llama},
    source::extract,
};

const PATTERNS: &[&str] = &["*.xcframework/**", "**/*.xcframework/**"];
const STAMP: &str = ".releases";

// The karrot0 forks publish the iOS packages under the same release tags as the desktop pins,
// built from the same upstream sources, so the generated bindings match both.
const ARCHIVES: [(&str, &str, &str); 2] = [
    (
        "karrot0/llama",
        llama::RELEASE,
        "aarch64-apple-ios-metal.tar.gz",
    ),
    (
        "karrot0/diffusion",
        diffusion::RELEASE,
        "aarch64-apple-ios-metal.tar.gz",
    ),
];

/// Downloads the XCFrameworks that an iOS build embeds into `destination`.
///
/// The directory is replaced whenever a package release changes so a bundle never mixes
/// frameworks from different releases.
pub async fn stage_frameworks(destination: &Path) -> Result<()> {
    let releases = ARCHIVES
        .iter()
        .map(|(repository, release, _)| format!("{repository} {release}\n"))
        .collect::<String>();
    if std::fs::read_to_string(destination.join(STAMP)).is_ok_and(|staged| staged == releases) {
        return Ok(());
    }
    if destination
        .try_exists()
        .with_context(|| format!("failed to inspect {}", destination.display()))?
    {
        std::fs::remove_dir_all(destination)
            .with_context(|| format!("failed to replace {}", destination.display()))?;
    }
    std::fs::create_dir_all(destination)?;

    for (repository, release, asset) in ARCHIVES {
        let url = format!("https://github.com/{repository}/releases/download/{release}/{asset}");
        let archive = tempfile::Builder::new().suffix(".tar.gz").tempfile()?;
        download::fetch(&url, archive.path()).await?;
        extract(archive.path(), destination, PATTERNS)
            .with_context(|| format!("failed to extract the iOS frameworks from {url}"))?;
    }
    std::fs::write(destination.join(STAMP), releases)?;
    Ok(())
}

/// Returns the bundle's framework directory once it holds a complete package.
#[cfg(target_os = "ios")]
pub(crate) fn frameworks(complete: impl Fn(&Path) -> bool) -> Result<PathBuf> {
    let executable = std::env::current_exe().context("failed to locate the app executable")?;
    let root = executable
        .parent()
        .context("the app executable has no bundle directory")?
        .join("Frameworks");
    anyhow::ensure!(
        complete(&root),
        "the app bundle is missing runtime frameworks in {}",
        root.display()
    );
    Ok(root)
}
