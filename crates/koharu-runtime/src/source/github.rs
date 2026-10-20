use std::path::PathBuf;

use crate::{Store, download, source::extract};

const COMPLETE: &str = ".complete";

/// A zip archive attached to a GitHub release, unpacked once into the store.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GitHubReleaseArchive<'a> {
    repository: &'a str,
    tag: &'a str,
    asset: &'a str,
}

impl<'a> GitHubReleaseArchive<'a> {
    #[must_use]
    pub const fn pinned(repository: &'a str, tag: &'a str, asset: &'a str) -> Self {
        Self {
            repository,
            tag,
            asset,
        }
    }

    /// Returns the directory holding the archive's contents, downloading it on first use.
    #[tracing::instrument(skip_all)]
    pub async fn resolve(self) -> anyhow::Result<PathBuf> {
        let target = Store::root()
            .join("github")
            .join(self.repository)
            .join(self.tag)
            .join(self.asset.trim_end_matches(".zip"));
        Store::directory(
            target,
            |path| path.join(COMPLETE).is_file(),
            move |stage| async move {
                let url = format!(
                    "https://github.com/{}/releases/download/{}/{}",
                    self.repository, self.tag, self.asset
                );
                let archive = tempfile::Builder::new().suffix(".zip").tempfile()?;
                download::fetch(&url, archive.path()).await?;
                extract(archive.path(), &stage, &["**"])?;
                std::fs::write(stage.join(COMPLETE), [])?;
                Ok(())
            },
        )
        .await
    }
}
