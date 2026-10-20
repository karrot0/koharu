use std::path::PathBuf;

use anyhow::Context as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let destination = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: stage-ios-frameworks <destination>")?;
    let destination = std::path::absolute(&destination)
        .with_context(|| format!("failed to resolve {}", destination.display()))?;
    koharu_runtime::ios::stage_frameworks(&destination).await
}
