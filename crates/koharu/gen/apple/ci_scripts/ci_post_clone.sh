#!/bin/sh
# Xcode Cloud runs this after cloning. Xcode builds the generated project directly, so nothing
# runs the Tauri `beforeBuildCommand`; this script installs the toolchains, runs that command to
# build the frontend and stage the runtime XCFrameworks, and lets the generated "Build Rust Code"
# phase find cargo.
set -eu

repository="${CI_PRIMARY_REPOSITORY_PATH:?}"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
  sh -s -- -y --profile minimal --target aarch64-apple-ios
curl -fsSL https://bun.sh/install | bash
export PATH="$HOME/.cargo/bin:$HOME/.bun/bin:$PATH"

cd "$repository"
# The root postinstall installs the pinned Tauri CLI that the build phase invokes.
bun install --frozen-lockfile
sh -c "$(plutil -extract build.beforeBuildCommand raw -o - crates/koharu/tauri.ios.conf.json)"

# Xcode script phases get a minimal PATH without the rustup install.
for project in crates/koharu/gen/apple/*.xcodeproj/project.pbxproj; do
  sed -i '' 's#shellScript = "cargo tauri#shellScript = "PATH=$HOME/.cargo/bin:$PATH cargo tauri#' "$project"
  grep -qF 'PATH=$HOME/.cargo/bin:$PATH cargo tauri' "$project" || {
    echo "the Build Rust Code phase in $project does not call cargo tauri" >&2
    exit 1
  }
done
