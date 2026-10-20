# koharu

`koharu` is Koharu's native composition package. It owns process startup,
diagnostics, Tauri build integration, and application configuration.

## Application boundary

```text
React
  | direct Tauri commands
  | typed IPC channels
  v
koharu-app
  | Tauri-managed project, canvas, pipeline, jobs, and channel state
  +-> koharu-scene
  +-> koharu-desktop -> koharu-canvas
  +-> koharu-renderer -> raster / koharu-psd

koharu -> koharu-app + koharu-desktop
```

Every operation has a named Tauri command. Commands that mutate a project take
its id and current revision directly. The frontend serializes those mutations
and uses the returned revision for the next call.

Native updates do not share an event envelope. `connect` binds independent
typed channels for project snapshots, canvas state, jobs, downloads,
preferences, resource telemetry, and cleanup reports. Tauri state is the only
application state container.

Thumbnails are read with `get_thumbnail`; the frontend creates a temporary
object URL from the returned bytes. There is no custom URI scheme or resource
protocol.

## Generated bindings

Rust command signatures and data types are authoritative:

```powershell
cargo run -p koharu-app --bin generate
```

Focused validation:

```powershell
cargo check -p koharu -p koharu-app -p koharu-desktop
bun x tsc --noEmit -p packages/koharu/tsconfig.json
cd packages/koharu
bun run test
```

## iOS

iOS runs the same application on the system WebKit engine (wry) instead of CEF.
Apple only executes code signed into the bundle, so Torch, llama.cpp, and
stable-diffusion.cpp ship as embedded XCFrameworks rather than downloaded
packages. Each package repository publishes an `aarch64-apple-ios-*` archive
beside its desktop archives; the iOS `beforeBuildCommand` stages them into
`gen/frameworks`, which `tauri.ios.conf.json` embeds.

The `iOS` workflow builds on a GitHub macOS runner: it generates the Xcode
project with `cargo tauri ios init`, builds and signs the IPA, and uploads it to
TestFlight. Push a tag such as `ios-1` to run it. Signing uses an App Store
Connect API key with the Admin role, stored as the `APPLE_API_KEY`,
`APPLE_API_ISSUER`, and `APPLE_API_KEY_P8` secrets beside
`APPLE_DEVELOPMENT_TEAM`.

llama.cpp and stable-diffusion.cpp use Metal. PyTorch builds MPS only for
macOS, so Torch models run on the CPU. iOS cannot pick a writable folder, so
exports are written to the app's Documents folder, which the Files app lists.
