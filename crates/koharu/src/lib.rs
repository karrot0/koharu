//! Koharu's process entrypoint and diagnostics.

pub mod panic;
pub mod sentry;
pub mod tracing;

use tracing_subscriber::layer::SubscriberExt as _;

/// Installs diagnostics and runs the application until its event loop exits.
pub fn run() {
    let _guard = sentry::initialize();
    panic::install();
    ::tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::filter::EnvFilter::builder()
                    .with_default_directive(::tracing::Level::INFO.into())
                    .from_env_lossy(),
            )
            .with(sentry::tracing_layer())
            .with(tracing::TimingLayer::new()),
    )
    .expect("failed to set the global tracing subscriber");
    koharu_app::run(tauri::generate_context!()).expect("failed to run the application");
}

/// The iOS host calls this instead of `main`.
#[cfg(target_os = "ios")]
#[tauri::mobile_entry_point]
fn start() {
    // SAFETY: the iOS host calls this on its main thread before Koharu starts any other thread,
    // which is why this entry point is not `async`: an async one runs on the started runtime.
    unsafe { koharu_runtime::ios::configure_downloads() };
    // Desktop runs the event loop inside a Tokio runtime context; keep that true on iOS.
    tauri::async_runtime::block_on(async { tokio::task::block_in_place(run) });
}
