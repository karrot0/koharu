//! Koharu's Tauri-managed application state, commands, and lifecycle.

mod app;
mod commands;

/// The Tauri runtime. Desktop renders through CEF; iOS permits only the system WebKit engine.
#[cfg(not(target_os = "ios"))]
pub type Runtime = tauri_runtime_cef::CefRuntime;
#[cfg(target_os = "ios")]
pub type Runtime = tauri_runtime_wry::WryRuntime;

pub use app::run;
pub use commands::bindings;
