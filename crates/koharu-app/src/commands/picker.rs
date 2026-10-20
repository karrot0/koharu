//! Where imported pages come from and where exports go. Desktop uses the system dialogs. iOS
//! cannot grant a writable folder outside the sandbox, so imports come from the document picker
//! and exports land in the app's Documents folder, which the Files app shows under Koharu.

use std::path::PathBuf;

use anyhow::Result;
use tauri::WebviewWindow;

use crate::Runtime;

/// Asks for page files to import.
#[cfg(not(target_os = "ios"))]
pub(crate) async fn files(
    window: &WebviewWindow<Runtime>,
    filter: &str,
    extensions: &[&str],
) -> Result<Option<Vec<PathBuf>>> {
    Ok(rfd::AsyncFileDialog::new()
        .add_filter(filter, extensions)
        .set_parent(window)
        .pick_files()
        .await
        .map(|files| {
            files
                .into_iter()
                .map(|file| file.path().to_owned())
                .collect()
        }))
}

/// Asks for page files to import.
#[cfg(target_os = "ios")]
pub(crate) async fn files(
    window: &WebviewWindow<Runtime>,
    filter: &str,
    extensions: &[&str],
) -> Result<Option<Vec<PathBuf>>> {
    use tauri_plugin_dialog::DialogExt as _;

    let (sender, receiver) = tokio::sync::oneshot::channel();
    window
        .dialog()
        .file()
        .add_filter(filter, extensions)
        .pick_files(move |files| {
            let _ = sender.send(files);
        });
    receiver
        .await?
        .map(|files| {
            files
                .into_iter()
                .map(|file| file.into_path().map_err(Into::into))
                .collect()
        })
        .transpose()
}

/// Asks for a folder to import pages from.
#[cfg(not(target_os = "ios"))]
pub(crate) async fn folder(window: &WebviewWindow<Runtime>) -> Result<Option<PathBuf>> {
    Ok(rfd::AsyncFileDialog::new()
        .set_parent(window)
        .pick_folder()
        .await
        .map(|folder| folder.path().to_owned()))
}

/// Asks for a folder to import pages from.
#[cfg(target_os = "ios")]
pub(crate) async fn folder(_window: &WebviewWindow<Runtime>) -> Result<Option<PathBuf>> {
    anyhow::bail!("iOS cannot import a folder; select the page files instead")
}

/// Chooses the folder that receives one exported file per page.
#[cfg(not(target_os = "ios"))]
pub(crate) async fn export_folder(
    window: &WebviewWindow<Runtime>,
    _name: &str,
) -> Result<Option<PathBuf>> {
    folder(window).await
}

/// Chooses the folder that receives one exported file per page.
#[cfg(target_os = "ios")]
pub(crate) async fn export_folder(
    window: &WebviewWindow<Runtime>,
    name: &str,
) -> Result<Option<PathBuf>> {
    let folder = documents(window, name)?;
    std::fs::create_dir_all(&folder)?;
    Ok(Some(folder))
}

/// Chooses the path of a single exported archive.
#[cfg(not(target_os = "ios"))]
pub(crate) async fn export_file(
    window: &WebviewWindow<Runtime>,
    filter: &str,
    extension: &str,
    name: &str,
) -> Result<Option<PathBuf>> {
    Ok(rfd::AsyncFileDialog::new()
        .set_parent(window)
        .add_filter(filter, &[extension])
        .set_file_name(format!("{name}.{extension}"))
        .save_file()
        .await
        .map(|file| file.path().to_owned()))
}

/// Chooses the path of a single exported archive.
#[cfg(target_os = "ios")]
pub(crate) async fn export_file(
    window: &WebviewWindow<Runtime>,
    _filter: &str,
    extension: &str,
    name: &str,
) -> Result<Option<PathBuf>> {
    Ok(Some(documents(window, &format!("{name}.{extension}"))?))
}

#[cfg(target_os = "ios")]
fn documents(window: &WebviewWindow<Runtime>, name: &str) -> Result<PathBuf> {
    use tauri::Manager as _;

    // A project name is free text; keep it one path component inside Documents.
    let name = name.replace(['/', ':'], "_");
    anyhow::ensure!(
        !name.is_empty() && name != "." && name != "..",
        "the project name cannot name an export"
    );
    Ok(window.path().document_dir()?.join(name))
}
