//! Apple's on-device Translation framework, on iOS.

use crate::{Model, Provider, TranslationRequest};

/// Apple Translation has no settings: its languages are downloaded in the system's Translate
/// settings.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, specta::Type)]
#[serde(default)]
pub struct AppleConfig {}

pub(crate) fn models() -> Vec<Model> {
    if cfg!(target_os = "ios") {
        vec![Model::service(Provider::Apple, "Apple Translation")]
    } else {
        Vec::new()
    }
}

#[cfg(target_os = "ios")]
pub(crate) async fn translate(request: &TranslationRequest) -> anyhow::Result<Vec<String>> {
    let segments = request.segments.clone();
    let source = request.source_language.map(crate::Language::tag);
    let target = request.target_language.tag();
    tokio_rayon::spawn(move || koharu_apple::translate(&segments, source, target)).await
}

#[cfg(not(target_os = "ios"))]
pub(crate) async fn translate(_request: &TranslationRequest) -> anyhow::Result<Vec<String>> {
    anyhow::bail!("Apple Translation is only available on iOS")
}
