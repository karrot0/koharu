//! Apple's on-device text recognition (Vision) and translation (Translation framework) on iOS.
//!
//! The frameworks are reached through a small Swift package that exports C functions, because
//! the Translation framework has no Objective-C interface. Both calls block, so run them on a
//! worker thread.
#![cfg(target_os = "ios")]

mod ffi;

use std::{ffi::CString, io::Cursor};

use anyhow::{Context, Result, anyhow};
use image::{DynamicImage, ImageFormat};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

/// Recognizes the text in an image, trying `languages` (BCP-47 tags) in order of preference.
pub fn recognize_text(image: &DynamicImage, languages: &[&str]) -> Result<String> {
    let mut encoded = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut encoded), ImageFormat::Png)
        .context("failed to encode the image for Vision")?;
    let languages = CString::new(languages.join(","))?;
    let recognition: Recognition = response(ffi::recognize_text(&encoded, &languages))?;
    Ok(recognition.text)
}

/// Translates `texts` from `source` to `target` (BCP-47 tags) with on-device models. Without a
/// source language, Apple's language recognizer picks one from the texts.
pub fn translate(texts: &[String], source: Option<&str>, target: &str) -> Result<Vec<String>> {
    let request = CString::new(serde_json::to_string(&TranslationJob {
        texts,
        source,
        target,
    })?)?;
    let translated: Translated = response(ffi::translate(&request))?;
    anyhow::ensure!(
        translated.texts.len() == texts.len(),
        "Apple Translation returned {} texts for {}",
        translated.texts.len(),
        texts.len()
    );
    Ok(translated.texts)
}

fn response<T: DeserializeOwned>(json: Option<String>) -> Result<T> {
    let json = json.ok_or_else(|| anyhow!("the Apple framework bridge returned no response"))?;
    let response: Response<T> = serde_json::from_str(&json)?;
    match (response.value, response.error) {
        (_, Some(error)) => Err(anyhow!(error)),
        (Some(value), None) => Ok(value),
        (None, None) => Err(anyhow!(
            "the Apple framework bridge returned an empty response"
        )),
    }
}

#[derive(Deserialize)]
struct Response<T> {
    value: Option<T>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct Recognition {
    text: String,
}

#[derive(Serialize)]
struct TranslationJob<'a> {
    texts: &'a [String],
    source: Option<&'a str>,
    target: &'a str,
}

#[derive(Deserialize)]
struct Translated {
    texts: Vec<String>,
}
