//! High-resolution manga layout instance segmentation with RF-DETR Seg 2XL.
//!
//! Checkpoint and strict Python loader:
//! https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/tree/aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2
//! RF-DETR upstream implementation:
//! https://github.com/roboflow/rf-detr/tree/4ab7c18729de9d02ffd0495795d0831b5630f01b
//!
//! Desktop runs the Torch port in `model.rs`. iOS has no Torch runtime and runs the same
//! checkpoint converted to Core ML (`coreml.rs`), published on the karrot0/koharu fork.

mod config;
#[cfg(target_os = "ios")]
mod coreml;
mod detections;
#[cfg(not(target_os = "ios"))]
mod model;
#[cfg(not(target_os = "ios"))]
mod processor;

use anyhow::{Context, Result};
use image::DynamicImage;

#[cfg(not(target_os = "ios"))]
pub use self::processor::KoharuLayoutRFDetrImageProcessor;
pub use self::{
    config::{KoharuLayoutRFDetrSeg2XLConfig, KoharuLayoutThresholds},
    detections::{KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask},
};

crate::model_repository!("mayocream/koharu-layout-rfdetr-seg-2xl-1152" @ "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2" {
    CONFIG = "inference_config.json",
});
#[cfg(not(target_os = "ios"))]
crate::model_repository!("mayocream/koharu-layout-rfdetr-seg-2xl-1152" @ "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2" {
    WEIGHTS = "model.safetensors",
});

/// The checkpoint above converted by `coreml/convert_koharu_layout.py` and compiled for iOS.
#[cfg(target_os = "ios")]
const CORE_ML_MODEL: koharu_runtime::GitHubReleaseArchive<'static> =
    koharu_runtime::GitHubReleaseArchive::pinned(
        "karrot0/koharu",
        "coreml-koharu-layout-rfdetr-seg-2xl-1152-1",
        "koharu-layout-rfdetr-seg-2xl-1152.mlmodelc.zip",
    );

#[derive(Debug)]
pub struct KoharuLayoutRFDetrSeg2XL {
    #[cfg(not(target_os = "ios"))]
    device: koharu_torch::Device,
    #[cfg(not(target_os = "ios"))]
    model: model::Model,
    #[cfg(not(target_os = "ios"))]
    processor: KoharuLayoutRFDetrImageProcessor,
    #[cfg(target_os = "ios")]
    network: coreml::CoreMlLayout,
}

impl KoharuLayoutRFDetrSeg2XL {
    #[cfg(not(target_os = "ios"))]
    pub async fn load(device: crate::Device) -> Result<Self> {
        use crate::backend::TryIntoDevice;

        let device: koharu_torch::Device = device.try_into_device()?;
        let config_path = CONFIG
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR inference config")?;
        let weights_path = WEIGHTS
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR weights")?;
        let config = KoharuLayoutRFDetrSeg2XLConfig::from_file(&config_path)?;
        let processor = KoharuLayoutRFDetrImageProcessor::new(&config)?;
        let mut model = model::Model::new(device);
        model
            .load(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;
        Ok(Self {
            device,
            model,
            processor,
        })
    }

    /// Core ML picks the Neural Engine, GPU, or CPU itself, so the runtime device is unused.
    #[cfg(target_os = "ios")]
    pub async fn load(_device: crate::Device) -> Result<Self> {
        let config_path = CONFIG
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR inference config")?;
        let model_path = CORE_ML_MODEL
            .resolve()
            .await
            .context("failed to resolve the KoharuLayout RF-DETR Core ML model")?
            .join("koharu-layout-rfdetr-seg-2xl-1152.mlmodelc");
        let config = KoharuLayoutRFDetrSeg2XLConfig::from_file(&config_path)?;
        let model = crate::coreml::CoreMlModel::load(&model_path)?;
        Ok(Self {
            network: coreml::CoreMlLayout::new(model, &config)?,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<KoharuLayoutDetections> {
        self.inference_with_thresholds(image, self.recommended_thresholds())
    }

    #[cfg(not(target_os = "ios"))]
    pub fn inference_with_thresholds(
        &self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        koharu_torch::no_grad(|| {
            let pixel_values = self.processor.preprocess(image, self.device)?;
            let output = self.model.forward(&pixel_values);
            self.processor
                .postprocess(&output, image.width(), image.height(), thresholds)
        })
    }

    #[cfg(target_os = "ios")]
    pub fn inference_with_thresholds(
        &self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        self.network.inference(image, thresholds)
    }

    pub fn recommended_thresholds(&self) -> KoharuLayoutThresholds {
        #[cfg(not(target_os = "ios"))]
        return self.processor.recommended_thresholds();
        #[cfg(target_os = "ios")]
        return self.network.recommended_thresholds();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::Result;

    use super::KoharuLayoutRFDetrSeg2XL;

    #[tokio::test]
    #[ignore = "downloads the checkpoint and requires CUDA"]
    async fn checkpoint_matches_rfdetr_upstream_structured_output() -> Result<()> {
        crate::init().await?;
        let image = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("benches/fixtures/object_detection/1.jpg"),
        )?;
        let model = KoharuLayoutRFDetrSeg2XL::load(crate::Device::cuda(0)).await?;
        let result = model.inference(&image)?;

        // RF-DETR 4ab7c18, CUDA BF16, shape=(1152, 1152), antialias disabled.
        // CUDA kernels vary across LibTorch releases, so compare structured
        // geometry and mask area in addition to bounded confidence differences.
        let best = &result.detections[0];
        assert_eq!(best.label, "bubble");
        assert!((best.score - 0.968_856_2).abs() < 0.01);
        for (actual, expected) in best
            .bbox
            .into_iter()
            .zip([566.220_7, 550.019_5, 691.044_9, 724.043])
        {
            assert!((actual - expected).abs() < 6.0);
        }
        assert!(best.area.abs_diff(18_882) < 100);

        let lower_panel = result
            .detections
            .iter()
            .find(|detection| detection.label == "panel" && detection.bbox[1] > 700.0)
            .expect("lower page panel");
        let middle_panel = result
            .detections
            .iter()
            .find(|detection| {
                detection.label == "panel" && detection.bbox[1] > 500.0 && detection.bbox[1] < 700.0
            })
            .expect("middle page panel");
        for (actual, expected) in lower_panel
            .bbox
            .into_iter()
            .zip([69.179_69, 799.453_1, 700.820_3, 1006.171_9])
        {
            assert!((actual - expected).abs() < 7.0);
        }
        for (actual, expected) in middle_panel
            .bbox
            .into_iter()
            .zip([70.683_59, 550.546_9, 699.316_4, 782.578_1])
        {
            assert!((actual - expected).abs() < 3.0);
        }
        assert!(lower_panel.area.abs_diff(136_734) < 1_000);
        assert!(middle_panel.area.abs_diff(142_179) < 2_000);
        Ok(())
    }
}
