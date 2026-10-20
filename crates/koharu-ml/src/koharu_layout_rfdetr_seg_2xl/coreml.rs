//! RF-DETR on Core ML, for iOS.
//!
//! The network is the checkpoint converted to an ML Program with the same input and outputs as
//! `model.rs`. Pre- and postprocessing reproduce `processor.rs` without a Torch runtime:
//! PyTorch's bilinear resize (`align_corners=False`, no antialiasing), sigmoid top-k over every
//! query/class pair, and per-mask bilinear projection thresholded at zero.

use anyhow::{Context, Result, bail, ensure};
use image::DynamicImage;

use super::{
    config::{KoharuLayoutRFDetrSeg2XLConfig, KoharuLayoutThresholds},
    detections::{
        KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask, validate_thresholds,
    },
};
use crate::coreml::{Array, CoreMlModel, Tensor};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const QUERIES: usize = 300;
/// Four labels and the checkpoint's background slot.
const CLASSES: usize = 5;
const MASK_SIZE: usize = 288;

#[derive(Debug)]
pub(super) struct CoreMlLayout {
    model: CoreMlModel,
    resolution: usize,
    num_select: usize,
    class_names: Vec<String>,
    recommended_thresholds: KoharuLayoutThresholds,
}

impl CoreMlLayout {
    pub(super) fn new(model: CoreMlModel, config: &KoharuLayoutRFDetrSeg2XLConfig) -> Result<Self> {
        ensure!(config.resolution > 0, "RF-DETR resolution must be positive");
        ensure!(config.num_select > 0, "RF-DETR num_select must be positive");
        Ok(Self {
            model,
            resolution: config.resolution as usize,
            num_select: config.num_select as usize,
            class_names: config.class_names(),
            recommended_thresholds: config.recommended_thresholds,
        })
    }

    pub(super) fn recommended_thresholds(&self) -> KoharuLayoutThresholds {
        self.recommended_thresholds
    }

    pub(super) fn inference(
        &self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        validate_thresholds(thresholds)?;
        let pixel_values = self.preprocess(image)?;
        let outputs = self.model.predict(
            &[("pixel_values", &pixel_values)],
            &["pred_logits", "pred_boxes", "pred_masks"],
        )?;
        let [logits, boxes, masks] = outputs.as_slice() else {
            bail!("RF-DETR Core ML model returned {} outputs", outputs.len());
        };
        self.postprocess(
            logits,
            boxes,
            masks,
            image.width(),
            image.height(),
            thresholds,
        )
    }

    fn preprocess(&self, image: &DynamicImage) -> Result<Tensor> {
        ensure!(
            image.width() > 0 && image.height() > 0,
            "cannot segment an empty image"
        );
        let image = image.to_rgb8();
        let (width, height) = (image.width() as usize, image.height() as usize);
        let pixels = image.as_raw();
        let size = self.resolution;
        let columns = bilinear_axis(width, size);
        let rows = bilinear_axis(height, size);
        let mut data = vec![0.0; 3 * size * size];
        for channel in 0..3 {
            let sample =
                |x: usize, y: usize| f32::from(pixels[(y * width + x) * 3 + channel]) / 255.0;
            let plane = &mut data[channel * size * size..(channel + 1) * size * size];
            for (y, row) in rows.iter().enumerate() {
                for (x, column) in columns.iter().enumerate() {
                    let value = row.weight[0]
                        * (column.weight[0] * sample(column.index[0], row.index[0])
                            + column.weight[1] * sample(column.index[1], row.index[0]))
                        + row.weight[1]
                            * (column.weight[0] * sample(column.index[0], row.index[1])
                                + column.weight[1] * sample(column.index[1], row.index[1]));
                    plane[y * size + x] = (value - MEAN[channel]) / STD[channel];
                }
            }
        }
        Ok(Tensor {
            shape: vec![1, 3, size, size],
            data,
        })
    }

    fn postprocess(
        &self,
        logits: &Array,
        boxes: &Array,
        masks: &Array,
        image_width: u32,
        image_height: u32,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        ensure!(
            logits.shape() == [1, QUERIES, CLASSES],
            "unexpected RF-DETR logits shape {:?}",
            logits.shape()
        );
        ensure!(
            boxes.shape() == [1, QUERIES, 4],
            "unexpected RF-DETR box shape {:?}",
            boxes.shape()
        );
        ensure!(
            masks.shape() == [1, QUERIES, MASK_SIZE, MASK_SIZE],
            "unexpected RF-DETR mask shape {:?}",
            masks.shape()
        );

        // PostProcess ranks every query/class pair, including the fifth background slot,
        // before applying the caller threshold.
        let scores = logits
            .read(&[])?
            .into_iter()
            .map(|logit| 1.0 / (1.0 + (-logit).exp()))
            .collect::<Vec<_>>();
        let mut ranked = (0..scores.len()).collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            scores[*right]
                .total_cmp(&scores[*left])
                .then(left.cmp(right))
        });
        ranked.truncate(self.num_select);

        let boxes = boxes.read(&[])?;
        let class_thresholds = thresholds.with_background();
        let (width, height) = (image_width as usize, image_height as usize);
        let columns = bilinear_axis(MASK_SIZE, width);
        let rows = bilinear_axis(MASK_SIZE, height);
        let mut detections = Vec::new();
        for index in ranked {
            let score = scores[index];
            let label = index % CLASSES;
            let query = index / CLASSES;
            if score <= class_thresholds[label] {
                continue;
            }
            let [center_x, center_y, box_width, box_height] = boxes[query * 4..query * 4 + 4]
                .try_into()
                .context("RF-DETR box is incomplete")?;
            let bbox = [
                (center_x - box_width / 2.0) * image_width as f32,
                (center_y - box_height / 2.0) * image_height as f32,
                (center_x + box_width / 2.0) * image_width as f32,
                (center_y + box_height / 2.0) * image_height as f32,
            ];
            let mask = project_mask(&masks.read(&[0, query])?, &columns, &rows);
            detections.push(KoharuLayoutDetection {
                label_id: label,
                label: self
                    .class_names
                    .get(label)
                    .cloned()
                    .unwrap_or_else(|| "__background__".to_owned()),
                score,
                bbox,
                area: mask.area,
                mask: mask.mask,
            });
        }

        Ok(KoharuLayoutDetections {
            image_width,
            image_height,
            detections,
        })
    }
}

/// The two source samples and weights PyTorch's `upsample_bilinear2d` reads for each output
/// index, with `align_corners=False` and no explicit scale.
struct Taps {
    index: [usize; 2],
    weight: [f32; 2],
}

fn bilinear_axis(input: usize, output: usize) -> Vec<Taps> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|destination| {
            let source = (scale * (destination as f32 + 0.5) - 0.5).max(0.0);
            let lower = source as usize;
            let upper = lower + usize::from(lower < input - 1);
            let fraction = source - lower as f32;
            Taps {
                index: [lower, upper],
                weight: [1.0 - fraction, fraction],
            }
        })
        .collect()
}

struct ProjectedMask {
    area: u32,
    mask: KoharuLayoutMask,
}

/// Projects one native mask onto the page and keeps the non-zero extent, matching the Torch
/// path's bilinear resize thresholded at zero.
fn project_mask(source: &[f32], columns: &[Taps], rows: &[Taps]) -> ProjectedMask {
    let value = |x: usize, y: usize| source[y * MASK_SIZE + x];
    // A page pixel can be positive only if one of its source taps is, so only output rows and
    // columns whose taps reach a positive source sample are evaluated.
    let positive_rows = (0..MASK_SIZE)
        .map(|y| (0..MASK_SIZE).any(|x| value(x, y) > 0.0))
        .collect::<Vec<_>>();
    let positive_columns = (0..MASK_SIZE)
        .map(|x| (0..MASK_SIZE).any(|y| value(x, y) > 0.0))
        .collect::<Vec<_>>();
    let reaches = |taps: &Taps, positive: &[bool]| taps.index.iter().any(|index| positive[*index]);
    let candidate_rows = rows
        .iter()
        .enumerate()
        .filter(|(_, taps)| reaches(taps, &positive_rows))
        .map(|(y, _)| y)
        .collect::<Vec<_>>();
    let candidate_columns = columns
        .iter()
        .enumerate()
        .filter(|(_, taps)| reaches(taps, &positive_columns))
        .map(|(x, _)| x)
        .collect::<Vec<_>>();

    let mut inside = Vec::new();
    for &y in &candidate_rows {
        let row = &rows[y];
        for &x in &candidate_columns {
            let column = &columns[x];
            let sample = row.weight[0]
                * (column.weight[0] * value(column.index[0], row.index[0])
                    + column.weight[1] * value(column.index[1], row.index[0]))
                + row.weight[1]
                    * (column.weight[0] * value(column.index[0], row.index[1])
                        + column.weight[1] * value(column.index[1], row.index[1]));
            if sample > 0.0 {
                inside.push((x, y));
            }
        }
    }

    let empty = ProjectedMask {
        area: 0,
        mask: KoharuLayoutMask {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            pixels: Vec::new(),
        },
    };
    let Some(left) = inside.iter().map(|(x, _)| *x).min() else {
        return empty;
    };
    let right = inside.iter().map(|(x, _)| *x).max().unwrap_or(left) + 1;
    let top = inside.iter().map(|(_, y)| *y).min().unwrap_or(0);
    let bottom = inside.iter().map(|(_, y)| *y).max().unwrap_or(top) + 1;
    let (mask_width, mask_height) = (right - left, bottom - top);
    let mut pixels = vec![0; mask_width * mask_height];
    for (x, y) in &inside {
        pixels[(y - top) * mask_width + (x - left)] = 255;
    }
    ProjectedMask {
        area: inside.len() as u32,
        mask: KoharuLayoutMask {
            x: left as u32,
            y: top as u32,
            width: mask_width as u32,
            height: mask_height as u32,
            pixels,
        },
    }
}
