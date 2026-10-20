"""Convert KoharuLayout RF-DETR Seg 2XL to Core ML and verify it against PyTorch.

`convert` (any OS) loads the pinned checkpoint strictly, exports rfdetr's ONNX-style forward as
an iOS 18 ML Program with the names Koharu's iOS loader reads, and records PyTorch's regular
eval forward on a real page. On macOS, `predict.swift` runs the compiled model on the same input.
`verify` compares those Core ML outputs with PyTorch's, raw and after Koharu's postprocessing,
then fails unless they agree.

    python convert_koharu_layout.py convert --image page.jpg --out build
    xcrun coremlcompiler compile build/koharu-layout-rfdetr-seg-2xl-1152.mlpackage build/compiled
    ./predict build/compiled/koharu-layout-rfdetr-seg-2xl-1152.mlmodelc build/pixel_values.f32 build/core-ml
    python convert_koharu_layout.py verify --out build
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY = "mayocream/koharu-layout-rfdetr-seg-2xl-1152"
REVISION = "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2"
NAME = "koharu-layout-rfdetr-seg-2xl-1152"
RESOLUTION = 1152
NUM_SELECT = 160
CLASSES = ["text", "onomatopoeia", "bubble", "panel"]
# The checkpoint's recommended thresholds, then the background slot.
THRESHOLDS = [0.25, 0.2, 0.5, 0.5, math.inf]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess(image: Image.Image) -> torch.Tensor:
    """Koharu's preprocessing: bilinear resize without antialiasing, then ImageNet norm."""
    import torch
    import torch.nn.functional as F

    pixels = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1)[None].float() / 255
    pixels = F.interpolate(
        pixels, size=(RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False, antialias=False
    )
    return (pixels - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


def load_network() -> torch.nn.Module:
    """The checkpoint's own strict loader (load_model.py in the model repository)."""
    from huggingface_hub import hf_hub_download
    from rfdetr import RFDETRSeg2XLarge
    from rfdetr.config import PretrainWeightsCompatibilityWarning
    from safetensors.torch import load_file

    weights = hf_hub_download(REPOSITORY, "model.safetensors", revision=REVISION)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PretrainWeightsCompatibilityWarning)
        model = RFDETRSeg2XLarge(
            pretrain_weights=None,
            resolution=RESOLUTION,
            num_select=NUM_SELECT,
            num_classes=len(CLASSES),
        )
    network = model.model.model
    incompatible = network.load_state_dict(load_file(weights, device="cpu"), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SystemExit(f"incompatible weights: {incompatible}")
    return network.eval()


def broadcast_meshgrid(*tensors, indexing="ij"):
    if indexing != "ij" or len(tensors) != 2:
        raise ValueError("only two-dimensional ij grids are supported")
    rows, columns = (tensor.reshape(-1) for tensor in tensors)
    shape = (rows.numel(), columns.numel())
    return rows[:, None].expand(shape), columns[None, :].expand(shape)


def single_image_deformable_attention(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None,
    input_spatial_shapes_hw=None,
):
    """rfdetr's MSDeformAttn.forward and ms_deform_attn_core_pytorch for a batch of one.

    Core ML tensors have at most five dimensions, and upstream's sampling offsets and locations
    are six-dimensional (batch, query, head, level, point, xy). The converted model always runs
    one image, so this drops the batch dimension and otherwise follows upstream step by step.
    """
    import torch
    import torch.nn.functional as F
    from rfdetr.utilities.tensors import _bilinear_grid_sample

    batch_size, len_query, _ = query.shape
    _, len_input, _ = input_flatten.shape
    if batch_size != 1:
        raise ValueError("the Core ML export supports a batch of one image")
    head_dim = self.d_model // self.n_heads

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    # query, head, level, point, xy
    offsets = self.sampling_offsets(query).view(len_query, self.n_heads, self.n_levels, self.n_points, 2)
    weights = self.attention_weights(query).view(len_query, self.n_heads, self.n_levels * self.n_points)
    reference = reference_points[0]
    if reference.shape[-1] == 2:
        normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        locations = reference[:, None, :, None, :] + offsets / normalizer[None, None, :, None, :]
    elif reference.shape[-1] == 4:
        locations = (
            reference[:, None, :, None, :2]
            + offsets / self.n_points * reference[:, None, :, None, 2:] * 0.5
        )
    else:
        raise ValueError(f"last dim of reference_points must be 2 or 4, got {reference.shape[-1]}")
    weights = F.softmax(weights, -1)

    value = value[0].transpose(0, 1).contiguous().view(self.n_heads, head_dim, len_input)
    shapes = input_spatial_shapes_hw if input_spatial_shapes_hw is not None else input_spatial_shapes
    values = value.split([height * width for height, width in shapes], dim=2)
    grids = 2 * locations - 1
    samples = []
    for level, (height, width) in enumerate(shapes):
        level_value = values[level].view(self.n_heads, head_dim, height, width)
        # head, query, point, xy
        level_grid = grids[:, :, level].transpose(0, 1)
        # head, head_dim, query, point
        samples.append(
            _bilinear_grid_sample(level_value, level_grid, padding_mode="zeros", align_corners=False)
        )
    weights = weights.transpose(0, 1).reshape(self.n_heads, 1, len_query, self.n_levels * self.n_points)
    sampled = torch.stack(samples, dim=-2).flatten(-2)
    output = (sampled * weights).sum(-1).view(1, self.n_heads * head_dim, len_query)
    return self.output_proj(output.transpose(1, 2).contiguous())


def convert(image: Path, out: Path) -> None:
    import coremltools as ct
    import torch
    import torch.nn.functional as F

    class Exported(torch.nn.Module):
        """rfdetr's export forward, reordered to Koharu's (logits, boxes, masks) outputs."""

        def __init__(self, network: torch.nn.Module):
            super().__init__()
            self.network = network

        def forward(self, pixel_values: torch.Tensor):
            boxes, logits, masks = self.network(pixel_values)
            return logits, boxes, masks

    out.mkdir(parents=True, exist_ok=True)
    pixel_values = preprocess(Image.open(image))
    network = load_network()
    with torch.no_grad():
        reference = network(pixel_values)
        network.export()
        # coremltools 9 rejects the linspace grids that rfdetr's encoder proposals pass to
        # torch.meshgrid. Broadcasting builds the same "ij" grid, and the export check below
        # compares the result against the unmodified eval forward.
        torch.meshgrid = broadcast_meshgrid
        # The segmentation head wraps its depthwise convolution in an autograd Function that only
        # keeps cuDNN out of training; tracing records it as an opaque Python op. Its forward is
        # a plain grouped convolution.
        from rfdetr.models.heads import segmentation

        segmentation._DepthwiseConvWithoutCuDNN.apply = staticmethod(F.conv2d)
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

        MSDeformAttn.forward = single_image_deformable_attention
        exported = Exported(network).eval()
        logits, boxes, masks = exported(pixel_values)
        traced = torch.jit.trace(exported, pixel_values)

    # The export path must match the eval forward that Koharu's Torch port reproduces.
    export_gap = {
        name: float((exported_value - reference[name]).abs().max())
        for name, exported_value in [
            ("pred_logits", logits),
            ("pred_boxes", boxes),
            ("pred_masks", masks),
        ]
    }
    print(json.dumps({"export_vs_forward_max_abs": export_gap}, indent=2))

    package = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=tuple(pixel_values.shape))],
        outputs=[
            ct.TensorType(name="pred_logits"),
            ct.TensorType(name="pred_boxes"),
            ct.TensorType(name="pred_masks"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        # Float16 compute reorders the two-stage proposal selection: on a real page, scores drift
        # by up to 0.17 against PyTorch and extra detections cross the thresholds.
        compute_precision=ct.precision.FLOAT32,
    )
    package.short_description = f"{REPOSITORY}@{REVISION}"
    package.save(str(out / f"{NAME}.mlpackage"))
    (out / "pixel_values.f32").write_bytes(pixel_values.numpy().astype("<f4").tobytes())
    np.savez_compressed(
        out / "reference.npz",
        pixel_values=pixel_values.numpy(),
        image_size=np.array(Image.open(image).size),
        **{name: reference[name].numpy() for name in ["pred_logits", "pred_boxes", "pred_masks"]},
    )


def resize_bilinear(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """PyTorch's bilinear interpolation with align_corners=False, without antialiasing.

    Written in NumPy so verification does not need PyTorch.
    """

    def axis(size: int, target: int):
        # PyTorch computes the scale and source coordinates in float32.
        scale = np.float32(size / target)
        source = np.maximum((np.arange(target, dtype=np.float32) + 0.5) * scale - 0.5, 0)
        low = source.astype(np.int64)
        high = np.minimum(low + 1, size - 1)
        weight = source - low.astype(np.float32)
        return low, high, 1 - weight, weight

    top, bottom, above, below = axis(values.shape[0], height)
    left, right, before, after = axis(values.shape[1], width)
    # Blend along x first, then y, in PyTorch's order.
    return above[:, None] * (
        before * values[top][:, left] + after * values[top][:, right]
    ) + below[:, None] * (before * values[bottom][:, left] + after * values[bottom][:, right])


def postprocess(logits, boxes, masks, width: int, height: int) -> list[dict]:
    """Koharu's RF-DETR postprocessing (processor.rs and coreml.rs)."""
    scores = 1 / (1 + np.exp(-logits[0].astype(np.float64)))
    flat = scores.reshape(-1)
    ranked = np.argsort(-flat, kind="stable")[:NUM_SELECT]
    detections = []
    for index in ranked:
        label, query = int(index % 5), int(index // 5)
        score = float(flat[index])
        if not score > THRESHOLDS[label]:
            continue
        cx, cy, box_width, box_height = boxes[0, query].astype(np.float64)
        mask = resize_bilinear(masks[0, query].astype(np.float32), height, width) > 0
        detections.append(
            {
                "label": CLASSES[label],
                "score": score,
                "bbox": [
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ],
                "area": int(mask.sum()),
                "mask": mask,
            }
        )
    return detections


def box_iou(left, right) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (
        right[3] - right[1]
    ) - inter
    return inter / union if union > 0 else 0.0


def verify(out: Path) -> None:
    reference = np.load(out / "reference.npz")
    width, height = (int(value) for value in reference["image_size"])
    core_ml = out / "core-ml"
    shapes = json.loads((core_ml / "shapes.json").read_text())
    predicted = {
        name: np.fromfile(core_ml / f"{name}.f32", dtype="<f4").reshape(shape)
        for name, shape in shapes.items()
    }

    raw = {
        name: {
            "shape": list(predicted[name].shape),
            "max_abs": float(np.abs(predicted[name] - reference[name]).max()),
            "mean_abs": float(np.abs(predicted[name] - reference[name]).mean()),
        }
        for name in ["pred_logits", "pred_boxes", "pred_masks"]
    }
    expected = postprocess(
        reference["pred_logits"], reference["pred_boxes"], reference["pred_masks"], width, height
    )
    actual = postprocess(
        predicted["pred_logits"], predicted["pred_boxes"], predicted["pred_masks"], width, height
    )

    failures = []
    for name, shape in [
        ("pred_logits", [1, 300, 5]),
        ("pred_boxes", [1, 300, 4]),
        ("pred_masks", [1, 300, 288, 288]),
    ]:
        if raw[name]["shape"] != shape:
            failures.append(f"{name} has shape {raw[name]['shape']}, expected {shape}")

    matches = []
    unmatched = list(actual)
    for detection in expected:
        candidates = [
            other for other in unmatched if other["label"] == detection["label"]
        ]
        best = max(candidates, key=lambda other: box_iou(detection["bbox"], other["bbox"]), default=None)
        threshold = THRESHOLDS[CLASSES.index(detection["label"])]
        near_threshold = detection["score"] < threshold + 0.05
        if best is None or box_iou(detection["bbox"], best["bbox"]) < 0.9:
            if not near_threshold:
                failures.append(f"no Core ML match for {detection['label']} {detection['score']:.3f}")
            continue
        unmatched.remove(best)
        inter = np.logical_and(detection["mask"], best["mask"]).sum()
        union = np.logical_or(detection["mask"], best["mask"]).sum()
        match = {
            "label": detection["label"],
            "score": [round(detection["score"], 4), round(best["score"], 4)],
            "box_iou": round(box_iou(detection["bbox"], best["bbox"]), 4),
            "mask_iou": round(float(inter / union) if union else 1.0, 4),
            "area": [detection["area"], best["area"]],
        }
        matches.append(match)
        if abs(detection["score"] - best["score"]) > 0.05:
            failures.append(f"score drift {match}")
        if match["mask_iou"] < 0.9 and detection["area"] > 2000:
            failures.append(f"mask drift {match}")
    for extra in unmatched:
        threshold = THRESHOLDS[CLASSES.index(extra["label"])]
        if extra["score"] >= threshold + 0.05:
            failures.append(f"extra Core ML detection {extra['label']} {extra['score']:.3f}")

    report = {
        "raw": raw,
        "detections": {"torch": len(expected), "core_ml": len(actual), "matched": len(matches)},
        "matches": matches,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    (out / "verification.json").write_text(json.dumps(report, indent=2))
    if failures:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    convert_parser = commands.add_parser("convert")
    convert_parser.add_argument("--image", type=Path, required=True)
    convert_parser.add_argument("--out", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "convert":
        convert(args.image, args.out)
    else:
        verify(args.out)


if __name__ == "__main__":
    main()
