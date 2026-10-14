#!/usr/bin/env python3
"""Export a public fish-disease image classifier (Hugging Face) to ONNX + labels.json.

The Rust `onnx` backend reads exactly these two files:

    model.onnx    ViT/ResNet image classifier, input NCHW float32, output logits
    labels.json   {index: "Class name"} in the ONNX logit order

Usage:
    python export/to_onnx.py <hf-model-id> <out-dir>

Example:
    python export/to_onnx.py panda992/fish_disease_datasets out/

Requires: pip install "optimum[onnxruntime]" transformers torch onnx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model_id = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {model_id} ...")
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()

    # labels.json straight from the model's id2label map, so the order matches ONNX logits
    id2label = getattr(model.config, "id2label", None) or {
        i: l for i, l in enumerate(getattr(model.config, "label_list", []))
    }
    labels = {str(int(k)): str(v) for k, v in id2label.items()}
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"wrote {out_dir/'labels.json'} ({len(labels)} classes)")

    # Prefer optimum's exporter (handles the ViT wrapper + dynamic axes for us).
    size = int(getattr(proc, "size", {}).get("height", 224) or 224)
    mean = list(getattr(proc, "image_mean", [0.5, 0.5, 0.5]))
    std = list(getattr(proc, "image_std", [0.5, 0.5, 0.5]))
    print(f"input size={size} mean={mean} std={std}")

    onnx_path = out_dir / "model.onnx"
    try:
        from optimum.exporters.onnx import main_export

        main_export(model_id, output=out_dir, task="image-classification", opset=17)
        # optimum names the file model.onnx by default
        if not onnx_path.exists():
            for cand in out_dir.glob("*.onnx"):
                cand.rename(onnx_path)
                break
    except Exception as exc:  # fall back to a manual torch.onnx.export
        print(f"optimum export failed ({exc}); falling back to torch.onnx.export")
        dummy = torch.zeros(1, 3, size, size, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

    print(f"wrote {onnx_path}")
    print(
        "\nRun the Rust binary with:\n"
        f"  export AQUA_ONNX_MODEL={onnx_path.resolve()}\n"
        f"  export AQUA_ONNX_LABELS={(out_dir/'labels.json').resolve()}\n"
        f"  export AQUA_ONNX_SIZE={size}\n"
        "  aqua-alert --features onnx triage <photo> --species <spp>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
