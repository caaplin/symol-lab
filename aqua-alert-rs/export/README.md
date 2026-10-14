# Wiring the deep ONNX model (optional `onnx` feature)

The Rust crate ships with the **heuristic** colour-statistics triage built in, so it runs
offline with zero ML dependencies. If you want the real deep model, the crate has an
`onnx` backend that uses the same **public** model datasets listed in the catalogue:

- `panda992/fish_disease_datasets` (Hugging Face)
- `allfahad/fish-disease-v1-original-20260608` (Hugging Face)
- Roboflow fish-disease datasets (e.g. `fish-disease-detection-3dqdg`)

Convert one of those to ONNX once, drop the two files next to the binary (or anywhere),
and point the binary at them. No model, tokenizer or Python is needed at runtime.

## 1. Export to ONNX

```bash
pip install "optimum[onnxruntime]" transformers torch onnx
python export/to_onnx.py panda992/fish_disease_datasets out/
```

This writes:

```
out/model.onnx      # the ViT image-classifier
out/labels.json     # index -> class label, exactly the order ONNX logits come back in
```

`to_onnx.py` is a thin wrapper over `optimum`'s exporter that also dumps `labels.json`
from the model's `id2label` map, so the Rust side never has to know the tokenizer.

## 2. Run the binary with the ONNX backend

```bash
cargo build --release --features onnx
export AQUA_ONNX_MODEL=$PWD/out/model.onnx
export AQUA_ONNX_LABELS=$PWD/out/labels.json
./target/release/aqua-alert triage assets/example_lesion_fish.jpg --species tilapia
```

The binary prefers, in order: `--model` flag → `AQUA_ONNX_MODEL` → `models/fish_disease.onnx`
→ the built-in heuristic. The `backend` field of the result tells you which one ran
(`onnx:<path>` vs `heuristic-fallback`), mirroring the Python service.

## Configuration (env)

| Var | Default | Meaning |
|-----|---------|---------|
| `AQUA_ONNX_MODEL` | `models/fish_disease.onnx` | path to `model.onnx` |
| `AQUA_ONNX_LABELS` | sibling `labels.json` | path to `labels.json` |
| `AQUA_ONNX_SIZE` | `224` | ViT square input size |
| `AQUA_ONNX_MEAN0/1/2` | `0.5` | ImageNet normalisation mean (per channel) |
| `AQUA_ONNX_STD0/1/2` | `0.5` | ImageNet normalisation std (per channel) |

## Note on the `ort` API

The backend targets **`ort` 2.0.0-rc.10**. That crate's tensor/IO API is still stabilising,
so if you build against a different rc you may need a one-line tweak in
`src/disease/classifier.rs` (`onnx_backend::predict_img`): the calls are
`session.run(ort::inputs![input.view()]?)` and `outputs[0].try_extract_tensor::<f32>()`.
The feature is off by default, so it never affects the single-binary heuristic build.
