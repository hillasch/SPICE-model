# `evaluation.py` Guide

This README is focused only on the evaluation flow in `evaluation.py`.

## What this file does
`evaluation.py` runs a full evaluation pipeline for selected dataset rows:
- Loads source image, comparison image, and `llm_edit` prompt.
- Builds a generated image using the project pipeline.
- Compares features in DINOv2 space.
- Prints per-sample metrics and final averages.

## Setup
Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision diffusers transformers sentence-transformers pandas pillow accelerate safetensors
```

If TAESD files are missing, download them:

```bash
curl -L -o taesd.py https://raw.githubusercontent.com/madebyollin/taesd/main/taesd.py
curl -L -o taesd_encoder.pth https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_encoder.pth
curl -L -o taesd_decoder.pth https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_decoder.pth
```

## Run evaluation
```bash
python evaluation.py
```

## Configure before running
Edit these values inside `evaluation.py`:
- `indexes`: dataset row indices to evaluate.
- `steps`: latent optimization steps.
- `num_steps`: diffusion refinement steps.

Example:

```python
indexes = [2000, 2001]
ine_dict = run_eval(image_index=idx, steps=450, num_steps=500)
```

## Output
For each run, the script prints metrics and returns a dictionary with:
- `cosine_our_src`
- `l2_our_src`
- `cosine_comp_src`
- `l2_comp_src`
- prompt and source/comparison URLs

At the end, it prints average metrics across all selected indices.

The script also opens 5 preview images:
- source image
- prompt search target image
- decoded midpoint image
- coherent (SD-refined) image
- comparison image

## Metric meaning
| Metric | Meaning | Better direction |
|---|---|---|
| `cosine_our_src` | Similarity between generated image and source image | Higher |
| `cosine_comp_src` | Similarity between baseline comparison image and source image | Higher |
| `l2_our_src` | Distance between generated image and source image | Lower |
| `l2_comp_src` | Distance between baseline comparison image and source image | Lower |

## Notes
- First run may download model assets (DINOv2, Stable Diffusion, search embeddings).
- GPU or Apple Silicon is recommended for reasonable runtime.
