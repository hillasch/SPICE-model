# Evaluation Guide (`evaluation.py`)

This README explains how to run `evaluation.py` and how it relies on `The_model.py` (optimization path only, without Gaussian blending).

## What `evaluation.py` does
`evaluation.py` runs a full evaluation pipeline for selected dataset rows:
- Loads source image, comparison image, and `llm_edit` prompt.
- Builds a generated image using the project pipeline.
- Compares features in DINOv2 space.
- Prints per-sample metrics and final averages.

## `The_model.py` flow 
The evaluation script calls functions from `The_model.py` using the optimization pipeline (`opt`).

Flow:
1. Load one CSV row (`src`, `comp`, `llm_edit`).
2. Retrieve a target image by semantic search - CLIP (`search_llm_edit_image`).
3. Encode source and target images into TAESD latents (`encode_image`).
4. Load frozen DINOv2 for semantic embeddings.
5. Optimize a latent midpoint (`optimize_semantic_midpoint`) so decoded image embedding matches the DINO midpoint target.
6. Decode optimized latent to image.
7. Refine with Stable Diffusion img2img (`make_image_coherent`).

## Main hyperparameters in `The_model.py`
Optimization:
- `--steps`  number of latent optimization steps.
- `--lr`  learning rate for Adam on latent parameters.
- `--grad-clip`  gradient clipping norm.

Semantic retrieval:
- `--search-top-k` (default: `1`) number of text-to-image search candidates.

Coherence refinement (SD img2img):
- `--cohere-prompt` (default: uses `llm_edit` prompt).
- `--cohere-strength` (default: `0.6`) how strongly SD changes the input image.
- `--cohere-guidance` (default: `7.5`) prompt guidance scale.
- `--cohere-steps` (default: `50`) diffusion steps.
- `--cohere-seed` (default: `42`) random seed for reproducibility.


## Setup

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
Edit these values inside `evaluation.py`(line - 155):
- `indexes`: dataset row indices to evaluate.
- `steps`: latent optimization steps.
- `num_steps`: diffusion refinement steps.

Example:

```python
index = 2000
num_steps = 450
steps = 500
ine_dict = run_eval(image_index=index, steps=steps, num_steps=num_steps)
```

## Output
For each run, the script prints metrics and returns a dictionary with:
- `cosine_our_src`
- `l2_our_src`
- `cosine_comp_src`
- `l2_comp_src`
- prompt and source/comparison URLs

At the end, it prints average metrics across all selected indices(if puts more than 1 picture).

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
