# Evaluation Guide (`evaluation.py`)

This README explains how to run `evaluation.py` and how it relies on `The_model.py`.

## What `evaluation.py` does
`evaluation.py` runs a full evaluation pipeline for selected dataset rows:
- Loads source image, comparison image (pipline 2 output), and `llm_edit` prompt.
- Builds a generated image using the project pipeline.
- comparison features in DINOv2 space.
- Prints dictionary metrics .

## `The_model.py` flow 
The evaluation script calls functions from `The_model.py` using the optimization pipeline.

* At the beginning of the project, we built a pipeline that blended the original image with the closest image from the prompt using Gaussian blending. The results were not satisfactory, which showed us that a more complex blending approach was needed. The method remains in the model but is not used for the final output.


Our model flow: 

1. Load one CSV row (`src`, `comp`, `llm_edit`).
2. Retrieve a target image by semantic search - CLIP (`search_llm_edit_image`)
   * Given a prompt, the function(CLIP) retrieves the closest image to the prompt using cosine similarity.
3. Encode source and target images into TAESD latents (`encode_image`).
4. Load frozen DINOv2 for semantic embeddings.
5. Optimize a latent midpoint (`optimize_semantic_midpoint`) so decoded image embedding matches the DINO midpoint target.
   * Too complex to explain here :-)
6. Decode optimized latent to image.
7. Refine with Stable Diffusion img2img (`make_image_coherent`).
      Given a prompt and the opimize image, the StableDiffusionImg2ImgPipeline function generates an image

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


# Setup

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

# Run evaluation
```bash
python evaluation.py
```

## Configure before running
Edit these values inside `evaluation.py`(The end of the file):
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
- `source/comparison URLs`
- `prompt`

At the end, it prints average metrics across all selected indices(if puts list of indices).

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
