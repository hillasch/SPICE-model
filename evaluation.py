# Setup and Imports
import torch
import torchvision.transforms.functional as TF
from pathlib import Path
import pandas as pd
import The_model as model
from pull_images import load_pair_from_csv
from dino import get_frozen_model


def run_eval(image_index: int, steps: int = 30, num_steps: int = 50):

    """
    Run a single evaluation for one dataset row (image_index).

    Flow:
    1) Load (src, comp, prompt) from CSV.
    2) Retrieve a "target image" using semantic search over llm_edit prompt.
    3) Encode source + target; optimize a semantic midpoint latent using DINO embeddings.
    4) Refine decoded midpoint with Stable Diffusion img2img ("make_image_coherent").
    5) Compute similarity metrics (cosine + L2) between:
       - our generated image vs source
       - comp (baseline comparison image) vs source
    6) Return a dict with metrics + URLs for logging/analysis.
    """
    # --- keep your defaults exactly as in the original cell ---
    # these are hyperparameters for the latent optimization + SD refinement stages
    lr = 0.01
    grad_clip = 1.0
    strength = 0.6
    guidance_scale = 7.5
    seed = 42

    dev = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # --- original flow (same order) ---

    # Load one row from the dataset CSV (src image, baseline comp image, and llm_edit prompt)
    src_img, comp_img, prompt, index = load_pair_from_csv(Path(model.DEFAULT_CSV), row_index=image_index)

    #  Retrieve the "closest" image to the prompt using semantic search function
    search_hit = model.search_llm_edit_image(prompt, k=1, show=False)
    target_image = search_hit["image"]
    print(f"_________llm_edit prompt:________ {prompt}")

    # Encode source and target images into tensors + latents
    # image_a_tensor/target_tensor are image tensors; latent_a/latent_b are latent-space representations.
    image_a_tensor, latent_a = model.encode_image(src_img)
    target_tensor, latent_b = model.encode_image(target_image)

    # DINOv2 setup model for semantic embeddings
    dino_model, dino_processor = get_frozen_model(device=dev)
    dino_model.eval()
    for p in dino_model.parameters():
        p.requires_grad_(False)

    # Optimize semantic midpoint latent
    decoded, losses = model.optimize_semantic_midpoint(
        image_a=image_a_tensor,
        image_b=target_tensor,
        latent_a=latent_a,
        latent_b=latent_b,
        dino_model=dino_model,
        dino_processor=dino_processor,
        steps=steps,
        lr=lr,
        grad_clip=grad_clip,
    )

    # Make result more coherent with SD img2img refinement
    prompt_for_sd = prompt
    sd_input = TF.to_pil_image(decoded[0].cpu())
    sd_image = model.make_image_coherent(
        image=sd_input,
        prompt=prompt_for_sd,
        strength=strength,
        guidance_scale=guidance_scale,
        num_steps=num_steps,
        seed=seed,
        device=str(dev),
    )

    # Metrics: OUR generated image vs SOURCE (semantic similarity)

    our_tensor, _ = model.encode_image(sd_image)
    first_image, _ = model.encode_image(src_img)

    # Embed with DINO
    with torch.no_grad():
        our_image = model.embed_tensor_with_dino(our_tensor, dino_model, dino_processor)
        src_image = model.embed_tensor_with_dino(first_image, dino_model, dino_processor)

    # Cosine similarity: higher means closer semantically
    cosine_sim_our_src = torch.nn.functional.cosine_similarity(
        our_image.unsqueeze(0), src_image.unsqueeze(0), dim=1
    )
    print(f"Cosine similarity between generated image and source image: {cosine_sim_our_src.item():.4f}")

    norm_l2_our_src = torch.nn.functional.pairwise_distance(
        our_image.unsqueeze(0), src_image.unsqueeze(0), p=2
    )
    print(f"L2 distance between generated image and source image: {norm_l2_our_src.item():.4f}")

    # Metrics: COMP (baseline) image vs SOURCE

    comp_tensor, _ = model.encode_image(comp_img)
    first_image, _ = model.encode_image(src_img)

    # Embed with DINO
    with torch.no_grad():
        comp_image = model.embed_tensor_with_dino(comp_tensor, dino_model, dino_processor)
        src_image = model.embed_tensor_with_dino(first_image, dino_model, dino_processor)

    cosine_sim_comp_src = torch.nn.functional.cosine_similarity(
        comp_image.unsqueeze(0), src_image.unsqueeze(0), dim=1
    )
    print(f"Cosine similarity between comparison comp image and source image: {cosine_sim_comp_src.item():.4f}")

    norm_l2_comp_src = torch.nn.functional.pairwise_distance(
        comp_image.unsqueeze(0), src_image.unsqueeze(0), p=2
    )
    print(f"L2 distance between comparison comp image and source image: {norm_l2_comp_src.item():.4f}")
    # the sorce image
    src_img.show()
    # the image from the semantic search hit (target for optimization)
    target_image.show()
    # the image from our optimization + SD refinement
    TF.to_pil_image(decoded[0].cpu()).show()
    # the final refined image from Stable Diffusion
    sd_image.show()
    # the baseline comparison image from the dataset
    comp_img.show()


    df = pd.read_csv("dataset_cap_edit_only.csv")

    return {
        "url_comp": df["comp_url"][image_index],
        "url_src": df["src_url"][image_index],
        "image_index evaluated": image_index,
        "prompt": prompt,
        "cosine_our_src": cosine_sim_our_src.item(),
        "l2_our_src": norm_l2_our_src.item(),
        "cosine_comp_src": cosine_sim_comp_src.item(),
        "l2_comp_src": norm_l2_comp_src.item(),

    }


# Lists to aggregate metrics across multiple evaluated indices
list_our_cosinos = []
list_comp_cosinos = []
list_our_cosinos_l2 = []
list_comp_cosinos_l2 = []


############################## Example evaluation run for one image index ##############################
index = 2000
num_steps = 450
steps = 500
ine_dict = run_eval(image_index=index, steps=steps, num_steps=num_steps)
print(ine_dict)
