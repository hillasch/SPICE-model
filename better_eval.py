import torch
import torchvision.transforms.functional as TF
from pathlib import Path
import pandas as pd
import model
from pull_images import load_pair_from_csv
from dino import get_frozen_model


def run_eval(image_index: int, steps: int = 30, num_steps: int = 50):
    # --- keep your defaults exactly as in the original cell ---
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
    src_img, comp_img, prompt, index = load_pair_from_csv(Path(model.DEFAULT_CSV), row_index=image_index)
    search_hit = model.search_llm_edit_image(prompt, k=1, show=False)
    target_image = search_hit["image"]
    print(f"_________llm_edit prompt:________ {prompt}")
   
    image_a_tensor, latent_a = model.encode_image(src_img)
    target_tensor, latent_b = model.encode_image(target_image)

    # dinov2 model for semantic embeddings
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
    # matrics for our image
    our_tensor, _ = model.encode_image(sd_image)
    first_image, _ = model.encode_image(src_img)

    with torch.no_grad():
        our_image = model.embed_tensor_with_dino(our_tensor, dino_model, dino_processor)
        src_image = model.embed_tensor_with_dino(first_image, dino_model, dino_processor)

    cosine_sim_our_src = torch.nn.functional.cosine_similarity(
        our_image.unsqueeze(0), src_image.unsqueeze(0), dim=1
    )
    print(f"Cosine similarity between generated image and source image: {cosine_sim_our_src.item():.4f}")

    norm_l2_our_src = torch.nn.functional.pairwise_distance(
        our_image.unsqueeze(0), src_image.unsqueeze(0), p=2
    )
    print(f"L2 distance between generated image and source image: {norm_l2_our_src.item():.4f}")

    # matrics for our image
    comp_tensor, _ = model.encode_image(comp_img)
    first_image, _ = model.encode_image(src_img)

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

    #src_img.show()

    sd_image.show()
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



list_our_cosinos = []
list_comp_cosinos = []
list_our_cosinos_l2 = []
list_comp_cosinos_l2 = []
indexes = [2000]
counter = 0
for idx in indexes:
    counter += 1
    print(f"_________Evaluating image index {idx} ({counter}/{len(indexes)})_________")
    ine_dict = run_eval(image_index=idx, steps=300, num_steps=300)
    list_our_cosinos.append(ine_dict["cosine_our_src"])
    list_comp_cosinos.append(ine_dict["cosine_comp_src"])
    list_our_cosinos_l2.append(ine_dict["l2_our_src"])
    list_comp_cosinos_l2.append(ine_dict["l2_comp_src"])
print(f"Average cosine similarity between generated images and source images: {sum(list_our_cosinos) / len(list_our_cosinos):.4f}")
print(f"Average cosine similarity between comparison comp images and source images: {sum(list_comp_cosinos) / len(list_comp_cosinos):.4f}") 
print(f"Average L2 distance between generated images and source images: {sum(list_our_cosinos_l2) / len(list_our_cosinos_l2):.4f}")
print(f"Average L2 distance between comparison comp images and source images: {sum(list_comp_cosinos_l2) / len(list_comp_cosinos_l2):.4f}")