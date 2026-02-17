# we wnat to get some evaluation metrics fot our model ' in the origin 
# paper they said that the new image dont save the oroginal context fporm the sorce iamge '
# so we want to the the similarity between the source image and the new image
# we can use the dino model to get the features of the images and then calculate the
# cosine similarity between the features of the source image and the new image
import torch
from model import main
from pull_images import load_pair_from_csv
from pathlib import Path
from model import DEFAULT_CSV
import model
from dino import get_frozen_model
import torchvision.transforms.functional as TF
steps = 30
lr = 0.01
grad_clip = 1.0
strength= 0.6
guidance_scale=7.5
num_steps=50
seed=42

dev = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
for i in range(1):
    src_img, comp_img, prompt, index = load_pair_from_csv(Path(DEFAULT_CSV), row_index=i)
    search_hit = model.search_llm_edit_image(prompt, k=1, show=False)
    target_image = search_hit["image"]
    print(f"llm_edit prompt: {prompt}")
    image_a_tensor, latent_a = model.encode_image(src_img)
    target_tensor, latent_b = model.encode_image(target_image)
    # dinov2 model for semantic embeddings
    dino_model, dino_processor = get_frozen_model(device=dev)
    dino_model.eval()
    for p in dino_model.parameters():
        p.requires_grad_(False)
    # Optimize semantic midpoint latent
    # we get the decoded image and the loss history
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

    
our_tensor, _ = model.encode_image(sd_image)
first_image, _ = model.encode_image(src_img)

with torch.no_grad():
    our_image = model.embed_tensor_with_dino(our_tensor, dino_model, dino_processor)
    there_image = model.embed_tensor_with_dino(first_image, dino_model, dino_processor)
cosine_sim = torch.nn.functional.cosine_similarity(
    our_image.unsqueeze(0), there_image.unsqueeze(0), dim=1
)
print(f"Cosine similarity between generated image and target image: {cosine_sim.item():.4f}")
norm_l2 = torch.nn.functional.pairwise_distance(
    our_image.unsqueeze(0), there_image.unsqueeze(0), p=2
)
print(f"L2 distance between generated image and target image: {norm_l2.item():.4f}")    
src_img.show()
sd_image.show()
comp_img.show()