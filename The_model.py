# Set up imports for:
# - Dataset loading (CSV + local image paths)
# - Semantic image retrieval (CLIP via SentenceTransformer)
# - Latent encoding/decoding (TAESD autoencoder)
# - Optional refinement back to a natural image manifold (Stable Diffusion img2img)
import argparse
import io
import os
import pickle
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import textwrap

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from sentence_transformers import SentenceTransformer, util

from diffusers import StableDiffusionImg2ImgPipeline
from dino import get_frozen_model

# Using pre - built function from pull_images that access the dataset csv and 
# load the source and target images, as well as the llm edit prompt
from pull_images import load_pair_from_csv

# TAESD is used to encode images into a compact latent space and decode latents back to images.
try:
    from taesd import TAESD
except ImportError as e:
    raise ImportError(
        "Could not import TAESD. Make sure taesd.py, taesd_encoder.pth, and taesd_decoder.pth "
        "are in the same directory as model.py."
    ) from e
# If TEASD files are missing run the following code to download them in Terminal:
#curl -L -o taesd.py https://raw.githubusercontent.com/madebyollin/taesd/main/taesd.py
#curl -L -o taesd_encoder.pth https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_encoder.pth
#curl -L -o taesd_decoder.pth https://raw.githubusercontent.com/madebyollin/taesd/main/taesd_decoder.pth




# --- Semantic search setup (Unsplash 25k + CLIP) ---
# Downloads only if files are missing to avoid repeated network calls.
PHOTO_FILENAME = "unsplash-25k-photos.zip"
EMB_FILENAME = "unsplash-25k-photos-embeddings.pkl"
SEARCH_MODEL = SentenceTransformer("clip-ViT-B-32")
print("Loaded sentence transformer model")


def _ensure_unsplash_data():
    """
    Ensure that the Unsplash dataset files exist locally.

    If the photo archive or the precomputed embeddings file are missing,
    they will be downloaded from the SBERT dataset repository.

    """
    if not os.path.exists(PHOTO_FILENAME):
        util.http_get("http://sbert.net/datasets/" + PHOTO_FILENAME, PHOTO_FILENAME)
    if not os.path.exists(EMB_FILENAME):
        util.http_get("http://sbert.net/datasets/" + EMB_FILENAME, EMB_FILENAME)


def _load_search_assets():

    _ensure_unsplash_data()

    # Open the ZIP file that contains all Unsplash images
    photo_zip = zipfile.ZipFile(PHOTO_FILENAME, "r")

    # Get all file names inside the ZIP and store them in a set
    zip_image_names = set(photo_zip.namelist())

    # Load precomputed image embeddings from a pickle file
    with open(EMB_FILENAME, "rb") as f:
        img_names, img_emb = pickle.load(f)
    print("Images:", len(img_names))

    # Return all loaded assets for semantic search
    return photo_zip, zip_image_names, img_names, img_emb


PHOTO_ZIP, ZIP_IMAGE_NAMES, IMG_NAMES, IMG_EMB = _load_search_assets()


def load_image_bytes(image_name: str) -> bytes:
    """
    Load image bytes by first checking the ZIP archive and falling back to a
    local 'photos' directory if needed.
    """

    if image_name in ZIP_IMAGE_NAMES:
        return PHOTO_ZIP.read(image_name)

    fallback_path = os.path.join("photos", image_name)
    if os.path.exists(fallback_path):  # Use an already-extracted copy if it is present
        with open(fallback_path, "rb") as f:
            return f.read()

    raise FileNotFoundError(f"Image {image_name} not found in zip or photos folder.")


def search(query: str, k: int = 1, show: bool = False):
    """
    Retrieve the top-k Unsplash images most semantically similar to a text query
    using CLIP embeddings (SentenceTransformer + precomputed image embeddings).

    Returns a list of dicts: {"name", "score", "image"}.
    """
    query_emb = SEARCH_MODEL.encode([query], convert_to_tensor=True, show_progress_bar=False)
    hits = util.semantic_search(query_emb, IMG_EMB, top_k=k)[0]

    results = []
    print(f"Query: {query}")
    for rank, hit in enumerate(hits, 1):
        image_name = IMG_NAMES[hit["corpus_id"]]
        image_bytes = load_image_bytes(image_name)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results.append({"name": image_name, "score": float(hit["score"]), "image": image})
        if show:
            print(f"#{rank}: {image_name} (score={hit['score']:.3f})")
            image.show()

    return results


def search_llm_edit_image(prompt: str, k: int = 1, show: bool = False):
    """
    Perform semantic search for an image based on a text prompt and return
    the top match.
    """

    hits = search(prompt, k=k, show=show)
    if not hits:
        raise RuntimeError(f"No search results for prompt: {prompt}")
    top = hits[0]
    print(f"Using search match: {top['name']} (score={top['score']:.3f})")
    return top


# --- Visualization helpers ---
# def caption_image(image: Image.Image, title: str, caption: str = "", wrap: int = 48):
#     """
#     Add a caption strip under an image so the prompt is visible beside it.
#     """
#     img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
#     font = ImageFont.load_default()
#     wrapped_caption = textwrap.fill(caption, width=wrap) if caption else ""
#
#     full_text = title if title else ""
#     if wrapped_caption:
#         full_text = f"{full_text}\n{wrapped_caption}" if full_text else wrapped_caption
#
#     draw = ImageDraw.Draw(img)
#     text_w, text_h = draw.multiline_textsize(full_text, font=font)
#     pad = 6
#     strip_h = text_h + 2 * pad if full_text else 0
#
#     out = Image.new("RGB", (img.width, img.height + strip_h), color=(24, 24, 24))
#     out.paste(img, (0, 0))
#     if full_text:
#         draw = ImageDraw.Draw(out)
#         draw.multiline_text((pad, img.height + pad), full_text, fill=(255, 255, 255), font=font)
#     return out


# def concat_horizontal(images):
#     """Concatenate PIL images horizontally with padding."""
#     if not images:
#         raise ValueError("No images to concatenate")
#     pad = 10
#     width = sum(im.width for im in images) + pad * (len(images) + 1)
#     height = max(im.height for im in images) + 2 * pad
#     canvas = Image.new("RGB", (width, height), color=(16, 16, 16))
#     x = pad
#     for im in images:
#         y = pad + (height - 2 * pad - im.height) // 2
#         canvas.paste(im, (x, y))
#         x += im.width + pad
#     return canvas

# Default hyperparameters
DEFAULT_CSV = Path("dataset_cap_edit_only.csv")
DEFAULT_SAVE_DIR = Path("gaussian_blend_outputs")
IMAGE_SIZE = 512
DEFAULT_SIGMA = 1
DEFAULT_STEPS = 10
DEFAULT_LR = 0.1


dev = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Initialize TAESD autoencoder (encoder/decoder)
# Used to move between pixel space (images) and latent space (where we blend/optimize)
taesd = TAESD().to(dev).eval()



def summarize_tensor(x):
    """
    Create a compact, color-formatted summary of a tensor for debugging,
    including its shape and basic statistics (min, mean, max).
    """
    return f"\033[34m{str(tuple(x.shape)).ljust(24)}\033[0m (\033[31mmin {x.min().item():+.4f}\033[0m / \033[32mmean {x.mean().item():+.4f}\033[0m / \033[33mmax {x.max().item():+.4f}\033[0m)"


def preprocess(image, size=IMAGE_SIZE):
    """
    Preprocess an input image by resizing it to a fixed square resolution
    required by the model.
    """
    return TF.resize(image, [size, size])


def encode_image(image: Image.Image):
    """
    Preprocess a PIL image (resize), convert it to a torch tensor (BCHW, 0-1),
    and encode it into TAESD latent space in inference mode

    Returns:
        image_tensor: [1, 3, IMAGE_SIZE, IMAGE_SIZE]
        latent:       TAESD latent representation (e.g., [1, 4, h, w])
    """
    image = preprocess(image)
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(dev)
    with torch.no_grad():
        latent = taesd.encoder(image_tensor)
    return image_tensor, latent


def decode_latent(latent: torch.Tensor, allow_grad: bool = False):
    """
    Decode a TAESD latent representation back into image space.

    If allow_grad is True, decoding is performed with gradients enabled
    (used during latent optimization). Otherwise, decoding runs in
    inference mode with no gradients.
    """
    if allow_grad:
        return taesd.decoder(latent).clamp(0, 1)
    with torch.no_grad():
        return taesd.decoder(latent).clamp(0, 1)




def save_tensor_image(tensor: torch.Tensor, path: Path, show: bool = True):
    """
    Save a torch image tensor to disk as a PIL image, with optional preview.
    """
    pil_img = TF.to_pil_image(tensor.detach().cpu())
    pil_img.save(path)
    if show:
        pil_img.show()


def make_image_coherent(
    image: Image.Image,
    prompt: str,
    strength: float = 0.6,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
    seed: int | None = None,
    device: str | None = None,
):
    """
    Refine an image using Stable Diffusion img2img to improve visual coherence.
    This step projects the blended or optimized result back onto a natural
    image manifold, guided by the given prompt.
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    ).to(device)

    pipe.enable_attention_slicing()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    image = image.convert("RGB")

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        generator=generator,
    )

    return result.images[0]

# DINO needs scaling and normalization
def prepare_for_dino(image_tensor: torch.Tensor, processor, device):
    """
    Prepare an image tensor for DINOv2 embedding.

    The function resizes the input tensor to the resolution expected by the
    DINOv2 processor and applies the required normalization (mean/std),
    while preserving gradients for downstream optimization.
    """
    size_cfg = processor.size
    if isinstance(size_cfg, dict):
        tgt_h = size_cfg.get("height") or size_cfg.get("shortest_edge") or 224
        tgt_w = size_cfg.get("width") or tgt_h
    else:
        tgt_h = tgt_w = int(size_cfg)
    x = F.interpolate(image_tensor, size=(tgt_h, tgt_w), mode="bicubic", align_corners=False)
    mean = torch.tensor(processor.image_mean, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def embed_tensor_with_dino(image_tensor: torch.Tensor, model, processor):
    """"
    Compute a differentiable DINOv2 embedding for an image tensor (BCHW, values in [0,1]).

    The input is resized/normalized using the DINO processor settings, then passed
    through the frozen DINOv2 model. We return a single global embedding by mean-pooling
    the last hidden states over tokens.
    """
    x = prepare_for_dino(image_tensor, processor, device=image_tensor.device)
    outputs = model(pixel_values=x, output_hidden_states=False)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)


def optimize_semantic_midpoint(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    dino_model,
    dino_processor,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
    grad_clip: float = 1.0,
):
    """
    Optimize a TAESD latent so that the decoded image lies at the semantic
    midpoint between a source image and a retrieved target image in DINOv2
    embedding space, using MSE loss on DINO embeddings.
    """
    # 1) Compute fixed target embedding = midpoint in DINO space
    with torch.no_grad():
        # DINO embeddings for the source image (A) and the retrieved target image (B)
        emb_a = embed_tensor_with_dino(image_a, dino_model, dino_processor)
        emb_b = embed_tensor_with_dino(image_b, dino_model, dino_processor)

        # Semantic "midpoint" target in embedding space
        target_emb = 0.5 * (emb_a + emb_b)

    # 2) Optimize a latent variable initialized at the midpoint of the two TAESD latents
    latent_param = torch.nn.Parameter(0.5 * (latent_a + latent_b))
    optimizer = torch.optim.Adam([latent_param], lr=lr)

    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        # 3) Decode current latent guess to an image tensor (keep gradients for optimization)
        decoded = decode_latent(latent_param, allow_grad=True)

        # 4) Embed the decoded image with DINO (differentiable path back to latent_param)
        emb_hat = embed_tensor_with_dino(decoded, dino_model, dino_processor)

        # 5) Loss: make decoded embedding match the target midpoint embedding
        loss = F.mse_loss(emb_hat, target_emb)
        loss.backward()

        # 6) Stabilize training by clipping latent gradients (optional)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_([latent_param], grad_clip)

        optimizer.step()

        # Track and print progress
        losses.append(loss.item())
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"[opt] step {step}/{steps} loss={loss.item():.6f}")

    # 7) Final decode in inference mode (no gradients)
    final_decoded = decode_latent(latent_param.detach(), allow_grad=False)
    return final_decoded, losses


# Hyperparameters options
def parse_args():
    parser = argparse.ArgumentParser(description="Gaussian-mask latent blending or semantic midpoint optimization.")
    parser.add_argument("--mode", choices=[ "opt"], default="opt", help="opt=semantic midpoint optimization.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV, help="Dataset CSV with local paths.")
    parser.add_argument("--row-index", type=int, default=0, help="Row to pull source/target images from.")
    parser.add_argument("--search-top-k", type=int, default=1, help="How many search results to fetch for llm_edit text (first hit is used).")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Where to save outputs.")
    parser.add_argument("--no-show", action="store_true", help="Do not open image previews.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Optimization steps (opt mode).")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Latent optimizer learning rate (opt mode).")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip max-norm for latent (opt mode).")
    parser.add_argument("--no-save", action="store_true", help="Skip saving images to disk.")
    parser.add_argument("--cohere", dest="cohere", action="store_true", default=True, help="Run SD img2img to project the output back to a natural image manifold.")
    parser.add_argument("--no-cohere", dest="cohere", action="store_false", help="Skip SD img2img refinement.")
    parser.add_argument("--cohere-prompt", type=str, default=None, help="Prompt for SD img2img; default uses llm_edit.")
    parser.add_argument("--cohere-strength", type=float, default=0.6 , help="Strength for SD img2img.")
    parser.add_argument("--cohere-guidance", type=float, default=7.5 , help="Guidance scale for SD img2img, mean that higher values encourage adherence to the prompt.")
    parser.add_argument("--cohere-steps", type=int, default=50 , help="Number of diffusion steps for SD img2img.")
    parser.add_argument("--cohere-seed", type=int, default=42 , help="Random seed for SD img2img.")
    return parser.parse_args()


def main():
    
    args = parse_args()
    # Load source + target images and llm_edit prompt from CSV
    image_a, _comp_img, llm_edit_text, row = load_pair_from_csv(args.csv_path, args.row_index)
    # Sanity check
    print(
        f"Loaded row {args.row_index} from {args.csv_path}:\n"
        f"- src_country={row.get('src_country', 'N/A')} target_country={row.get('target_country', 'N/A')}\n"
        f"- filename={row.get('filename')}\n"
        f"- llm_edit={llm_edit_text}"
    )

    show_search = not args.no_show
    # Search for target image using llm_edit prompt
    search_hit = search_llm_edit_image(llm_edit_text, k=args.search_top_k, show=show_search)
    target_image = search_hit["image"]
    print(f"llm_edit prompt: {llm_edit_text}")
    # Encode both source and retrieved target images into TAESD latent space
    image_a_tensor, latent_a = encode_image(image_a)
    target_tensor, latent_b = encode_image(target_image)
    print("latent A", summarize_tensor(latent_a[0]))
    print("latent target", summarize_tensor(latent_b[0]))

    args.save_dir.mkdir(exist_ok=True, parents=True)

    if not args.no_save:
        save_tensor_image(image_a_tensor[0], args.save_dir / "image_source.png", show=not args.no_show)
        save_tensor_image(target_tensor[0], args.save_dir / "image_llm_edit.png", show=not args.no_show and not show_search)
    else:
        if not args.no_show:
            TF.to_pil_image(image_a_tensor[0].cpu()).show()
            if not show_search:
                TF.to_pil_image(target_tensor[0].cpu()).show()
    # Combine source and target in latent space:
    # - opt:   optimize a latent so the decoded image is the DINOv2 semantic midpoint (main method)
    # Load a frozen DINOv2 model to compute semantic embeddings (no training of DINO parameters)
    dino_model, dino_processor = get_frozen_model(device=dev)
    dino_model.eval()
    for p in dino_model.parameters():
        p.requires_grad_(False)
    # Optimize a latent so that the decoded image lies at the DINOv2 semantic midpoint
    # between the source image and the retrieved target image
    decoded, losses = optimize_semantic_midpoint(
        image_a=image_a_tensor,
        image_b=target_tensor,
        latent_a=latent_a,
        latent_b=latent_b,
        dino_model=dino_model,
        dino_processor=dino_processor,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )
    print(f"final loss={losses[-1]:.6f} after {len(losses)} steps")
    if not args.no_save:
        save_tensor_image(decoded[0], args.save_dir / "semantic_midpoint.png", show=not args.no_show)
    elif not args.no_show:
        TF.to_pil_image(decoded[0].cpu()).show()
# Stable Diffusion img2img refinement to improve visual coherence/realism
    if args.cohere:
        prompt_for_sd = args.cohere_prompt if args.cohere_prompt is not None else llm_edit_text
        sd_input = TF.to_pil_image(decoded[0].cpu())
        sd_image = make_image_coherent(
            image=sd_input,
            prompt=prompt_for_sd,
            strength=args.cohere_strength,
            guidance_scale=args.cohere_guidance,
            num_steps=args.cohere_steps,
            seed=args.cohere_seed,
            device=str(dev),
        )
        if not args.no_save:
            out_name = "coherent_blend.png" if args.mode == "blend" else "coherent_semantic_midpoint.png"
            sd_image.save(args.save_dir / out_name)
        if not args.no_show:
            sd_image.show()

    if not args.no_save:
        print(f"Saved outputs to {args.save_dir.resolve()}")


if __name__ == "__main__":
    main()