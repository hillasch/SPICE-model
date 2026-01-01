import argparse
from pathlib import Path
from typing import Optional

import torch

from diffusion_edit import ImageTextDiffusionConfig, ImageTextDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an edited image conditioned on a source image and text instruction."
    )
    parser.add_argument("--source-image", required=True, help="Path to the source image to edit.")
    parser.add_argument("--instruction", required=True, help="Edit instruction (text prompt).")
    parser.add_argument("--negative-prompt", default="", help="Optional negative prompt to avoid content.")
    parser.add_argument(
        "--model-id",
        default="runwayml/stable-diffusion-1-5",
        help="Diffusion backbone to use (Hugging Face repo id).",
    )
    parser.add_argument("--num-steps", type=int, default=30, help="Denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--strength", type=float, default=0.6, help="Noise strength for img2img.")
    parser.add_argument("--image-cond-scale", type=float, default=1.0, help="Scale for the DINO image tokens.")
    parser.add_argument("--image-tokens", type=int, default=4, help="How many projected image tokens to append.")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--output", default="edited_outputs", help="Output file or directory.")
    parser.add_argument("--device", default=None, help="Override device (cpu, cuda, mps).")
    return parser.parse_args()


def save_images(images, target: Path):
    if len(images) == 1:
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(target)
            print(f"Saved edited image to {target}")
        else:
            target.mkdir(parents=True, exist_ok=True)
            out_path = target / "edited.png"
            images[0].save(out_path)
            print(f"Saved edited image to {out_path}")
        return

    # Multiple images → write all to a directory
    out_dir = target.parent if target.suffix else target
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = target.stem if target.suffix else "edited"
    suffix = target.suffix if target.suffix else ".png"
    for idx, img in enumerate(images):
        out_path = out_dir / f"{stem}_{idx}{suffix}"
        img.save(out_path)
        print(f"Saved edited image to {out_path}")


def main():
    args = parse_args()

    config = ImageTextDiffusionConfig(
        model_id=args.model_id,
        num_image_tokens=args.image_tokens,
        image_cond_scale=args.image_cond_scale,
        device=args.device,
    )
    pipeline = ImageTextDiffusionPipeline(config)

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

    result = pipeline(
        image=args.source_image,
        prompt=args.instruction,
        negative_prompt=args.negative_prompt or None,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        num_images_per_prompt=args.num_images,
        generator=generator,
    )

    save_images(result.images, Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
