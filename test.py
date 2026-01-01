import argparse
import textwrap
import time
from pathlib import Path
from typing import Optional

import torch

from diffusion_edit import ImageTextDiffusionConfig, ImageTextDiffusionPipeline
from pro_deep import preview_model_predictions, train_edit_model
##### to apply the test.py script for comparing diffusion results with your trained model, run:
#python test.py --mode compare \
#  --csv-path final_dataset_clean.csv \
#  --num-samples 3 \
#  --save-comparisons eval_previews/diffusion_triplets \
#  --no-show \
#  --guidance-scale 7.5 --strength 0.6 --image-cond-scale 1.0 --image-tokens 4 --num-steps 30
##############################################################################################

def run_training(args):
    run_start = time.time()
    model = train_edit_model(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        max_samples=args.max_samples,
        progress_every=args.progress_every,
        progress_every_seconds=args.progress_every_seconds,
    )
    model.save_pretrained(args.save_dir)

    elapsed = time.time() - run_start
    print(f"Training script finished in {elapsed/60:.2f}m ({elapsed:.1f}s)")

    preview_model_predictions(
        model,
        csv_path=args.csv_path,
        samples_to_show=1,
        save_dir=Path("eval_previews"),
        show=False,  # set True to pop up matplotlib windows
    )


def compare_samples_with_diffusion(
    csv_path: Path,
    num_samples: int = 3,
    save_dir: Optional[Path] = None,
    show: bool = True,
    model_id: str = "runwayml/stable-diffusion-1-5",
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.6,
    image_cond_scale: float = 1.0,
    image_tokens: int = 4,
    seed: Optional[int] = None,
    negative_prompt: str = "",
):
    """
    Visualize source, target, and diffusion-generated images side by side for qualitative checks.
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        from PIL import Image, ImageOps
    except ImportError as exc:
        print(f"Missing dependency for visualization: {exc}")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["src_local_path", "comp_local_path", "llm_edit"])
    df = df[
        df["src_local_path"].apply(lambda p: Path(p).exists())
        & df["comp_local_path"].apply(lambda p: Path(p).exists())
    ]
    if df.empty:
        print("No valid samples found for diffusion comparison.")
        return

    samples = df.sample(n=min(num_samples, len(df)))

    config = ImageTextDiffusionConfig(
        model_id=model_id,
        num_image_tokens=image_tokens,
        image_cond_scale=image_cond_scale,
    )
    pipeline = ImageTextDiffusionPipeline(config)

    generator: Optional[torch.Generator] = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for _, row in samples.iterrows():
        src_path = Path(row["src_local_path"])
        tgt_path = Path(row["comp_local_path"])
        instruction = row["llm_edit"]

        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")

        result = pipeline(
            image=src_img,
            prompt=instruction,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=1,
            generator=generator,
        )
        gen_img = result.images[0]

        # Resize copies for uniform display only; originals on disk remain untouched.
        display_src = src_img.copy()
        display_tgt = ImageOps.contain(tgt_img, display_src.size)
        display_gen = ImageOps.contain(gen_img, display_src.size)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(display_src)
        axes[0].set_title(f"Source ({row.get('src_country', 'N/A')})", fontsize=10)
        axes[1].imshow(display_tgt)
        axes[1].set_title(f"Target ({row.get('target_country', 'N/A')})", fontsize=10)
        axes[2].imshow(display_gen)
        axes[2].set_title("Generated (Diffusion)", fontsize=10)

        for ax in axes:
            ax.axis("off")

        caption = textwrap.fill(f"Edit instruction: {instruction}", width=90)
        fig.suptitle(caption, fontsize=11)
        fig.tight_layout()

        if save_dir:
            filename = row.get("filename", src_path.stem)
            out_name = f"{Path(filename).stem}_{row.get('target_country', 'target')}.png"
            out_path = save_dir / out_name
            fig.savefig(out_path, bbox_inches="tight")
            print(f"Saved comparison to {out_path}")

        if show:
            plt.show()
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Training + qualitative diffusion comparison utility.")
    parser.add_argument("--mode", choices=["train", "compare"], default="train", help="Run training or diffusion viz.")
    parser.add_argument("--csv-path", type=Path, default=Path("final_dataset_clean.csv"), help="Dataset CSV path.")

    # Training options
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--progress-every-seconds", type=int, default=15)
    parser.add_argument("--save-dir", type=Path, default=Path("edit_model_final"))

    # Diffusion comparison options
    parser.add_argument("--num-samples", type=int, default=2, help="How many random rows to visualize.")
    parser.add_argument("--save-comparisons", type=Path, default=None, help="Where to save comparison figures.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures, only save if path given.")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-1-5")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--image-cond-scale", type=float, default=1.0)
    parser.add_argument("--image-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--negative-prompt", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    else:
        compare_samples_with_diffusion(
            csv_path=args.csv_path,
            num_samples=args.num_samples,
            save_dir=args.save_comparisons,
            show=not args.no_show,
            model_id=args.model_id,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            image_cond_scale=args.image_cond_scale,
            image_tokens=args.image_tokens,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
        )


if __name__ == "__main__":
    raise SystemExit(main())
