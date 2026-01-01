"""
Utility script for building and visualizing the image-transcreation dataset locally.

This version is macOS-friendly and avoids Colab-specific code. It can:
1. Clone the dataset repo (image-transcreation).
2. Build a cleaned CSV with local download paths.
3. Optionally display random samples for quick inspection.
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import unquote

import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time

try:
    from tqdm import tqdm
except ImportError:  # Fallback when tqdm is not installed
    def tqdm(iterable=None, total=None):
        return iterable


REPO_URL = "https://github.com/simran-khanuja/image-transcreation.git"
REPO_NAME = "image-transcreation"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_DIR = BASE_DIR / REPO_NAME
DEFAULT_OUTPUT_CSV = BASE_DIR / "final_dataset_clean.csv"
DEFAULT_IMAGES_DIR = BASE_DIR / "images"
DEFAULT_MAX_WORKERS = 10


def format_duration(seconds: float) -> str:
    """Return a compact human-readable duration like '1h 05m 07s'."""
    seconds = int(round(seconds))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or mins:
        parts.append(f"{mins:02d}m")
    parts.append(f"{secs:02d}s")
    return " ".join(parts)


def get_filename(path_or_url: Optional[str]) -> Optional[str]:
    """Extract the filename from a path or URL."""
    if path_or_url is None or (isinstance(path_or_url, float) and pd.isna(path_or_url)):
        return None
    name = Path(str(path_or_url)).name
    return unquote(name).strip()


def clone_repo(repo_dir: Path, repo_url: str = REPO_URL) -> None:
    """Clone the dataset repo if it is not already present."""
    if repo_dir.exists():
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {repo_url} into {repo_dir} ...")
    result = subprocess.run(
        ["git", "clone", repo_url, str(repo_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git clone failed: {result.stderr.strip()}")


def build_dataset_v2(
    repo_dir: Path,
    images_dir: Path,
    output_csv: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    """
    Build a cleaned dataset from the image-transcreation repo and download images locally.
    Returns the resulting dataframe.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    # Keyed by "<filename>_<target_country>" for quick cross-referencing
    master_data: Dict[str, Dict[str, Optional[str]]] = {}

    meta_files = list(repo_dir.rglob("metadata.csv"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata.csv files found under {repo_dir}")

    for mf in meta_files:
        try:
            target_country = mf.parent.name
            df = pd.read_csv(mf)
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                src_path = row.get("src_image_path")
                fname = get_filename(src_path)
                if not fname:
                    continue

                unique_key = f"{fname}_{target_country}"
                entry = master_data.get(
                    unique_key,
                    {
                        "unique_id": unique_key,
                        "filename": fname,
                        "src_url": None,
                        "target_country": target_country,
                        "src_country": None,
                        "caption": None,
                        "llm_edit": None,
                    },
                )

                # Only overwrite fields if the new value is present (avoid nuking with NaN/None)
                for field, value in {
                    "src_url": src_path,
                    "src_country": row.get("src_country"),
                    "caption": row.get("caption"),
                    "llm_edit": row.get("llm_edit"),
                }.items():
                    if value is not None and not (isinstance(value, float) and pd.isna(value)):
                        entry[field] = value

                master_data[unique_key] = entry
        except Exception as exc:
            print(f"Skipping {mf}: {exc}")

    # Locate split/output CSVs that contain generated comparison image URLs
    output_csv_candidates = [
        p for p in repo_dir.rglob("*.csv") if p.name != "metadata.csv" and "outputs" in p.parts
    ]

    for sf in output_csv_candidates:
        try:
            country_from_path = sf.parent.name
            df = pd.read_csv(sf)
            if "src_image_path" not in df.columns or "model_path_2" not in df.columns:
                continue
            for _, row in df.iterrows():
                fname = get_filename(row.get("src_image_path"))
                comp_url = row.get("model_path_2")
                if not fname or not comp_url:
                    continue

                unique_key = f"{fname}_{country_from_path}"
                if unique_key in master_data and pd.notna(comp_url):
                    master_data[unique_key]["comp_url"] = comp_url
        except Exception as exc:
            print(f"Skipping {sf}: {exc}")

    df = pd.DataFrame(master_data.values())
    df = df.dropna(subset=["llm_edit"])

    # Derive deterministic local file paths for downloads
    df["src_local_path"] = df["filename"].apply(lambda x: str(images_dir / f"src_{x}"))
    df["comp_local_path"] = df.apply(
        lambda row: str(images_dir / f"comp_{row['target_country']}_{row['filename']}")
        if pd.notna(row.get("comp_url"))
        else None,
        axis=1,
    )

    downloads = []

    def download_task(url, path):
        # Skip empty rows and allow restarting without re-downloading
        if url is None or pd.isna(url) or path is None:
            return "Skip"

        path = Path(path)
        if path.exists():
            return "Exists"

        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            response.raise_for_status()
            path.write_bytes(response.content)
            return "Downloaded"
        except Exception:
            return "Error"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in df.iterrows():
            downloads.append(executor.submit(download_task, row["src_url"], row["src_local_path"]))
            if pd.notna(row.get("comp_url")):
                downloads.append(executor.submit(download_task, row["comp_url"], row["comp_local_path"]))

        results = {"Downloaded": 0, "Exists": 0, "Error": 0, "Skip": 0}
        for future in tqdm(as_completed(downloads), total=len(downloads)):
            res = future.result()
            results[res] = results.get(res, 0) + 1

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    print(f"Download summary: {results}")
    return df


def show_dataset_samples(csv_path: Path, samples_to_show: int = 3) -> None:
    """Display random source/target pairs for a quick visual check."""
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        import textwrap
    except ImportError as exc:
        print(f"Install pillow and matplotlib to display samples ({exc}).")
        return

    if not csv_path.exists():
        print(f"CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Only keep rows that have both source and comparison images
    df_complete = df[df["comp_local_path"].notna()].copy()
    if df_complete.empty:
        print("No complete pairs (source + target) found yet.")
        return

    # Random subset for a quick visual spot-check
    samples = df_complete.sample(n=min(samples_to_show, len(df_complete)))
    print(f"Displaying {len(samples)} random samples out of {len(df_complete)} complete pairs...")

    for _, row in samples.iterrows():
        src_path = Path(row["src_local_path"])
        comp_path = Path(row["comp_local_path"])
        if not src_path.exists() or not comp_path.exists():
            print(f"Skipping missing files for {row['filename']}")
            continue

        try:
            img_src = Image.open(src_path)
            img_comp = Image.open(comp_path)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_src)
            axes[0].set_title(
                f"Source: {row.get('src_country', 'Original')}\nFilename: {row['filename']}",
                fontsize=10,
            )
            axes[0].axis("off")

            axes[1].imshow(img_comp)
            axes[1].set_title(
                f"Target: {row['target_country']}",
                fontsize=10,
                color="blue",
                fontweight="bold",
            )
            axes[1].axis("off")

            edit_text = row.get("llm_edit", "No instruction")
            wrapped_text = "\n".join(textwrap.wrap(f"Instruction: {edit_text}", width=80))

            plt.suptitle(wrapped_text, fontsize=12, y=1.05)
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            print("-" * 80)
        except Exception as exc:
            print(f"Error displaying image for {row['filename']}: {exc}")


def preview_model_predictions(
    model: nn.Module,
    csv_path: Path,
    samples_to_show: int = 3,
    save_dir: Optional[Path] = None,
    show: bool = True,
    device=None,
):
    """
    Visualize how the trained model edits embeddings by comparing predicted vs target images.
    This does NOT generate new images; it shows the source and ground-truth target images while
    reporting the L2 distance between the model's predicted embedding and the target embedding.
    """
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        import textwrap
    except ImportError as exc:
        print(f"Install pillow and matplotlib to display previews ({exc}).")
        return

    device = device or get_device()
    model = model.to(device).eval()

    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df[df["comp_local_path"].notna() & df["src_local_path"].notna()].copy()
    df = df[
        df["src_local_path"].apply(lambda p: Path(p).exists())
        & df["comp_local_path"].apply(lambda p: Path(p).exists())
    ]
    if df.empty:
        print("No valid rows with source/target image files found for preview.")
        return

    tokenizer, text_encoder = load_text_encoder(device=device)
    dino_model, dino_processor = get_frozen_model(device=device)

    save_dir_path = Path(save_dir) if save_dir else None
    if save_dir_path:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    samples = df.sample(n=min(samples_to_show, len(df)))
    print(f"Previewing {len(samples)} samples...")

    for _, row in samples.iterrows():
        src_path = Path(row["src_local_path"])
        tgt_path = Path(row["comp_local_path"])
        try:
            src_img = Image.open(src_path).convert("RGB")
            tgt_img = Image.open(tgt_path).convert("RGB")

            x_src = embed_image_with_dino(
                src_path, model=dino_model, processor=dino_processor, device=device, pool=True
            )
            t_emb = embed_llm_edit(
                row["llm_edit"], tokenizer=tokenizer, text_encoder=text_encoder, device=device, pool=True
            )
            x_tgt = embed_image_with_dino(
                tgt_path, model=dino_model, processor=dino_processor, device=device, pool=True
            )

            with torch.no_grad():
                _, x_hat = model(x_src.unsqueeze(0).to(device), t_emb.unsqueeze(0).to(device))
            x_hat = x_hat.squeeze(0).cpu()
            x_tgt = x_tgt.cpu()
            dist = torch.norm(x_hat - x_tgt, p=2).item()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(src_img)
            axes[0].set_title(f"Source: {row.get('src_country', 'original')}\n{row['filename']}", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(tgt_img)
            axes[1].set_title(f"Target: {row['target_country']}\nL2 dist={dist:.3f}", fontsize=10, color="blue")
            axes[1].axis("off")

            edit_text = row.get("llm_edit", "No instruction")
            wrapped_text = "\n".join(textwrap.wrap(f"Instruction: {edit_text}", width=80))
            plt.suptitle(wrapped_text, fontsize=11, y=1.02)
            plt.tight_layout()

            if save_dir_path:
                out_path = save_dir_path / f"preview_{row['filename']}_{row['target_country']}.png"
                plt.savefig(out_path, bbox_inches="tight")
                print(f"Saved preview to {out_path} (L2 dist={dist:.3f})")
                plt.close(fig)
            elif show:
                plt.show()
            else:
                plt.close(fig)
        except Exception as exc:
            print(f"Error previewing {row['filename']}: {exc}")


def get_device():
    """Return the best available torch device (lazy import to avoid hard dependency)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_frozen_model(device=None):
    """Load and freeze DINOv2 base model for feature extraction."""
    import torch
    from transformers import AutoImageProcessor, AutoModel

    device = device or get_device()

    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, processor


def load_text_encoder(device=None):
    """Load and freeze the CLIP text encoder."""
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    device = device or get_device()
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)

    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False

    return tokenizer, text_encoder


def get_text_embedding(text: str, tokenizer, text_encoder, device=None):
    """Produce a text embedding suitable for Stable Diffusion cross-attention."""
    device = device or get_device()
    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = text_encoder(**tokens)

    return outputs.last_hidden_state


def embed_llm_edit(text: str, tokenizer=None, text_encoder=None, device=None, pool: bool = True):
    """
    Convenience wrapper: load CLIP text encoder (if needed) and return the text embedding.
    If pool=True, returns a single vector (masked mean over tokens); otherwise returns the full sequence.
    """
    device = device or get_device()
    if tokenizer is None or text_encoder is None:
        tokenizer, text_encoder = load_text_encoder(device=device)

    seq = get_text_embedding(text, tokenizer, text_encoder, device=device)
    if not pool:
        return seq

    mask = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)["attention_mask"].unsqueeze(-1)  # [B, seq, 1]

    masked = seq * mask
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
    pooled = masked.sum(dim=1, keepdim=True) / denom  # [B, 1, hidden]
    return pooled.squeeze()


def embed_image_with_dino(image_path, model=None, processor=None, device=None, pool: bool = True):
    """
    Embed a source image with DINOv2. Returns a pooled vector by default.

    pool=True returns a single vector (mean-pooled last_hidden_state); False returns the full sequence.
    """
    from PIL import Image

    device = device or get_device()
    if model is None or processor is None:
        model, processor = get_frozen_model(device=device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state

    if pool:
        return outputs.mean(dim=1).squeeze(0)  # [hidden_size]
    return outputs.squeeze(0)  # [seq_len, hidden_size]


class TextGuidedDelta(nn.Module):
    """Residual MLP that predicts a text-conditioned edit in embedding space."""

    def __init__(self, d_img: int, d_txt: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.hidden = hidden
        self.dropout = dropout
        self.text_proj = nn.Sequential(
            nn.Linear(d_txt, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_img),
        )
        self.delta_mlp = nn.Sequential(
            nn.Linear(d_img * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_img),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t_proj = self.text_proj(t)
        fused = torch.cat([x, t_proj], dim=-1)
        delta = self.delta_mlp(fused)
        x_hat = x + delta
        return delta, x_hat

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save weights and minimal config in a HuggingFace-like layout.
        Produces save_directory/pytorch_model.bin with state_dict + config.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        state = {k: v.cpu() for k, v in self.state_dict().items()}
        payload = {
            "state_dict": state,
            "config": {
                "d_img": self.d_img,
                "d_txt": self.d_txt,
                "hidden": self.hidden,
                "dropout": self.dropout,
            },
        }
        torch.save(payload, save_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path], map_location=None):
        """
        Load weights saved via save_pretrained. map_location is forwarded to torch.load.
        """
        load_dir = Path(load_directory)
        checkpoint_path = load_dir / "pytorch_model.bin"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        cfg = checkpoint.get("config")
        if not cfg:
            raise ValueError("Checkpoint missing config; cannot reconstruct model.")

        model = cls(
            d_img=cfg["d_img"],
            d_txt=cfg["d_txt"],
            hidden=cfg.get("hidden", 512),
            dropout=cfg.get("dropout", 0.1),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model


class EditDataset(Dataset):
    """
    Minimal dataset that embeds source/target images with DINO and llm_edit text with CLIP.
    Expects CSV rows to have: src_local_path, comp_local_path, llm_edit.
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        text_encoder,
        dino_model,
        dino_processor,
        device=None,
        max_samples: int = None,
    ):
        self.device = device or get_device()
        df = pd.read_csv(csv_path)
        # Require non-null paths
        df = df[df["comp_local_path"].notna() & df["src_local_path"].notna()].copy()
        # Keep only rows where both source/target files exist on disk
        df = df[
            df["src_local_path"].apply(lambda p: Path(p).exists())
            & df["comp_local_path"].apply(lambda p: Path(p).exists())
        ].copy()
        if max_samples:
            df = df.head(max_samples)
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.dino_model = dino_model
        self.dino_processor = dino_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_path = Path(row["src_local_path"])
        tgt_path = Path(row["comp_local_path"])
        text = row["llm_edit"]

        x = embed_image_with_dino(
            src_path, model=self.dino_model, processor=self.dino_processor, device=self.device, pool=True
        )
        x_tgt = embed_image_with_dino(
            tgt_path, model=self.dino_model, processor=self.dino_processor, device=self.device, pool=True
        )
        t_emb = embed_llm_edit(text, tokenizer=self.tokenizer, text_encoder=self.text_encoder, device=self.device)

        return x.cpu(), t_emb.cpu(), x_tgt.cpu()


def train_edit_model(
    csv_path: Path,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 1e-4,
    hidden: int = 512,
    dropout: float = 0.1,
    num_workers: int = 0,
    max_samples: int = None,
    progress_every: int = 25,
    progress_every_seconds: int = 30,
    device=None,
):
    """
    Train the edit model to minimize L2 distance between predicted edited embedding and target DINO embedding.
    """
    device = device or get_device()
    csv_path = Path(csv_path)

    # Simple run counter + timer so repeated calls are labeled and timed.
    run_id = getattr(train_edit_model, "_run_counter", 0) + 1
    train_edit_model._run_counter = run_id
    run_start = time.time()

    needs_build = False
    if not csv_path.exists():
        needs_build = True
    else:
        try:
            needs_build = pd.read_csv(csv_path).empty
        except Exception:
            needs_build = True

    if needs_build:
        print("CSV missing or empty. Rebuilding dataset with build_dataset_v2...")
        clone_repo(DEFAULT_REPO_DIR, REPO_URL)
        build_dataset_v2(
            repo_dir=DEFAULT_REPO_DIR,
            images_dir=DEFAULT_IMAGES_DIR,
            output_csv=csv_path,
            max_workers=DEFAULT_MAX_WORKERS,
        )

    tokenizer, text_encoder = load_text_encoder(device=device)
    dino_model, dino_processor = get_frozen_model(device=device)

    d_img = dino_model.config.hidden_size
    d_txt = text_encoder.config.hidden_size

    dataset = EditDataset(
        csv_path=Path(csv_path),
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        dino_model=dino_model,
        dino_processor=dino_processor,
        device=device,
        max_samples=max_samples,
    )
    if len(dataset) == 0:
        raise ValueError("No valid samples found (missing files or comp_local_path is null). Rebuild/download data first.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = TextGuidedDelta(d_img=d_img, d_txt=d_txt, hidden=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(
        f"[Run {run_id}] Starting training with {len(dataset)} samples, "
        f"batch_size={batch_size}, epochs={epochs}"
    )
    global_start = time.time()
    total_steps = epochs * len(loader)
    global_step = 0
    log_every = max(1, min(len(loader), progress_every))
    last_log = run_start

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()
        for step_idx, (x, t, x_tgt) in enumerate(loader, 1):
            x = x.to(device)
            t = t.to(device)
            x_tgt = x_tgt.to(device)

            optimizer.zero_grad()
            _, x_hat = model(x, t)
            # L2 loss in embedding space between predicted edit and DINO embedding of target image
            loss = torch.norm(x_hat - x_tgt, p=2, dim=-1).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            global_step += 1

            now = time.time()
            should_log = (
                step_idx % log_every == 0
                or global_step == total_steps
                or (now - last_log) >= progress_every_seconds
            )

            if should_log:
                elapsed = now - run_start
                remaining_steps = total_steps - global_step
                avg_step = elapsed / max(global_step, 1)
                eta_secs = max(0.0, avg_step * remaining_steps)
                last_log = now
                print(
                    f"[Run {run_id}] Epoch {epoch+1}/{epochs} step {step_idx}/{len(loader)} "
                    f"loss={loss.item():.4f} ETA {format_duration(eta_secs)}"
                )

        avg_loss = total_loss / len(dataset)
        epoch_secs = time.time() - epoch_start
        total_secs = time.time() - global_start
        print(
            f"[Run {run_id}] Epoch {epoch+1}/{epochs} - train_loss={avg_loss:.4f} "
            f"- epoch_time={epoch_secs:.1f}s - total_time={total_secs/60:.1f}m"
        )

    print(f"[Run {run_id}] Completed in {format_duration(time.time() - run_start)}")

    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and inspect the image-transcreation dataset locally.")
    parser.add_argument("--repo-url", default=REPO_URL, help="Repository URL to clone.")
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR), help="Where to clone the repository.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR), help="Directory to store downloaded images.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Path to save the cleaned CSV.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel downloads.")
    parser.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help="If > 0, display this many random source/target pairs after building.",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip cloning the repo (useful if already cloned or offline).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip rebuilding the CSV (only show samples from an existing CSV).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_dir = Path(args.repo_dir).expanduser()
    images_dir = Path(args.images_dir).expanduser()
    output_csv = Path(args.output_csv).expanduser()

    if not args.skip_clone:
        clone_repo(repo_dir, args.repo_url)
    elif not repo_dir.exists():
        print(f"Repo directory {repo_dir} does not exist. Remove --skip-clone or fix the path.")
        return 1

    if args.skip_build:
        if not output_csv.exists():
            print(f"CSV not found at {output_csv}; cannot skip build.")
            return 1
    else:
        build_dataset_v2(repo_dir, images_dir, output_csv, max_workers=args.max_workers)

    if args.show_samples > 0:
        show_dataset_samples(output_csv, samples_to_show=args.show_samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


    
