from pathlib import Path
from urllib.parse import urlparse
from PIL import Image
from sentence_transformers import SentenceTransformer, util

def load_pair_from_csv(csv_path: Path, row_index: int = 9):
    """
    Load source + comparison image pair and LLM edit prompt from the dataset CSV.
    """
    import csv
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index {row_index} out of range (CSV has {len(rows)} rows)")

    row = rows[row_index]
    prompt = row.get("llm_edit", "").strip()

    def _resolve_image(local_key: str, url_key: str, cache_dir: Path, label: str) -> Path:
        candidate = (row.get(local_key) or row.get(url_key) or "").strip()
        if not candidate:
            raise FileNotFoundError(f"No {label} path/url provided in row {row_index}")

        # Normalize common scheme typos ("https:/...") so downloads succeed.
        if candidate.startswith("https:/") and not candidate.startswith("https://"):
            candidate = candidate.replace("https:/", "https://", 1)
        if candidate.startswith("http:/") and not candidate.startswith("http://"):
            candidate = candidate.replace("http:/", "http://", 1)

        # Download remote files to a local cache when only a URL is provided.
        if candidate.startswith("http://") or candidate.startswith("https://"):
            parsed = urlparse(candidate)
            filename = Path(parsed.path).name or f"{label}.jpg"
            cache_dir.mkdir(parents=True, exist_ok=True)
            local_path = cache_dir / filename
            if not local_path.exists():
                print(f"Downloading {label} image from {candidate} -> {local_path}")
                util.http_get(candidate, str(local_path))
            return local_path

        path = Path(candidate).expanduser()
        if not path.is_absolute():
            candidate_rel = csv_path.parent / path
            if candidate_rel.exists():
                return candidate_rel
        if path.exists():
            return path

        raise FileNotFoundError(f"Missing {label} image at {path}")

    cache_root = csv_path.parent / "image_cache"
    src_path = _resolve_image("src_local_path", "src_url", cache_root, "source")
    comp_path = _resolve_image("comp_local_path", "comp_url", cache_root, "comparison/target")

    src_img = Image.open(src_path).convert("RGB")
    comp_img = Image.open(comp_path).convert("RGB")
    return src_img, comp_img, prompt, row

