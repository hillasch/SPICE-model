import torch
from pro_deep import DeltaEditModel
from pathlib import Path
from pro_deep import preview_model_predictions

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ⚠️ חייב להיות אותם פרמטרים כמו באימון
model = DeltaEditModel(
    image_dim=768,
    text_dim=768,
    hidden_dim=1024,
)

state = torch.load(
    "edit_model_final/pytorch_model.bin",
    map_location=device
)
model.load_state_dict(state)

model = model.to(device).eval()

preview_model_predictions(
    model=model,   # ← מודל אמיתי, לא Path
    csv_path="final_dataset_clean.csv",
    out_dir="eval_previews",
    num_samples=1,
    show=False,
)