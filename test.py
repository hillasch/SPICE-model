from pathlib import Path
from pro_deep import train_edit_model
##### yuval test

model = train_edit_model(
    csv_path=Path("final_dataset_clean.csv"),
    batch_size=4,
    epochs=3,
    lr=1e-4,
    max_samples=500  # optional subset
)
model.save_pretrained("edit_model_final")