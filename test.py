from pathlib import Path
import time
from pro_deep import train_edit_model

##### yuval test
run_start = time.time()

model = train_edit_model(
    csv_path=Path("final_dataset_clean.csv"),
    batch_size=1,
    epochs=1,
    lr=1e-4,
    max_samples=500,  # optional subset
    progress_every=10,  # log roughly every 10 steps
    progress_every_seconds=15,  # or every 15s, whichever comes first
)
model.save_pretrained("edit_model_final")

elapsed = time.time() - run_start
print(f"Training script finished in {elapsed/60:.2f}m ({elapsed:.1f}s)")
