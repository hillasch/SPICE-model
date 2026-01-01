from pathlib import Path
import time
from pro_deep import train_edit_model, preview_model_predictions

##### yuval test
run_start = time.time()

model = train_edit_model(
    csv_path=Path("final_dataset_clean.csv"),
    batch_size=1,
    epochs=2,
    lr=1e-4,
    max_samples=1000,  # optional subset
    progress_every=100,  # log roughly every 10 steps
    progress_every_seconds=15,  # or every 15s, whichever comes first
)
model.save_pretrained("edit_model_final")

elapsed = time.time() - run_start
print(f"Training script finished in {elapsed/60:.2f}m ({elapsed:.1f}s)")

# Save a few preview figures comparing source vs ground-truth target with predicted embedding distance.
preview_model_predictions(
    model,
    csv_path=Path("final_dataset_clean.csv"),
    samples_to_show=5,
    save_dir=Path("eval_previews"),
    show=False,  # set True to pop up matplotlib windows
)
