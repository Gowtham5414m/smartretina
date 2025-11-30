# utils/create_aptos_splits.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("data/aptos")
csv_in = ROOT / "train.csv"
out_train = ROOT / "train_split.csv"
out_val = ROOT / "val_split.csv"

print("Reading:", csv_in)
df = pd.read_csv(csv_in)
print("Rows in original train.csv:", len(df))
# The CSV has columns: id_code, diagnosis
# Create image_path relative to dataset root: train_images/<id>.png (or .jpg if needed)
def find_ext(root, img_id):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        p = root / "train_images" / f"{img_id}{ext}"
        if p.exists():
            return f"train_images/{img_id}{ext}"
    return None

# Build image_path column
img_paths = []
missing = []
for img_id in df["id_code"].astype(str):
    p = find_ext(ROOT, img_id)
    if p is None:
        missing.append(img_id)
        img_paths.append("")  # placeholder
    else:
        img_paths.append(p)

df["image_path"] = img_paths
# Drop rows with missing images
if missing:
    print(f"Warning: {len(missing)} ids not found. Example: {missing[:5]}")
    df = df[df["image_path"] != ""]

# Rename label column to 'label' (integer)
if "diagnosis" in df.columns:
    df = df.rename(columns={"diagnosis":"label"})
else:
    # try common alternatives
    possible = [c for c in df.columns if c.lower() in ("label","labels","diagnosis","severity","level")]
    if possible:
        df = df.rename(columns={possible[0]:"label"})
    else:
        raise RuntimeError("No label column found in APTOS train.csv")

df = df[["image_path","label"]]

# stratified split if labels exist
if df["label"].dtype.kind in "ifubc":  # numeric
    strat = df["label"]
else:
    strat = None

train, val = train_test_split(df, test_size=0.1, random_state=42, stratify=strat)
train.to_csv(out_train, index=False)
val.to_csv(out_val, index=False)
print("Wrote:", out_train, "rows:", len(train))
print("Wrote:", out_val, "rows:", len(val))
