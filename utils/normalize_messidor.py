# utils/normalize_messidor.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("data/messidor")
CSV = ROOT / "messidor_data.csv"   # your file
OUT = ROOT / "train_norm.csv"
OUT_TRAIN = ROOT / "train_split.csv"
OUT_VAL = ROOT / "val_split.csv"

print("Reading:", CSV)
df = pd.read_csv(CSV)
print("Columns:", list(df.columns))
# Expect columns like id_code,diagnosis,...

# Which column contains filename?
if "id_code" in df.columns:
    fname_col = "id_code"
else:
    # fallback to first column
    fname_col = df.columns[0]

# Which column contains label?
label_col = None
for c in ["diagnosis", "label", "labels", "target", "severity"]:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    # fallback: try second column
    label_col = df.columns[1] if len(df.columns) > 1 else None

print("Using filename column:", fname_col)
print("Using label column:", label_col)

# build a map: lowercase filename -> relative path
print("Scanning local image files under", ROOT, "this may take a few seconds...")
file_map = {}
for p in ROOT.rglob("*"):
    if p.is_file():
        name = p.name.lower()
        file_map[name] = p

def find_local_path(name):
    if not isinstance(name, str):
        name = str(name)
    key = name.strip().lower()
    # try direct match
    if key in file_map:
        return str(file_map[key].relative_to(ROOT))
    # try adding common extensions
    base = Path(name).stem.lower()
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        k = base + ext
        if k in file_map:
            return str(file_map[k].relative_to(ROOT))
    # try name with prefix/suffix changes: return None if not found
    return None

rows = []
missing = []
for _, r in df.iterrows():
    fname = r[fname_col]
    local = find_local_path(fname)
    if local is None:
        missing.append(str(fname))
        continue
    if label_col is None:
        lab = 0
    else:
        lab = r[label_col]
        # try convert to int if possible
        try:
            lab = int(lab)
        except Exception:
            # if string labels, keep as-is
            lab = str(lab)
    rows.append({"image_path": local.replace("\\","/"), "label": lab})

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)
print("Wrote normalized CSV:", OUT, "rows:", len(out_df))
print("Missing files count (not found locally):", len(missing))
if missing:
    print("Sample missing:", missing[:20])

# Show label distribution
if len(out_df) > 0:
    try:
        print("Label value counts:")
        print(out_df["label"].value_counts().to_dict())
    except Exception as e:
        print("Could not compute label counts:", e)

print("Sample rows:")
print(out_df.head(10))

# Create stratified splits if label column appears numeric / discrete
try:
    if out_df["label"].dtype.kind in "iubc":
        train, val = train_test_split(out_df, test_size=0.1, random_state=42, stratify=out_df["label"])
        train.to_csv(OUT_TRAIN, index=False)
        val.to_csv(OUT_VAL, index=False)
        print("Wrote train/val splits:", OUT_TRAIN, OUT_VAL, "rows:", len(train), len(val))
    else:
        print("Label column not numeric-discrete; skipping stratified split. Create splits manually if needed.")
except Exception as e:
    print("Could not create splits:", e)
