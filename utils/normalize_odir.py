# utils/normalize_odir.py  (robust version)
import pandas as pd
from pathlib import Path
import ast
import numpy as np

ROOT = Path("data/odir")
full = ROOT / "full_df.csv"  # inspector found this file
out = ROOT / "train_norm.csv"

print("Reading:", full)
df = pd.read_csv(full)

print("Columns in full_df.csv:", list(df.columns)[:40])

# Heuristic: prefer 'filename' or 'filepath' as image reference
img_col = None
for c in ["filename", "filepath", "Left-Fundus", "Right-Fundus", "left", "right"]:
    if c in df.columns:
        img_col = c
        break
if img_col is None:
    for c in df.columns:
        if "file" in c.lower() or "path" in c.lower():
            img_col = c
            break
if img_col is None:
    raise RuntimeError("No image column found automatically. Check full_df.csv and set img_col manually in script.")

print("Using image column:", img_col)

# label column: try 'target' or 'labels' or 'labels' like multi-hot
label_col = None
for c in ["target", "labels", "labels_text", "diagnosis"]:
    if c in df.columns:
        label_col = c
        break
if label_col is None:
    # try single-letter multi-label flags (N,D,G,...)
    for c in ["N","D","G","C","A","H","M","O"]:
        if c in df.columns:
            label_col = None  # we'll fall back to building labels from those columns
            break

print("Using label column:", label_col)

def normalize_path(val):
    s = str(val)
    # if it's already a local relative path inside ROOT, normalize it
    p = Path(s)
    if (ROOT / p).exists():
        return str((ROOT / p).relative_to(ROOT))
    # if it is absolute and exists
    if p.is_file():
        try:
            return str(p.relative_to(ROOT))
        except Exception:
            return str(p.name)
    # if path contains 'Training Images' or 'Testing Images' convert to local layout
    if "Training Images" in s or "Testing Images" in s:
        name = Path(s).name
        candidates = list(ROOT.rglob(name))
        if candidates:
            return str(candidates[0].relative_to(ROOT))
    # if it's just a filename, try to find it recursively
    name = Path(s).name
    candidates = list(ROOT.rglob(name))
    if candidates:
        return str(candidates[0].relative_to(ROOT))
    # fallback: return filename only
    return name

def parse_label(raw):
    # handle numeric
    if pd.isna(raw):
        return 0
    # already integer-like
    try:
        return int(raw)
    except Exception:
        pass
    # maybe it's a string representation of a list e.g. "[1, 0, 0, 0]"
    try:
        parsed = ast.literal_eval(str(raw))
        # if parsed is list or tuple
        if isinstance(parsed, (list, tuple, np.ndarray)):
            arr = np.array(parsed, dtype=float)
            # choose argmax (first highest) as single-label fallback
            idx = int(np.argmax(arr))
            return int(idx)
        # if parsed is dict? attempt to use first key or 'target'
    except Exception:
        pass
    # If we reach here, try to interpret as comma separated tokens: "1 0 0" or "1,0,0"
    s = str(raw).replace(",", " ").split()
    if len(s) > 1:
        try:
            arr = np.array([float(x) for x in s])
            return int(np.argmax(arr))
        except Exception:
            pass
    # fallback: 0
    return 0

rows = []
missing_img = []
for _, r in df.iterrows():
    imgref = r.get(img_col, "")
    imgpath = normalize_path(imgref)
    # determine label
    if label_col and label_col in r:
        lab_raw = r[label_col]
        lab = parse_label(lab_raw)
    else:
        # attempt to create label from multi-columns N,D,G,C,A,H,M,O by finding which column is 1
        # prefer the first one with value==1, else 0
        multi_cols = [c for c in ["N","D","G","C","A","H","M","O"] if c in df.columns]
        lab = 0
        for i, c in enumerate(multi_cols):
            try:
                if int(r[c]) == 1:
                    lab = i
                    break
            except Exception:
                continue
    # check if image exists
    if not (ROOT / imgpath).exists():
        missing_img.append(imgpath)
    rows.append({"image_path": imgpath, "label": int(lab)})

out_df = pd.DataFrame(rows)
out_df.to_csv(out, index=False)
print("Wrote normalized ODIR CSV:", out, "rows:", len(out_df))
print("Missing image count (sample <=20):", len(missing_img), "sample:", missing_img[:20])
print("Label distribution (value counts):")
print(out_df["label"].value_counts().to_dict())
print("Sample rows:\n", out_df.head(10))
