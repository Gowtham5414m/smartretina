# utils/merge_odir_drishti.py
"""
Merge ODIR + Drishti into data/client3_combined.
Creates symlinks (preferred) or copies images into data/client3_combined/images/,
and writes train.csv, train_split.csv, val_split.csv with columns: image_path,label.

Usage:
    python utils/merge_odir_drishti.py         # default: symlink mode
    python utils/merge_odir_drishti.py --mode copy   # force copying files instead of symlinks

Notes:
- Requires pandas, sklearn (for splitting).
- Symlinks require appropriate permissions on Windows; if symlink creation fails, the script falls back to copying.
"""
from pathlib import Path
import pandas as pd
import shutil
import argparse
import sys
import os
from sklearn.model_selection import train_test_split

ROOT = Path(".").resolve()
ODIR_ROOT = ROOT / "data" / "odir"
DRISHTI_ROOT = ROOT / "data" / "drishti"
COMBINED_ROOT = ROOT / "data" / "client3_combined"
COMBINED_IMAGES = COMBINED_ROOT / "images"
ODIR_NORM = ODIR_ROOT / "train_norm.csv"

def read_odir():
    if not ODIR_NORM.exists():
        raise FileNotFoundError(f"{ODIR_NORM} not found. Run normalize_odir.py first.")
    df = pd.read_csv(ODIR_NORM)
    # image paths in ODIR CSV are relative to data/odir; convert to absolute source paths
    def src_path(rel):
        p = ODIR_ROOT / Path(rel)
        if p.exists():
            return p
        # try searching by name
        name = Path(rel).name
        found = list(ODIR_ROOT.rglob(name))
        return found[0] if found else p
    df["src"] = df["image_path"].apply(src_path)
    # normalize label to int
    df["label"] = df["label"].astype(int)
    return df[["src","label"]]

def read_drishti():
    """
    Heuristic: find any CSV or XLSX in data/drishti that maps filenames to labels.
    If found, use it. Else look for Training images and include them with label 0 (warn).
    """
    dr = DRISHTI_ROOT
    if not dr.exists():
        print("Drishti folder not found:", dr)
        return pd.DataFrame(columns=["src","label"])
    # look for CSVs/XLSX
    files = list(dr.rglob("*.csv")) + list(dr.rglob("*.xlsx")) + list(dr.rglob("*.xls"))
    # prefer CSV first
    chosen = None
    for f in files:
        name = f.name.lower()
        if "diagnos" in name or "label" in name or "train" in name:
            chosen = f
            break
    if chosen is None and files:
        chosen = files[0]
    if chosen:
        print("Found Drishti label file:", chosen)
        # read with pandas
        try:
            if chosen.suffix.lower() in (".xls", ".xlsx"):
                df0 = pd.read_excel(chosen)
            else:
                df0 = pd.read_csv(chosen)
            cols = list(df0.columns)
            print("Drishti label file columns:", cols[:20])
            # find a filename column
            fname_col = None
            for c in cols:
                if any(k in c.lower() for k in ("file","image","name","id","filename")):
                    fname_col = c
                    break
            label_col = None
            for c in cols:
                if any(k in c.lower() for k in ("label","diagnos","grade","class","target")):
                    label_col = c
                    break
            if fname_col is None:
                # try to detect image folder layout - fallback
                print("Could not detect filename column in Drishti label file; will attempt to find images in Training/ folder.")
            else:
                print("Using Drishti filename column:", fname_col, "label column:", label_col)
                # build rows
                rows = []
                for _, r in df0.iterrows():
                    fname = str(r[fname_col])
                    # try to find file under Drishti root
                    candidates = list(dr.rglob(fname))
                    if not candidates:
                        # try common extensions
                        base = Path(fname).stem
                        found = []
                        for ext in [".png",".jpg",".jpeg",".tif",".tiff"]:
                            found = list(dr.rglob(base + ext))
                            if found:
                                candidates = found
                                break
                    if candidates:
                        src = candidates[0]
                        if label_col and label_col in r:
                            lab = r[label_col]
                            try:
                                lab = int(lab)
                            except Exception:
                                # if string label, map later or set 0 placeholder
                                lab = str(lab)
                        else:
                            lab = 0
                        rows.append({"src": src, "label": lab})
                dfd = pd.DataFrame(rows)
                print("Drishti mapped rows from label file:", len(dfd))
                return dfd
        except Exception as e:
            print("Error reading Drishti label file:", e)
    # Fallback: include images under Training folder with label 0
    print("No usable label file detected for Drishti. Scanning Training folder and assigning label 0 (UNLABELED).")
    imgs = []
    for p in dr.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff"):
            # prefer Training subfolder
            if "training" in str(p).lower() or "train" in str(p).lower():
                imgs.append(p)
    if not imgs:
        # take any images
        imgs = [p for p in dr.rglob("*") if p.is_file() and p.suffix.lower() in (".png",".jpg",".jpeg",".tif",".tiff")]
    dfr = pd.DataFrame([{"src":p, "label":0} for p in imgs])
    print("Drishti fallback rows:", len(dfr))
    return dfr

def ensure_dir(p:Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src:Path, dst:Path, mode="symlink"):
    try:
        if mode == "symlink":
            # remove dst if exists
            if dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
            return "symlink"
        else:
            # copy
            shutil.copy2(src, dst)
            return "copy"
    except Exception as e:
        # fallback to copy
        try:
            shutil.copy2(src, dst)
            return "copy"
        except Exception as e2:
            return f"error:{e2}"

def main(mode="symlink"):
    print("Mode:", mode)
    odir = read_odir()
    drishti = read_drishti()

    print("ODIR rows:", len(odir))
    print("Drishti rows:", len(drishti))

    # concat
    combined = pd.concat([odir, drishti], ignore_index=True)
    # drop rows with missing src files
    combined = combined[combined["src"].apply(lambda p: Path(p).exists())]
    print("Combined rows after removing missing files:", len(combined))

    # prepare combined images folder
    ensure_dir(COMBINED_IMAGES)

    # fill combined images by creating symlink or copies
    new_rel_paths = []
    seen_names = set()
    actions = {"symlink":0, "copy":0}
    for idx, row in combined.iterrows():
        src: Path = Path(row["src"])
        lab = row["label"]
        # dest filename: use a prefix to avoid name clashes: odir_0001_right.jpg or drishti_dr1.png etc
        src_name = src.name
        src_parent = src.parts[-3] if len(src.parts) >=3 else src.parent.name
        # produce unique name if collision
        base = f"{src_parent}_{src_name}"
        if base in seen_names:
            # add idx
            base = f"{src_parent}_{idx}_{src_name}"
        seen_names.add(base)
        dst = COMBINED_IMAGES / base
        result = link_or_copy(src, dst, mode=mode)
        if result.startswith("error"):
            print("Failed to link/copy", src, "->", dst, result)
        else:
            actions[result] = actions.get(result,0)+1
        # store path relative to COMBINED_ROOT
        new_rel_paths.append((Path("images")/base, int(lab) if (isinstance(lab,(int,float)) or str(lab).isdigit()) else str(lab)))

    # build final dataframe
    final_df = pd.DataFrame(new_rel_paths, columns=["image_path","label"])
    # try to cast label to int if possible
    try:
        final_df["label"] = final_df["label"].astype(int)
    except Exception:
        pass

    # save full CSV
    ensure_dir(COMBINED_ROOT)
    full_csv = COMBINED_ROOT / "train.csv"
    final_df.to_csv(full_csv, index=False)
    print("Wrote combined train.csv:", full_csv, "rows:", len(final_df))
    # show label distribution
    try:
        print("Label counts:", final_df["label"].value_counts().to_dict())
    except Exception:
        print("Label column is non-numeric sample:", final_df["label"].head(10).tolist())

    # create stratified splits if labels numeric
    if final_df["label"].dtype.kind in "iubc":
        train, val = train_test_split(final_df, test_size=0.1, random_state=42, stratify=final_df["label"])
        train.to_csv(COMBINED_ROOT / "train_split.csv", index=False)
        val.to_csv(COMBINED_ROOT / "val_split.csv", index=False)
        print("Wrote train_split.csv and val_split.csv (stratified). Rows:", len(train), len(val))
    else:
        # simple split without stratify
        train = final_df.sample(frac=0.9, random_state=42)
        val = final_df.drop(train.index)
        train.to_csv(COMBINED_ROOT / "train_split.csv", index=False)
        val.to_csv(COMBINED_ROOT / "val_split.csv", index=False)
        print("Wrote train_split.csv and val_split.csv (non-stratified). Rows:", len(train), len(val))

    print("Link/copy summary:", actions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("symlink","copy"), default="symlink", help="use symlink (fast) or copy images into combined folder")
    args = parser.parse_args()
    try:
        main(mode=args.mode)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
