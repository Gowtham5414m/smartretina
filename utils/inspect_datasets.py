# utils/inspect_datasets.py  (fixed)
import json
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os
import statistics

DATA_ROOTS = {
    "aptos": Path("data/aptos"),
    "messidor": Path("data/messidor"),
    "odir": Path("data/odir"),
    "drishti": Path("data/drishti"),
}

REPORT_JSON = Path("utils/dataset_report.json")
REPORT_MD = Path("utils/dataset_report.md")

def inspect_images(folder):
    exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    files = [p for e in exts for p in folder.rglob(f"*{e}")]
    sizes = []
    corrupt = []
    sample_files = files[:5]
    for f in files:
        try:
            with Image.open(f) as im:
                sizes.append(im.size)  # (W,H)
        except (UnidentifiedImageError, OSError):
            corrupt.append(str(f))
    widths = [s[0] for s in sizes] if sizes else []
    heights = [s[1] for s in sizes] if sizes else []
    return {
        "file_count": len(files),
        "sample_files": [str(p.relative_to(folder)) for p in sample_files],
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "width_median": int(statistics.median(widths)) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "height_median": int(statistics.median(heights)) if heights else None,
        "corrupt_files_count": len(corrupt),
        "corrupt_files": corrupt[:10],
    }

def inspect_csv(csv_path, root_path):
    if not csv_path.exists():
        return {"exists": False}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"exists": True, "error": str(e)}
    cols = list(df.columns)
    # try to detect image column
    img_cols = [c for c in cols if "image" in c.lower() or "file" in c.lower() or "path" in c.lower() or "filename" in c.lower()]
    label_cols = [c for c in cols if c.lower() in ("label","labels","diagnosis","diagnose","severity","target","class","level")]
    # sample rows
    sample = df.head(5).to_dict(orient="records")
    # class balance if label exists and discrete
    balance = {}
    if label_cols:
        lab = label_cols[0]
        try:
            counts = df[lab].value_counts().to_dict()
            balance = counts
        except Exception:
            balance = {}
    # verify that image files referenced exist
    missing = []
    if img_cols:
        imgc = img_cols[0]
        # modern pandas: use .items()
        for i, val in df[imgc].head(200).items():  # quick check first 200
            p = root_path / str(val)
            if not p.exists():
                missing.append(str(val))
        missing = missing[:20]
    return {"exists": True, "columns": cols, "image_column_candidates": img_cols, "label_column_candidates": label_cols, "sample_rows": sample, "balance_sample": balance, "missing_sample": missing}

def main():
    report = {}
    for name, root in DATA_ROOTS.items():
        info = {"root": str(root), "exists": root.exists()}
        if not root.exists():
            report[name] = info
            continue
        # Inspect images in common subfolders (train_images, images, Training, etc.)
        candidates = ["train_images", "train", "images", "Training", ""]
        imgs_summary = {}
        for c in candidates:
            folder = root / c if c else root
            if folder.exists():
                imgs_summary[c if c else "root"] = inspect_images(folder)
        info["images"] = imgs_summary

        # Inspect CSVs commonly found
        csvs = ["train.csv", "labels.csv", "full_df.csv", "train_labels.csv", "val.csv", "test.csv"]
        csv_report = {}
        for csv in csvs:
            p = root / csv
            csv_report[csv] = inspect_csv(p, root)
        info["csvs"] = csv_report

        # rough disk size
        total_bytes = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        info["size_mb"] = round(total_bytes / (1024*1024), 2)
        report[name] = info

    # Save json + md
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    # Also write a short markdown report
    md = ["# Dataset inspection report\n"]
    for k, v in report.items():
        md.append(f"## {k}\n")
        if not v.get("exists"):
            md.append(f"- Path {v['root']} does not exist\n")
            continue
        md.append(f"- size (MB): {v.get('size_mb')}\n")
        md.append("- Image folders inspected:\n")
        for fkey, fsummary in v["images"].items():
            md.append(f"  - `{fkey}`: {fsummary['file_count']} images; sample: {fsummary['sample_files']}\n")
            md.append(f"    - size (w x h): min {fsummary['width_min']} max {fsummary['width_max']} median ({fsummary['width_median']}, {fsummary['height_median']})\n")
            md.append(f"    - corrupt files (count): {fsummary['corrupt_files_count']}\n")
        md.append("- CSVs:\n")
        for csvname, crec in v["csvs"].items():
            md.append(f"  - `{csvname}`: exists={crec.get('exists')}; cols={crec.get('columns') if crec.get('exists') else 'N/A'}; sample missing refs: {crec.get('missing_sample')}\n")
        md.append("\n")
    REPORT_MD.write_text("\n".join(md))
    print("Report saved to", REPORT_JSON, "and", REPORT_MD)
    print("Printed summary below:\n")
    print(REPORT_MD.read_text())

if __name__ == "__main__":
    main()
