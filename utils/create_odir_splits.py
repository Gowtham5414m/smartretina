# utils/create_odir_splits.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("data/odir")
csv_in = ROOT / "train_norm.csv"
out_train = ROOT / "train_split.csv"
out_val = ROOT / "val_split.csv"

print("Reading:", csv_in)
df = pd.read_csv(csv_in)
print("Rows:", len(df))
# stratified split by label (works since label is integer 0..7)
train, val = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
train.to_csv(out_train, index=False)
val.to_csv(out_val, index=False)
print("Wrote:", out_train, "rows:", len(train))
print("Wrote:", out_val, "rows:", len(val))
