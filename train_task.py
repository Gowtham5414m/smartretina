# train_task.py  (fixed: casts numeric config values from YAML)
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as torch_optim
import pandas as pd
import os

# ---- Dataset helper (CSV with image_path,label) ----
class FundusDataset(Dataset):
    def __init__(self, root:Path, csv_path:Path, image_col='image_path', label_col='label', size=384, is_regression=False):
        self.root = Path(root)
        df = pd.read_csv(csv_path)
        # try detect columns if default names missing
        if image_col not in df.columns or label_col not in df.columns:
            img_cols = [c for c in df.columns if "image" in c.lower() or "file" in c.lower() or "name" in c.lower()]
            lab_cols = [c for c in df.columns if c.lower() in ("label","labels","diagnosis","severity","target","class","level")]
            if img_cols:
                image_col = img_cols[0]
            if lab_cols:
                label_col = lab_cols[0]
        self.items = df[[image_col, label_col]].values.tolist()
        self.size = int(size)
        self.is_regression = is_regression
        self.tf = T.Compose([T.Resize((self.size,self.size)), T.RandomHorizontalFlip(), T.RandomRotation(15), T.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_rel, label = self.items[idx]
        img = Image.open(self.root / str(img_rel)).convert("RGB")
        x = self.tf(img)
        if self.is_regression:
            return x, torch.tensor(float(label), dtype=torch.float32)
        else:
            return x, torch.tensor(int(label), dtype=torch.long)

# ---- Model factory (choose task-specific class names) ----
def get_model(task_name, num_classes=2, device='cpu'):
    if task_name == "dr":
        from models.dr import DRModel as M
        model = M(num_classes=num_classes)
    else:
        from models.dr import DRModel as M
        model = M(num_classes=num_classes)
    return model.to(device)

# ---- Training loop ----
def train_one_round(cfg_path, task_name, out_checkpoint):
    cfg = yaml.safe_load(open(cfg_path))
    dataset_path = Path(cfg["dataset_path"])
    train_csv = dataset_path / cfg.get("train_csv", "labels.csv")
    val_csv = dataset_path / cfg.get("val_csv", "val_labels.csv")

    # cast numeric config fields safely (they may be strings)
    def as_int(x, default):
        try:
            return int(x)
        except Exception:
            return default
    def as_float(x, default):
        try:
            return float(x)
        except Exception:
            return default

    image_size = as_int(cfg.get("image_size", 384), 384)
    batch_size = as_int(cfg.get("batch_size", 8), 8)
    local_epochs = as_int(cfg.get("local_epochs", 1), 1)
    lr = as_float(cfg.get("lr", 1e-4), 1e-4)
    weight_decay = as_float(cfg.get("weight_decay", 1e-5), 1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_reg = bool(cfg.get("is_regression", False))
    num_classes = as_int(cfg.get("num_classes", 2), 2)

    ds = FundusDataset(dataset_path, train_csv, size=image_size, is_regression=is_reg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = get_model(task_name, num_classes=num_classes, device=device)
    optimizer = torch_optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if is_reg:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    model.train()
    epochs = local_epochs
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits, internals = model(xb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, internals = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / max(1, len(ds))
        print(f"[{task_name}] epoch {epoch} avg_loss {avg_loss:.4f}")

    # save checkpoint as state_dict
    Path(out_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_checkpoint)
    print("Saved checkpoint:", out_checkpoint)
    return out_checkpoint

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)             # e.g. configs/client1.yaml
    p.add_argument("--task", required=True)            # e.g. dr
    p.add_argument("--out", required=True)             # e.g. checkpoints/local/client1_dr_round1.pth
    args = p.parse_args()
    train_one_round(args.cfg, args.task, args.out)
