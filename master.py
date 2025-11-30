# master.py  (DEBUG-ready)
import argparse
from pathlib import Path
import sys
print("master.py start - Python:", sys.executable)   # debug: which python runs

def train():
    print("Entered train()")
    from data.preprocess import load_image
    from models.dummy_model import DummyModel
    import torch

    train_folder = Path("data/aptos/train_images")
    print("Looking for images in:", train_folder.resolve())
    imgs = list(train_folder.glob("*.jpg")) + list(train_folder.glob("*.png"))
    print("Found images count:", len(imgs))
    if len(imgs) == 0:
        print("NO IMAGE FOUND in", train_folder)
        return

    img_path = imgs[0]
    print("Using sample image:", img_path)

    model = DummyModel()
    x = load_image(str(img_path), size=256)
    print("Loaded tensor shape:", x.shape)
    y = torch.tensor([0])

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        logits = model(x)
        loss = loss_fn(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")

def infer():
    print("Entered infer()")
    # small debug stub
    print("Infer stub - nothing implemented yet")

if __name__ == "__main__":
    print("main guard reached")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","infer"], default="train")
    args = parser.parse_args()
    print("mode chosen:", args.mode)
    if args.mode == "train":
        train()
    else:
        infer()
    print("master.py finished")
