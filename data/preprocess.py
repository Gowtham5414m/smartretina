# data/preprocess.py
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch
import sys

def load_image(path, size=256):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p.resolve()}")
    img = Image.open(p).convert("RGB")
    tf = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tf(img).unsqueeze(0)  # batch dim

if __name__ == "__main__":
    # Choose image path here. You can pass path as first arg:
    # python data/preprocess.py path/to/image.jpg
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # <--- EDIT THIS fallback to an image that exists in your project.
        # Example possibilities (change if your folder names differ):
        # "data/aptos/train_images/00001_left.jpg"
        # "data/drishti/Drishti-GS1_files/Training/some_image.jpg"
        img_path = "C:\project\smartretina\data\aptos\train_images\0a4e1a29ffff.png"

    try:
        t = load_image(img_path, size=256)
        print("Loaded:", img_path)
        print("Tensor shape:", t.shape)
    except Exception as e:
        print("ERROR:", e)
        # helpful hint for user
        print("Tip: pass a valid image path. Example:")
        print('  python data/preprocess.py "C:\project\smartretina\data\aptos\train_images\0a4e1a29ffff.png"')
