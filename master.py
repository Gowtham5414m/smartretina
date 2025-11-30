import argparse

def train():
    print("Training mode — add training loop here")

def infer():
    print("Infer mode — add inference here")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","infer"], default="train")
    args = parser.parse_args()
    if args.mode == "train":
        train()
    else:
        infer()
