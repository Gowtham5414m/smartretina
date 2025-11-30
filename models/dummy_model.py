# models/dummy_model.py
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    m = DummyModel(num_classes=2)
    x = torch.randn(1, 3, 256, 256)  # random input
    out = m(x)
    print("Dummy forward output shape:", out.shape)
