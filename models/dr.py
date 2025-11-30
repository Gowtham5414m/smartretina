# models/dr.py
from models.dummy_model import DummyModel
import torch.nn as nn

class DRModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = DummyModel(num_classes=num_classes)

    def forward(self, x):
        logits = self.backbone(x)
        internals = None
        return logits, internals
