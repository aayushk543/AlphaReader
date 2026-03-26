"""Shared constants and configuration."""

import string
import torch
from torchvision import transforms

MODEL_PATH = "alphabet_cnn.pth"
LABELS = list(string.ascii_uppercase)          # A-Z  (index 0-25)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform used for inference (resize + normalise)
TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
