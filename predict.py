"""Inference utilities — load model and predict characters from images."""

import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from config import DEVICE, MODEL_PATH, LABELS, TRANSFORM
from model import AlphabetCNN


# ──────────────────────────── helpers ──────────────────────────────
def load_model() -> AlphabetCNN:
    """Load saved model weights and return the model in eval mode."""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file '{MODEL_PATH}' not found. "
              f"Run `python train.py` first.")
        sys.exit(1)

    model = AlphabetCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,
                                     weights_only=True))
    model.eval()
    return model


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a model-ready tensor (1, 1, 28, 28).

    EMNIST images are stored transposed, so we transpose the input to
    match the distribution the model was trained on.
    """
    img = img.convert("L")
    img = img.transpose(Image.TRANSPOSE)
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    return tensor


def predict_character(model: AlphabetCNN, img: Image.Image):
    """Return (predicted_letter, confidence%) for a single character image."""
    tensor = preprocess_image(img)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)
    letter     = LABELS[idx.item()]
    confidence = conf.item() * 100
    return letter, confidence


# ──────────────────────────── single-char CLI ─────────────────────
def predict_single(image_path: str):
    """Load an image and predict the single character it contains."""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    model = load_model()
    img   = Image.open(image_path)
    letter, confidence = predict_character(model, img)

    print(f"\n🔍 Predicted character: {letter}")
    print(f"   Confidence: {confidence:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    predict_single(sys.argv[1])
