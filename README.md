# OCR Scanner

A handwritten character recognition system built with PyTorch. Trains a CNN on the **EMNIST-Letters** dataset and uses it to recognise A–Z characters from real images.

## Project Structure

```
OCR/
├── Scanner.py   # CLI entry point
├── config.py    # Shared constants (device, paths, transforms)
├── model.py     # AlphabetCNN architecture
├── train.py     # Training & evaluation on EMNIST
├── predict.py   # Single-character inference
├── scan.py      # Full-image segmentation & OCR
└── README.md
```

## Setup

### Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- NumPy
- OpenCV *(only for the `scan` command)*

### Install Dependencies

```bash
pip install torch torchvision pillow numpy opencv-python
```

## Usage

All commands are accessed through `Scanner.py`:

### 1. Train the Model

```bash
python Scanner.py train
```

| Flag           | Default | Description              |
|----------------|---------|--------------------------|
| `--epochs`     | 5       | Number of training epochs |
| `--batch-size` | 64      | Training batch size       |
| `--lr`         | 0.001   | Learning rate             |

```bash
# Example: train for 10 epochs with a smaller learning rate
python Scanner.py train --epochs 10 --lr 0.0005
```

The trained weights are saved to `alphabet_cnn.pth`.

### 2. Predict a Single Character

```bash
python Scanner.py predict <image_path>
```

```bash
python Scanner.py predict letter_a.png
#    Predicted character: A
#    Confidence: 98.3%
```

### 3. Scan a Full Image

```bash
python Scanner.py scan <image_path>
```

This segments individual characters from the image using adaptive thresholding and contour detection, groups them into lines, and prints the recognised text.

```bash
python Scanner.py scan handwritten_note.png
#    Detected 12 character(s):
#
#    HELLO
#    WORLD
```

## Model Architecture

**AlphabetCNN** — a lightweight CNN for 28×28 greyscale character images:

| Layer         | Details              |
|---------------|----------------------|
| Conv2d        | 1 → 32, kernel 3×3  |
| Conv2d        | 32 → 64, kernel 3×3 |
| MaxPool2d     | 2×2                 |
| Dropout       | 0.25                |
| Linear        | 9216 → 128          |
| Linear        | 128 → 26 (A–Z)      |

## How It Works

1. **Training** — The model is trained on EMNIST-Letters (26 classes, A–Z) with cross-entropy loss and Adam optimiser.
2. **Preprocessing** — Input images are converted to greyscale, resized to 28×28, transposed (to match EMNIST orientation), and normalised.
3. **Scanning** — OpenCV applies adaptive thresholding and finds contours to isolate individual characters. Each crop is fed through the CNN for prediction. Characters are grouped into lines based on vertical position, and word boundaries are detected via horizontal gaps.
