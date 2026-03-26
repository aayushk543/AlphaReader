# AlphaReader

Handwritten character recognition using PyTorch. Trains a CNN on the EMNIST-Letters dataset to recognise A-Z characters from images.

## Structure

```
├── Scanner.py       # CLI entry point
├── config.py        # Constants and transforms
├── model.py         # AlphabetCNN architecture
├── train.py         # Training and evaluation
├── predict.py       # Single character prediction
├── scan.py          # Full image segmentation and OCR
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python Scanner.py train
python Scanner.py train --epochs 10 --lr 0.0005
```

**Predict a single character:**
```bash
python Scanner.py predict letter.png
```

**Scan a full image:**
```bash
python Scanner.py scan page.png
```

## Model

AlphabetCNN - a CNN for 28x28 greyscale character images:

| Layer     | Details            |
|-----------|--------------------|
| Conv2d    | 1 → 32, kernel 3x3 |
| Conv2d    | 32 → 64, kernel 3x3 |
| MaxPool2d | 2x2               |
| Dropout   | 0.25              |
| Linear    | 9216 → 128        |
| Linear    | 128 → 26 (A-Z)    |

Trained on EMNIST-Letters with cross-entropy loss and Adam optimiser. For scanning, OpenCV segments characters via adaptive thresholding and contour detection, then each crop is classified by the CNN.
