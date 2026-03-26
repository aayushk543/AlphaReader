import os
import sys

import numpy as np
from PIL import Image

from predict import load_model, predict_character


def segment_characters(image_path):
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for scanning. Install it: pip install opencv-python")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)

    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_h, img_w = grey.shape
    min_area = (img_h * img_w) * 0.0005
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area and h > 5 and w > 5:
            boxes.append((x, y, w, h))

    if not boxes:
        print("No characters detected in the image.")
        return [], []

    line_height = np.median([h for _, _, _, h in boxes])
    boxes.sort(key=lambda b: (round(b[1] / line_height), b[0]))

    characters = []
    for (x, y, w, h) in boxes:
        pad = max(4, int(0.15 * max(w, h)))
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, img_w)
        y1 = min(y + h + pad, img_h)

        crop = grey[y0:y1, x0:x1]
        characters.append(Image.fromarray(crop))

    return characters, boxes


def scan_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model = load_model()
    characters, boxes = segment_characters(image_path)

    if not characters:
        return

    print(f"\nDetected {len(characters)} character(s):\n")

    line_height = np.median([h for _, _, _, h in boxes])
    results = []
    current_line = []
    prev_row = None

    for char_img, (x, y, w, h) in zip(characters, boxes):
        row = round(y / line_height)
        if prev_row is not None and row != prev_row:
            results.append(current_line)
            current_line = []
        letter, conf = predict_character(model, char_img)
        current_line.append((letter, conf, x))
        prev_row = row

    if current_line:
        results.append(current_line)

    for line_chars in results:
        line_text = ""
        for i, (letter, conf, x) in enumerate(line_chars):
            if i > 0:
                prev_x = line_chars[i - 1][2]
                gap = x - prev_x
                avg_char_width = np.mean([b[2] for b in boxes]) if boxes else 20
                if gap > avg_char_width * 1.5:
                    line_text += " "
            line_text += letter
        print(f"   {line_text}")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan.py <image_path>")
        sys.exit(1)
    scan_image(sys.argv[1])
