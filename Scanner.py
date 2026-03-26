"""
OCR Scanner — Train an alphabet CNN on EMNIST and use it to recognise
handwritten / printed characters from real images.

Usage:
    python Scanner.py train              # Train the model and save weights
    python Scanner.py predict <image>    # Predict a single character
    python Scanner.py scan <image>       # Segment & recognise all characters
"""

import argparse
import sys

from train import train_model
from predict import predict_single
from scan import scan_image


def main():
    parser = argparse.ArgumentParser(
        description="OCR Scanner — Train, predict, or scan handwritten text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python Scanner.py train                        # Train the model (EMNIST)
  python Scanner.py train --epochs 10            # Train for more epochs
  python Scanner.py predict letter_a.png         # Predict one character
  python Scanner.py scan   handwritten_note.png  # Scan full image
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── train ──
    train_parser = subparsers.add_parser("train", help="Train the AlphabetCNN model")
    train_parser.add_argument("--epochs", type=int, default=5,
                              help="Number of training epochs (default: 5)")
    train_parser.add_argument("--batch-size", type=int, default=64,
                              help="Training batch size (default: 64)")
    train_parser.add_argument("--lr", type=float, default=0.001,
                              help="Learning rate (default: 0.001)")

    # ── predict ──
    predict_parser = subparsers.add_parser("predict",
                                           help="Predict a single character")
    predict_parser.add_argument("image", help="Path to a character image")

    # ── scan ──
    scan_parser = subparsers.add_parser("scan",
                                        help="Scan & recognise text from an image")
    scan_parser.add_argument("image", help="Path to the image to scan")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.command == "predict":
        predict_single(args.image)
    elif args.command == "scan":
        scan_image(args.image)


if __name__ == "__main__":
    main()
