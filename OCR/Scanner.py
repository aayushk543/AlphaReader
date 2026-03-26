import argparse
import sys

from train import train_model
from predict import predict_single
from scan import scan_image


def main():
    parser = argparse.ArgumentParser(description="OCR Scanner")

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=0.001)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("image")

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("image")

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
