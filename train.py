"""Train AlphabetCNN on the EMNIST-Letters dataset."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from config import DEVICE, MODEL_PATH
from model import AlphabetCNN


def train_model(epochs: int = 5, batch_size: int = 64, lr: float = 0.001):
    """Train the model, evaluate on the test set, and save weights."""
    print(f"Running on: {DEVICE}\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_dataset = datasets.EMNIST(root='./data', split='letters',
                                    train=True, download=True,
                                    transform=transform)
    test_dataset  = datasets.EMNIST(root='./data', split='letters',
                                    train=False, download=True,
                                    transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1000,
                                               shuffle=False)

    print(f"Training samples: {len(train_dataset)} | "
          f"Test samples: {len(test_dataset)}\n")

    model     = AlphabetCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── training loop ──
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = target - 1              # EMNIST labels 1-26 → 0-25

            optimizer.zero_grad()
            outputs = model(data)
            loss    = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] — Loss: {avg_loss:.4f}")

    # ── evaluation ──
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            target = target - 1
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total   += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")

    # ── save ──
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved as '{MODEL_PATH}'")


if __name__ == "__main__":
    train_model()
