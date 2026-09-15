"""Train a small convolutional neural network on MNIST."""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limit-train", type=int, default=20_000,
                        help="0 uses the full training set")
    parser.add_argument("--limit-test", type=int, default=5_000,
                        help="0 uses the full test set")
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/outputs/cnn"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU")
    return torch.device(requested)


def maybe_subset(dataset, limit: int, seed: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def save_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, output_path: Path
) -> None:
    images, labels = next(iter(loader))
    with torch.inference_mode():
        predictions = model(images.to(device)).argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for axis, image, label, prediction in zip(
        axes.flat, images[:10], labels[:10], predictions[:10]
    ):
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(f"true={label.item()}, pred={prediction.item()}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, epochs={args.epochs}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_set = maybe_subset(train_set, args.limit_train, args.seed)
    test_set = maybe_subset(test_set, args.limit_test, args.seed + 1)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 2,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={running_loss / len(train_set):.4f} "
            f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.2%}"
        )

    model_path = args.output_dir / "mnist_cnn.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, model_path)
    save_predictions(model, test_loader, device, args.output_dir / "predictions.png")
    print(f"saved={model_path}")


if __name__ == "__main__":
    main()
