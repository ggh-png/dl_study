"""Train an RNN, GRU, or LSTM to predict the next value of a sine wave."""

import argparse
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SequenceRegressor(nn.Module):
    def __init__(self, cell: str, hidden_size: int) -> None:
        super().__init__()
        cells = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        self.recurrent = cells[cell](1, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(x)
        return self.output(sequence[:, -1]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("rnn", "gru", "lstm"), default="rnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--samples", type=int, default=5_000)
    parser.add_argument("--output-dir", type=Path, default=Path("/outputs/rnn"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a GPU")
    return torch.device(requested)


def make_dataset(samples: int, sequence_length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.rand(samples, generator=generator) * (12 * math.pi)
    offsets = torch.linspace(0, 2.5, sequence_length + 1)
    waves = torch.sin(starts[:, None] + offsets[None, :])
    noise = torch.randn(waves.shape, generator=generator) * 0.03
    waves = waves + noise
    return waves[:, :-1, None], waves[:, -1]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, cell={args.cell}, epochs={args.epochs}")

    inputs, targets = make_dataset(args.samples, args.sequence_length, args.seed)
    split = int(len(inputs) * 0.8)
    train_loader = DataLoader(
        TensorDataset(inputs[:split], targets[:split]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_inputs, test_targets = inputs[split:], targets[split:]

    model = SequenceRegressor(args.cell, args.hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_inputs.size(0)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_mse={total_loss / split:.6f}")

    model.eval()
    with torch.inference_mode():
        predictions = model(test_inputs.to(device)).cpu()
    rmse = torch.sqrt(nn.functional.mse_loss(predictions, test_targets)).item()
    print(f"test_rmse={rmse:.6f}")

    model_path = args.output_dir / f"sine_{args.cell}.pt"
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, model_path)

    count = min(200, len(test_targets))
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.plot(test_targets[:count].numpy(), label="target", linewidth=2)
    axis.plot(predictions[:count].numpy(), label="prediction", alpha=0.8)
    axis.set(title=f"{args.cell.upper()} sine-wave next-value prediction", xlabel="test sample")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f"sine_{args.cell}_predictions.png", dpi=150)
    plt.close(fig)
    print(f"saved={model_path}")


if __name__ == "__main__":
    main()
