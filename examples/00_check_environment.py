"""Print the PyTorch environment and run a small tensor operation."""

import platform

import torch
import torchvision


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Python      : {platform.python_version()}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")
    print(f"CUDA build  : {torch.version.cuda}")
    print(f"CUDA ready  : {torch.cuda.is_available()}")
    print(f"Device      : {device}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    result = (a @ b).mean()
    print(f"Matrix test : OK (mean={result.item():.6f})")


if __name__ == "__main__":
    main()
