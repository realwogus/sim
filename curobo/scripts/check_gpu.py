#!/usr/bin/env python3
"""Check that CUDA, PyTorch, and cuRobo are usable in the container."""

import torch

import curobo


def main() -> None:
    print(f"cuRobo={curobo.__version__}")
    print(f"PyTorch={torch.__version__} CUDA={torch.version.cuda}")
    print(f"CUDA available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available inside the container")
    print(f"GPU={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
