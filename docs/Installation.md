# Installation Guide

## Requirements

- Python >= 3.8
- pip or conda package manager

## Installation from PyPI (Recommended)

The easiest way to install debye3d is from PyPI:

```bash
pip install debye3d
```

## Installation from Source

For development or latest features:

```bash
git clone https://github.com/nicoratel/Debye3D.git
cd Debye3D
pip install -e .
```

This will install the core dependencies:
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- ase >= 3.22.0
- pyyaml >= 5.4.0
- tqdm >= 4.60.0

## Installation with GPU Support

For GPU-accelerated calculations using PyTorch:

```bash
# From PyPI
pip install debye3d[gpu]

# From source
pip install -e ".[gpu]"
```

This adds:
- torch >= 1.9.0

**Note**: For CUDA support, you may need to install PyTorch separately following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Installation with CPU Acceleration

For fast CPU calculations using Numba:

```bash
# From PyPI
pip install debye3d[fast]

# From source
pip install -e ".[fast]"
```

This adds:
- numba >= 0.54.0

## Installation with Detector Support

For pyFAI integration (azimuthal integration):

```bash
pip install -e ".[detector]"
```

This adds:
- pyFAI >= 0.20.0

## Full Installation

To install all optional dependencies:

```bash
# From PyPI
pip install debye3d[full]

# From source
pip install -e ".[full]"
```

This includes:
- torch >= 1.9.0
- numba >= 0.54.0
- pyFAI >= 0.20.0
- debyecalculator >= 1.0.0

## Development Installation

For development with testing and code quality tools:

```bash
pip install -e ".[dev]"
```

This adds:
- pytest >= 7.0.0
- pytest-cov >= 3.0.0
- black >= 22.0.0
- flake8 >= 4.0.0
- mypy >= 0.950

## Conda Installation

If you prefer using conda:

```bash
# Create a new environment
conda create -n debye3d python=3.10
conda activate debye3d

# Install dependencies
conda install numpy scipy matplotlib ase pyyaml tqdm -c conda-forge

# Optional: GPU support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Optional: CPU acceleration
conda install numba -c conda-forge

# Install debye3d
cd /path/to/debye3d
pip install -e .
```

## Verification

Test your installation:

```python
import debye3d
print(f"Debye3D version: {debye3d.__version__}")

from debye3d import Experiment, Debye3D
print("Installation successful!")
```

## Troubleshooting

### ImportError for torch

If you get GPU-related import errors but don't need GPU support, the package will automatically fall back to CPU mode.

### PyFAI installation issues

pyFAI can be tricky to install. If you encounter issues:

```bash
conda install pyfai -c conda-forge
```

### Numba compilation errors

Ensure you have a compatible compiler installed. On Linux:

```bash
sudo apt-get install gcc
```

## Next Steps

Once installed, proceed to the [Quick Start guide](Quick-Start.md).
