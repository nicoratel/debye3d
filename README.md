# Debye3D

3D scattering calculator using the Debye equation for X-ray and electron scattering.

## Description

`debye3d` is a Python package for computing 2D projections of 3D scattering patterns from atomic structures using the Debye scattering equation. It supports GPU (PyTorch) and CPU (Numba) acceleration for high-performance calculations.

## Installation

### Installation from PyPI

```bash
pip install debye3d
```

### Installation with GPU acceleration

```bash
pip install debye3d[gpu]
```

### Installation with all features

```bash
pip install debye3d[full]
```

### Installation from source (development)

```bash
git clone https://github.com/nicoratel/Debye3D.git
cd Debye3D
pip install -e .
```

## Dependencies

### Required
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- ase >= 3.22.0
- pyyaml >= 5.4.0
- tqdm >= 4.60.0

### Optional
- **GPU**: torch >= 1.9.0
- **CPU acceleration**: numba >= 0.54.0
- **Detectors**: pyFAI >= 0.20.0
- **DebyeCalculator**: debyecalculator >= 1.0.0

## Usage

Refer to Jupyter notebooks for application examples

## Key Features

- 🔬 3D scattering calculation using Debye equation
- ⚡ GPU support (PyTorch) and CPU acceleration (Numba)
- 🎯 Atomic form factors (Cromer-Mann and Lobato parametrizations)
- 🌐 Orientation averaging with adaptive Fibonacci grid
- 📊 Integration with pyFAI for detector simulations
- 💾 Utilities for saving and visualizing data

## Package Structure

```
debye3d/
├── debye3d.py                          # Main Debye calculation functions
├── compute_f0.py                       # Atomic form factors
├── lobato_scattering.py                # Lobato parametrizations
├── adaptative_fibonacci.py             # Adaptive Fibonacci grid
├── generate_paracrystal_assembly.py    # Supercell generation
├── utilities.py                        # Utilities (I/O, visualization)
└── elements_info.yaml                  # Element database
```

