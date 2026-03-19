# Changelog

All notable changes to debye3d will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-19

### Added
- Initial release of debye3d
- 3D scattering calculation using Debye equation
- Support for X-ray and electron scattering
- GPU acceleration via PyTorch
- CPU acceleration via Numba
- Atomic form factors (Cromer-Mann and Lobato parametrizations)
- Orientation averaging with adaptive Fibonacci grid
- Integration with pyFAI for detector simulations
- Utilities for saving and visualizing scattering data
- Elements database (elements_info.yaml)
- Paracrystal assembly generation
- Full documentation (README, Installation guide, API reference)

### Core modules
- `debye3d.py` - Main Debye calculation functions
- `compute_f0.py` - Atomic form factors
- `lobato_scattering.py` - Lobato parametrizations
- `adaptative_fibonacci.py` - Adaptive Fibonacci grid for orientation averaging
- `generate_paracrystal_assembly.py` - Supercell generation utilities
- `utilities.py` - I/O and visualization utilities

[Unreleased]: https://github.com/nicoratel/Debye3D/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nicoratel/Debye3D/releases/tag/v0.1.0
