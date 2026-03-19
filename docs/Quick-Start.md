# Quick Start Guide

This guide will get you started with Debye3D in minutes.

## Basic Workflow

The typical workflow involves:

1. Define experimental setup (`Experiment`)
2. Load atomic structure
3. Create `Debye3D` calculator
4. Compute scattering intensity
5. Visualize and analyze results

## Example: Simple 2D Scattering Pattern

```python
from debye3d import Experiment, Debye3D
from ase.cluster import Decahedron
import matplotlib.pyplot as plt

# 1. Define experimental setup
exp = Experiment(
    wl=1.54,           # Wavelength in Angstroms (Cu K-alpha)
    distance=0.36,     # Sample-detector distance in meters
    npix=500,          # Number of pixels (500x500)
    pixel_size=0.0006  # Pixel size in meters (600 μm)
)

# 2. Create atomic structure
atoms = Decahedron('Au', 5, 25, 0)
atoms.write('structure.xyz')

# 3. Create Debye3D calculator
d3d = Debye3D(
    'structure.xyz',
    wl=exp.wl,
    distance=exp.distance,
    npix=exp.npix,
    pixel_size=exp.pixel_size
)

# 4. Compute 2D scattering intensity
I = d3d.compute_intensity()

# 5. Visualize
d3d.plot_intensity(I, vmin=-6)
plt.show()
```

## Example: Isotropic Powder Pattern

```python
# Compute orientation-averaged intensity
q_iso, i_iso = d3d.compute_isotropic_intensity_fibonacci(
    n_orient=500  # Number of orientations for averaging
)

# Plot I(q)
plt.figure()
plt.loglog(q_iso, i_iso)
plt.xlabel('q (Å⁻¹)')
plt.ylabel('Intensity')
plt.title('Isotropic Scattering Pattern')
plt.show()
```

## Example: Rotating the Structure

```python
# Rotate structure by 90° around Y-axis
d3d.rotate_positions(alpha=0, beta=90, gamma=0)

# Compute new scattering pattern
I_rotated = d3d.compute_intensity()
d3d.plot_intensity(I_rotated, vmin=-6)
```

## Example: Integration with pyFAI

```python
# Azimuthal integration to get I(q)
q, i = d3d.integrate_with_pyfai(I)

# Plot
plt.figure()
plt.loglog(q, i)
plt.xlabel('q (Å⁻¹)')
plt.ylabel('Intensity')
plt.show()
```

## Example: Using GPU Acceleration

```python
# Automatic GPU detection
d3d = Debye3D('structure.xyz', wl=1.54, distance=0.36, npix=500, pixel_size=0.0006)

# Compute with GPU if available
I = d3d.compute_intensity(use_gpu=True)
```

## Example: Save and Load Results

```python
from debye3d import save_intensity_npz, plot_from_npz

# Save intensity map
save_intensity_npz('output.npz', I, d3d.Qx, d3d.Qz)

# Load and plot
plot_from_npz('output.npz')
```

## Common Parameters

### Experiment Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `wl` | X-ray wavelength (Å) | 1.54 (Cu Kα), 0.71 (Mo Kα) |
| `distance` | Sample-detector distance (m) | 0.1 - 1.0 |
| `npix` | Detector pixels | 256, 512, 1024 |
| `pixel_size` | Pixel size (m) | 5e-5 - 1e-3 |

### Debye3D Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `structure_file` | Path to XYZ file | required |
| `verbose` | Print information | False |
| `torch_device` | GPU device | 'cuda' if available |
| `scattering_type` | 'xray' or 'electron' | 'xray' |

## Next Steps

- Explore [detailed examples](Examples.md)
- Check the [API reference](API-Reference.md)
- Learn about [advanced features](Advanced-Usage.md)
