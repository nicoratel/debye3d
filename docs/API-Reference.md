# API Reference

Complete reference for all classes and functions in Debye3D.

## Core Classes

### Experiment

```python
class Experiment(npix=250, wl=1.0, distance=0.5, pixel_size=0.0001, verbose=True)
```

Defines the experimental setup including detector geometry and accessible Q-range.

**Parameters:**
- `npix` (int): Number of detector pixels (creates npix × npix detector)
- `wl` (float): X-ray wavelength in Angstroms
- `distance` (float): Sample-to-detector distance in meters
- `pixel_size` (float): Detector pixel size in meters
- `verbose` (bool): Print setup information

**Attributes:**
- `Qx`, `Qz` (ndarray): Q-space coordinates for detector pixels
- `Qy` (ndarray): Out-of-plane Q component
- `qvecs` (ndarray): Array of Q-vectors for all pixels (shape: npix²×3)
- `q_min`, `q_max` (float): Accessible Q-range

**Example:**
```python
exp = Experiment(wl=1.54, distance=0.36, npix=500, pixel_size=0.0006)
print(f"Q-range: {exp.q_min:.3f} to {exp.q_max:.3f} Å⁻¹")
```

---

### Debye3D

```python
class Debye3D(structure_file, npix=250, wl=1.0, distance=0.5, 
              pixel_size=0.0001, verbose=False, torch_device=None, 
              scattering_type='xray')
```

Main calculator for Debye scattering patterns. Inherits from `Experiment`.

**Parameters:**
- `structure_file` (str): Path to atomic structure file (XYZ format)
- `npix` (int): Number of detector pixels
- `wl` (float): Wavelength in Angstroms
- `distance` (float): Sample-detector distance in meters
- `pixel_size` (float): Pixel size in meters
- `verbose` (bool): Print calculation details
- `torch_device` (str or None): Torch device ('cuda', 'cpu', or None for auto)
- `scattering_type` (str): 'xray' or 'electron'

**Attributes:**
- `atoms` (ase.Atoms): ASE Atoms object
- `positions` (ndarray): Atomic positions (N×3)
- `elements` (list): Chemical symbols
- `nb_atoms` (int): Number of atoms
- `device` (torch.device or None): PyTorch device if available

**Example:**
```python
d3d = Debye3D('nanoparticle.xyz', wl=1.54, distance=0.36, 
              npix=500, pixel_size=0.0006, verbose=True)
```

---

## Debye3D Methods

### Structure Manipulation

#### rotate_positions

```python
d3d.rotate_positions(alpha=0, beta=0, gamma=0)
```

Rotate atomic positions using Euler angles (ZYZ convention).

**Parameters:**
- `alpha` (float): Rotation around Z-axis (degrees)
- `beta` (float): Rotation around Y-axis (degrees)
- `gamma` (float): Rotation around Z-axis (degrees)

**Example:**
```python
# Rotate 90° around Y-axis
d3d.rotate_positions(alpha=0, beta=90, gamma=0)
```

---

### Scattering Calculations

#### compute_intensity

```python
I = d3d.compute_intensity(use_gpu=False, batch_size=None, 
                          use_lobato=False, formula=None)
```

Compute 2D scattering intensity for current particle orientation.

**Parameters:**
- `use_gpu` (bool): Use GPU acceleration if available
- `batch_size` (int or None): GPU batch size for memory management
- `use_lobato` (bool): Use Lobato parametrization for form factors
- `formula` (str or None): Chemical formula (e.g., 'Au')

**Returns:**
- `I` (ndarray): Intensity map (npix×npix, flattened)

**Example:**
```python
I = d3d.compute_intensity(use_gpu=True)
```

---

#### compute_isotropic_intensity_fibonacci

```python
q_iso, i_iso = d3d.compute_isotropic_intensity_fibonacci(
    n_orient=400, use_lobato=False, formula=None, backend='numba'
)
```

Compute orientation-averaged I(q) using Fibonacci sphere sampling.

**Parameters:**
- `n_orient` (int): Number of orientations for averaging
- `use_lobato` (bool): Use Lobato form factors
- `formula` (str or None): Chemical formula
- `backend` (str): 'numba', 'torch', or 'numpy'

**Returns:**
- `q_iso` (ndarray): Q values
- `i_iso` (ndarray): Orientation-averaged intensity

**Example:**
```python
q, i = d3d.compute_isotropic_intensity_fibonacci(n_orient=500)
```

---

#### compute_intensity_uniaxial_ODF

```python
I_dist = d3d.compute_intensity_uniaxial_ODF(
    n_samples=100, sigma_y=5, sigma_z=5
)
```

Compute scattering with uniaxial orientation distribution (Gaussian ODF).

**Parameters:**
- `n_samples` (int): Number of Monte Carlo samples
- `sigma_y` (float): Gaussian width for Y-axis rotation (degrees)
- `sigma_z` (float): Gaussian width for Z-axis rotation (degrees)

**Returns:**
- `I_dist` (ndarray): Averaged 2D intensity

**Example:**
```python
# Oriented along X with ±5° spread in Y
I = d3d.compute_intensity_uniaxial_ODF(n_samples=100, sigma_y=5, sigma_z=0)
```

---

#### compute_Iq_debyecalc

```python
q_dc, i_dc = d3d.compute_Iq_debyecalc()
```

Compute I(q) using the DebyeCalculator package (if installed).

**Returns:**
- `q_dc` (ndarray): Q values
- `i_dc` (ndarray): Intensity values

**Example:**
```python
q, i = d3d.compute_Iq_debyecalc()
```

---

### Integration and Analysis

#### integrate_with_pyfai

```python
q, i = d3d.integrate_with_pyfai(I)
```

Azimuthal integration using pyFAI to obtain I(q) from 2D pattern.

**Parameters:**
- `I` (ndarray): 2D intensity map (flattened)

**Returns:**
- `q` (ndarray): Q values
- `i` (ndarray): Azimuthally integrated intensity

**Example:**
```python
I_2d = d3d.compute_intensity()
q, i = d3d.integrate_with_pyfai(I_2d)
```

---

### Visualization

#### plot_intensity

```python
d3d.plot_intensity(I, vmin=None, vmax=None, cmap='jet')
```

Plot 2D scattering pattern.

**Parameters:**
- `I` (ndarray): Intensity map
- `vmin` (float or None): Minimum value for log scale
- `vmax` (float or None): Maximum value for log scale
- `cmap` (str): Matplotlib colormap

**Example:**
```python
I = d3d.compute_intensity()
d3d.plot_intensity(I, vmin=-6, vmax=0, cmap='viridis')
```

---

## Utility Functions

### Form Factors

#### load_elements_yaml

```python
elements = load_elements_yaml(path)
```

Load element database containing Cromer-Mann coefficients.

**Parameters:**
- `path` (str): Path to elements_info.yaml

**Returns:**
- `elements` (dict): Element data dictionary

---

#### f0_from_Q

```python
f0 = f0_from_Q(Q, element, yaml_table)
```

Compute atomic form factor from Q using Cromer-Mann parametrization.

**Parameters:**
- `Q` (float or ndarray): Scattering vector magnitude (Å⁻¹)
- `element` (str): Element symbol (e.g., 'Au')
- `yaml_table` (dict): Element database from `load_elements_yaml`

**Returns:**
- `f0` (float or ndarray): Atomic form factor

---

#### neutron_scattering_length

```python
b = neutron_scattering_length(element, yaml_table)
```

Get neutron scattering length.

**Parameters:**
- `element` (str): Element symbol
- `yaml_table` (dict): Element database

**Returns:**
- `b` (float): Neutron scattering length

---

### I/O Functions

#### save_intensity_npz

```python
save_intensity_npz(filename, I_flat, Qx, Qz)
```

Save intensity map to compressed NPZ file.

**Parameters:**
- `filename` (str): Output filename (.npz)
- `I_flat` (ndarray): Flattened intensity array
- `Qx`, `Qz` (ndarray): Q-space coordinates

---

#### plot_from_npz

```python
plot_from_npz(filename)
```

Load and plot intensity from NPZ file.

**Parameters:**
- `filename` (str): NPZ file to load

---

## Advanced Functions

### Fibonacci Sphere

#### fibonacci_sphere

```python
directions = fibonacci_sphere(n_orient)
```

Generate Fibonacci sphere sampling points for orientation averaging.

**Parameters:**
- `n_orient` (int): Number of points

**Returns:**
- `directions` (ndarray): Unit vectors (n_orient×3)

---

## Constants and Configuration

### Available Backends

- `'numba'`: Fast CPU calculation with Numba JIT
- `'torch'`: GPU/CPU calculation with PyTorch
- `'numpy'`: Pure NumPy (slowest, most compatible)

### Scattering Types

- `'xray'`: X-ray scattering (default)
- `'electron'`: Electron scattering

---

## Notes

- All Q values are in Å⁻¹
- All distances in meters unless specified
- Angles in degrees
- Wavelengths in Angstroms
