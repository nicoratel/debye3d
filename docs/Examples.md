# Examples and Tutorials

Complete examples demonstrating Debye3D capabilities.

## Table of Contents

1. [Basic 2D Scattering Pattern](#example-1-basic-2d-scattering-pattern)
2. [Anisotropic Nanoparticles](#example-2-anisotropic-nanoparticles)
3. [Orientation Averaging](#example-3-orientation-averaging)
4. [Orientation Distribution Functions](#example-4-orientation-distribution-functions)
5. [GPU-Accelerated Calculations](#example-5-gpu-accelerated-calculations)
6. [Comparison with DebyeCalculator](#example-6-comparison-with-debyecalculator)

---

## Example 1: Basic 2D Scattering Pattern

Compute and visualize a 2D scattering pattern for a gold nanoparticle.

```python
from debye3d import Experiment, Debye3D
from ase.cluster import Decahedron
import matplotlib.pyplot as plt

# Create experimental setup
exp = Experiment(
    wl=1.54,        # Cu K-alpha wavelength
    distance=0.36,  # 36 cm
    npix=500,
    pixel_size=0.0006  # 600 μm pixels
)

# Generate gold decahedron
atoms = Decahedron('Au', 5, 25, 0)
print(f'Structure contains {len(atoms)} atoms')
atoms.write('decahedron.xyz')

# Create Debye3D calculator
d3d = Debye3D(
    'decahedron.xyz',
    wl=exp.wl,
    distance=exp.distance,
    npix=exp.npix,
    pixel_size=exp.pixel_size
)

# Compute intensity
I = d3d.compute_intensity()

# Plot
d3d.plot_intensity(I, vmin=-6)
plt.title('2D Scattering Pattern - Gold Decahedron')
plt.show()
```

---

## Example 2: Anisotropic Nanoparticles

Study the effect of particle orientation on scattering patterns.

```python
from debye3d import Experiment, Debye3D
from ase.cluster import Decahedron
import matplotlib.pyplot as plt

# Setup
wl = 1.54
distance = 0.36
npix = 500
pix_size = 0.0006

# Create elongated decahedron
atoms = Decahedron('Au', 5, 25, 0)  # Initially aligned along Z
atoms.write('particle.xyz')

# Create calculator
d3d = Debye3D('particle.xyz', wl=wl, distance=distance, 
              npix=npix, pixel_size=pix_size)

# Compute for initial orientation (along Z)
I_z = d3d.compute_intensity()

# Rotate 90° around Y-axis to align along X
d3d.rotate_positions(alpha=0, beta=90, gamma=0)
I_x = d3d.compute_intensity()

# Rotate 90° around X-axis to align along Y
d3d.rotate_positions(alpha=0, beta=0, gamma=90)
I_y = d3d.compute_intensity()

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

d3d.plot_intensity(I_z, vmin=-6)
axes[0].set_title('Aligned along Z')

d3d.plot_intensity(I_x, vmin=-6)
axes[1].set_title('Aligned along X')

d3d.plot_intensity(I_y, vmin=-6)
axes[2].set_title('Aligned along Y')

plt.tight_layout()
plt.show()
```

---

## Example 3: Orientation Averaging

Compute powder-like isotropic scattering patterns.

```python
from debye3d import Debye3D
from ase.cluster import Decahedron
import matplotlib.pyplot as plt

# Create structure
atoms = Decahedron('Au', 5, 25, 0)
atoms.write('nanoparticle.xyz')

# Create calculator
d3d = Debye3D('nanoparticle.xyz', wl=1.54, distance=0.36, 
              npix=500, pixel_size=0.0006)

# Compute single orientation
I_single = d3d.compute_intensity()
q_single, i_single = d3d.integrate_with_pyfai(I_single)

# Compute orientation-averaged intensity
q_iso, i_iso = d3d.compute_isotropic_intensity_fibonacci(n_orient=500)

# Optional: Compare with DebyeCalculator
try:
    q_dc, i_dc = d3d.compute_Iq_debyecalc()
except:
    q_dc, i_dc = None, None

# Plot comparison
plt.figure(figsize=(10, 6))
plt.loglog(q_single, i_single, '--', alpha=0.7, label='Single orientation')
plt.loglog(q_iso, i_iso, label='Isotropic (Fibonacci)', linewidth=2)

if q_dc is not None:
    plt.loglog(q_dc, i_dc, ':', label='DebyeCalculator', alpha=0.7)

plt.xlabel('q (Å⁻¹)', fontsize=12)
plt.ylabel('I(q)', fontsize=12)
plt.legend(fontsize=10)
plt.title('Orientation Averaging Comparison')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Example 4: Orientation Distribution Functions

Model partially oriented nanoparticles with Gaussian ODFs.

```python
from debye3d import Debye3D
from ase.cluster import Decahedron
import matplotlib.pyplot as plt

# Create structure
atoms = Decahedron('Au', 5, 25, 0)
atoms.write('particle.xyz')

# Create calculator and rotate to align along X
d3d = Debye3D('particle.xyz', wl=1.54, distance=0.36, 
              npix=500, pixel_size=0.0006)
d3d.rotate_positions(alpha=0, beta=90, gamma=0)

# Case 1: Perfectly aligned along X
I_aligned = d3d.compute_intensity()

# Case 2: Small distribution around Y (±5°)
I_narrow = d3d.compute_intensity_uniaxial_ODF(
    n_samples=100, sigma_y=5, sigma_z=0
)

# Case 3: Large distribution around Y (±15°)
I_wide = d3d.compute_intensity_uniaxial_ODF(
    n_samples=100, sigma_y=15, sigma_z=0
)

# Case 4: Isotropic distribution
q_iso, i_iso = d3d.compute_isotropic_intensity_fibonacci(n_orient=300)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

d3d.plot_intensity(I_aligned, vmin=-6)
axes[0, 0].set_title('Perfect Alignment')

d3d.plot_intensity(I_narrow, vmin=-6)
axes[0, 1].set_title('Narrow ODF (σ=5°)')

d3d.plot_intensity(I_wide, vmin=-6)
axes[1, 0].set_title('Wide ODF (σ=15°)')

# For isotropic, show I(q)
q_narrow, i_narrow = d3d.integrate_with_pyfai(I_narrow)
q_wide, i_wide = d3d.integrate_with_pyfai(I_wide)

axes[1, 1].loglog(q_narrow, i_narrow, label='σ=5°')
axes[1, 1].loglog(q_wide, i_wide, label='σ=15°')
axes[1, 1].loglog(q_iso, i_iso, label='Isotropic')
axes[1, 1].set_xlabel('q (Å⁻¹)')
axes[1, 1].set_ylabel('I(q)')
axes[1, 1].legend()
axes[1, 1].set_title('I(q) Comparison')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Example 5: GPU-Accelerated Calculations

Leverage GPU for faster computations.

```python
from debye3d import Debye3D
from ase.cluster import Icosahedron
import time

# Create larger structure for GPU benefit
atoms = Icosahedron('Au', noshells=7)  # ~1500 atoms
print(f'Structure contains {len(atoms)} atoms')
atoms.write('large_particle.xyz')

# Create calculator with GPU
d3d = Debye3D('large_particle.xyz', wl=1.54, distance=0.36,
              npix=512, pixel_size=0.0005, verbose=True)

# CPU calculation
start = time.time()
I_cpu = d3d.compute_intensity(use_gpu=False)
cpu_time = time.time() - start
print(f'CPU time: {cpu_time:.2f} seconds')

# GPU calculation
start = time.time()
I_gpu = d3d.compute_intensity(use_gpu=True)
gpu_time = time.time() - start
print(f'GPU time: {gpu_time:.2f} seconds')

print(f'Speedup: {cpu_time/gpu_time:.1f}x')

# Verify results match
import numpy as np
diff = np.max(np.abs(I_cpu - I_gpu))
print(f'Maximum difference: {diff:.2e}')
```

---

## Example 6: Comparison with DebyeCalculator

Validate results against the DebyeCalculator package.

```python
from debye3d import Debye3D
from ase.cluster import FaceCenteredCubic
import matplotlib.pyplot as plt

# Create FCC nanoparticle
surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
layers = [6, 9, 5]
atoms = FaceCenteredCubic('Au', surfaces, layers)
atoms.write('fcc_particle.xyz')

# Debye3D calculation
d3d = Debye3D('fcc_particle.xyz', wl=1.54, distance=0.36,
              npix=500, pixel_size=0.0006)

# Method 1: Fibonacci averaging
q_fib, i_fib = d3d.compute_isotropic_intensity_fibonacci(n_orient=500)

# Method 2: DebyeCalculator
q_dc, i_dc = d3d.compute_Iq_debyecalc()

# Plot comparison
plt.figure(figsize=(10, 6))
plt.loglog(q_fib, i_fib, label='Debye3D (Fibonacci)', linewidth=2)
plt.loglog(q_dc, i_dc, '--', label='DebyeCalculator', linewidth=2, alpha=0.7)

plt.xlabel('q (Å⁻¹)', fontsize=12)
plt.ylabel('I(q)', fontsize=12)
plt.legend(fontsize=11)
plt.title('Debye3D vs DebyeCalculator')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate relative difference
import numpy as np
i_fib_interp = np.interp(q_dc, q_fib, i_fib)
rel_diff = np.abs(i_fib_interp - i_dc) / i_dc
print(f'Mean relative difference: {np.mean(rel_diff)*100:.2f}%')
print(f'Max relative difference: {np.max(rel_diff)*100:.2f}%')
```

---

## Example 7: Save and Load Results

Efficient data management for computed patterns.

```python
from debye3d import Debye3D, save_intensity_npz, plot_from_npz
from ase.cluster import Decahedron

# Compute intensity
atoms = Decahedron('Au', 5, 25, 0)
atoms.write('particle.xyz')

d3d = Debye3D('particle.xyz', wl=1.54, distance=0.36,
              npix=500, pixel_size=0.0006)

I_dist = d3d.compute_intensity_uniaxial_ODF(
    n_samples=100, sigma_y=5, sigma_z=0
)

# Save to file
save_intensity_npz('oriented_pattern.npz', I_dist, d3d.Qx, d3d.Qz)
print('Saved to oriented_pattern.npz')

# Later: Load and plot
plot_from_npz('oriented_pattern.npz')
```

---

## Tips and Best Practices

### Choosing n_orient for Fibonacci Averaging

- Small particles (<100 atoms): 200-500 orientations
- Medium particles (100-1000 atoms): 500-1000 orientations
- Large particles (>1000 atoms): 1000-2000 orientations

### GPU vs CPU

- GPU is faster for large structures (>500 atoms) and high resolution (npix > 512)
- CPU with Numba is excellent for medium structures
- Pure NumPy works for small structures and testing

### Memory Management

For very large calculations:

```python
# Use batch processing
I = d3d.compute_intensity(use_gpu=True, batch_size=10000)
```

### Convergence Testing

Always test orientation averaging convergence:

```python
for n in [100, 200, 500, 1000]:
    q, i = d3d.compute_isotropic_intensity_fibonacci(n_orient=n)
    plt.loglog(q, i, label=f'n={n}')
plt.legend()
plt.show()
```
