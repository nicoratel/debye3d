# Advanced Usage

Advanced topics and techniques for power users.

## Custom Form Factors

### Using Lobato Parametrization

For electron scattering or improved accuracy:

```python
from debye3d import Debye3D

d3d = Debye3D('structure.xyz', wl=1.54, distance=0.36, 
              npix=500, pixel_size=0.0006,
              scattering_type='electron')

# Use Lobato form factors
I = d3d.compute_intensity(use_lobato=True, formula='Au')
```

### Implementing Custom Form Factors

Modify the `get_scattering_factor` method:

```python
import numpy as np
from debye3d import Debye3D

class CustomDebye3D(Debye3D):
    def get_scattering_factor(self, q, use_lobato=False, formula=None):
        # Your custom implementation
        f_custom = np.ones_like(q)  # Example: constant form factor
        return f_custom

# Use custom calculator
d3d = CustomDebye3D('structure.xyz', wl=1.54, distance=0.36,
                    npix=500, pixel_size=0.0006)
I = d3d.compute_intensity()
```

## Optimizing Performance

### Backend Selection

```python
# Test different backends
backends = ['numpy', 'numba', 'torch']
times = {}

for backend in backends:
    try:
        import time
        start = time.time()
        q, i = d3d.compute_isotropic_intensity_fibonacci(
            n_orient=500, backend=backend
        )
        times[backend] = time.time() - start
        print(f'{backend}: {times[backend]:.2f}s')
    except:
        print(f'{backend}: not available')
```

### Memory-Efficient Calculations

For very large structures:

```python
# Process in chunks
def compute_intensity_chunked(d3d, chunk_size=5000):
    n_pixels = d3d.npix ** 2
    I_total = np.zeros(n_pixels)
    
    for start_idx in range(0, n_pixels, chunk_size):
        end_idx = min(start_idx + chunk_size, n_pixels)
        # Process chunk
        qvecs_chunk = d3d.qvecs[start_idx:end_idx]
        # ... compute intensity for chunk
        I_total[start_idx:end_idx] = I_chunk
    
    return I_total
```

### Parallel Processing

```python
from multiprocessing import Pool
import numpy as np

def compute_for_orientation(args):
    """Compute intensity for one orientation"""
    d3d, alpha, beta, gamma = args
    d3d.rotate_positions(alpha, beta, gamma)
    return d3d.compute_intensity()

def parallel_orientation_average(d3d, n_orient=500, n_processes=4):
    """Parallel orientation averaging"""
    # Generate orientations
    from debye3d import fibonacci_sphere
    directions = fibonacci_sphere(n_orient)
    
    # Convert to Euler angles
    orientations = []
    for direction in directions:
        # ... convert direction to Euler angles
        orientations.append((d3d, alpha, beta, gamma))
    
    # Parallel computation
    with Pool(n_processes) as pool:
        intensities = pool.map(compute_for_orientation, orientations)
    
    # Average
    I_avg = np.mean(intensities, axis=0)
    return I_avg
```

## Custom Orientation Distributions

### Non-Gaussian ODF

```python
import numpy as np
from scipy.stats import vonmises

def custom_odf_sampling(d3d, n_samples=100, kappa=5.0):
    """
    Sample orientations from von Mises distribution
    kappa: concentration parameter
    """
    I_sum = np.zeros(d3d.npix ** 2)
    
    for _ in range(n_samples):
        # Sample angles from von Mises distribution
        beta = vonmises.rvs(kappa, loc=0) * 180/np.pi
        gamma = vonmises.rvs(kappa, loc=0) * 180/np.pi
        
        # Compute intensity
        d3d.rotate_positions(alpha=0, beta=beta, gamma=gamma)
        I_sum += d3d.compute_intensity()
    
    return I_sum / n_samples

# Use
d3d = Debye3D('structure.xyz', wl=1.54, distance=0.36,
              npix=500, pixel_size=0.0006)
I_custom = custom_odf_sampling(d3d, n_samples=100, kappa=5.0)
```

### Texture Analysis

Model preferred orientations:

```python
def texture_odf(d3d, n_samples=100, texture_axis='z', spread=10):
    """
    Simulate textured sample with preferred orientation
    
    Parameters
    ----------
    texture_axis : str
        Preferred axis ('x', 'y', or 'z')
    spread : float
        Angular spread (degrees) around preferred axis
    """
    I_sum = np.zeros(d3d.npix ** 2)
    
    for _ in range(n_samples):
        if texture_axis == 'z':
            alpha = np.random.normal(0, spread)
            beta = np.random.normal(0, spread)
            gamma = np.random.uniform(0, 360)  # Random in-plane rotation
        elif texture_axis == 'x':
            alpha = np.random.uniform(0, 360)
            beta = np.random.normal(90, spread)
            gamma = np.random.normal(0, spread)
        # ... add other axes
        
        d3d.rotate_positions(alpha, beta, gamma)
        I_sum += d3d.compute_intensity()
    
    return I_sum / n_samples
```

## Integration with Other Tools

### ASE Integration

```python
from ase.io import read, write
from ase.build import make_supercell
import numpy as np

# Load structure
atoms = read('unit_cell.cif')

# Create supercell
supercell_matrix = np.diag([3, 3, 3])
supercell = make_supercell(atoms, supercell_matrix)

# Add defects or modifications
# ... modify supercell

# Save and compute
supercell.write('supercell.xyz')
d3d = Debye3D('supercell.xyz', wl=1.54, distance=0.36,
              npix=500, pixel_size=0.0006)
I = d3d.compute_intensity()
```

### PyFAI Advanced Integration

```python
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector

# Create custom detector
detector = Detector(
    pixel1=0.0006,  # 600 μm
    pixel2=0.0006,
    max_shape=(500, 500)
)

# Create azimuthal integrator
ai = AzimuthalIntegrator(
    dist=0.36,  # meters
    detector=detector,
    wavelength=1.54e-10  # Convert to meters
)

# Custom integration
I_2d = d3d.compute_intensity().reshape(500, 500)
q, i = ai.integrate1d(
    I_2d,
    npt=1000,
    unit='q_A^-1',
    method='csr'
)
```

## Batch Processing

### Multiple Structures

```python
import glob
from debye3d import Debye3D
import numpy as np

# Process all XYZ files in directory
xyz_files = glob.glob('structures/*.xyz')

results = {}
for xyz_file in xyz_files:
    d3d = Debye3D(xyz_file, wl=1.54, distance=0.36,
                  npix=500, pixel_size=0.0006)
    
    q, i = d3d.compute_isotropic_intensity_fibonacci(n_orient=500)
    results[xyz_file] = (q, i)
    
    # Save individual results
    output_name = xyz_file.replace('.xyz', '_scattering.npz')
    np.savez(output_name, q=q, i=i)

print(f'Processed {len(results)} structures')
```

### Parameter Sweeps

```python
import numpy as np
import matplotlib.pyplot as plt

# Sweep over particle sizes
sizes = [5, 10, 15, 20, 25]
results = {}

for size in sizes:
    from ase.cluster import Icosahedron
    atoms = Icosahedron('Au', noshells=size)
    atoms.write(f'ico_{size}.xyz')
    
    d3d = Debye3D(f'ico_{size}.xyz', wl=1.54, distance=0.36,
                  npix=500, pixel_size=0.0006)
    q, i = d3d.compute_isotropic_intensity_fibonacci(n_orient=500)
    results[size] = (q, i)

# Plot
plt.figure(figsize=(10, 6))
for size, (q, i) in results.items():
    plt.loglog(q, i, label=f'{size} shells')
plt.xlabel('q (Å⁻¹)')
plt.ylabel('I(q)')
plt.legend()
plt.show()
```

## Debugging and Validation

### Check Q-range Coverage

```python
d3d = Debye3D('structure.xyz', wl=1.54, distance=0.36,
              npix=500, pixel_size=0.0006)

print(f'Q-range: {d3d.q_min:.3f} to {d3d.q_max:.3f} Å⁻¹')
print(f'Number of Q-vectors: {len(d3d.qvecs)}')

# Visualize Q-space coverage
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(d3d.Qx.ravel(), d3d.Qz.ravel(), s=1, alpha=0.5)
plt.xlabel('Qx (Å⁻¹)')
plt.ylabel('Qz (Å⁻¹)')
plt.title('Q-space Coverage')
plt.axis('equal')
plt.show()
```

### Form Factor Validation

```python
from debye3d import load_elements_yaml, f0_from_Q
import numpy as np
import matplotlib.pyplot as plt

# Load elements
elements = load_elements_yaml('debye3d/elements_info.yaml')

# Plot form factors
Q = np.linspace(0, 5, 1000)
elements_list = ['H', 'C', 'Au', 'Pt']

plt.figure(figsize=(10, 6))
for element in elements_list:
    f0 = f0_from_Q(Q, element, elements)
    plt.plot(Q, f0, label=element)

plt.xlabel('Q (Å⁻¹)')
plt.ylabel('f₀(Q)')
plt.legend()
plt.title('Atomic Form Factors')
plt.grid(True, alpha=0.3)
plt.show()
```

## Performance Profiling

```python
import time
import cProfile
import pstats

def profile_calculation():
    d3d = Debye3D('structure.xyz', wl=1.54, distance=0.36,
                  npix=500, pixel_size=0.0006)
    q, i = d3d.compute_isotropic_intensity_fibonacci(n_orient=500)
    return q, i

# Profile
profiler = cProfile.Profile()
profiler.enable()
q, i = profile_calculation()
profiler.disable()

# Print statistics
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```
