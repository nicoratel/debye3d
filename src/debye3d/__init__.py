"""
debye3d - Debye 3D Scattering Calculator

A Python package for computing 3D X-ray and neutron scattering patterns
using the Debye scattering equation with support for GPU/CPU acceleration.
"""

__version__ = "0.1.0"

# Main classes
from .debye3d import (
    Experiment,
    Debye3D    
)

__all__ = [
    "__version__",
    # Classes
    "Experiment",
    "Debye3D",
    # Functions    
    "save_intensity_npz",
    "plot_from_npz"
    
]
