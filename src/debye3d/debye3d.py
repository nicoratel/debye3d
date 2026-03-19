import math
import time
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from .compute_f0 import f0_from_Q, load_elements_yaml
from tqdm import tqdm
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
from .adaptative_fibonacci import compute_isotropic_intensity_adaptative_fibonacci
try:
    from debyecalculator import DebyeCalculator
except Exception:
    DebyeCalculator = None
try:
    from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    from pyFAI.detectors import Detector
except Exception:
    AzimuthalIntegrator = None
    Detector = None

# --- Torch (GPU/CPU backend) ---
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# --- Numba (CPU-parallel backend) ---
try:
    from numba import njit, prange, set_num_threads
    import numba as nb
    import os
    max_threads = os.cpu_count()
    num_threads = max(1, max_threads - 2) if max_threads else 1
    set_num_threads(num_threads)
    NUMBA_AVAILABLE = True
except Exception:
    nb = None
    NUMBA_AVAILABLE = False

import re

# ===============================================================
# Utility: fibonacci sphere directions
# ===============================================================
def fibonacci_sphere(n_orient):
    indices = np.arange(n_orient)
    phi = (1 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / phi
    z = 1.0 - 2.0 * (indices + 0.5) / n_orient
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    dirs = np.stack((r * np.cos(theta), r * np.sin(theta), z), axis=1)
    return dirs

def parse_formula(formula):
    """
    Parse a chemical formula string into elements and their stoichiometric ratios.
    
    Args:
        formula (str): Chemical formula like "SiO2", "Al2O3", etc.
    
    Returns:
        tuple: (elements_list, ratios_list) where ratios are normalized to sum to 1
    
    Example:
        parse_formula("SiO2") -> (['Si', 'O'], [0.333, 0.667])
    """
    tokens = re.findall(r'([A-Z][a-z]*)([0-9.]+)?', formula)
    elements = []
    counts = []
    
    for (elem, count) in tokens:
        elements.append(elem)
        counts.append(float(count) if count else 1.0)
    
    counts = np.array(counts)
    ratios = counts / counts.sum()
    
    return elements, ratios.tolist()


def compute_avg_scattering_factor_lobato(
    formula,
    x_max=3.0,
    x_step=0.005,
    qvalues=False,
    plot=False,
    xray=False):
    """
    Compute average scattering factor using Lobato parametrization.
    
    Parameters
    ----------
    formula : str
        Chemical formula (e.g., "SiO2", "Al2O3")
    x_max : float
        Maximum q or s value
    x_step : float
        Step size for q or s
    qvalues : bool
        If True, x_max and x_step are in q units (Å⁻¹), else in s units
    plot : bool
        If True, plot individual and average scattering factors
    xray : bool
        If True, use X-ray scattering factors; if False, use electron scattering factors
    
    Returns
    -------
    q_array : np.ndarray
        q values (Å⁻¹)
    avg_scattering_factor : np.ndarray
        Averaged scattering factor values
    """
    from .lobato_scattering import LobatoScatteringCalculator
    
    elements, ratios = parse_formula(formula)
    
    # Convert to s (Lobato internal variable)
    if qvalues:
        s_max = x_max / (2 * np.pi)
        s_step = x_step / (2 * np.pi)
    else:
        s_max = x_max
        s_step = x_step
    
    parametrization = LobatoScatteringCalculator()
    name = "x_ray_scattering_factor" if xray else "scattering_factor"
    
    sf = parametrization.line_profiles(
        elements,
        cutoff=s_max,
        sampling=s_step,
        name=name
    )
    
    npts = sf.array.shape[1]
    
    # Build axis
    s = np.arange(npts) * s_step
    q = 2 * np.pi * s
    
    # Weighted average
    favg = np.zeros(npts)
    for i in range(len(elements)):
        favg += ratios[i] * sf.array[i]
    
    if plot:
        fig, ax = plt.subplots()
        for i, symbol in enumerate(elements):
            ax.plot(q, sf.array[i], label=symbol)
        ax.plot(q, favg, label="<f(q)>", lw=2)
        ax.set_xlabel(r"$Q\ (\AA^{-1})$")
        ax.set_ylabel("Scattering factor")
        ax.legend()
        plt.show()
    
    return q, favg
# ===============================================================
# Experiment
# ===============================================================
class Experiment:
    def __init__(self, npix=250, wl=1.0, distance=0.5, pixel_size=0.0001, verbose=True):
        self.npix = int(npix)
        self.wl = float(wl)
        self.distance = float(distance)
        self.pixel_size = float(pixel_size)
        self.D = self.distance

        # Detector grid
        i_vals = np.arange(-self.npix // 2, self.npix // 2)
        j_vals = np.arange(-self.npix // 2, self.npix // 2)
        I, J = np.meshgrid(i_vals, j_vals, indexing="xy")
        delta_i = I * self.pixel_size
        delta_j = J * self.pixel_size
        denom = np.sqrt(self.D**2 + delta_i**2 + delta_j**2)
        a = 2.0 * np.pi / self.wl

        self.Qx = (a / denom) * delta_i
        self.Qz = (a / denom) * delta_j
        self.Qy = (a / denom) * (self.D - denom)

        self.qvecs = np.stack([self.Qx.ravel(), self.Qy.ravel(), self.Qz.ravel()], axis=1)
        Q_magnitude = np.linalg.norm(self.qvecs, axis=1)
        self.q_min = float(Q_magnitude.min())
        self.q_max = float(Q_magnitude.max())

        if verbose:
            print("----------------------------------------------------")
            print(" Detector configuration / accessible Q-range")
            print("----------------------------------------------------")
            print(f" Wavelength λ = {self.wl:.4f} Å")
            print(f" Sample-detector distance = {self.D*1e3:.2f} mm")
            print(f" Pixel size = {self.pixel_size*1e3:.3f} mm")
            print(f" Number of pixels = {self.npix} x {self.npix}")
            print(f" |Q| range : {self.q_min:.4f} → {self.q_max:.4f} Å⁻¹")
            print("----------------------------------------------------\n")


# ===============================================================
# Debye3D hybrid class
# ===============================================================
class Debye3D(Experiment):
    def __init__(self, structure_file, npix=250, wl=1.0, distance=0.5, pixel_size=0.0001,
                 verbose=False, torch_device=None, scattering_type='xray'):
        """
        Initialize Debye3D calculator.
        
        Parameters
        ----------
        structure_file : str
            Path to structure file
        npix : int
            Number of detector pixels
        wl : float
            Wavelength (Å)
        distance : float
            Sample-detector distance (m)
        pixel_size : float
            Pixel size (m)
        verbose : bool
            Print information
        torch_device : str or None
            Torch device specification
        scattering_type : str
            Type of scattering factor: 'xray' (default) or 'electron'
        """
        super().__init__(npix=npix, wl=wl, distance=distance, pixel_size=pixel_size, verbose=verbose)
        self.atoms = read(structure_file)
        self.file = structure_file
        self.positions = self.atoms.get_positions()
        self.elements = self.atoms.get_chemical_symbols()
        self.nb_atoms = len(self.positions)
        self.scattering_type = scattering_type  # NEW: store scattering type
       
        if verbose:
            print(f"\n Structure contains {self.nb_atoms} atoms.")
            print(f" Scattering type: {self.scattering_type}\n")

        # Torch device setup
        if TORCH_AVAILABLE:
            if torch_device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(torch_device)
        else:
            self.device = None

        # Warm-up
        if self.device is not None and self.device.type == "cuda":
            _ = torch.zeros(1, device=self.device)
            del _
            if verbose:
                print(f"[Debye3D] Use of GPU: {torch.cuda.get_device_name(self.device)}")
        self.verbose = verbose

    
    # ==============================================================
    # Utility: view_structure
    # ==============================================================
    def view_structure(self):
        """
        Write a temporary XYZ file and launch an external viewer (jmol).

        Notes
        -----
        This function calls an external 'jmol' binary using os.system and
        requires jmol to be installed and accessible in the PATH.
        """
        self.save_structure_as_xyz('./file.xyz')
        os.system('jmol file.xyz')
        os.remove('file.xyz')

    def save_structure_as_xyz(self, filename):
        natoms=self.positions.shape[0]
        line2write=f'{natoms}\n\n'
        for i in range(len(self.elements)):
            line2write+=f'{self.elements[i]}\t{self.positions[i,0]:.8f}\t{self.positions[i,1]:.8f}\t{self.positions[i,2]:.8f}\n'
        with open(filename,'w') as f:
            f.write(line2write)

    def update_structure(self,coords,element):
        """ 
        Update strcutre associated to the classe given a list of coordinates and a single element
        Parameters
        ----------
        coords: tuple, ndarray
            List of atomic coordinates
        element: string
        """
        self.positions = coords
        self.nb_atoms= self.positions.shape[0]
        self.elements = np.full(self.nb_atoms,element)


    
    # ===============================================================
    # Auto batch size based on available VRAM
    # ===============================================================
    def auto_batch_size(self, target_fraction=0.8, reference=200_000, reference_mem_gb=8.0):
        """
        Automatic batch size determination adapted to available VRAM size.
        """
        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            if self.device:
                if self.verbose:                    
                    print(f"[auto_batch_size] No detected GPU {self.device}. Default_batch_size = 200_000")
            return 200_000

        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9
        batch_size = int(reference * (free_gb / reference_mem_gb) * target_fraction)
        if self.verbose:
            print(f"[auto_batch_size] free VRAM : {free_gb:.1f} GB → batch ≈ {batch_size}")
        return max(50_000, min(batch_size, 1_000_000))

    # ===============================================================
    # Atomic form factor
    # ===============================================================
    def get_scattering_factor(self, q, use_lobato=True, formula=None):
        """
        Get scattering factors for given q values.
        Uses Lobato parametrization by default for both X-ray and electron scattering.
        
        Parameters
        ----------
        q : np.ndarray
            q magnitude values (Å⁻¹)
        use_lobato : bool
            If True (default), use Lobato parametrization
            If False, use traditional YAML method (X-ray only, legacy)
        formula : str or None
            Chemical formula (for compounds). If None, uses first element.
        
        Returns
        -------
        f_q : np.ndarray
            Scattering factor values
        """
        if use_lobato:
            # Determine formula
            if formula is None:
                # Use first element if no formula provided
                formula = self.elements[0]
            
            # Get q range
            q_min, q_max = q.min(), q.max()
            q_step = (q_max - q_min) / len(q) if len(q) > 1 else 0.01
            
            # Compute Lobato scattering factors
            xray = (self.scattering_type == 'xray')
            q_lobato, f_lobato = compute_avg_scattering_factor_lobato(
                formula,
                x_max=q_max * 1.1,  # Add 10% margin
                x_step=q_step,
                qvalues=True,
                plot=False,
                xray=xray
            )
            
            # Interpolate to requested q values
            f_q = np.interp(q, q_lobato, f_lobato)
            
        else:
            # Legacy: Use traditional X-ray form factor from YAML
            import warnings
            warnings.warn(
                "Using legacy YAML form factors. Consider using Lobato parametrization (use_lobato=True) "
                "for more accurate and consistent results.",
                UserWarning
            )
            
            path = "./elements_info.yaml"
            table = load_elements_yaml(path)
            element = self.elements[0]
            f_q = f0_from_Q(q, element, table)
        
        return f_q

    # ===============================================================
    # GPU version (PyTorch)
    # ===============================================================
    def _to_torch(self, arr, dtype=None):
        if not TORCH_AVAILABLE or self.device is None:
            raise RuntimeError("PyTorch is required for GPU mode.")
        if dtype is None:
            dtype = torch.float32
        return torch.from_numpy(np.asarray(arr)).to(device=self.device, dtype=dtype)

       # ===============================================================
    # Intensity calculation
    # ===============================================================
    def compute_intensity(self, use_gpu=True, batch_size=None, use_lobato=True, formula=None):
        """
        Compute scattering intensity.
        
        Parameters
        ----------
        use_gpu : bool
            Use GPU acceleration if available
        batch_size : int or None
            Batch size for GPU computation
        use_lobato : bool
            If True (default), use Lobato parametrization
            If False, use legacy YAML method
        formula : str or None
            Chemical formula (e.g., 'SiO2'). If None, uses first element.
        
        Returns
        -------
        I : np.ndarray
            Intensity values
        """
        qvecs = self.qvecs
        q_mags = np.linalg.norm(qvecs, axis=1)
        f_atom = self.get_scattering_factor(q_mags, use_lobato=use_lobato, formula=formula)

        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            if batch_size is None:
                batch_size = self.auto_batch_size()
            return self._compute_intensity_torch(batch_size, f_atom)
        else:
            return compute_intensity_numba(self.positions, f_atom, qvecs)
        
    # ===============================================================
    # GPU (PyTorch)
    # ===============================================================
    def _compute_intensity_torch(self, batch_size=None, f_atom=None, atom_chunk=None, verbose=True):
        """
        High-performance GPU version for computing X-ray scattering intensity.
        Automatically adjusts batch and atom chunk sizes to prevent CUDA OOM errors.

        Parameters
        ----------
        batch_size : int or None
            Number of q-vectors processed per batch. Auto-scaled if None.
        f_atom : array-like or None
            Atomic scattering factors (precomputed). If None, computed internally.
        atom_chunk : int or None
            Number of atoms processed per sub-batch. Auto-scaled if None.
        verbose : bool
            If True, prints progress and GPU memory diagnostics.
        """

        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            raise RuntimeError("GPU not available or PyTorch not compiled with CUDA.")

        t0 = time.time()
        device = self.device
        Q = self.qvecs.astype(np.float32)
        Nq = Q.shape[0]
        Nat = self.nb_atoms

        # --- GPU memory diagnostics ---
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
        except RuntimeError:
            raise RuntimeError("Unable to query GPU memory. Check NVML / driver installation.")
        free_gb = free_mem / 1e9

        # --- adaptive batch and chunk sizes ---
        if batch_size is None:
            batch_size = int(200_000 * (free_gb / 40.0))
            batch_size = max(20_000, min(batch_size, 400_000))

        if atom_chunk is None:
            atom_chunk = int(100_000 * (free_gb / 40.0))
            atom_chunk = max(5_000, min(atom_chunk, 300_000))

        # --- Prevent excessive matrix sizes (batch_size × atom_chunk)
        # Allow up to 40% of free VRAM for the main phases tensor
        max_bytes = int(0.4 * free_mem)
        max_elements = max_bytes // 4
        if batch_size * atom_chunk > max_elements:
            scale = (max_elements / (batch_size * atom_chunk)) ** 0.5
            batch_size = int(batch_size * scale)
            atom_chunk = int(atom_chunk * scale)
            if self.verbose:
                print(f"[⚠️] Adjusted batch sizes to avoid OOM: "
                    f"batch_q={batch_size}, chunk_atoms={atom_chunk}")

        if self.verbose:
            print(f"[Debye3D GPU] {Nat} atoms, {Nq} q-vectors")
            print(f" → batch_q = {batch_size}, chunk_atoms = {atom_chunk}, free VRAM ≈ {free_gb:.1f} GB")

        # --- Move data to GPU ---
        positions_t = torch.tensor(self.positions, dtype=torch.float32, device=device)

        if f_atom is None:
            q_mags = np.linalg.norm(Q, axis=1)
            f_atom = self.get_scattering_factor(q_mags)
        f_atom_t = torch.tensor(f_atom, dtype=torch.float32, device=device)

        I_acc = torch.zeros(Nq, dtype=torch.float32, device=device)

        # --- Main computation ---
        with torch.no_grad():
            try:
                for q_start in tqdm(range(0, Nq, batch_size), desc="Computing intensity", disable=not verbose):
                    q_stop = min(Nq, q_start + batch_size)
                    Qbatch = torch.tensor(Q[q_start:q_stop], dtype=torch.float32, device=device)

                    Re_acc = torch.zeros(Qbatch.shape[0], dtype=torch.float32, device=device)
                    Im_acc = torch.zeros_like(Re_acc)

                    for a_start in range(0, Nat, atom_chunk):
                        a_stop = min(Nat, a_start + atom_chunk)
                        pos_chunk = positions_t[a_start:a_stop]

                        # Large dot product: (batch_q × atom_chunk)
                        phases = torch.matmul(Qbatch, pos_chunk.T)
                        Re_acc += torch.cos(phases).sum(dim=1)
                        Im_acc += torch.sin(phases).sum(dim=1)

                        del pos_chunk, phases
                        torch.cuda.empty_cache()

                    I_batch = (Re_acc**2 + Im_acc**2) * (f_atom_t[q_start:q_stop] ** 2)
                    I_acc[q_start:q_stop] = I_batch

                    del Qbatch, Re_acc, Im_acc, I_batch
                    torch.cuda.empty_cache()
                    gc.collect()

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    if self.verbose:
                        print("[⚠️] CUDA OOM detected — reducing batch size and retrying on CPU.")
                    torch.cuda.empty_cache()
                    return self._compute_intensity_cpu(verbose=verbose)
                else:
                    raise e

        if self.verbose:
            dt = time.time() - t0
            print(f"[✓] GPU computation completed in {dt/60:.2f} minutes")

        return I_acc.cpu().numpy()

    def Iq_to_2D(self, q_values, I_q, blockdirect=False, q_min_mask=0.5, mask_value=np.nan):
        """
        Convertit une courbe I(q) 1D en image 2D avec anneaux concentriques.
        Utile pour visualiser le résultat de la moyenne orientationnelle
        (Fibonacci ou autre) sous forme d'image de diffusion.
        
        Parameters:
        -----------
        q_values : array 1D
            Valeurs de q correspondantes (ex: 500 points)
        I_q : array 1D
            Intensités moyennées en fonction de q (ex: 500 points)
        plot : bool, optional
            Si True, affiche l'image 2D et la courbe I(q) (défaut: False)
        blockdirect : bool, optional
            Si True, masque les pixels avec |q| < q_min_mask (défaut: False)
        q_min_mask : float, optional
            Seuil de q pour le masquage (défaut: 0.5 Å⁻¹)
        mask_value : float, optional
            Valeur à utiliser pour les pixels masqués (défaut: np.nan)
            Peut être np.nan, 0, ou toute autre valeur
        
        Returns:
        --------
        image_2d : array 1D
            Image de diffusion 2D aplatie (npix*npix)
        
        Example:
        --------
        >>> debye = Debye3D('structure.xyz', npix=250)
        >>> q, I_q = debye.compute_isotropic_intensity_fibonacci(n_q=500, n_orient=1000)
        >>> image_2d = debye.Iq_to_2D(q, I_q, plot=True, blockdirect=True)
        """
        # Calculer les magnitudes q pour chaque pixel du détecteur
        qvecs = self.qvecs
        q_mags = np.linalg.norm(qvecs, axis=1)
        q_mags_2d = q_mags.reshape(self.npix, self.npix)
        if self.verbose:
            print(f'q_mags_2d.shape={q_mags_2d.shape}, min={q_mags_2d.min()}, max = {q_mags_2d.max()}')
        # Interpoler I(q) sur la grille 2D du détecteur
        image_2d = np.interp(q_mags_2d, q_values, I_q)
        
        # Appliquer le masque du faisceau direct si demandé
        if blockdirect:
            q_perp = np.sqrt(self.Qx**2 + self.Qz**2)
            mask = q_perp < q_min_mask
            image_2d[mask] = mask_value
            
                       
        return image_2d.flatten()


    # ===============================================================
    # Isotropic (Fibonacci)
    # ===============================================================
     
    def compute_isotropic_intensity_fibonacci(
        self, n_q=500,
        n_orient=1000,
        use_gpu=True,
        batch_orient=None,
        atom_chunk=None,
        verbose=True,
        use_lobato=True,
        formula=None):
        """
        Isotropic intensity computation using Fibonacci sphere sampling.
        
        Parameters
        ----------
        n_q : int
            Number of q magnitudes
        n_orient : int
            Number of orientations (for q_max < 1)
        use_gpu : bool
            Use GPU acceleration
        batch_orient : int or None
            Orientation batch size
        atom_chunk : int or None
            Atom chunk size
        verbose : bool
            Print progress
        use_lobato : bool
            If True (default), use Lobato scattering factors
        formula : str or None
            Chemical formula. If None, uses first element.
        
        Returns
        -------
        q_vals : np.ndarray
            q values
        Iq : np.ndarray
            Isotropic intensity
        """
        q_vals = np.linspace(self.q_min, self.q_max, n_q)
        f_q = self.get_scattering_factor(q_vals, use_lobato=use_lobato, formula=formula)
        Nat = self.nb_atoms

        # ==================== CPU FALLBACK ====================
        if not (use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()):
            if self.verbose:
                print("→ Isotropic average (CPU) over orientations")
            if self.q_max < 1:
                dirs = fibonacci_sphere(n_orient)
                return q_vals, compute_intensity_fibonacci_numba(self.positions, f_q, q_vals, dirs)
            else:
                if self.verbose:
                    print('-> Using adaptive Fibonacci algorithm (CPU) to compute orientation averaging')
                return q_vals, compute_isotropic_intensity_adaptative_fibonacci(
                    self.positions, f_q, q_vals,
                    n_base=400, q_ref=0.75, scaling_power=2,
                    n_min=400, n_max=20000)

        # ==================== GPU COMPUTATION ====================
        
        # --- GPU memory info ---
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9

        # --- Adaptive batching ---
        if batch_orient is None:
            batch_orient = int(10_000 * (free_gb / 40.0))
            batch_orient = max(500, min(batch_orient, 50_000))
        if atom_chunk is None:
            atom_chunk = int(100_000 * (free_gb / 40.0))
            atom_chunk = max(5_000, min(atom_chunk, 300_000))

        # --- Cap to avoid excessive tensor sizes (batch_orient × atom_chunk) ---
        max_bytes = int(0.4 * free_mem)
        max_elements = max_bytes // 4
        if batch_orient * atom_chunk > max_elements:
            scale = (max_elements / (batch_orient * atom_chunk)) ** 0.5
            batch_orient = int(batch_orient * scale)
            atom_chunk = int(atom_chunk * scale)
            if self.verbose:
                print(f"[⚠️] Adjusted batches to fit VRAM: orient={batch_orient}, atoms={atom_chunk}")

        device = self.device
        positions_t = torch.tensor(self.positions, dtype=torch.float32, device=device)
        q_vals_t = torch.tensor(q_vals.astype(np.float32), dtype=torch.float32, device=device)
        Iq = torch.zeros_like(q_vals_t)
        t0 = time.time()

        # ==================== CHOIX ALGORITHME ====================
        
        if self.q_max < 1:
            # --- ALGORITHME STANDARD (nombre fixe d'orientations) ---
            if self.verbose:
                print(f"[Debye3D GPU - Standard] {Nat} atoms, {n_orient} orientations, {n_q} q-points")
                print(f" → batch_orient = {batch_orient}, chunk_atoms = {atom_chunk}, free ≈ {free_gb:.1f} GB")
            
            dirs = fibonacci_sphere(n_orient)
            dirs_t = torch.tensor(dirs.astype(np.float32), dtype=torch.float32, device=device)
            
            with torch.no_grad():
                for iq in tqdm(range(n_q), disable=not verbose):
                    q = q_vals_t[iq]
                    f = f_q[iq]
                    Q_dirs = q * dirs_t

                    I_partial_sum = 0.0
                    count = 0

                    # --- batch over orientations ---
                    for o_start in range(0, n_orient, batch_orient):
                        o_stop = min(n_orient, o_start + batch_orient)
                        Q_batch = Q_dirs[o_start:o_stop]

                        Re_acc = torch.zeros(Q_batch.shape[0], device=device)
                        Im_acc = torch.zeros_like(Re_acc)

                        # --- sub-batch over atoms ---
                        for a_start in range(0, Nat, atom_chunk):
                            a_stop = min(Nat, a_start + atom_chunk)
                            pos_chunk = positions_t[a_start:a_stop]
                            phases = torch.matmul(Q_batch, pos_chunk.T)
                            Re_acc += torch.cos(phases).sum(dim=1)
                            Im_acc += torch.sin(phases).sum(dim=1)

                            del pos_chunk, phases
                            torch.cuda.empty_cache()

                        I_batch = (Re_acc**2 + Im_acc**2)
                        I_partial_sum += I_batch.sum().item()
                        count += I_batch.shape[0]

                        del Q_batch, Re_acc, Im_acc, I_batch
                        torch.cuda.empty_cache()
                        gc.collect()

                    I_mean = (I_partial_sum / count) * (f**2)
                    Iq[iq] = I_mean
        
        else:
            # --- ALGORITHME ADAPTATIF (nombre variable d'orientations selon q) ---
            if self.verbose:
                print(f"[Debye3D GPU - Adaptive] {Nat} atoms, adaptive orientations, {n_q} q-points")
                print(f" → batch_orient = {batch_orient}, chunk_atoms = {atom_chunk}, free ≈ {free_gb:.1f} GB")
            
            # Paramètres adaptatifs (ajustables)
            n_base = 400
            q_ref = 0.75
            scaling_power = 0.5
            n_min = 150
            n_max = 10000
            
            with torch.no_grad():
                for iq in tqdm(range(n_q), disable=not verbose):
                    q = q_vals_t[iq]
                    f = f_q[iq]
                    
                    # Calcul du nombre d'orientations pour ce q
                    n_orient_q = int(n_base * (q.item() / q_ref) ** scaling_power)
                    n_orient_q = min(max(n_orient_q, n_min), n_max)
                    
                    # Génération de la grille Fibonacci pour ce q
                    dirs = fibonacci_sphere(n_orient_q)
                    dirs_t = torch.tensor(dirs.astype(np.float32), dtype=torch.float32, device=device)
                    Q_dirs = q * dirs_t

                    I_partial_sum = 0.0
                    count = 0

                    # --- batch over orientations ---
                    for o_start in range(0, n_orient_q, batch_orient):
                        o_stop = min(n_orient_q, o_start + batch_orient)
                        Q_batch = Q_dirs[o_start:o_stop]

                        Re_acc = torch.zeros(Q_batch.shape[0], device=device)
                        Im_acc = torch.zeros_like(Re_acc)

                        # --- sub-batch over atoms ---
                        for a_start in range(0, Nat, atom_chunk):
                            a_stop = min(Nat, a_start + atom_chunk)
                            pos_chunk = positions_t[a_start:a_stop]
                            phases = torch.matmul(Q_batch, pos_chunk.T)
                            Re_acc += torch.cos(phases).sum(dim=1)
                            Im_acc += torch.sin(phases).sum(dim=1)

                            del pos_chunk, phases
                            torch.cuda.empty_cache()

                        I_batch = (Re_acc**2 + Im_acc**2)
                        I_partial_sum += I_batch.sum().item()
                        count += I_batch.shape[0]

                        del Q_batch, Re_acc, Im_acc, I_batch
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Nettoyage de la grille pour ce q
                    del dirs_t, Q_dirs
                    torch.cuda.empty_cache()

                    I_mean = (I_partial_sum / count) * (f**2)
                    Iq[iq] = I_mean

        if self.verbose:
            dt = time.time() - t0
            print(f"[✓] Completed isotropic average in {dt/60:.2f} minutes")

        return q_vals, Iq.cpu().numpy()


    # ===============================================================
    # Uniaxial
    # ===============================================================
   
    def compute_intensity_uniaxial_ODF(
        self,n_samples=200,
        sigma_y=2.0,sigma_z=2.0,
        use_gpu=True,
        batch_q=None,atom_chunk=None,
        verbose=True):
        """
        Uniaxial ODF intensity computation with full double batching (q × atom).
        Prevents CUDA OOM by adaptively scaling both batch dimensions.

        Parameters
        ----------
        n_samples : int
            Number of random orientation samples.
        sigma_y, sigma_z : float
            Angular standard deviations (degrees).
        use_gpu : bool
            Whether to use GPU acceleration.
        batch_q : int or None
            Number of q-vectors per batch (auto-scaled if None).
        atom_chunk : int or None
            Number of atoms per sub-batch (auto-scaled if None).
        verbose : bool
            If True, print progress and GPU diagnostics.
        """

        import torch, math, gc, time
        import numpy as np
        from tqdm import tqdm

        sigma_y_rad = np.deg2rad(sigma_y)
        sigma_z_rad = np.deg2rad(sigma_z)
        rng = np.random.default_rng()
        theta_y = rng.normal(0, sigma_y_rad, n_samples)
        theta_z = rng.normal(0, sigma_z_rad, n_samples)

        qvecs = self.qvecs
        q_mags = np.linalg.norm(qvecs, axis=1)
        f_atom = self.get_scattering_factor(q_mags)
        Nat = self.nb_atoms

        if not (use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()):
            if self.verbose:
                print("→ Uniaxial average (CPU)")
            return compute_intensity_uniaxial_numba(self.positions, f_atom, qvecs, theta_y, theta_z)

        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9

        if batch_q is None:
            batch_q = int(100_000 * (free_gb / 40.0))
            batch_q = max(10_000, min(batch_q, 400_000))
        if atom_chunk is None:
            atom_chunk = int(100_000 * (free_gb / 40.0))
            atom_chunk = max(5_000, min(atom_chunk, 300_000))

        # Avoid huge tensors (batch_q × atom_chunk)
        max_bytes = int(0.4 * free_mem)
        max_elements = max_bytes // 4
        if batch_q * atom_chunk > max_elements:
            scale = (max_elements / (batch_q * atom_chunk)) ** 0.5
            batch_q = int(batch_q * scale)
            atom_chunk = int(atom_chunk * scale)
            if self.verbose:
                print(f"[⚠️] Adjusted batches to fit VRAM: q={batch_q}, atoms={atom_chunk}")

        if self.verbose:
            print(f"[Debye3D GPU - Uniaxial] {Nat} atoms, {len(qvecs)} q-points, {n_samples} orientations")
            print(f" → batch_q = {batch_q}, chunk_atoms = {atom_chunk}, free ≈ {free_gb:.1f} GB")

        device = self.device
        positions_t = torch.tensor(self.positions, dtype=torch.float32, device=device)
        Q_t_full = torch.tensor(qvecs.astype(np.float32), dtype=torch.float32, device=device)
        f_t_full = torch.tensor(f_atom.astype(np.float32), dtype=torch.float32, device=device)
        Nq = Q_t_full.shape[0]
        I_acc = torch.zeros(Nq, dtype=torch.float32, device=device)

        t0 = time.time()
        with torch.no_grad():
            for i in tqdm(range(n_samples), disable=not verbose):
                ty, tz = float(theta_y[i]), float(theta_z[i])
                cy, sy = math.cos(ty), math.sin(ty)
                cz, sz = math.cos(tz), math.sin(tz)
                R = torch.tensor(
                    [
                        [cz * cy, -sz, cz * sy],
                        [sz * cy,  cz, sz * sy],
                        [-sy,      0.,     cy],
                    ],
                    dtype=torch.float32,
                    device=device,
                )

                # Rotate positions once per sample
                pos_rot = positions_t @ R.T

                # --- batch over q-vectors ---
                for q_start in range(0, Nq, batch_q):
                    q_stop = min(Nq, q_start + batch_q)
                    Q_batch = Q_t_full[q_start:q_stop]
                    f_batch = f_t_full[q_start:q_stop]

                    Re_acc = torch.zeros(Q_batch.shape[0], device=device)
                    Im_acc = torch.zeros_like(Re_acc)

                    # --- sub-batch over atoms ---
                    for a_start in range(0, Nat, atom_chunk):
                        a_stop = min(Nat, a_start + atom_chunk)
                        pos_chunk = pos_rot[a_start:a_stop]
                        phases = torch.matmul(Q_batch, pos_chunk.T)
                        Re_acc += torch.cos(phases).sum(dim=1)
                        Im_acc += torch.sin(phases).sum(dim=1)

                        del pos_chunk, phases
                        torch.cuda.empty_cache()

                    I_batch = (Re_acc**2 + Im_acc**2) * (f_batch**2)
                    I_acc[q_start:q_stop] += I_batch

                    del Q_batch, f_batch, Re_acc, Im_acc, I_batch
                    torch.cuda.empty_cache()
                    gc.collect()

        I_mean = I_acc / float(n_samples)
        if self.verbose:
            dt = time.time() - t0
            print(f"[✓] Completed uniaxial average in {dt/60:.2f} minutes")

        return I_mean.cpu().numpy()

    # ===============================================================
    # Miscellaneous
    # ===============================================================

    def compute_Iq_debyecalc(self):
        """
        Compute I(q) using the external DebyeCalculator (if available).

        The method builds a q-grid consistent with the detector Q-range and
        invokes DebyeCalculator.iq to compute a Debye scattering curve.

        Returns
        -------
        q_dc : numpy.ndarray
            q-grid returned by the DebyeCalculator (Å⁻¹).
        i_dc_scaled : numpy.ndarray
            Intensity returned by DebyeCalculator, scaled to be comparable to
            other computations (arbitrary units). If DebyeCalculator is not
            available, raises RuntimeError.
        """
        if DebyeCalculator is None:
            raise RuntimeError("DebyeCalculator is not available. Install the optional dependency 'debyecalculator'.")

        n = self.npix / 2.0
        qstep = (self.q_max - self.q_min) / n
        calc = DebyeCalculator(qmin=self.q_min, qmax=self.q_max, qstep=qstep, biso=0, device='cuda')

        q_dc, i_dc = calc.iq(self.file)

        K=2
        return q_dc, i_dc * K

    # ===============================================================
    # Plot
    # ===============================================================
    # Dans votre fichier debye3d.py, remplacez la méthode plot_intensity par celle-ci:

    def plot_intensity(self, I_flat, log=True, vmin=-6, vmax=0, qmax=None, 
                    interpolation='bicubic', filename=None, handle_nan=True, cmap='jet',
                    blockdirect=False, qmin=0.0):
        """
        Plot a 2D intensity map on the detector Q-plane.
        Properly handles NaN values from beamstop masking.
        
        Parameters
        ----------
        I_flat : numpy.ndarray, shape (npix*npix,)
            Flattened intensity map to plot.
        log : bool, optional
            If True, plot on a logarithmic color scale (default True).
        vmin : float, optional
            Lower bound exponent (10**vmin) for LogNorm (default -6).
        vmax : float, optional
            Upper bound exponent (10**vmax) for LogNorm (default 0).
        qmax : float or None, optional
            If provided, restrict the upper extent of the displayed Qx/Qz axes to qmax.
        interpolation : str, optional
            Matplotlib interpolation method for imshow (default 'bicubic').
        filename : str, optional
            Full destination path to save the plot.
        handle_nan : bool, optional
            If True, properly handles NaN values in the data (default True).
        blockdirect : bool, optional
            If True, mask pixels where q < qmin (default False).
        qmin : float, optional
            Minimum q value - pixels with q < qmin will be masked if blockdirect=True (default 0.0).
        
        Returns
        -------
        None
        """
        # Reshape I_flat to match detector shape
        I_map = I_flat.reshape(self.Qx.shape)
        
        # Apply qmin masking if requested
        if blockdirect and qmin > 0:
            # Calculate q for each pixel
            q_map = np.sqrt(self.Qx**2 + self.Qz**2)
            # Mask pixels below qmin
            qmin_mask = q_map < qmin
            I_map = np.where(qmin_mask, np.nan, I_map)
        
        # Handle NaN values (from beamstop masking and/or qmin masking)
        if handle_nan and np.any(np.isnan(I_map)):
            valid_mask = ~np.isnan(I_map)
            
            if not np.any(valid_mask):
                raise ValueError("All values are NaN - cannot plot")
            
            # Normalize only using valid values
            I_max = np.max(I_map[valid_mask])
            I_map_normalized = I_map / I_max
            
            # Clip valid values, keep NaN as is temporarily
            I_map_plot = np.where(valid_mask, 
                                np.clip(I_map_normalized, 1e-12, 1.0),
                                np.nan)
            
            # Replace NaN with very small value for display
            I_map_display = np.where(np.isnan(I_map_plot), 
                                    1e-12,  # Small value for masked region
                                    I_map_plot)
        else:
            # Original behavior without NaN
            I_map_display = np.clip(I_map / np.max(I_map), 1e-12, 1.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Set up normalization
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10.0 ** vmin, vmax=10.0 ** vmax)
        else:
            norm = None
        
        # Set extent
        if qmax is None:
            extent = [self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()]
        else:
            extent = [self.Qx.min(), qmax, self.Qz.min(), qmax]
        
        # Plot
        im = ax.imshow(
            I_map_display,
            extent=extent,
            origin='lower',
            cmap=cmap,
            norm=norm,
            interpolation=interpolation
        )
        
        plt.colorbar(im, ax=ax, label="Intensity (log)" if log else "Intensity")
        ax.set_xlabel(r"$Q_x$ (Å$^{-1}$)")
        ax.set_ylabel(r"$Q_z$ (Å$^{-1}$)")
        
        # Status bar formatter
        def format_coord(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.degrees(np.arctan2(y, x))
            if r > 0:
                d = 2.0 * np.pi / (10.0 * r)
                return f"q={r:.4f} Å⁻¹, d={d:.4f} nm, θ={theta:.1f}°"
            else:
                return f"q={r:.4f} Å⁻¹, θ={theta:.1f}°"
        
        ax.format_coord = format_coord
        
        # Show masking info if applicable
        if handle_nan and np.any(np.isnan(I_map)):
            n_masked = np.sum(np.isnan(I_map))
            pct_masked = 100 * n_masked / I_map.size
            mask_text = f'Masked: {n_masked} pixels ({pct_masked:.1f}%)'
            if blockdirect and qmin > 0:
                mask_text += f'\n(beamstop + q<{qmin:.3f})'
            
            ax.text(0.02, 0.98, mask_text,
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {filename}")

    def plot_intensity_old(self, I_flat, log=True, vmin=-6, vmax=0, qmax=None, 
                    interpolation='bicubic', filename=None, handle_nan=True,cmap='jet'):
        """
        Plot a 2D intensity map on the detector Q-plane.
        Properly handles NaN values from beamstop masking.
        
        Parameters
        ----------
        I_flat : numpy.ndarray, shape (npix*npix,)
            Flattened intensity map to plot.
        log : bool, optional
            If True, plot on a logarithmic color scale (default True).
        vmin : float, optional
            Lower bound exponent (10**vmin) for LogNorm (default -6).
        vmax : float, optional
            Upper bound exponent (10**vmax) for LogNorm (default 0).
        qmax : float or None, optional
            If provided, restrict the upper extent of the displayed Qx/Qz axes to qmax.
        interpolation : str, optional
            Matplotlib interpolation method for imshow (default 'bicubic').
        filename : str, optional
            Full destination path to save the plot.
        handle_nan : bool, optional
            If True, properly handles NaN values in the data (default True).
        
        Returns
        -------
        None
        """
        # Reshape I_flat to match detector shape
        I_map = I_flat.reshape(self.Qx.shape)
        
        # Handle NaN values (from beamstop masking)
        if handle_nan and np.any(np.isnan(I_map)):
            valid_mask = ~np.isnan(I_map)
            
            if not np.any(valid_mask):
                raise ValueError("All values are NaN - cannot plot")
            
            # Normalize only using valid values
            I_max = np.max(I_map[valid_mask])
            I_map_normalized = I_map / I_max
            
            # Clip valid values, keep NaN as is temporarily
            I_map_plot = np.where(valid_mask, 
                                np.clip(I_map_normalized, 1e-12, 1.0),
                                np.nan)
            
            # Replace NaN with very small value for display
            I_map_display = np.where(np.isnan(I_map_plot), 
                                    1e-12,  # Small value for masked region
                                    I_map_plot)
        else:
            # Original behavior without NaN
            I_map_display = np.clip(I_map / np.max(I_map), 1e-12, 1.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Set up normalization
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10.0 ** vmin, vmax=10.0 ** vmax)
        else:
            norm = None
        
        # Set extent
        if qmax is None:
            extent = [self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()]
        else:
            extent = [self.Qx.min(), qmax, self.Qz.min(), qmax]
        
        # Plot
        im = ax.imshow(
            I_map_display,
            extent=extent,
            origin='lower',
            cmap=cmap,
            norm=norm,
            interpolation=interpolation
        )
        
        plt.colorbar(im, ax=ax, label="Intensity (log)" if log else "Intensity")
        ax.set_xlabel(r"$Q_x$ (Å$^{-1}$)")
        ax.set_ylabel(r"$Q_z$ (Å$^{-1}$)")
        
        # Status bar formatter
        def format_coord(x, y):
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.degrees(np.arctan2(y, x))
            if r > 0:
                d = 2.0 * np.pi / (10.0 * r)
                return f"q={r:.4f} Å⁻¹, d={d:.4f} nm, θ={theta:.1f}°"
            else:
                return f"q={r:.4f} Å⁻¹, θ={theta:.1f}°"
        
        ax.format_coord = format_coord
        
        # Show masking info if applicable
        if handle_nan and np.any(np.isnan(I_map)):
            n_masked = np.sum(np.isnan(I_map))
            pct_masked = 100 * n_masked / I_map.size
            ax.text(0.02, 0.98, f'Masked: {n_masked} pixels ({pct_masked:.1f}%)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {filename}")
        
        
    
    def plot_intensity_old(self, I_flat, log=True, vmin=-6, vmax=0, qmax=None, interpolation='bicubic',filename = None):
        """
        Plot a 2D intensity map on the detector Q-plane.

        Parameters
        ----------
        I_map : numpy.ndarray, shape (npix, npix)
            Intensity map to plot.
        log : bool, optional
            If True, plot on a logarithmic color scale (default True).
        vmin : float, optional
            Lower bound exponent (10**vmin) for LogNorm (default -6).
        vmax : float, optional
            Upper bound exponent (10**vmax) for LogNorm (default 0).
        qmax : float or None, optional
            If provided, restrict the upper extent of the displayed Qx/Qz axes to qmax.
        interpolation : str, optional
            Matplotlib interpolation method for imshow (default 'bicubic').
        filename: str, optional
            Full destination path to save the plot

        Returns
        -------
        None
        """
        I_map = I_flat.reshape(self.Qx.shape) # reshape I_flat to match detector shape
        I_map = np.clip(I_map / np.max(I_map), 1e-12, 1.0)
        fig, ax = plt.subplots(figsize=(6, 5))
        if log:
            norm = plt.matplotlib.colors.LogNorm(vmin=10.0 ** vmin, vmax=10.0 ** vmax)
        else:
            norm = None

        if qmax is None:
            extent = [self.Qx.min(), self.Qx.max(), self.Qz.min(), self.Qz.max()]
        else:
            extent = [self.Qx.min(), qmax, self.Qz.min(), qmax]

        im = ax.imshow(
            I_map,
            extent=extent,
            origin='lower',
            cmap='jet',
            norm=norm,
            interpolation=interpolation
        )

        plt.colorbar(im, ax=ax, label="Intensity (log)" if log else "Intensity")
        ax.set_xlabel("$q_x$ (Å⁻¹)")
        ax.set_ylabel("$q_z$ (Å⁻¹)")

        def format_coord(x, y):
            """
            Status-bar coordinate formatter that prints q, real-space d and angle.
            """
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.degrees(np.arctan2(y, x))
            if r > 0:
                d = 2.0 * np.pi / (10.0 * r)
                return f"q={r:.4f} Å⁻¹, d={d:.4f} nm, θ={theta:.1f}°"
            else:
                return f"q={r:.4f} Å⁻¹, θ={theta:.1f}°"
        ax.format_coord = format_coord

        if filename:
            plt.savefig(filename)

    # -----------------------------
    # Euler rotations
    # -----------------------------
    @staticmethod
    def euler_rotation_matrix(alpha, beta, gamma):
        """
        Return a rotation matrix for Z-Y-Z Euler angles (degrees).

        Parameters
        ----------
        alpha : float
            Rotation angle (degrees) about Z for the first rotation.
        beta : float
            Rotation angle (degrees) about Y (in the middle).
        gamma : float
            Rotation angle (degrees) about Z for the last rotation.

        Returns
        -------
        R : numpy.ndarray, shape (3, 3)
            Rotation matrix corresponding to the Z-Y-Z Euler sequence.
        """
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)
        ca, cb, cg = np.cos([alpha, beta, gamma])
        sa, sb, sg = np.sin([alpha, beta, gamma])
        Rz_alpha = np.array([[ca, -sa, 0],
                             [sa, ca, 0],
                             [0, 0, 1]])
        Ry_beta = np.array([[cb, 0, sb],
                            [0, 1, 0],
                            [-sb, 0, cb]])
        Rz_gamma = np.array([[cg, -sg, 0],
                             [sg, cg, 0],
                             [0, 0, 1]])
        return Rz_alpha @ Ry_beta @ Rz_gamma

    def rotate_positions(self, alpha, beta, gamma):
        """
        Rotate the stored atomic positions in-place using Euler angles.

        Parameters
        ----------
        alpha, beta, gamma : float
            Euler angles in degrees (ZYZ convention).

        Returns
        -------
        rotated_positions : numpy.ndarray, shape (N_atoms, 3)
            The rotated positions (also stored in self.positions).
        """
        R = self.euler_rotation_matrix(alpha, beta, gamma)
        self.positions = self.positions @ R.T
        return self.positions

    def shake_positions(self, frac_a, frac_b, frac_c, reference_length=None, seed=None):
        """
        Apply small random displacements ("shake") to atomic positions,
        using fractions of a characteristic interatomic distance.

        Parameters
        ----------
        frac_a, frac_b, frac_c : float
            Fractional amplitudes of random displacement along x, y, z.
            Each atom is shifted by a random amount within:
            [-frac_a * ref, +frac_a * ref] along x,
            [-frac_b * ref, +frac_b * ref] along y,
            [-frac_c * ref, +frac_c * ref] along z.
        reference_length : float, optional
            Characteristic distance used as reference (e.g. lattice constant).
            If None, it is estimated as the mean nearest-neighbor distance.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        shaken_positions : numpy.ndarray, shape (N_atoms, 3)
            The shaken (disordered) atomic positions (also stored in self.positions).
        """
        import numpy as np
        from scipy.spatial import cKDTree

        if seed is not None:
            np.random.seed(seed)

        coords = self.positions

        # Estimate reference length if not given: mean nearest-neighbor distance
        if reference_length is None:
            tree = cKDTree(coords)
            dists, _ = tree.query(coords, k=2)  # k=1 is self (distance 0), k=2 is nearest neighbor
            reference_length = np.mean(dists[:, 1])
        #print(f'Reference length is {reference_length:.2f}')
        # Compute maximum displacements per axis
        dx_max = frac_a * reference_length
        dy_max = frac_b * reference_length
        dz_max = frac_c * reference_length

        # Random displacements for each atom
        dx = np.random.uniform(-dx_max, dx_max, size=len(coords))
        dy = np.random.uniform(-dy_max, dy_max, size=len(coords))
        dz = np.random.uniform(-dz_max, dz_max, size=len(coords))

        # Apply displacements
        displacements = np.column_stack((dx, dy, dz))
        self.positions = coords + displacements

        return self.positions
    
    # ===========================================================
    # Zone axis tools
    # ===========================================================
    @staticmethod
    def indices_to_cartesian(uvw, crystal_system, lattice_params):
        """
        Fonction unifiée pour convertir les indices de Miller en cartésien
        pour TOUS les systèmes cristallins.
        
        Parameters
        ----------
        uvw : tuple
            Indices (u,v,w) ou (h,k,i,l) pour hexagonal
        crystal_system : str
            'cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 
            'monoclinic', 'triclinic'
        lattice_params : dict
            Paramètres selon le système :
            - cubic: {'a'}
            - tetragonal: {'a', 'c'}
            - orthorhombic: {'a', 'b', 'c'}
            - hexagonal: {'a', 'c'}
            - monoclinic: {'a', 'b', 'c', 'beta'}
            - triclinic: {'a', 'b', 'c', 'alpha', 'beta', 'gamma'}
        
        Returns
        -------
        x, y, z : float
            Coordonnées cartésiennes
        """
        
        if crystal_system == 'cubic':
            u, v, w = uvw
            a = lattice_params['a']
            return a*u, a*v, a*w
        
        elif crystal_system == 'tetragonal':
            u, v, w = uvw
            a = lattice_params['a']
            c = lattice_params['c']
            return a*u, a*v, c*w
        
        elif crystal_system == 'orthorhombic':
            u, v, w = uvw
            a = lattice_params['a']
            b = lattice_params['b']
            c = lattice_params['c']
            return a*u, b*v, c*w
        
        elif crystal_system == 'hexagonal':
            h, k, i, l = uvw
            a = lattice_params['a']
            c = lattice_params['c']
            x = a * (h - k/2)
            y = a * (k * np.sqrt(3) / 2)
            z = c * l
            return x, y, z
        
        elif crystal_system == 'monoclinic':
            u, v, w = uvw
            a = lattice_params['a']
            b = lattice_params['b']
            c = lattice_params['c']
            beta = lattice_params['beta']
            
            beta_rad = np.radians(beta)
            cos_beta = np.cos(beta_rad)
            sin_beta = np.sin(beta_rad)
            
            x = a * u + c * w * cos_beta
            y = b * v
            z = c * w * sin_beta
            return x, y, z
        
        elif crystal_system == 'triclinic':
            u, v, w = uvw
            a = lattice_params['a']
            b = lattice_params['b']
            c = lattice_params['c']
            alpha = lattice_params['alpha']
            beta = lattice_params['beta']
            gamma = lattice_params['gamma']
            
            alpha_rad = np.radians(alpha)
            beta_rad = np.radians(beta)
            gamma_rad = np.radians(gamma)
            
            cos_alpha = np.cos(alpha_rad)
            cos_beta = np.cos(beta_rad)
            cos_gamma = np.cos(gamma_rad)
            sin_gamma = np.sin(gamma_rad)
            
            V = a * b * c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 
                                    + 2*cos_alpha*cos_beta*cos_gamma)
            
            c_x = c * cos_beta
            c_y = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
            c_z = V / (a * b * sin_gamma)
            
            x = a * u + b * v * cos_gamma + c_x * w
            y = b * v * sin_gamma + c_y * w
            z = c_z * w
            return x, y, z
        
        else:
            raise ValueError(f"Système '{crystal_system}' non supporté")

    
    def zone_axis_to_rotation_matrix(self, uvw, crystal_system, lattice_params):
        """
        Calcul robuste de la matrice de rotation via optimisation axis-angle.
        Plus robuste que les angles d'Euler (pas de singularités).
        
        Parameters
        ----------
        uvw : tuple
            Indices de l'axe de zone
        crystal_system : str
            Système cristallin
        lattice_params : dict
            Paramètres de maille
        
        Returns
        -------
        R : numpy.ndarray, shape (3, 3)
            Matrice de rotation
        """
        # Conversion en cartésien
        x, y, z = self.indices_to_cartesian(uvw, crystal_system, lattice_params)
        
        target = np.array([x, y, z], dtype=float)
        norm = np.linalg.norm(target)
        
        if norm < 1e-10:
            raise ValueError("Vecteur nul !")
        
        target = target / norm
        desired = np.array([0, 1, 0])
        
        # Optimiser directement sur la représentation axis-angle (3 paramètres)
        def objective_rotvec(rotvec):
            """
            Optimiser sur rotvec = angle * axis (représentation compacte et robuste).
            """
            rot = Rotation.from_rotvec(rotvec)
            rotated = rot.apply(target)
            return np.linalg.norm(rotated - desired)**2
        
        # Initialisation : solution analytique (produit vectoriel)
        cross = np.cross(target, desired)
        cross_norm = np.linalg.norm(cross)
        
        if cross_norm > 1e-8:
            axis = cross / cross_norm
            angle = np.arccos(np.clip(np.dot(target, desired), -1.0, 1.0))
            initial_rotvec = angle * axis
        else:
            initial_rotvec = np.array([0, 0, 0])
        
        # Optimiser
        bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
        result = differential_evolution(
            objective_rotvec,
            bounds,
            x0=initial_rotvec,
            seed=42,
            atol=1e-12,
            tol=1e-12,
            maxiter=1000
        )
        
        # Créer la matrice de rotation
        rot_final = Rotation.from_rotvec(result.x)
        R = rot_final.as_matrix()
        
        return R
    
    def zone_axis_to_euler(self,uvw, crystal_system, lattice_params, gamma_fixed=None):
        """
        Calcul des angles d'Euler à partir de la matrice de rotation robuste.
        
        Parameters
        ----------
        uvw : tuple
            Indices de l'axe de zone
        crystal_system : str
            Système cristallin
        lattice_params : dict
            Paramètres de maille
        gamma_fixed : float or None
            Non utilisé dans cette version (gardé pour compatibilité)
        
        Returns
        -------
        alpha, beta, gamma : float
            Angles d'Euler (degrés) - convertis à partir de la matrice de rotation
        """
        # Obtenir la matrice de rotation robuste
        R = self.zone_axis_to_rotation_matrix(uvw, crystal_system, lattice_params)
        
        # Convertir en angles d'Euler pour affichage
        rot_obj = Rotation.from_matrix(R)
        euler_angles = rot_obj.as_euler('zyz', degrees=True)
        alpha, beta, gamma = euler_angles
        
        return alpha, beta, gamma
    
    #-------------------------------------------------------------------------------------------------------------
    # Rotate to zone axis
    def rotate_to_zone_axis(self, uvw, crystal_system='cubic', lattice_params=None):
        """
        Rotate the structure so that the specified zone axis aligns with the y-axis.
        Utilise une approche robuste basée sur l'optimisation axis-angle.

        Parameters
        ----------
        uvw : tuple
            Indices of the zone axis.
        crystal_system : str
            Crystal system.
        lattice_params : dict
            Lattice parameters.

        Returns
        -------
        rotated_positions : numpy.ndarray, shape (N_atoms, 3)
            The rotated atomic positions (also stored in self.positions).
        """
        # Obtenir la matrice de rotation robuste (axis-angle)
        R = self.zone_axis_to_rotation_matrix(uvw, crystal_system, lattice_params)
        
        # Appliquer la rotation
        self.positions = self.positions @ R.T
        return self.positions
    
    def write_xyz(self, output_file):
        """
        Write the current structure to an XYZ file.

        Parameters
        ----------
        output_file : str
            Path to the output XYZ file.
        """
        line2with = f"{len(self.positions)}\n\n"
        with open(output_file, 'w') as f:
            f.write(line2with)
            for elem, pos in zip(self.elements, self.positions):
                f.write(f"{elem}\t{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\n")

    # -----------------------------
    # pyFAI convenience
    # -----------------------------
    def ai(self):
        """
        Create and return a pyFAI AzimuthalIntegrator configured for the detector.

        Returns
        -------
        ai : pyFAI.AzimuthalIntegrator
            Configured integrator instance.

        Notes
        -----
        pyFAI must be installed for this method to work.
        """
        if AzimuthalIntegrator is None or Detector is None:
            raise RuntimeError("pyFAI is not available. Install 'pyFAI' to use integrate functions.")
        detector = Detector(pixel1=self.pixel_size, pixel2=self.pixel_size)
        ai = AzimuthalIntegrator(dist=self.D, detector=detector)
        ai.setFit2D(self.D * 1000.0, self.npix / 2.0, self.npix / 2.0, wavelength=self.wl)
        return ai

    def integrate_with_pyfai(self, I_flat, plot=False):
        """
        Integrate a 2D intensity map radially (1D integration) using pyFAI.

        Parameters
        ----------
        I : numpy.ndarray, shape (npix, npix)
            2D intensity map in detector coordinates.
        plot : bool, optional
            If True, plot the integrated 1D I(q).

        Returns
        -------
        q : numpy.ndarray
            1D q-grid returned by pyFAI (Å⁻¹).
        i : numpy.ndarray
            Radially integrated intensity values.
        """
        I = I_flat.reshape(self.Qx.shape)
        ai = self.ai()
        q, i = ai.integrate1d(I, npt=1000, unit="q_A^-1")
        if plot:
            plt.figure()
            plt.loglog(q, i, 'k-')
            plt.xlabel("Q (Å⁻¹)")
            plt.ylabel("I(Q) (a.u.)")
            plt.xlim(self.q_min, self.q_max)
            plt.title("Radial integration in Q-space")
            plt.grid(True)
            plt.show()
        return q, i

    # ===========================================================
    # Structure factor computation
    # -==========================================================
    def compute_structure_factor(self,N,Z,use_gpu=True):
        """ 
        N:int
            number of atoms in particle
        Z: int
            atomic number of atoms in particle        
        """
        return N*Z**2*self.compute_intensity(use_gpu=use_gpu)


# ===============================================================
# Numba core
# ===============================================================

if NUMBA_AVAILABLE:

    @njit(parallel=True, fastmath=True, nogil=True)
    def compute_intensity_numba(positions, f_atom, Q):
        n_q = Q.shape[0]
        n_atoms = positions.shape[0]
        I = np.empty(n_q, dtype=np.float64)
        for iq in prange(n_q):
            qx, qy, qz = Q[iq]
            Re, Im = 0.0, 0.0
            for ia in range(n_atoms):
                phase = qx * positions[ia, 0] + qy * positions[ia, 1] + qz * positions[ia, 2]
                Re += math.cos(phase)
                Im += math.sin(phase)
            I[iq] = (Re ** 2 + Im ** 2) * (f_atom[iq] ** 2)
        return I

    @njit(parallel=True, fastmath=True, nogil=True)
    def compute_intensity_fibonacci_numba(positions, f_q, q_vals, dirs):
        n_atoms = positions.shape[0]
        n_q = len(q_vals)
        n_orient = dirs.shape[0]
        Iq = np.zeros(n_q, dtype=np.float64)

        for iq in prange(n_q):
            q = q_vals[iq]
            f = f_q[iq]
            I_sum = 0.0
            for io in range(n_orient):
                qx, qy, qz = q * dirs[io, :]
                Re = 0.0
                Im = 0.0
                for ia in range(n_atoms):
                    phase = qx * positions[ia, 0] + qy * positions[ia, 1] + qz * positions[ia, 2]
                    Re += math.cos(phase)
                    Im += math.sin(phase)
                I_sum += (Re ** 2 + Im ** 2)
            Iq[iq] = (I_sum / n_orient) * (f ** 2)
        return Iq

    @njit(parallel=True, fastmath=True, nogil=True)
    def compute_intensity_uniaxial_numba(positions, f_atom, Q, theta_y, theta_z):
        n_samples = len(theta_y)
        n_q = Q.shape[0]
        n_atoms = positions.shape[0]
        I_accum = np.zeros(n_q, dtype=np.float64)

        for isamp in prange(n_samples):
            ty = theta_y[isamp]
            tz = theta_z[isamp]
            cy, sy = math.cos(ty), math.sin(ty)
            cz, sz = math.cos(tz), math.sin(tz)
            R = np.array([
                [cz * cy, -sz, cz * sy],
                [sz * cy,  cz, sz * sy],
                [-sy,      0.0, cy]
            ])
            pos_rot = np.zeros_like(positions)
            for i in range(n_atoms):
                x, y, z = positions[i]
                pos_rot[i, 0] = R[0, 0]*x + R[0, 1]*y + R[0, 2]*z
                pos_rot[i, 1] = R[1, 0]*x + R[1, 1]*y + R[1, 2]*z
                pos_rot[i, 2] = R[2, 0]*x + R[2, 1]*y + R[2, 2]*z

            for iq in range(n_q):
                qx, qy, qz = Q[iq]
                f = f_atom[iq]
                Re, Im = 0.0, 0.0
                for ia in range(n_atoms):
                    phase = qx * pos_rot[ia, 0] + qy * pos_rot[ia, 1] + qz * pos_rot[ia, 2]
                    Re += math.cos(phase)
                    Im += math.sin(phase)
                I_accum[iq] += (Re ** 2 + Im ** 2) * (f ** 2)
        return I_accum / n_samples


# ===============================================================
# Script entry point
# ===============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debye3D hybrid CPU/GPU with orientation averaging")
    parser.add_argument("structure", help="Structure file (XYZ, CIF, ... supported by ASE)")
    parser.add_argument("--device", default=None, help="Torch device (e.g. cuda:0 or cpu)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--npix", type=int, default=250, help="Detector pixel count")
    parser.add_argument("--wl", type=float, default=1.0, help="Wavelength (Å)")
    parser.add_argument("--distance", type=float, default=0.5, help="Sample-detector distance (m)")
    parser.add_argument("--pixel", type=float, default=0.0001, help="Pixel size (m)")
    parser.add_argument("--fib", type=int, default=0, help="Average over N orientations using Fibonacci grid (0 = disabled)")
    parser.add_argument("--uniax", type=int, default=0, help="Average over N uniaxial orientations (0 = disabled)")
    parser.add_argument("--sigma_y", type=float, default=2.0, help="σy for uniaxial distribution (deg)")
    parser.add_argument("--sigma_z", type=float, default=2.0, help="σz for uniaxial distribution (deg)")
    args = parser.parse_args()

    # 1️⃣ Initialisation de l’expérience
    exp = Experiment(npix=args.npix, wl=args.wl, distance=args.distance,
                     pixel_size=args.pixel, verbose=True)

    # 2️⃣ Création du modèle Debye3D
    model = Debye3D(structure_file=args.structure,
                    npix=exp.npix,
                    wl=exp.wl,
                    distance=exp.distance,
                    pixel_size=exp.pixel_size,
                    verbose=True,
                    torch_device=args.device)

    print("\n==============================================================")
    print(f" Using device : {model.device}")
    print(f" GPU available : {torch.cuda.is_available() if TORCH_AVAILABLE else False}")
    print("==============================================================\n")

    # 3️⃣ Choix automatique du batch
    batch = model.auto_batch_size()

    # 4️⃣ Calcul de l’intensité
    if args.fib > 0:
        print(f"\n>>> Calcul orientationnel (Fibonacci, {args.fib} directions)...\n")
        I = model.compute_intensity_fibonacci(n_orient=args.fib, use_gpu=args.use_gpu, batch_size=batch)
    elif args.uniax > 0:
        print(f"\n>>> Calcul uniaxial ({args.uniax} échantillons, σy={args.sigma_y}°, σz={args.sigma_z}°)...\n")
        I = model.compute_intensity_uniaxial(n_samples=args.uniax,
                                             sigma_y=args.sigma_y,
                                             sigma_z=args.sigma_z,
                                             use_gpu=args.use_gpu,
                                             batch_size=batch)
    else:
        print("\n>>> Calcul d’intensité simple (aucune moyenne orientationnelle)...\n")
        I = model.compute_intensity(use_gpu=args.use_gpu, batch_size=batch)

    # 5️⃣ Affichage de la carte d’intensité
    model.plot_intensity(I)
    print("\n✅ Calcul terminé.")
