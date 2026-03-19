from ase.spacegroup import crystal
from ase.io import write
import numpy as np

def generate_supercell(cellpar, spacegroup = 225, supercell_size=(10, 10, 10), output_file="supercell.xyz"):
    """
    Génère une supermaille monoatomique à partir d'un groupe d'espace et de paramètres de maille.

    Paramètres
    ----------
    cellpar : list ou tuple de 6 floats
        Paramètres de maille [a, b, c, alpha, beta, gamma] en Å et degrés
    spacegroup : int
        Numéro du groupe d'espace (1 à 230)
    supercell_size : tuple de 3 int, optionnel
        Facteur de répétition dans les directions (nx, ny, nz)
    output_file : str, optionnel
        Nom du fichier de sortie (.xyz)
    """

    # Création de la maille élémentaire monoatomique
    atoms = crystal('Au',
                    basis=[(0, 0, 0)],
                    spacegroup=spacegroup,
                    cellpar=cellpar)

    # Création de la supermaille
    supercell = atoms.repeat(supercell_size)

    # Sauvegarde au format XYZ
    write(output_file, supercell)
    print(f"Supermaille générée : {output_file} ({len(supercell)} atomes)")
    return output_file

def generate_paracrystal_supercell(
    cellpar,
    spacegroup=225,
    supercell_size=(10, 10, 10),
    g=0.05,
    output_file="supercell_paracrystal.xyz"
):
    """
    Génère une supermaille paracristalline 3D avec désordre cumulatif (paramètre g).
    g ∈ [0.01, 0.25]
    """

    # Maille élémentaire parfaite
    atoms = crystal(
        'Au',
        basis=[(0, 0, 0)],
        spacegroup=spacegroup,
        cellpar=cellpar
    )

    # Supermaille parfaite
    supercell = atoms.repeat(supercell_size)

    # Vecteurs de maille
    cell = atoms.get_cell()
    a1, a2, a3 = cell

    # Positions idéales
    positions = supercell.get_positions()

    # Indices de répétition
    nx, ny, nz = supercell_size
    indices = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                indices.append((ix, iy, iz))
    indices = np.array(indices)

    # Génération du désordre cumulatif
    # Créer les displacements pour chaque cellule unitaire
    cell_displacements = (
        indices[:, 0][:, None] * np.random.normal(0, g, size=(len(indices), 1)) * a1 +
        indices[:, 1][:, None] * np.random.normal(0, g, size=(len(indices), 1)) * a2 +
        indices[:, 2][:, None] * np.random.normal(0, g, size=(len(indices), 1)) * a3
    )
    
    # Répliquer les displacements pour chaque atome dans la cellule FCC (4 atomes)
    num_atoms_per_cell = len(supercell) // len(indices)
    displacements = np.repeat(cell_displacements, num_atoms_per_cell, axis=0)

    positions += displacements

    supercell.set_positions(positions)

    write(output_file, supercell)
    print(f"Supermaille paracristalline générée : {output_file}")
    return output_file

def honeycomb(a, n_y, n_z, n_layers, dx, yz_noise=0.5, seed=None):
    """
    Génère un empilement de lamelles hexagonales 2D le long de x,
    avec des positions (y,z) légèrement aléatoires d'une lamelle à l'autre.

    Parameters
    ----------
    a : float
        Paramètre de maille hexagonale (distance entre voisins dans la lamelle).
    n_y, n_z : int
        Nombre de points le long de y et z dans chaque lamelle.
    n_layers : int
        Nombre de lamelles le long de x.
    dx : float
        Distance entre les lamelles le long de x.
    yz_noise : float
        Amplitude du bruit aléatoire ajouté à y et z pour chaque lamelle.
    seed : int, optional
        Seed pour reproductibilité.

    Returns
    -------
    coords : numpy.ndarray, shape (N_atoms, 3)
        Coordonnées (x,y,z) de tous les points du réseau.
    """
    if seed is not None:
        np.random.seed(seed)

    coords = []

    # vecteurs de base du réseau hexagonal 2D (y,z)
    a1 = np.array([a, 0])
    a2 = np.array([a/2, a*np.sqrt(3)/2])

    for k in range(n_layers):
        x_offset = k * dx
        # génération d'un petit décalage aléatoire pour cette lamelle
        delta_yz = np.random.uniform(-yz_noise, yz_noise, size=2)

        for i in range(n_y):
            for j in range(n_z):
                y, z = i*a1[0] + j*a2[0], i*a1[1] + j*a2[1]
                # on ajoute le petit décalage aléatoire spécifique à cette lamelle
                coords.append([x_offset, y + delta_yz[0], z + delta_yz[1]])

    return np.array(coords)

def honeycomb_disordered(
    a=5,
    n_y=20,
    n_z=20,
    n_layers=10,
    dx=100,
    yz_noise=0.5,
    layer_shift_frac=0.05,
    atomic_disorder_frac=(0.05, 0.05, 0.05),
    seed=None
):
    """
    Generate a stack of 2D hexagonal (honeycomb-like) layers along the x-axis,
    with random positional disorder at both the layer and atomic levels.
    Atomic disorder can be isotropic or anisotropic.

    Parameters
    ----------
    a : float
        Lattice constant of the 2D hexagonal structure (distance between nearest neighbors).
    n_y, n_z : int
        Number of lattice points along the y and z directions within each layer.
    n_layers : int
        Number of layers stacked along the x direction.
    dx : float
        Interlayer spacing along x.
    layer_shift_frac : float, optional
        Amplitude of global random shift applied to each layer in the (y, z) plane,
        expressed as a fraction of `a`.
        Example: 0.05 → each layer is shifted by ±5% of `a`.
    atomic_disorder_frac : float or tuple of 3 floats, optional
        Fractional amplitude of local random displacement applied to each atom.
        - If float: isotropic disorder applied to x, y, z.
        - If tuple (fx, fy, fz): anisotropic disorder along x, y, z.
        Example: (0.03, 0.05, 0.02) → ±3% of dx along x, ±5% of a along y, ±2% of a along z.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    coords : numpy.ndarray of shape (N_atoms, 3)
        Cartesian coordinates (x, y, z) of all atoms in the disordered structure.
    """
    if seed is not None:
        np.random.seed(seed)

    # handle isotropic vs anisotropic disorder
    if isinstance(atomic_disorder_frac, (int, float)):
        frac_x = frac_y = frac_z = atomic_disorder_frac
    elif isinstance(atomic_disorder_frac, (tuple, list)) and len(atomic_disorder_frac) == 3:
        frac_x, frac_y, frac_z = atomic_disorder_frac
    else:
        raise ValueError("atomic_disorder_frac must be a float or a tuple/list of 3 floats")

    coords = []

    # 2D hexagonal lattice basis vectors (y-z plane)
    a1 = np.array([a, 0])
    a2 = np.array([a / 2, a * np.sqrt(3) / 2])

    # amplitudes of local atomic disorder
    noise_x = frac_x * dx
    noise_y = frac_y * a
    noise_z = frac_z * a

    for k in range(n_layers):
        # global shift for the layer (fraction of a)
        delta_yz = np.random.uniform(-yz_noise * a, yz_noise * a, size=2)
        delta_x = np.random.uniform(-layer_shift_frac*dx,layer_shift_frac*dx)
        
        x_offset = k * dx + delta_x

        
        for i in range(n_y):
            for j in range(n_z):
                # ideal lattice positions
                y = i * a1[0] + j * a2[0]
                z = i * a1[1] + j * a2[1]

                # anisotropic atomic disorder
                dx_rand = np.random.uniform(-noise_x, noise_x)
                dy_rand = np.random.uniform(-noise_y, noise_y)
                dz_rand = np.random.uniform(-noise_z, noise_z)

                coords.append([
                    x_offset + dx_rand,
                    y + delta_yz[0] + dy_rand,
                    z + delta_yz[1] + dz_rand
                ])

    return np.array(coords)



