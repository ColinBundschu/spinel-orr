
import dataclasses
import fractions
from typing import Iterable

import numpy as np

import studies.configs as configs
from ecat_types import (AdsorbateInterface, Atoms, Calc, Lattice, LatticeType,
                        Material)
from ecat_types.atom import Atom
from jdftx.jdftx import VAC_A_DEFAULT
from studies.spinel.spinel import Spinel


@dataclasses.dataclass(frozen=True, order=True)
class SurfaceConfig:
    max_supercell: int
    suggested_area: float
    x_count: int | None
    y_count: int | None
    z_count: int
    bounds_L: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    removals: tuple[tuple[float, float, float], ...]
    lock_depth_L: float

    def __post_init__(self):
        bounds = np.array(self.bounds_L)
        if bounds.shape != (3, 2):
            raise ValueError(f'Bounds should be a 3x2 array, not {bounds.shape}')
        if not all(bounds[:, 0] >= 0):
            raise ValueError(f'Bounds should be positive, not {bounds}')
        if not all(bounds[:, 1] <= 1):
            raise ValueError(f'Bounds should be less than 1, not {bounds}')


def add_graphene(lattice_A: Lattice, atoms_L: Atoms, adsorbate: AdsorbateInterface | None, *, height_A: float = 2.15,
                 mismatch_tol: float = 0.04, xy_offset_A: tuple[float, float] = (0, 0), graphene_unit_x_A: float = 4.27,
                 graphene_unit_y_A: float = 2.47) -> Atoms:
    '''Add a graphene layer to the top of the slab'''
    if lattice_A.lattice_type not in [LatticeType.Cubic, LatticeType.Orthorhombic]:
        raise NotImplementedError('Only cubic and orthorhombic lattices are supported by this method')

    off_x_L = xy_offset_A[0] / lattice_A.a
    off_y_L = xy_offset_A[1] / lattice_A.b

    x_count = round(lattice_A.a / graphene_unit_x_A)
    graphene_x_A = x_count * graphene_unit_x_A
    if abs(graphene_x_A - lattice_A.a) / lattice_A.a > mismatch_tol:
        raise ValueError(f'{x_count}-fold x tiling mismatch: graphene {graphene_x_A} A vs substrate {lattice_A.a} A')

    y_count = round(lattice_A.b / graphene_unit_y_A)
    graphene_y_A = y_count * graphene_unit_y_A
    if abs(graphene_y_A - lattice_A.b) / lattice_A.b > mismatch_tol:
        raise ValueError(f'{y_count}-fold y tiling mismatch: graphene {graphene_y_A} A vs substrate {lattice_A.b} A')

    new_atoms_L = []
    z_L = max(atom.z for atom in atoms_L) + height_A/lattice_A.c
    for x in range(x_count):
        x0_L = (x + 1/2)/x_count + off_x_L
        x1_L = (x + 2/3)/x_count + off_x_L
        x2_L = x/x_count + off_x_L
        x3_L = (x + 1/6)/x_count + off_x_L
        for y in range(y_count):
            y0_L = (y + 1/2)/y_count + off_y_L
            y1_L = y/y_count + off_y_L
            new_atoms_L.append(Atom(6, (x0_L, y0_L, z_L)))
            new_atoms_L.append(Atom(6, (x1_L, y1_L, z_L)))
            new_atoms_L.append(Atom(6, (x2_L, y1_L, z_L)))
            new_atoms_L.append(Atom(6, (x3_L, y0_L, z_L)))

    if str(adsorbate).startswith('y1-'):
        new_atoms_L = [atom for atom in new_atoms_L if not 0.46 < atom.x < 0.58]

    if any([str(adsorbate).startswith(name) for name in ['s1-', 'xs2-', 't4-']]):
        target_xy = (0.5, 0.5)
        target = min(new_atoms_L, key=lambda atom: (atom.x - target_xy[0]) ** 2 + (atom.y - target_xy[1]) ** 2)
        new_atoms_L.remove(target)
        # Find the n atoms closest to the removed atom and remove them as well
        n = int(str(adsorbate).split('-', maxsplit=1)[0][-1]) - 1
        closest_n_atoms = sorted(new_atoms_L, key=lambda atom: (atom.x - target.x) ** 2 + (atom.y - target.y) ** 2)[:n]
        for atom in closest_n_atoms:
            new_atoms_L.remove(atom)

    return Atoms(atoms_L + tuple(new_atoms_L))


def make_default_calc(material: Material) -> Calc:
    try:
        return material.default_config
    except NotImplementedError:
        return getattr(configs, material.basename)()


def compute_slab_geometry(slab_material: Material, adsorbate: AdsorbateInterface | None,
                          base_calc: Calc, base_geo: str, *, vacuum_z_A: float = VAC_A_DEFAULT) -> Calc:
    '''The normal vector of the slab is assumed to be in the +z direction'''

    if base_geo == 'bulk':
        surface_config = make_fc_config(slab_material, base_calc.lattice_A)
        calc = orient_crystal(slab_material.miller_indices, base_calc, surface_config, slab_material)

        atoms_L = perform_removals(calc.atoms_L, surface_config)
        if slab_material.frozen_layers:
            atoms_L = Atoms([dataclasses.replace(atom, frozen=atom.z <= surface_config.lock_depth_L)
                            for atom in atoms_L])
        if slab_material.basename == 'Ni' and slab_material.geo == 'slab':
            atoms_L = add_graphene(calc.lattice_A, atoms_L, adsorbate)
        if slab_material.replacement_layers:
            # compute the fraction of replacement layers vs total layers:
            replacement_fraction = slab_material.replacement_layers / slab_material.layers
            z_max = max(atom.z for atom in atoms_L)
            z_min = min(atom.z for atom in atoms_L)
            z_replacement = z_min + replacement_fraction * (z_max - z_min)
            atoms_L = Atoms([dataclasses.replace(atom, atomic_num=slab_material.replacement_atomic_num)
                             if atom.z <= z_replacement else atom for atom in atoms_L])

        unspaced_calc = Calc(calc.lattice_A, atoms_L)
    elif base_geo == 'slab':
        unspaced_calc = base_calc
    else:
        raise NotImplementedError()

    spaced_calc = adjust_vacuum_and_recenter(unspaced_calc, (None, None, vacuum_z_A))
    return adsorbate.attach(spaced_calc) if adsorbate else spaced_calc


def compute_bulk_geometry(material: Material, primitive_calc: Calc) -> Calc:
    if material.facet:
        surface_config = make_fc_config(material, primitive_calc.lattice_A)
        return orient_crystal(material.miller_indices, primitive_calc, surface_config, material)
    return primitive_calc


def calculate_miller_matrix_Z(z_miller: tuple[float, float, float], R: np.ndarray, *,
                              target_area: float = 0, MAX_SUPERCELL: int = 8) -> np.ndarray:
    '''
    Given a Miller index in the primitive cell coordinates (z_miller)
    and the encoding of the lattice vectors in a Cartesian system,
    construct a supercell that is axis-aligned in Cartesian coordinates.
    '''
    EPS = 1e-9

    # Step 1: Generate indices for the supercell using np.mgrid
    grid_range = np.mgrid[-MAX_SUPERCELL:MAX_SUPERCELL+1,
                          -MAX_SUPERCELL:MAX_SUPERCELL+1,
                          -MAX_SUPERCELL:MAX_SUPERCELL+1]

    # Reshape the grid into an (N x 3) array where N is the number of points in the grid
    indices = grid_range.reshape(3, -1).T

    # Convert to Cartesian coordinates using the lattice matrix R
    raw_candidates = indices @ R.T
    non_zero_mask = np.linalg.norm(raw_candidates, axis=1) > EPS
    indices = indices[non_zero_mask]
    candidates = raw_candidates[non_zero_mask]
    z_R = R @ np.array(z_miller)

    # Step 1: Filter by orthogonality of x and z
    z_ortho_mask = np.abs(np.dot(candidates, z_R)) < EPS
    N_ortho = np.sum(z_ortho_mask)
    ortho_i = indices[z_ortho_mask]
    ortho_R = candidates[z_ortho_mask]

    # Step 2: Generate the x and y combinations for filtering
    x_i = np.repeat(ortho_i, len(indices), axis=0) # AAABBBCCC
    x_R = np.repeat(ortho_R, len(indices), axis=0) # AAABBBCCC
    y_i = np.tile(indices, (N_ortho, 1)) # ABCABCABC
    y_R = np.tile(candidates, (N_ortho, 1)) # ABCABCABC

    # Step 3: Filter out combinations that are not square enough
    ratios = np.linalg.norm(x_R, axis=1) / np.linalg.norm(y_R, axis=1)
    square_enough = (0.5 < ratios) & (ratios < 2)
    x_i = x_i[square_enough]
    x_R = x_R[square_enough]
    y_i = y_i[square_enough]
    y_R = y_R[square_enough]

    # Step 4: Filter by orthogonality of x and y
    x_cross_z_R = np.cross(x_R, z_R)
    x_cross_z_R_normalized = x_cross_z_R / np.linalg.norm(x_cross_z_R, axis=1)[:, np.newaxis]
    y_R_normalized = y_R / np.linalg.norm(y_R, axis=1)[:, np.newaxis]
    y_alignment = np.abs(np.einsum('ij,ij->i', x_cross_z_R_normalized, y_R_normalized))
    xy_ortho_mask = y_alignment > max(y_alignment) - EPS
    x_i = x_i[xy_ortho_mask]
    x_R = x_R[xy_ortho_mask]
    y_i = y_i[xy_ortho_mask]
    y_R = y_R[xy_ortho_mask]

    # Step 5: Filter by handedness (positive dot product with z_R)
    xy_crosses_R = np.cross(x_R, y_R)
    handedness_mask = np.einsum('ij,j->i', xy_crosses_R, z_R) > 0
    x_i = x_i[handedness_mask]
    x_R = x_R[handedness_mask]
    y_i = y_i[handedness_mask]
    y_R = y_R[handedness_mask]
    xy_crosses_R = xy_crosses_R[handedness_mask]

    # Step 6: Filter for the area of cross products
    xy_areas = np.linalg.norm(xy_crosses_R, axis=1)
    area_deltas = abs(xy_areas - target_area)
    area_mask = area_deltas < EPS + min(area_deltas)
    x_i = x_i[area_mask]
    x_R = x_R[area_mask]
    y_i = y_i[area_mask]
    y_R = y_R[area_mask]

    # Step 7: Choose the pair with the most square shape
    vector_lengths_diff = np.abs(np.linalg.norm(x_R, axis=1) - np.linalg.norm(y_R, axis=1))
    min_length_diff = np.min(vector_lengths_diff)
    square_mask = np.abs(vector_lengths_diff - min_length_diff) < EPS
    x_i = x_i[square_mask]
    y_i = y_i[square_mask]

    # Final result: Miller matrix aligned with the Cartesian system
    miller_matrix = np.array([x_i[0], y_i[0], z_miller], dtype=int).T
    return miller_matrix


def populate_supercell(Z: np.ndarray, frac_coord_atoms: Atoms, flip_y: bool) -> Atoms:
    Z_inv_t = np.linalg.inv(Z).T  # Transpose Z_inv to prepare it for broadcasting
    N = round(np.linalg.det(Z))

    # Generate all possible lattice shifts using mgrid
    ranges = [slice(-N, N+1) for _ in range(3)]
    lattice_shifts = np.mgrid[ranges].reshape(3, -1).T

    # Add a new axis to atom_positions and lattice_shifts_all to enable broadcasting
    atom_positions = np.array([(atom.x + 1e-4, atom.y + 1e-5, atom.z + 1e-6) for atom in frac_coord_atoms])
    atom_positions = atom_positions[:, np.newaxis, :]
    lattice_shifts = lattice_shifts[np.newaxis, :, :]
    shifted_atom_positions = np.dot(atom_positions + lattice_shifts, Z_inv_t)

    # Compute the mask of valid positions and apply it to shifted_atom_positions
    valid_mask = np.all((shifted_atom_positions < 1) & (shifted_atom_positions >= 0), axis=2)
    valid_positions = shifted_atom_positions[valid_mask]
    if len(valid_positions) != len(frac_coord_atoms) * N:
        raise ValueError(f'Invalid positions: {valid_positions}')

    # Adjustments made after selecting positions
    if flip_y:
        valid_positions[:, 1] = 1 - valid_positions[:, 1]
    valid_positions[valid_positions > 0.99] = 0


    # Prepare the final array of atoms valid positions list atoms in blocks of N ordered by type
    new_atoms = Atoms([atom.with_new_xyz(tuple(valid_positions[N*i+n]))
                       for i, atom in enumerate(frac_coord_atoms)
                       for n in range(N)])
    return new_atoms


def clean_miller_vec(millers: Iterable) -> np.ndarray[int]:
    '''Convert miller indices to integers and reduce to lowest terms. Accepts a single vector'''
    fracs = [fractions.Fraction(x).limit_denominator(100000) for x in millers]
    d = np.lcm.reduce([f.denominator for f in fracs])
    ints = np.array([int(f * d) for f in fracs])
    return ints // np.gcd.reduce(ints)


def orient_crystal(z_cubic: tuple[int, int, int], bulk_calc: Calc, surface_config: SurfaceConfig,
                   material: Material) -> Calc:
    '''The resulting lattice will be in cartesian coordinates'''
    R_A = bulk_calc.lattice_A.R_A
    if bulk_calc.lattice_A.lattice_type in [LatticeType.Cubic, LatticeType.Orthorhombic]:
        flip_y = False
        transform = np.identity(3)
        Z = np.stack([clean_miller_vec(v) for v in np.linalg.inv(R_A).T], axis=-1)
        Z[:, 2] *= surface_config.z_count
        a, b, c = np.diag(R_A @ Z)
    elif z_cubic == (1, 0, 0):
        flip_y = False
        transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Shife the z index left 1 to be 0 0 1
        Z = np.stack([clean_miller_vec(v) for v in np.linalg.inv(R_A).T], axis=-1)
        Z[:, 0] *= surface_config.z_count
        Z[:, 1] *= surface_config.x_count
        Z[:, 2] *= surface_config.y_count
        c, a, b = np.diag(R_A @ Z)
    elif z_cubic == (0, 1, 0):
        flip_y = False
        transform = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # Shift the z index right 1 to be 0 0 1
        Z = np.stack([clean_miller_vec(v) for v in np.linalg.inv(R_A).T], axis=-1)
        Z[:, 1] *= surface_config.z_count
        b, c, a = np.diag(R_A @ Z)
    elif z_cubic == (0, 0, 1):
        flip_y = False
        transform = np.identity(3)
        Z = np.stack([clean_miller_vec(v) for v in np.linalg.inv(R_A).T], axis=-1)
        Z[:, 2] *= surface_config.z_count
        a, b, c = np.diag(R_A @ Z)
    elif z_cubic == (1, 1, 1) and material.geo == 'slab' and isinstance(material, Spinel):
        transform = np.identity(3)
        if material.facet == '111b':
            Z = np.array([[-1,  1,  1], [0, -2,  1], [1,  1,  1]])
        elif material.facet == '111s':
            Z = np.array([[-2,  1,  1], [ 0, -2,  1], [ 2,  1,  1]])
        else:
            raise ValueError(f'Invalid facet: {material.facet}')
        Z[:, 2] *= surface_config.z_count
        a, b, c = np.linalg.norm(R_A @ Z, axis=0)
        flip_y = Z[2,0] == 0
    else:
        transform = np.identity(3)
        Z = calculate_miller_matrix_Z(z_cubic, R_A, target_area=surface_config.suggested_area,
                                        MAX_SUPERCELL=surface_config.max_supercell)
        Z[:, 2] *= surface_config.z_count
        a, b, c = np.linalg.norm(R_A @ Z, axis=0)
        if not np.isclose(a*b, surface_config.suggested_area):
            raise ValueError(f'Area mismatch: {a*b} vs {surface_config.suggested_area}')
        flip_y = Z[2,0] == 0

    supercell_atoms = populate_supercell(Z, bulk_calc.atoms_L, flip_y).transformed(transform)
    cubic = np.isclose(a, b) and np.isclose(a, c)
    lattice_A = Lattice(LatticeType.Cubic, a=a) if cubic else Lattice(LatticeType.Orthorhombic, a=a, b=b, c=c)
    return Calc(lattice_A, supercell_atoms)


def adjust_vacuum_and_recenter(calc: Calc, desired_vac_A: tuple[float, float, float]) -> Calc:
    '''desired_vac_A should be a tuple of desired vacuum widths'''
    if calc.lattice_A.lattice_type not in [LatticeType.Cubic, LatticeType.Orthorhombic]:
        raise NotImplementedError('Only cubic and orthorhombic lattices are supported by this method')

    vacuum_indices = [i for i, j in enumerate(desired_vac_A) if j is not None]
    R_A = calc.lattice_A.R_A

    min_bounds, max_bounds = calc.atoms_L.bounds.T
    thicknesses_L = max_bounds - min_bounds
    thicknesses_A = R_A @ thicknesses_L
    vac_A = R_A @ (1 - thicknesses_L)
    target_vac_A = np.copy(vac_A)
    for i in vacuum_indices:
        target_vac_A[i] = desired_vac_A[i]

    scaling = np.diag((thicknesses_A + target_vac_A) / (thicknesses_A + vac_A))
    R_new = scaling @ R_A
    R_new_from_old = np.linalg.inv(R_new) @ R_A
    positions_new = (R_new_from_old @ calc.atoms_L.Xi.T).T
    centers = R_new_from_old @ (max_bounds + min_bounds)/2
    for i in vacuum_indices:
        positions_new[:, i] += 0.5 - centers[i]

    atoms_new_L = calc.atoms_L.repositioned(positions_new)
    if atoms_new_L.Xi.max() > 1 or atoms_new_L.Xi.min() < 0:
        raise ValueError(f'Bounds of atoms after vacuums spacing are not valid: {atoms_new_L.bounds}')

    a, b, c = np.diag(R_new)
    lattice_A = Lattice(LatticeType.Orthorhombic, a=a, b=b, c=c)
    return Calc(lattice_A, atoms_new_L)


def perform_removals(atoms_L: Atoms, surface_config: SurfaceConfig, *, frac_tol: float = 0.05) -> Atoms:
    bounds = np.array(surface_config.bounds_L)
    atoms_L = Atoms([atom for atom in atoms_L if all(atom.xyz >= bounds[:, 0]) and all(atom.xyz <= bounds[:, 1])])

    if not surface_config.removals:
        return atoms_L

    atoms_xyz = atoms_L.Xi
    matched_indices = []
    for xyz in surface_config.removals:
        matches = np.full(len(atoms_L), True, dtype=bool)
        for i, x in [(i, x) for i, x in enumerate(xyz) if x is not None]:
            abs_diff = np.abs(atoms_xyz[:, i] - x) % 1
            matches &= (abs_diff < frac_tol) | (abs_diff > 1 - frac_tol)
        # Implicity errors if there is not exactly one match
        [matched_i] = np.nonzero(matches)
        matched_indices.append(matched_i)
    return Atoms([atom for i, atom in enumerate(atoms_L) if i not in matched_indices])


def make_fc_config(material: Material, lattice_A: Lattice, *, frac_tol: float = 0.01) -> SurfaceConfig:
    '''
    For facets of the form '###x', the x can be:
    's' stoichiometric. This is a surface that terminates with a stoichiometry the same
    as the bulk material.
    'o' oxidized. This is a surface with more oxygens, which happens in oxygen rich
    environments such as alkaline solutions.
    'r' reduced. Acidic and with low oxygen. More cations on the surface.

    In spinels, when there is enough variety of cations you can terminate 
    as s,r, and o in multiple ways. These are designated with the prefix a, b, etc. 
    So 'ar' and 'br' would be two variants of the reduced surface. 
    '''
    if material.frozen_layers > material.layers:
        raise ValueError(f'Frozen layers {material.frozen_layers} cannot be more than layers {material.layers}')

    removals, target_area, x_count, y_count, z_count = None, None, 1, 1, 1
    lock_depth_L = 1/2 - frac_tol
    # Cubic spinel facets need an odd number of layers as implemented
    if isinstance(material, Spinel):
        max_supercell = 4
        if lattice_A.lattice_type not in [LatticeType.FCC, LatticeType.FCO]:
            raise ValueError(f'Only FCC and FCO lattices are supported for spinel, not {lattice_A.lattice_type}')

        if material.facet in ['100s', '100o']:
            if (material.frozen_layers and
                not (material.layers == 3 and material.frozen_layers == 2) and
                not (material.layers == 4 and material.frozen_layers == 2) and
                    not (material.layers == 5 and material.frozen_layers == 3)):
                raise NotImplementedError('Need to implement better calculation method for layers, hard coded as is')
            z_count = 3
            one_layer = 1 / (4 * z_count)
            slab_low = 1/2 - (material.layers - 2) * one_layer
            slab_hi = 1/2 + 2 * one_layer
            removal_number = 0.75 if material.layers % 2 else 0
            removals = ((None, removal_number, slab_low), (None, 0, slab_hi))
            if material.facet == '100o':
                removals += ((None, (removal_number + 0.5) % 1, slab_low), (None, 0.5, slab_hi))
        elif material.facet == '111s' and material.layers == 8:
            if material.frozen_layers and material.frozen_layers != 5:
                raise NotImplementedError('Need to implement better calculation method for layers, hard coded as is')
            target_area = lattice_A.a**2 if lattice_A.lattice_type == LatticeType.FCC else lattice_A.a * lattice_A.b
            target_area *= np.sqrt(3)
            z_count = 2
            slab_hi = 1/2 + 7/48
            slab_low = 1/2 - 3/16
            removals = ((0, 1/3, slab_hi), (1/4, 5/6, slab_hi), (1/2, 1/3, slab_hi), (3/4, 5/6, slab_hi))
        elif material.facet == '111b' and material.layers == 9:
            if material.frozen_layers and material.frozen_layers != 4:
                raise NotImplementedError('Need to implement better calculation method for layers, hard coded as is')
            target_area = lattice_A.a**2 if lattice_A.lattice_type == LatticeType.FCC else lattice_A.a * lattice_A.b
            target_area *= np.sqrt(3)
            z_count = 2
            slab_hi = 0.58
            slab_low = 0.22
            lock_depth_L = 0.4 - frac_tol
            removals = tuple([])
        else:
            raise NotImplementedError()
    elif material.facet == '111s':
        max_supercell = 8
        if lattice_A.lattice_type != LatticeType.FCC:
            raise NotImplementedError(f'TODO: target area for {lattice_A.lattice_type} on {material.facet}')
        if material.layers % 3 != 0:
            raise NotImplementedError(f'Only multiples of 3 layers are supported for {material.facet}')
        if material.frozen_layers != (material.layers + 1) // 2:
            raise NotImplementedError(f'Only half frozen is current supported for {material.facet}')

        target_area = 15 * np.sqrt(3) * lattice_A.a**2 * (2 / 3) # 2/3 of the full sized sheet to save memory
        z_count = material.layers // 3
        slab_low = 0
        slab_hi = 1
    elif material.facet == '100s':
        max_supercell = 1
        if lattice_A.lattice_type != LatticeType.FCC:
            raise NotImplementedError(f'TODO: target area for {lattice_A.lattice_type} on {material.facet}')
        if material.layers % 2 != 0:
            raise NotImplementedError(f'Only multiples of 2 layers are supported for {material.facet}')
        x_count = 2
        y_count = 2
        z_count = material.layers // 2
        slab_low = 0
        slab_hi = 1
        lock_depth_L = material.frozen_layers / material.layers - frac_tol
    else:
        raise NotImplementedError()

    # Each matched case should assign a z_count, so this acts as a proxy for checking
    # if we matched anything or not
    if z_count is None:
        raise NotImplementedError(f'Material {material} not yet supported')

    bounds = ((0, 1), (0, 1), (max(0, slab_low-frac_tol), min(1, slab_hi+frac_tol)))
    return SurfaceConfig(max_supercell, target_area, x_count, y_count, z_count, bounds, removals, lock_depth_L)
