
import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

import constants


@dataclass(frozen=True, order=True)
class Atom():
    atomic_num: int
    xyz: tuple[float, float, float]
    _: dataclasses.KW_ONLY
    magnetic_moment: float | None = None
    oxidation_state: float | None = None
    frozen: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'xyz', tuple(float(x) for x in self.xyz))

    def with_new_xyz(self, new_xyz: tuple[float, float, float]) -> 'Atom':
        return Atom(self.atomic_num, new_xyz, magnetic_moment=self.magnetic_moment,
                    oxidation_state=self.oxidation_state, frozen=self.frozen)

    @property
    def symbol(self) -> str:
        return constants.ATOMIC_SYMBOLS[self.atomic_num]

    @property
    def x(self) -> float:
        return self.xyz[0]

    @property
    def y(self) -> float:
        return self.xyz[1]

    @property
    def z(self) -> float:
        return self.xyz[2]


class Atoms(tuple[Atom, ...]):
    def to_firestore_dict(self) -> dict[str, list]:
        # Initialize lists for each attribute
        atoms_frozen = []
        atoms_mag = []
        atoms_num = []
        atoms_ox = []
        atoms_xyz = []

        # Iterate over each Atom and populate the lists
        for atom in self:
            atoms_frozen.append(atom.frozen)
            atoms_mag.append(atom.magnetic_moment)
            atoms_num.append(atom.atomic_num)
            atoms_ox.append(atom.oxidation_state)
            atoms_xyz += [atom.x, atom.y, atom.z]

        return {
            'atoms_frozen': atoms_frozen,
            'atoms_mag': atoms_mag,
            'atoms_num': atoms_num,
            'atoms_ox': atoms_ox,
            'atoms_xyz': atoms_xyz,
        }

    def translated(self, delta_xyz: tuple[float, float, float]) -> 'Atoms':
        return Atoms([atom.with_new_xyz(tuple(x + delta for x, delta in zip(atom.xyz, delta_xyz))) for atom in self])

    def transformed(self, transformation: np.ndarray) -> 'Atoms':
        return Atoms([atom.with_new_xyz(tuple(transformation @ np.array(atom.xyz))) for atom in self])

    def rotated(self, rotation: Rotation | tuple[float, float, float, float]) -> 'Atoms':
        return Atoms([atom.with_new_xyz(tuple(Rotation(rotation).apply(atom.xyz))) for atom in self])

    def repositioned(self, positions):
        return Atoms([atom.with_new_xyz(xyz) for atom, xyz in zip(self, positions)])

    @property
    def nums(self):
        return [atom.atomic_num for atom in self]

    @property
    def symbols(self) -> list[str]:
        return [atom.symbol for atom in self]

    @property
    def unique_symbols(self):
        return list(dict.fromkeys(atom.symbol for atom in self))

    @property
    def magnetic_moments(self):
        return [atom.magnetic_moment for atom in self]

    @property
    def oxidation_states(self):
        return [atom.oxidation_state for atom in self]

    @property
    def frozens(self):
        return [atom.frozen for atom in self]

    @property
    def bounds(self):
        max_bounds = np.amax(self.Xi, axis=0)
        min_bounds = np.amin(self.Xi, axis=0)
        return np.array(list(zip(min_bounds, max_bounds)))

    @property
    def Xi(self):
        return np.array([atom.xyz for atom in self])


def bond_length_A(atom_a_L: Atom, atom_b_L: Atom, R_A: np.ndarray):
    a_to_b_L = np.array(atom_b_L.xyz).T - np.array(atom_a_L.xyz).T
    return np.linalg.norm(R_A @ a_to_b_L)
