

import dataclasses
from dataclasses import dataclass
from enum import Enum

import numpy as np
from structlog import BoundLogger

from constants import MAX_LATTICE_CHANGE_A


class LatticeType(Enum):
    Cubic = 'Cubic'
    Orthorhombic = 'Orthorhombic'
    FCC = 'Face-Centered Cubic'
    FCO = 'Face-Centered Orthorhombic'


@dataclass(frozen=True, order=True)
class Lattice:
    lattice_type: LatticeType
    _: dataclasses.KW_ONLY
    a: float | None = None
    b: float | None = None
    c: float | None = None

    def __post_init__(self):
        lattice_str = f'a={self.a}, b={self.b}, c={self.c}'
        match self.lattice_type:
            case LatticeType.Cubic | LatticeType.FCC:
                if not self.a or any([self.b, self.c]):
                    raise ValueError(f'FCC/Cubic should specify exactly "a": {lattice_str}')
            case LatticeType.FCO | LatticeType.Orthorhombic:
                if not all([self.a, self.b, self.c]):
                    raise ValueError(f'FCO/Orthorhombic should specify exactly "a", "b", "c": {lattice_str}')
            case _:
                raise NotImplementedError(f'Lattice type {self.lattice_type} not implemented')

    @property
    def abc_str(self) -> str:
        return f'({self.a:.2f}, {self.b:.2f}, {self.c:.2f})'

    @property
    def R_A(self) -> np.ndarray:
        match self.lattice_type:
            case LatticeType.Cubic:
                return self.a * np.identity(3)
            case LatticeType.Orthorhombic:
                return np.array([[self.a, 0, 0], [0, self.b, 0], [0, 0, self.c]])
            case LatticeType.FCC:
                return (self.a/2) * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
            case LatticeType.FCO:
                return np.array([[0, self.a/2, self.a/2], [self.b/2, 0, self.b/2], [self.c/2, self.c/2, 0]])
            case _:
                raise NotImplementedError(f'No lattice vectors defined for {self.lattice_type}')

    def to_firestore_dict(self, last: bool):
        firestore_dict = {}
        firestore_dict['Lattice_abc' + ('' if last else '_start')] = [self.a, self.b, self.c]
        firestore_dict['Lattice_type' + ('' if last else '_start')] = self.lattice_type.name
        return firestore_dict

    def close_under_symmetry(self, other: 'Lattice', *, allow_FCO_swap: bool = False) -> bool:
        max_lat_abs_diff_A = np.max(np.abs(self.R_A - other.R_A))
        if max_lat_abs_diff_A < MAX_LATTICE_CHANGE_A:
            return True

        if self.lattice_type != other.lattice_type:
            return False

        if allow_FCO_swap and self.lattice_type == LatticeType.FCO:
            a0, b0, c0 = sorted([self.a, self.b, self.c])
            ordered_self = Lattice(self.lattice_type, a=a0, b=b0, c=c0)
            a1, b1, c1 = sorted([other.a, other.b, other.c])
            ordered_other = Lattice(self.lattice_type, a=a1, b=b1, c=c1)
            max_lat_abs_diff_A = np.max(np.abs(ordered_self.R_A - ordered_other.R_A))
            return max_lat_abs_diff_A < MAX_LATTICE_CHANGE_A

        return False
