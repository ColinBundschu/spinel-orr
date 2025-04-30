import dataclasses
from dataclasses import dataclass

import numpy as np

from constants import (UNIT_CELL_FRAC_CONV, MAX_ENERGY_CHANGE_eV,
                       MAX_FORCE_LATTICE_eVpA, MAX_FORCE_SLAB_eVpA)

from .atom import Atoms
from .lattice import Lattice
from .minimization_point import MinimizationPoint
from .status import Status


@dataclass(frozen=True, order=True)
class Calc:
    lattice_A: Lattice
    atoms_L: Atoms
    _: dataclasses.KW_ONLY
    forces_max_L2_eVpA: float | None = None
    min_data: tuple[MinimizationPoint] | None = None
    status: Status = Status.Fresh
    lattice_A_start: Lattice | None = None
    origin: str | None = None

    @property
    def is_converged(self) -> bool:
        # Loads from similar states are not considered converged
        if self.origin not in ['db', 'local']:
            return False

        if self.forces_max_L2_eVpA is None:
            return False

        if self.status == Status.Lattice_Converged:
            lattice_min_data = [x for x in self.min_data if x.type == 'L']

            V0 = lattice_min_data[0].unit_cell_volume_A3
            V1 = lattice_min_data[-1].unit_cell_volume_A3
            volume_converged = abs(V0 - V1) / ((V0 + V1) / 2) < UNIT_CELL_FRAC_CONV

            E0 = lattice_min_data[0].Etot_eV
            E1 = lattice_min_data[-1].Etot_eV
            energy_converged = abs(E0 - E1) < MAX_ENERGY_CHANGE_eV

            lattice_converged = self.lattice_A.close_under_symmetry(self.lattice_A_start, allow_FCO_swap=True)
            forces_converged = self.forces_max_L2_eVpA < MAX_FORCE_LATTICE_eVpA
            if energy_converged and lattice_converged and forces_converged and not volume_converged:
                raise ValueError('Lattice converged to significantly non-orthogonal basis. Need manual intervention.')
            return energy_converged and lattice_converged and forces_converged

        if self.status == Status.Ionic_Converged:
            ionic_min_data = [x for x in self.min_data if x.type == 'I']
            E0 = ionic_min_data[0].Etot_eV
            E1 = ionic_min_data[-1].Etot_eV
            energy_converged = abs(E0 - E1) < MAX_ENERGY_CHANGE_eV
            forces_converged = self.forces_max_L2_eVpA < MAX_FORCE_SLAB_eVpA
            return energy_converged and forces_converged

        return False

    def as_xsf_lines(self) -> list[str]:
        '''Create an xsf file representing the calc state'''
        lines = []
        lines.append('CRYSTAL')
        lines.append('PRIMVEC')
        for v0, v1, v2 in self.lattice_A.R_A:
            lines.append(f' {v0:<11.7} {v1:<11.7} {v2:<.7}')

        lines.append('PRIMCOORD')
        lines.append(f'{len(self.atoms_L)} 1')

        for atom in self.atoms_L:
            x, y, z = self.lattice_A.R_A @ np.array(atom.xyz)
            lines.append(f'{atom.atomic_num} {x:<11.7} {y:<11.7} {z:<.7}')

        return lines
