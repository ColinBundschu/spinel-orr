from dataclasses import dataclass

from ecat_types import Material, Sites
from ecat_types.atom import Atom, Atoms
from ecat_types.calc import Calc
from ecat_types.kpoints import KPoints
from ecat_types.lattice import Lattice, LatticeType

from .Pt_sites import PtSites

@dataclass(frozen=True, order=True)
class Pd4Pt(Material):
    def __post_init__(self):
        if self.basename is None:
            object.__setattr__(self, 'basename', 'Pd4Pt')
        object.__setattr__(self, 'replacement_atomic_num', 46)
        object.__setattr__(self, 'replacement_layers', 4)
        super().__post_init__()

    @property
    def KPOINT_DEFAULT(self) -> KPoints:
        if self.geo == 'slab':
            return KPoints(4, 4, 1)
        raise NotImplementedError()

    @property
    def KPOINT_DEFAULT(self) -> KPoints:
        if self.geo == 'bulk':
            return KPoints(24, 24, 24)
        if self.geo == 'slab':
            return KPoints(4, 4, 1)
        raise NotImplementedError()

    @property
    def dft_kwargs(self) -> dict:
        kwargs = {}
        if self.geo == 'slab':
            kwargs['coulomb_truncation_embed'] = '0 0 0.5'
        return kwargs

    @property
    def dft_U(self) -> str | None:
        return None

    @property
    def sites(self) -> Sites:
        return PtSites()

    @property
    def default_config(self) -> Calc:
        lattice_A = Lattice(LatticeType.FCC, a=3.92)
        atoms = Atoms([Atom(78, (0, 0, 0))])
        return Calc(lattice_A, atoms)
