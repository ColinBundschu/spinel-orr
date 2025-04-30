from dataclasses import dataclass

from ecat_types import Atom, Atoms, Calc, Lattice, LatticeType, Material, Sites
from ecat_types.kpoints import KPoints

from .Ni_sites import NiSites


@dataclass(frozen=True, order=True)
class Ni(Material):
    def __post_init__(self):
        if self.basename is None:
            object.__setattr__(self, 'basename', 'Ni')
        super().__post_init__()

    @property
    def KPOINT_DEFAULT(self) -> KPoints:
        if self.geo == 'bulk':
            return KPoints(12, 12, 12)
        if self.geo == 'slab':
            return KPoints(2, 3, 1)
        raise NotImplementedError()

    @property
    def dft_kwargs(self) -> dict:
        kwargs = {}
        if self.geo == 'slab':
            kwargs['coulomb_truncation_embed'] = '0 0 0.5'
        return kwargs

    @property
    def dft_U(self) -> str | None:
        return None  # No add-U in a pure nickel alloy

    @property
    def sites(self) -> Sites:
        return NiSites()

    @property
    def default_config(self) -> Calc:
        lattice_A = Lattice(LatticeType.FCC, a=3.52)
        atoms = Atoms([Atom(28, (0, 0, 0))])
        return Calc(lattice_A, atoms)
