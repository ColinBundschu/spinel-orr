import dataclasses
from dataclasses import dataclass

from .adsorbate import AdsorbateInterface
from .free_species import FreeSpecies


@dataclass(frozen=True, order=True)
class Step:
    index: int
    adsorbate: str | AdsorbateInterface
    _free_species: tuple[tuple[FreeSpecies, int], ...]
    _: dataclasses.KW_ONLY
    Efree0V_eV: float | None = None

    def __post_init__(self):
        object.__setattr__(self, '_free_species', tuple(sorted(self.free_species.items())))

    def __str__(self):
        parts = [str(self.index), str(self.adsorbate)]
        for species, count in self.free_species.items():
            parts.append(f'({count},{species})')
        return ' '.join(parts)

    @property
    def free_species(self) -> dict[FreeSpecies, int]:
        return dict(self._free_species)

    def Efree_eV(self, potential_V: float) -> float | None:
        if self.Efree0V_eV is None:
            return None
        return self.Efree0V_eV - potential_V * self.free_species.get(FreeSpecies('e-', None), 0)

    # def compute_or_fetch_fs_free_energy(self, fs_name: str) -> float:
    #     if fs_name in self.fsFreeEnergies_eV:
    #         return self.fsFreeEnergies_eV[fs_name]
    #     fs = self.freeSpecies.get(fs_name)
    #     if not fs:
    #         raise ValueError(f"No free species found for {fs_name}")
    #     # Compute the free energy and store it
    #     fs_free_energy = fs.calculate_free_energy(self.temperature_K)
    #     self.fsFreeEnergies_eV[fs_name] = fs_free_energy
    #     return fs_free_energy
