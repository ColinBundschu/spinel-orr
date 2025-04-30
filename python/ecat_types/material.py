
import dataclasses
from dataclasses import dataclass
from typing import Iterable

from ecat_types.calc import Calc
from ecat_types.kpoints import KPoints

from .sites import Sites


@dataclass(frozen=True, order=True)
class Material:
    geo: str | None
    _: dataclasses.KW_ONLY
    facet: str | None = None
    layers: int | None = None
    xy_count: str | None = None
    basename: str | None = None
    frozen_layers: int = 0
    replacement_layers: int = 0
    replacement_atomic_num: int = 0

    GEOS = (None, 'molecule', 'slab', 'bulk')

    def __post_init__(self):
        if self.geo not in Material.GEOS:
            raise ValueError(f'Invalid geo: {self.geo}. Geo must be in {Material.GEOS}')

        if self.basename is None:
            raise ValueError('basename must be specified on base Materials or if subclass does not define a default')

    def __str__(self):
        parts = [self.basename]
        if self.geo_str:
            parts.append(self.geo_str)
        return '_'.join(parts)

    @property
    def KPOINT_DEFAULT(self) -> KPoints:
        raise NotImplementedError()

    @property
    def do_lattice_min(self) -> bool:
        return False

    @property
    def dft_kwargs(self) -> dict:
        return {}

    # Any material with facets should implement this
    @property
    def sites(self) -> Sites:
        raise NotImplementedError()

    @property
    def default_config(self) -> Calc:
        raise NotImplementedError()

    @property
    def dft_U(self) -> str | None:
        return None

    @property
    def miller_indices(self) -> tuple[int, int, int]:
        assert (self.facet[:3].isdigit() and not any(char.isdigit() for char in self.facet[3:]))
        return tuple(int(char) for char in self.facet[:3])

    @property
    def geo_str(self) -> str:
        geo_str_parts = []
        if self.facet is not None:
            geo_str_parts.append(self.facet)

        if self.layers is not None:
            geo_str_parts.append(str(self.layers))
            if self.frozen_layers:
                geo_str_parts[-1] += f'x{self.frozen_layers}'

        if self.xy_count is not None:
            geo_str_parts.append(self.xy_count)

        if self.geo != 'slab' and self.geo is not None:
            geo_str_parts.append(self.geo)

        return '_'.join(geo_str_parts)

    @staticmethod
    def parse_geo_str(geo_str: str) -> tuple[str, str | None, int | None, int | None]:
        if geo_str in ['bulk', 'molecule']:
            return geo_str, None, None, None

        facet, layers = geo_str.split('_')
        if 'x' in layers:
            layers, frozen_layers = layers.split('x')
        else:
            frozen_layers = 0
        return 'slab', facet, int(layers), int(frozen_layers)


def dft_U(metals: Iterable[str]) -> str | None:
    # 10.1016/j.mtla.2019.100381 {Co 3}
    # 10.1021/acs.jpcc.5b02298 {Co 3}
    # Peng oxidation: 10.1021/acscatal.1c00214 {Co 2.8, Fe 3.1, Ni 4.2}
    # Wang (Ceder) oxidation: 10.1103/PhysRevB.73.195107 {Co 3.3, Cr 3.5, Mn 3.5/3.8/4, Fe 3.9/4.1, Cu 4.0, Ni 6.4}
    # Anubhav (Ceder) oxiation: 10.1103/PhysRevB.84.045115 {Co 3.4, Cr 3.5, Mn 3.9, Fe 4, Cu 4.0, Ni 6}
    # Stevanovic oxidation: 10.1103/PhysRevB.85.115104 {3 for all except 5 eV for Cu}
    U_map_d_orbital_H = {
        'Al': None,  # No U as its not strongly correlated
        'Ga': None,  # No U as its not strongly correlated
        'Mg': None,  # No U as its not strongly correlated
        'Mn': 0.129,  # 3.5 eV
        'Fe': 0.129,  # 3.5 eV
        'Cu': 0.147,  # 4 eV
        'Co': 0.11,  # 3 eV
        'Cr': 0.12,  # 3.25 eV, since the Ceder papers put it between Co and Mn/Fe
        'Ni': 0.184,  # 5 eV
        'Zn': None,  # No U as its not strongly correlated
    }

    dft_Us = []
    for element in set(metals):
        if element not in U_map_d_orbital_H:
            raise NotImplementedError(f'No Hubbard U value specified for {element}')
        if U_map_d_orbital_H[element] is not None:
            dft_Us.append(f'{element} d {U_map_d_orbital_H[element]}')

    if dft_Us:
        return ' '.join(dft_Us)
    return None
