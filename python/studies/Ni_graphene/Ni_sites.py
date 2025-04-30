
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from ecat_types import AttachSite, Sites
# pylint: disable=multiple-statements, line-too-long

@dataclass(frozen=True, order=True)
class NiSites(Sites):
    # Attach sites
    def Ni(self): return AttachSite('Ni', 0, 'Ni', (0, 0, 1.68), use_angle=False)
    def s1(self): return AttachSite('Ni', 0, 's1', (1.5, 0, 2.15), use_angle=False, config='linear')
    def xs2(self): return AttachSite('Ni', 0, 'xs2', (1.5, 0, 2.15), use_angle=False, config='linear')
    def t4(self): return AttachSite('Ni', 0, 't4', (1.5, 0, 2.15), use_angle=False, config='linear')
    def y1(self): return AttachSite('Ni', 0, 'y1', (0.9, 0, 2.15), use_angle=False, config='linear', rotation=R.from_euler('z', 60, degrees=True))

    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.Ni(), self.s1(), self.xs2(), self.t4(), self.y1()]
        if key == 2:
            return []
        if key == '2Bridges':
            return []
        raise NotImplementedError(f'Invalid key: {key}')
