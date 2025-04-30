
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R

from ecat_types import AttachSite, Sites

#pylint: disable=multiple-statements, line-too-long

@dataclass(frozen=True, order=True)
class MonoReg100oSites(Sites):
    B: str

    # Attach sites
    def Oct(self, i): return AttachSite(self.B, i, f'Oct{i}', (0, 0, 1.68), rotation=R.from_euler('z', 280, degrees=True))
    def OctOctT(self): return AttachSite(self.B, 0, 'OctOctT', (-1.1, 0.8, 1.6))
    def OctOctO2B(self): return AttachSite(self.B, 0, 'OctOctO2B', (0.45, -0.45, 1.7), rotation=R.from_euler('yz', (-90, 135), degrees=True))
    def OctOctT2B(self): return AttachSite(self.B, 0, 'OctOctT2B', (-0.45, 0.45, 1.7), rotation=R.from_euler('yz', (-90, -45), degrees=True))
    def O(self, i): return AttachSite('O', i, f'O{i}', (0, 0, 1))

    def H_spec(self, i): return self.O(i)

    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.Oct(0), self.OctOctT()]
        elif key == 2:
            return [
                [self.Oct(0), self.Oct(1)],
                [self.Oct(0), self.Oct(2)],
            ]
        elif key == 3:
            return [[self.Oct(0), self.Oct(1), self.Oct(2)]]
        elif key == 4:
            return [[self.Oct(0), self.Oct(1), self.Oct(2), self.Oct(3)]]
        elif key == '2Bridges':
            return [self.OctOctO2B(), self.OctOctT2B()]
        elif 5 <= key <= 8:
            return []
        raise NotImplementedError(f'Invalid key: {key}')
