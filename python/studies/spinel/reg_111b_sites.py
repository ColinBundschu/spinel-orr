
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R

from ecat_types import AttachSite, Sites

#pylint: disable=multiple-statements, line-too-long

@dataclass(frozen=True, order=True)
class Reg111bSites(Sites):
    A: str

    # Attach sites
    def Oct(self, i): return AttachSite(self.A, i, f'o{i}', (0, 0, 1.68), rotation=R.from_euler('z', 180, degrees=True))

    def H_spec(self, i): raise NotImplementedError('H_spec not implemented for Reg111bSites (No O sites)')

    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.Oct(0)]
        if key == 2:
            return [
                [self.Oct(0), self.Oct(1)],
                [self.Oct(0), self.Oct(4)],
            ]
        if key == 3:
            return [
                [self.Oct(0), self.Oct(1), self.Oct(2)],
                [self.Oct(0), self.Oct(1), self.Oct(3)],
                [self.Oct(0), self.Oct(1), self.Oct(5)],
            ]
        if key == 4:
            return [
                [self.Oct(0), self.Oct(1), self.Oct(2), self.Oct(3)],
                [self.Oct(0), self.Oct(1), self.Oct(3), self.Oct(4)],
                [self.Oct(0), self.Oct(1), self.Oct(3), self.Oct(5)],
            ]
        if key == 5:
            return [
                [self.Oct(0), self.Oct(1), self.Oct(2), self.Oct(3), self.Oct(4)],
            ]
        if key == 6:
            return [
                [self.Oct(0), self.Oct(1), self.Oct(2), self.Oct(3), self.Oct(4), self.Oct(5)],
            ]
        else:
            return []
