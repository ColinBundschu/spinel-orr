
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R

from ecat_types import AttachSite, Sites

#pylint: disable=multiple-statements, line-too-long

def add_i(name: str, i: int) -> str:
    return name if i == 0 else name + str(i)

@dataclass(frozen=True, order=True)
class Striped100sSites(Sites):
    A: str
    B0: str
    B1: str

    def __post_init__(self):
        if self.B0 == self.B1:
            raise ValueError(f'Invalid sites: {self}')

    # Attach sites
    def Tet(self): return AttachSite(self.A, 0, 'Tet', (0, 0, 1.68))
    def TetA(self): return AttachSite(self.A, 0, 'TetA', (1.225, -1.225, 0.8), rotation=R.from_euler('x', 120, degrees=True), config='linear')
    def TetB(self): return AttachSite(self.A, 0, 'TetB', (-1.225, 1.225, 0.8), rotation=R.from_euler('x', -120, degrees=True), config='linear')
    def TetO(self): return AttachSite(self.A, 0, 'TetO', (1.1, -1.1, 0.7), rotation=R.from_euler('x', 90, degrees=True), config='linear')
    def TetOct02B(self): return AttachSite(self.A, 0, f'TetOct{self.B0}2B', (1.2, 0.1, 1.3), rotation=R.from_euler('yz', (110, 20), degrees=True))
    def TetOct12B(self): return AttachSite(self.A, 0, f'TetOct{self.B1}2B', (-1.2, -0.1, 1.3), rotation=R.from_euler('yz', (-110, 20), degrees=True))
    def Oct0(self, i=0): return AttachSite(self.B0, self.oct_i(self.B0) + i, add_i(f'Oct{self.B0}', i), (0, 0, 1.68), rotation=R.from_euler('z', 280, degrees=True))
    def Oct1(self, i=0): return AttachSite(self.B1, self.oct_i(self.B1) + i, add_i(f'Oct{self.B1}', i), (0, 0, 1.68), rotation=R.from_euler('z', 100, degrees=True))
    def OctOctT(self): return AttachSite(self.B0, self.oct_i(self.B0), 'OctOctT', (-1.1, 0.8, 1.6))
    def OctOctO2B(self): return AttachSite(self.B0, self.oct_i(self.B0), 'OctOctO2B', (0.45, -0.45, 1.7), rotation=R.from_euler('yz', (-90, 135), degrees=True))
    def OctOctT2B(self): return AttachSite(self.B0, self.oct_i(self.B0), 'OctOctT2B', (-0.45, 0.45, 1.7), rotation=R.from_euler('yz', (-90, -45), degrees=True))
    def O(self, i): return AttachSite('O', i, add_i('O', i), (0, 0, 1))

    def H_spec(self, i): return self.O(i)

    def oct_i(self, element: str):
        return 1 if self.A == element else 0

    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.Tet(), self.Oct0(), self.Oct1(), self.TetO(), self.OctOctT()]
        elif key == 2:
            return [[self.TetA(), self.TetB()],
                    [self.Oct0(), self.Oct1()],
                    [self.Oct0(), self.Oct1(1)],
                    [self.Oct0(), self.Tet()],
                    [self.Oct1(), self.Tet()]]
        elif key == '2Bridges':
            return [self.OctOctO2B(), self.OctOctT2B(), self.TetOct02B(), self.TetOct12B()]
        raise NotImplementedError(f'Invalid key: {key}')
