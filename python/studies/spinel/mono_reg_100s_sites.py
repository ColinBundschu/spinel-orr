
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R

from ecat_types import AttachSite, Sites

#pylint: disable=multiple-statements, line-too-long

def add_i(name: str, i: int) -> str:
    return name if i == 0 else name + str(i)

@dataclass(frozen=True, order=True)
class MonoReg100sSites(Sites):
    A: str
    B: str

    # Attach sites
    def Tet(self): return AttachSite(self.A, 0, 'Tet', (0, 0, 1.68))
    def TetA(self): return AttachSite(self.A, 0, 'TetA', (1.225, -1.225, 0.8), rotation=R.from_euler('x', 120, degrees=True), config='linear')
    def TetB(self): return AttachSite(self.A, 0, 'TetB', (-1.225, 1.225, 0.8), rotation=R.from_euler('x', -120, degrees=True), config='linear')
    def TetO(self): return AttachSite(self.A, 0, 'TetO', (1.1, -1.1, 0.7), rotation=R.from_euler('x', 90, degrees=True), config='linear')
    def TetOct2B(self): return AttachSite(self.A, 0, 'TetOct2B', (1.2, 0.1, 1.3), rotation=R.from_euler('yz', (110, 20), degrees=True))
    def Oct(self, i=0): return AttachSite(self.B, self.oct_i + i, add_i('Oct', i), (0, 0, 1.68), rotation=R.from_euler('z', 280, degrees=True))
    def OctOctT(self): return AttachSite(self.B, self.oct_i, 'OctOctT', (-1.1, 0.8, 1.6))
    def OctOctO2B(self): return AttachSite(self.B, self.oct_i, 'OctOctO2B', (0.45, -0.45, 1.7), rotation=R.from_euler('yz', (-90, 135), degrees=True))
    def OctOctT2B(self): return AttachSite(self.B, self.oct_i, 'OctOctT2B', (-0.45, 0.45, 1.7), rotation=R.from_euler('yz', (-90, -45), degrees=True))
    def O(self, i): return AttachSite('O', i, add_i('O', i), (0, 0, 1))

    def H_spec(self, i): return self.O(i)

    @property
    def oct_i(self):
        return 1 if self.A == self.B else 0

    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.Tet(), self.Oct(), self.TetO(), self.OctOctT()]
        elif key == 2:
            return [[self.TetA(), self.TetB()],
                    [self.Oct(), self.Oct(2)],
                    [self.Oct(), self.Oct(3)],
                    [self.Oct(), self.Tet()]]
        elif key == 3:
            return [[self.Tet(), self.Oct(), self.Oct(2)],
                    [self.Oct(), self.Oct(2), self.Oct(3)]]
        elif key == 4:
            return [
                [self.Tet(), self.Oct(), self.Oct(2), self.Oct(3)],
                [self.TetA(), self.Oct(), self.Oct(2), self.Oct(3)],
                [self.TetB(), self.Oct(), self.Oct(2), self.Oct(3)],
                [self.Oct(), self.Oct(1), self.Oct(2), self.Oct(3)],
            ]
        elif key == 5:
            return [
                [self.Tet(), self.Oct(), self.Oct(1), self.Oct(2), self.Oct(3)],
                [self.TetA(), self.Oct(), self.Oct(1), self.Oct(2), self.Oct(3)],
                [self.TetB(), self.Oct(), self.Oct(1), self.Oct(2), self.Oct(3)],
                [self.TetA(), self.TetB(), self.Oct(), self.Oct(2), self.Oct(3)],
            ]
        elif key == 6:
            return [[self.TetA(), self.TetB(), self.Oct(), self.Oct(1), self.Oct(2), self.Oct(3)]]
        elif key == '2Bridges':
            return [self.OctOctO2B(), self.OctOctT2B(), self.TetOct2B()]
        elif key == 7:
            return []
        elif key == 8:
            return []
        raise NotImplementedError(f'Invalid key: {key}')
