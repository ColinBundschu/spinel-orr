
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R

from ecat_types import AttachSite, Sites

#pylint: disable=multiple-statements, line-too-long

@dataclass(frozen=True, order=True)
class Reg111sSites(Sites):
    A: str

    # Attach sites
    def Tet(self, i): return AttachSite(self.A, i, f't{i}', (0, 0, 1.68), rotation=R.from_euler('z', 180, degrees=True))
    def TetA(self, i): return AttachSite(self.A, 0, f'tA{i}', (-1.6, 0.6, 1.3), rotation=R.from_euler('x', 90, degrees=True), config='linear')
    def TetB(self, i): return AttachSite(self.A, 0, f'tB{i}', (1.6, 0.7, 1.3), rotation=R.from_euler('y', 110, degrees=True), config='linear')
    def TetC(self, i): return AttachSite(self.A, i, f'tC{i}', (0, -1.7, 1), rotation=R.from_euler('x', 110, degrees=True), config='linear')
    def O(self, i): return AttachSite('O', i, f'O{i}', (0, 0, 1))

    def H_spec(self, i): return self.O(i)

    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.Tet(0)]
        elif key == 2:
            return [
                [self.TetA(0), self.TetB(0)],
                [self.TetB(0), self.TetC(3)],
                [self.Tet(0), self.Tet(1)],
                [self.Tet(0), self.Tet(2)],
            ]
        elif key == 3:
            return [
                [self.Tet(0), self.Tet(1), self.Tet(2)],
            ]
        elif key == 4:
            return [
                [self.Tet(0), self.Tet(1), self.Tet(2), self.Tet(3)],
            ]
        elif key == 5:
            return [
                [self.TetA(0), self.TetB(0), self.Tet(1), self.Tet(2), self.Tet(3)],
            ]
        elif key == '2Bridges':
            return []
        elif key >= 6:
            return []
        raise NotImplementedError(f'Invalid key: {key}')
