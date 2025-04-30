
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R

from ecat_types import AttachSite, Sites

# pylint: disable=multiple-statements, line-too-long

def add_i(name: str, i: int) -> str:
    return name if i == 0 else name + str(i)

@dataclass(frozen=True, order=True)
class MN4CSites(Sites):
    m: str

    # Attach sites
    def M(self): return AttachSite(self.m, 0, 'M', (0, 0, 1.68))
    def MA(self): return AttachSite(self.m, 0, 'MA', (0.1, -1.1, 1.4), rotation=R.from_euler('xz', (120, -55), degrees=True), config='linear')
    def MB(self): return AttachSite(self.m, 0, 'MB', (-0.1, 1.1, 1.4), rotation=R.from_euler('xz', (-120, -55), degrees=True), config='linear')
    def MN(self): return AttachSite(self.m, 0, 'MN', (0.6, 0.7, 1.4), config='linear')
    def N(self, i): return AttachSite('N', i, add_i('N', i), (0, 0, 1))

    def H_spec(self, i): return self.N(i)

    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.M(), self.MN()]
        elif key == 2:
            return [[self.MA(), self.MB()]]
        elif key == '2Bridges':
            return []
        raise NotImplementedError(f'Invalid key: {key}')
