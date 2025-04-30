
from dataclasses import dataclass

from ecat_types import AttachSite, Sites
# pylint: disable=multiple-statements, line-too-long

@dataclass(frozen=True, order=True)
class PtSites(Sites):
    # Attach sites
    def Atop(self, i): return AttachSite('Pt', i, f'a{i}', (0, 0, 1.68))
    def Bridge(self, i): return AttachSite('Pt', i, f'b{i}', (-1, -1, 1.6))
    def Hollow(self, i): return AttachSite('Pt', i, f'h{i}', (-2, 0, 1.2))

    def H_spec(self, i: int) -> AttachSite:
        return None
    
    def matching_sites(self, key: int | str) -> list[AttachSite | list[AttachSite]]:
        if key == 1:
            return [self.Atop(0), self.Hollow(0), self.Bridge(0)]
        if key == 2:
            return [
                [self.Atop(0), self.Atop(1)],
                [self.Atop(0), self.Atop(2)],
                [self.Bridge(0), self.Bridge(1)],
                [self.Bridge(0), self.Bridge(2)],
                [self.Hollow(0), self.Hollow(1)],
                [self.Hollow(0), self.Hollow(2)],
                ]
        if key == '2Bridges':
            return []
        if key == 3:
            return [
                [self.Atop(0), self.Atop(1), self.Atop(2)],
                [self.Atop(0), self.Atop(1), self.Atop(3)],
                [self.Bridge(0), self.Bridge(1), self.Bridge(2)],
                [self.Bridge(0), self.Bridge(1), self.Bridge(3)],
                [self.Hollow(0), self.Hollow(1), self.Hollow(2)],
                [self.Hollow(0), self.Hollow(1), self.Hollow(3)],
                ]
        if key == 4:
            return [
                [self.Atop(0), self.Atop(1), self.Atop(2), self.Atop(3)],
                [self.Atop(0), self.Atop(2), self.Atop(4), self.Atop(6)],
                [self.Bridge(0), self.Bridge(1), self.Bridge(2), self.Bridge(3)],
                [self.Bridge(0), self.Bridge(2), self.Bridge(4), self.Bridge(6)],
                [self.Hollow(0), self.Hollow(1), self.Hollow(2), self.Hollow(3)],
                [self.Hollow(0), self.Hollow(2), self.Hollow(4), self.Hollow(6)],
            ]
        if key == 5:
            return [
                [self.Atop(i) for i in range(key)],
                [self.Bridge(i) for i in range(key)],
                [self.Hollow(i) for i in range(key)],
                [self.Atop(i) for i in [1,3,5,6,7]],
                [self.Bridge(i) for i in [1,3,5,6,7]],
                [self.Hollow(i) for i in [1,3,5,6,7]],
                [self.Atop(i) for i in [1,4,5,6,7]],
                [self.Bridge(i) for i in [1,4,5,6,7]],
                [self.Hollow(i) for i in [1,4,5,6,7]],
                [self.Atop(i) for i in [1,3,4,6,7]],
                [self.Bridge(i) for i in [1,3,4,6,7]],
                [self.Hollow(i) for i in [1,3,4,6,7]],
            ]
        if key == 6:
            return [
                [self.Atop(i) for i in range(key)],
                [self.Bridge(i) for i in range(key)],
                [self.Hollow(i) for i in range(key)],
                [self.Atop(i) for i in [1,3,4,5,6,7]],
                [self.Bridge(i) for i in [1,3,4,5,6,7]],
                [self.Hollow(i) for i in [1,3,4,5,6,7]],
            ]
        if key in [7, 8]:
            return [
                [self.Atop(i) for i in range(key)],
                [self.Bridge(i) for i in range(key)],
                [self.Hollow(i) for i in range(key)],
            ]
        raise NotImplementedError(f'Invalid key: {key}')
