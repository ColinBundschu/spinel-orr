
import dataclasses
from dataclasses import dataclass

from scipy.spatial.transform import Rotation


@dataclass(frozen=True, order=True)
class AttachSite:
    element: str
    z_depth_index: int
    label: str
    site_start_As: tuple[float, float, float]
    _: dataclasses.KW_ONLY
    config: str = None
    rotation: tuple[float, float, float, float] = tuple(Rotation.identity().as_quat())
    use_angle: bool = True

    def __post_init__(self):
        if isinstance(self.rotation, Rotation):
            object.__setattr__(self, 'rotation', tuple(self.rotation.as_quat()))  # pylint: disable=E1101
