from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class KPoints:
    x: int
    y: int
    z: int

    def __post_init__(self):
        for x in (self.x, self.y, self.z):
            if not isinstance(x, int):
                raise TypeError(f'Expected int, got {type(x)}')
            if x < 1:
                raise ValueError(f'Expected value >= 1, got {x}')

    def __repr__(self) -> str:
        if self.x < 10 and self.y < 10 and self.z < 10:
            return f'{self.x}{self.y}{self.z}'
        return f'{self.x}x{self.y}x{self.z}'

    @property
    def tuple(self) -> tuple[int, int, int]:
        return (self.x, self.y, self.z)
