import dataclasses
from dataclasses import dataclass

import molmass

from constants import ATOMIC_SYMBOLS, AVOGADRO, P_ATM_eVpA3, kB_eVpK


@dataclass(frozen=True, order=True)
class Token:
    symbol: str
    count: int


def make_token(symbol: str, count_str: str) -> Token:
    if not symbol or symbol not in ATOMIC_SYMBOLS:
        raise ValueError(f'Unknown atomic symbol {symbol}')

    count = int(count_str) if count_str else 1
    if count < 1:
        raise ValueError(f'Invalid count {count_str} for atomic symbol {symbol}')

    return Token(symbol, count)


def tokenize_molecule(molecule: str) -> list[Token]:
    tokens: list[Token] = []
    symbol = ''
    count_str = ''

    for char in molecule:
        if 'A' <= char <= 'Z':  # Start of a new atomic symbol
            if symbol:
                tokens.append(make_token(symbol, count_str))
            symbol = char
            count_str = ''
        elif symbol and 'a' <= char <= 'z':  # Part of an atomic symbol
            symbol += char
        elif symbol and '0' <= char <= '9':  # Part of a count
            count_str += char
        else:
            raise ValueError(f'Unexpected character {char} in molecule {molecule}')

    # Add the last token
    if symbol:
        tokens.append(make_token(symbol, count_str))

    return tokens


@dataclass(frozen=True, order=True)
class FreeSpecies:
    name: str
    phase: str
    _: dataclasses.KW_ONLY
    bond_length_A: float | None = None
    Etot_eV: float | None = None
    Efree_eV: float | None = None

    def __post_init__(self):
        if self.phase not in ['gas', 'aq', None]:
            raise ValueError(f'phase {self.phase} not supported. Species: {self.name}')

        if self.name == 'e-' and self.phase is not None:
            raise ValueError(f'e- should not have a phase, but {self.phase} was given')

        if self.name == 'H+' and self.phase != 'aq':
            raise ValueError('H+ can only exist in an aq phase')

        if self.name == 'OH-' and self.phase != 'aq':
            raise ValueError('OH- can only exist in an aq phase')

        if self.name in ['H2', 'O2'] and self.phase == 'aq':
            raise ValueError(f'aq solvation not supported for {self.name}')

    def __repr__(self) -> str:
        if self.phase is None:
            return self.name
        return f'{self.name}_{self.phase}'

    @property
    def is_charged(self) -> bool:
        return self.name[-1] in ['-', '+']

    @property
    def is_diatomic(self) -> bool:
        return self.is_homodiatomic or self.is_heterodiatomic

    @property
    def is_homodiatomic(self) -> bool:
        tokens = tokenize_molecule(self.name)
        return len(tokens) == 1 and tokens[0].count == 2

    @property
    def is_heterodiatomic(self) -> bool:
        tokens = tokenize_molecule(self.name)
        return len(tokens) == 2 and tokens[0].count == 1 and tokens[1].count == 1

    @property
    def molar_mass_gpMol(self):
        return molmass.Formula(self.name).mass

    def merge_into_dict(self, count: int, fs_dict: dict['FreeSpecies', int]) -> None:
        if self in fs_dict:
            fs_dict[self] += count
            if fs_dict[self] == 0:
                del fs_dict[self]
        else:
            fs_dict[self] = count

    def number_density_pA3(self, temperature_K: float) -> float:
        if self.phase == 'aq':
            if not hasattr(self, 'molar_mass_gpMol') or self.molar_mass_gpMol is None:
                raise ValueError(f"Molar mass of {str(self)} is not defined.")
            if self.name == 'H2O':
                density_gpcm3 = 0.997048  # www.wolframalpha.com/input?i=water+density
                density_gpA3 = density_gpcm3 * 1E-24
                return (density_gpA3 * AVOGADRO) / self.molar_mass_gpMol

        if self.phase == 'gas':
            pressure_eVpA3 = None
            vapor_pressure_atm = 1
            if self.name == 'O2':
                vapor_pressure_atm = 0.207  # 21 kPa, en.wikipedia.org/wiki/Vapor_pressure
                pressure_eVpA3 = vapor_pressure_atm * P_ATM_eVpA3
            elif self.name == 'H2':
                # The Standard Hydrogen Electrode (SHE) defines the pressure of H2 to be 1 atm
                pressure_eVpA3 = P_ATM_eVpA3
            elif self.name == 'H2O':
                vapor_pressure_atm = 0.0227  # 2.3 kPa, en.wikipedia.org/wiki/Vapor_pressure
                pressure_eVpA3 = vapor_pressure_atm * P_ATM_eVpA3
            else:
                raise ValueError(f"Vapor Pressure of {str(self)} has not yet been implemented.")

            return pressure_eVpA3 / (kB_eVpK * temperature_K)  # from ideal gas law: V/N = P/kT

        raise ValueError(f"Number Density of {str(self)} has not yet been implemented.")
