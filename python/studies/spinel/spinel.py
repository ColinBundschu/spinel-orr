import re
from dataclasses import dataclass

import torch
from torch import Tensor

import constants
import ecat_types.material
from ecat_types import Atom, Atoms, Calc, Lattice, LatticeType, Material
from ecat_types.kpoints import KPoints
from ecat_types.sites import Sites
from studies.spinel import (mono_reg_100o_sites, mono_reg_100s_sites, reg_111b_sites, reg_111s_sites,
                            striped_100o_sites, striped_100s_sites)


def all_same(l):
    return all(l[0] == x for x in l)


def concatenate_symbols(atoms: list[str]) -> str:
    if not atoms:
        return ''
    result = ''
    current_symbol = atoms[0]
    count = 1
    for symbol in atoms[1:]:
        if symbol == current_symbol:
            count += 1
        else:
            result += current_symbol + (str(count) if count > 1 else '')
            current_symbol = symbol
            count = 1
    result += current_symbol + (str(count) if count > 1 else '')
    return result


A_SITES = ('Co', 'Mn', 'Fe', 'Zn', 'Ni')
B_SITES = ('Co', 'Mn', 'Fe', 'Ga', 'Al', 'Ni', 'Cr')
CATEGORIES = (len(A_SITES), len(B_SITES), len(B_SITES))
OHE_LENGTH = sum(CATEGORIES)

@dataclass(frozen=True, order=True)
class Spinel(Material):
    AABBBB: tuple[str]

    def __post_init__(self):
        if len(self.AABBBB) != 6:
            raise ValueError(f'Spinels have exactly 6 cations, but stoich={self.AABBBB}')

        if (not all(x in A_SITES for x in self.AABBBB[:2]) or
            not all(x in B_SITES for x in self.AABBBB[2:4]) or
                not all(x in B_SITES for x in self.AABBBB[4:])):
            raise ValueError(f'Invalid cation in stoich: {self.AABBBB}')

        if self.basename is None:
            object.__setattr__(self, 'basename', self._make_basename())
        super().__post_init__()

    @property
    def KPOINT_DEFAULT(self) -> KPoints:
        if self.geo == 'bulk':
            return KPoints(6, 6, 6)
        if self.geo == 'slab':
            if '100' in self.facet:
                return KPoints(3, 3, 1)
            if '111' in self.facet:
                return KPoints(3, 3, 1)
        raise NotImplementedError()

    @property
    def is_striped(self) -> bool:
        return self.octs[0] != self.octs[1]
    
    @property
    def is_simple(self) -> bool:
        return all_same(self.tets) and all_same(self.octs[:2]) and all_same(self.octs[2:])

    @property
    def dft_U(self) -> str | None:
        return ecat_types.material.dft_U(set(self.AABBBB))

    @property
    def tets(self) -> list[str]:
        return self.AABBBB[:2]

    @property
    def octs(self) -> list[str]:
        return self.AABBBB[2:]

    @property
    def A(self) -> str:
        if not all_same(self.tets):
            raise ValueError('A is ambiguous if tetrahedral sites differ')
        return self.tets[0]

    @property
    def B(self) -> str:
        if not all_same(self.octs):
            raise ValueError('B is ambiguous if octahedral sites differ')
        return self.octs[0]

    @property
    def sites(self) -> Sites:
        if self.facet == '100s':
            if self.is_striped:
                return striped_100s_sites.Striped100sSites(self.tets[0], self.octs[0], self.octs[1])
            return mono_reg_100s_sites.MonoReg100sSites(self.tets[0], self.octs[0])
        if self.facet == '100o':
            if self.is_striped:
                return striped_100o_sites.Striped100oSites(self.octs[0], self.octs[1])
            return mono_reg_100o_sites.MonoReg100oSites(self.octs[0])
        if self.facet == '111s':
            return reg_111s_sites.Reg111sSites(self.tets[1])
        if self.facet == '111b':
            if not all_same(self.octs):
                raise ValueError('B is ambiguous if octahedral sites differ')
            return reg_111b_sites.Reg111bSites(self.octs[0])
        raise NotImplementedError()

    @property
    def default_config(self) -> Calc:
        atoms = Atoms([
            # Tetrahedral (A-Site)
            Atom(constants.ATOMIC_NUMS[self.tets[0]], (0.000, 0.000, 0.000)),
            # Tetrahedral (A-Site)
            Atom(constants.ATOMIC_NUMS[self.tets[1]], (0.250, 0.250, 0.250)),
            # Octahedral (B-Site)
            Atom(constants.ATOMIC_NUMS[self.octs[0]], (0.625, 0.625, 0.125)),
            # Octahedral (B-Site)
            Atom(constants.ATOMIC_NUMS[self.octs[1]], (0.625, 0.125, 0.625)),
            # Octahedral (B-Site)
            Atom(constants.ATOMIC_NUMS[self.octs[2]], (0.125, 0.625, 0.625)),
            # Octahedral (B-Site)
            Atom(constants.ATOMIC_NUMS[self.octs[3]], (0.625, 0.625, 0.625)),
            # Relaxed at 444 bulk Co3O4
            Atom(8,  (0.86, 0.86, 0.41)),
            Atom(8,  (0.38, 0.38, 0.38)),
            Atom(8,  (0.41, 0.86, 0.86)),
            Atom(8,  (0.86, 0.86, 0.86)),
            Atom(8,  (0.86, 0.41, 0.86)),
            Atom(8,  (0.83, 0.38, 0.38)),
            Atom(8,  (0.38, 0.38, 0.83)),
            Atom(8,  (0.38, 0.83, 0.38))])
        return Calc(Lattice(LatticeType.FCO, a=8.0, b=8.2, c=8.4), atoms)

    def _make_basename(self) -> str:
        if self.octs[0] > self.octs[1]:
            raise ValueError(
                'Swapping Oct0 and Oct1 is the same under symmetry, so only allow Oct0 <= Oct1')
        if self.octs[2] > self.octs[3]:
            raise ValueError(
                'Swapping Oct2 and Oct3 is the same under symmetry, so only allow Oct2 <= Oct3')

        # A3O4: All tet and oct sites are the same element
        if all_same(self.AABBBB):
            return f'{self.A}3O4'

        # AB2O4: Regular Spinel: All oct sites are the same, all tet sites are the same, but tet != oct
        if all_same(self.tets) and all_same(self.octs):
            return f'{self.A}{self.B}2O4'

        # A2BO4: The surface looks like a monoatomic regular spinel (all A)
        if all_same(self.tets + self.octs[:2]) and all_same(self.octs[2:]):
            return f'{self.A}2{self.octs[2]}O4'

        # ABAO4: The surface looks like a regular spinel (A tet, B oct)
        if all_same(self.tets + self.octs[2:]) and all_same(self.octs[:2]):
            return f'{self.A}{self.octs[0]}{self.A}O4'

        # ABCO4: The surface looks like a regular spinel (A tet, B oct)
        if all_same(self.tets) and all_same(self.octs[:2]) and all_same(self.octs[2:]):
            return f'{self.A}{self.octs[0]}{self.octs[2]}O4'

        # AAABABO4: The surface oct sites look striped with A and B
        if all_same(self.tets + (self.octs[0], self.octs[2])) and self.octs[1] == self.octs[3]:
            return f'{self.A}2x{self.octs[1]}O4'

        # AABCBCO4: The surface oct sites look striped with B and C
        if all_same(self.tets) and self.octs[0] == self.octs[2] and self.octs[1] == self.octs[3]:
            return f'{self.A}{self.octs[0]}x{self.octs[1]}O4'

        return f'{concatenate_symbols(self.AABBBB)}O8'

    def one_hot_encoding(self, simple_only: bool) -> Tensor:
        if simple_only:
            if not self.is_simple:
                raise ValueError('Currently only testing simple spinels')
            encoded_A = torch.zeros(len(A_SITES))
            encoded_B0 = torch.zeros(len(B_SITES))
            encoded_B1 = torch.zeros(len(B_SITES))
            encoded_A[A_SITES.index(self.AABBBB[0])] += 1
            encoded_B0[B_SITES.index(self.AABBBB[2])] += 1
            encoded_B1[B_SITES.index(self.AABBBB[4])] += 1
            return torch.cat((encoded_A, encoded_B0, encoded_B1)).to(dtype=torch.double)

        encoded_A0 = torch.zeros(len(A_SITES))
        encoded_A1 = torch.zeros(len(A_SITES))
        encoded_B0 = torch.zeros(len(B_SITES))
        encoded_B1 = torch.zeros(len(B_SITES))
        encoded_B2 = torch.zeros(len(B_SITES))
        encoded_B3 = torch.zeros(len(B_SITES))
        encoded_A0[A_SITES.index(self.AABBBB[0])] += 1
        encoded_A1[A_SITES.index(self.AABBBB[1])] += 1
        encoded_B0[B_SITES.index(self.AABBBB[2])] += 1
        encoded_B1[B_SITES.index(self.AABBBB[3])] += 1
        encoded_B2[B_SITES.index(self.AABBBB[4])] += 1
        encoded_B3[B_SITES.index(self.AABBBB[5])] += 1
        return torch.cat((encoded_A0, encoded_A1, encoded_B0, encoded_B1, encoded_B2, encoded_B3)).to(dtype=torch.double)

    @property
    def ratio_encoding(self) -> Tensor:
        element_list = sorted(
            list(set(A_SITES).union(set(B_SITES)).union(set(B_SITES))))
        encoding = torch.zeros(len(element_list), dtype=torch.double)
        for element in self.AABBBB:
            encoding[element_list.index(element)] += 1
        return encoding

    @property
    def property_encoding(self) -> tuple[float]:
        # if not (all_same(self.tets) and all_same(self.octs[:2]) and all_same(self.octs[2:])):
        #     raise ValueError('Currently only testing simple spinels')
        enc = [A_SITES.index(self.AABBBB[0]), A_SITES.index(self.AABBBB[1]),
               B_SITES.index(self.AABBBB[2]), B_SITES.index(self.AABBBB[3]),
               B_SITES.index(self.AABBBB[4]), B_SITES.index(self.AABBBB[5]),
               ]
        return tuple(enc)

    def adsorbate_full_encoding(self, adsorbate: str, simple_only: bool) -> tuple[Tensor]:
        """
        Encodes an adsorbate string into a 1D tensor of length N_O_sites + N_oct_sites,
        where each tensor represents 1 for indices that match the adsorbate sites and 0 otherwise.

        Args:
            adsorbate (str): The adsorbate string to encode. Expects to only have H bound to O sites,
                and OH bound to octahedral sites.

        Returns:
            Tensor: A 1D tensor of length N_O_sites + N_oct_sites.
        """

        if not self.facet == '100o':
            raise ValueError('Currently only testing 100o spinels')

        O_tensor, oct_tensor = self.encode_adsorbate(adsorbate)
        return torch.cat([self.one_hot_encoding(simple_only), O_tensor, oct_tensor])

    def encode_adsorbate(self, adsorbate):
        N_O_sites = 8
        O_tensor = torch.zeros(N_O_sites, dtype=torch.double)
        N_oct_sites = 4
        oct_tensor = torch.zeros(N_oct_sites, dtype=torch.double)

        # Parse the location of the adsorbed H on oxygen sites
        O_matches = [int(match)
                     for match in re.findall(r'O(\d+)-xH', adsorbate)]
        if max(O_matches, default=-1) >= N_O_sites:
            raise ValueError(f'Invalid adsorbate: {adsorbate}')

        # Create tensors for O sites and octahedral sites
        for idx in O_matches:
            O_tensor[idx] = 1

        if self.AABBBB[2] == self.AABBBB[3]:
            # Parse the location of the adsorbed OH on octahedral sites
            oct_matches = [int(match)
                           for match in re.findall(r'Oct(\d+)-xOH', adsorbate)]
            if max(oct_matches, default=-1) >= N_oct_sites:
                raise ValueError(f'Invalid adsorbate: {adsorbate}')
        else:
            # Parse the location of the adsorbed OH on octahedral sites
            oct0_matches = [int(match) for match in re.findall(
                f'Oct{self.AABBBB[2]}(\\d+)-xOH', adsorbate)]
            oct1_matches = [int(match) + N_oct_sites // 2 for match in re.findall(
                f'Oct{self.AABBBB[3]}(\\d+)-xOH', adsorbate)]
            oct_matches = oct0_matches + oct1_matches
            if not (max(oct0_matches, default=-1) < N_oct_sites // 2
                    <= max(oct1_matches, default=N_oct_sites // 2) < N_oct_sites):
                raise ValueError(f'Invalid adsorbate: {adsorbate}')

        for idx in oct_matches:
            oct_tensor[idx] = 1

        # Ensure the total number of matches corresponds to the adsorbate structure
        if len(O_matches) + len(oct_matches) != adsorbate.count('_') + 1 and adsorbate != 'clean':
            raise ValueError(f'Failed to parse adsorbate: {adsorbate}')
        return O_tensor,oct_tensor


def spinel_from_name(name: str, facet: str) -> Spinel:
    cations, oxygens = name[:-2], name[-2:]
    if oxygens != 'O4' and oxygens != 'O8':
        raise ValueError(f'Invalid spinel name: {name}')

    is_striped = 'x' in cations
    cations = cations.replace('x', '')
    if is_striped and oxygens == 'O8':
        raise ValueError(
            f'Invalid spinel name (Only O4 specifies a striped surface): {name}')

    # This regular expression matches an element's symbol (one uppercase followed by optional lowercase letters)
    # followed optionally by a number which captures the count of the element
    pattern = re.compile(r"([A-Z][a-z]*)(\d*)")
    cation_list = []
    for element, count in pattern.findall(cations):
        count = int(count) if count else 1
        cation_list.extend([element] * count)
    # Stupid hack to avoid linter warning with unpacking below
    cation_tuple = tuple([x for x in cation_list])

    if oxygens == 'O8':
        if len(cation_tuple) != 6:
            raise ValueError(
                f'Invalid spinel name (O8 requires 6 cations): {name}')
        return Spinel(None, cation_tuple, facet=facet)

    if len(cation_tuple) != 3:
        raise ValueError(
            f'Invalid spinel name (O4 requires 3 cations): {name}')
    A, B0, B1 = cation_tuple

    if is_striped:
        return Spinel(None, (A, A, B0, B1, B0, B1), facet=facet)
    return Spinel(None, (A, A, B0, B0, B1, B1), facet=facet)
