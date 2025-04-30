
import functools
import itertools
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cmp_to_key

import numpy as np

import constants

from .atom import Atom, Atoms
from .attach_site import AttachSite
from .calc import Calc
from .material import Material
from .sites import Sites


@dataclass(frozen=True, order=True)
class AdsorbateInterface(ABC):
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def attach(self, calc: Calc) -> Calc:
        pass


@dataclass(frozen=True, order=True)
class Adsorbate(AdsorbateInterface):
    group: str | None
    attach_site: AttachSite | None

    def __post_init__(self):
        if self.group is None and self.attach_site is not None:
            raise ValueError('If attach_site is defined, then group must also be defined')
        if self.group is not None and self.attach_site is None:
            raise ValueError('If group is defined, then attach_site must also be defined')

    def __str__(self):
        if not self.group:
            return 'clean'
        group = self.group.replace('*', 'x')
        return f'{self.attach_site.label}-{group}'

    # Positions are in angstroms
    @staticmethod
    def compute_atoms(group: str, attach_site: AttachSite, *,
                      perturbation: tuple[float, float, float] = (0.05, -0.03, 0)) -> Atoms:
        extracted_symbol = re.match(r'\*?([a-zA-Z][a-zA-Z]?)\d?', group)
        if extracted_symbol and extracted_symbol.groups()[0] in constants.ATOMIC_SYMBOLS:
            if len(extracted_symbol.groups()) != 1:
                raise ValueError(f'Invalid group: {group}')
            symbol = extracted_symbol.groups()[0]
            atoms = Atoms([Atom(constants.ATOMIC_NUMS[symbol], (0, 0, 0))])

        elif group in ['*OO']:
            if attach_site.config == 'linear':
                atoms = Atoms([Atom(8, (0, 0, 0)), Atom(8, (0, 0, 1.22442))])
            else:
                atoms = Atoms([Atom(8, (-0.47, 0, 0.1)), Atom(8, (0.61, 0,  1.14))])

        elif group in ['*O*O', '*OO*']:
            atoms = Atoms([Atom(8, (0, 0, 0)), Atom(8, (0, 0, 1.22442))])

        elif group in ['O*O']:
            atoms = Atoms([Atom(8, (-0.7, 0, -0.3)), Atom(8, (0.7, 0, -0.3))])

        elif group.startswith('*OH'):
            if attach_site.config == 'linear':
                atoms = Atoms([Atom(8, (0, 0, 0)), Atom(1, (0, 0, 1))])
            else:
                atoms = Atoms([Atom(8, (0.32896603, 0, 0.13315941)),
                               Atom(1, (-0.42, 0, 0.75))])

        elif group in ['*OOH', '*O*OH']:
            atoms = Atoms([Atom(8, (0.4677784392541898, 0, 0.10767426)),
                           Atom(8, (-0.611551891, 0,  1.14415604)),
                           Atom(1, (-0.16365483, 0, 1.96511914))])

        elif group == '*OO*H':
            atoms = Atoms([Atom(8, (0, 0, 0)),
                           Atom(8, (0, 0, 1.22442)),
                           Atom(1, (0.9, 0, 1.52))])

        elif group in ['O*OH']:
            atoms = Atoms([Atom(8, (0, 0, 0.215092)),
                           Atom(8, (1.164335, 0, 0.972334)),
                           Atom(1, (-0.741185, 0, 0.977334))])

        elif group in ['*HOH']:
            atoms = Atoms([Atom(1, (0, 0, 0)), Atom(8, (0, 0, 0.98159)), Atom(1, (0.9542, 0, 1.21187))])

        elif group in ['*NO']:
            atoms = Atoms([Atom(7, (0, 0, 0)), Atom(8, (0, 0, 1.15))])

        elif group in ['*ON']:
            atoms = Atoms([Atom(8, (0, 0, 0)), Atom(7, (0, 0, 1.15))])

        elif group in ['*']:
            atoms = Atoms([])

        else:
            raise NotImplementedError

        atoms = atoms.rotated(attach_site.rotation)
        atoms = atoms.translated(attach_site.site_start_As)
        if perturbation is not None:
            atoms = atoms.translated(perturbation)
        return atoms

    @staticmethod
    def attach_site_comparison(a: tuple[float, float, float], b: tuple[float, float, float], *,
                               tol_L: float = 0.02, angle_tol: float = math.radians(2), use_angle: bool = True) -> int:
        '''Order surface atoms in a reproducible way to enable consistent adsorbate attachments across materials'''
        x1, y1, z1 = a
        x2, y2, z2 = b

        # First compare depth: large z values are considered "smaller" (closer to the slab's surface, assumed +Z)
        if abs(z1 - z2) > tol_L:
            return -1 if z1 > z2 else 1

        if use_angle:
            # Calculate the angle from a very arbitrarily chosen axis centered around the middle of the slab (0.5, 0.5)
            # The offset means items on the axis get picked even if they are off by a bit
            branch_cut_offset = 2 * angle_tol + math.pi/8
            angle1 = (math.atan2(y1 - 0.5, x1 - 0.5) - branch_cut_offset) % (2 * math.pi)
            angle2 = (math.atan2(y2 - 0.5, x2 - 0.5) - branch_cut_offset) % (2 * math.pi)
            if abs(angle1 - angle2) > 2 * math.pi - angle_tol:
                raise ValueError(
                    f'Comparison Undefined: Atoms at {a} and {b} are close in angle and straddle branch cut. '
                    'This comparitor results in contradictions if this happens, e.g., if angle_tol=2 and the '
                    ' angles mod 360 are A:0.5, B:180, C:359.5 and dist(C) < dist(A), then A < B < C < A')
            # The atom with the larger angle is "smaller"
            if abs(angle1 - angle2) > angle_tol:
                return -1 if angle1 > angle2 else 1

        # In the case of the same angle, the "smaller" atom is the one closer to the middle of the slab
        dist1 = math.sqrt((x1 - 0.5)**2 + (y1 - 0.5)**2)
        dist2 = math.sqrt((x2 - 0.5)**2 + (y2 - 0.5)**2)
        return -1 if dist1 < dist2 else 1

    def attach(self, calc: Calc) -> Calc:
        if not self.group:
            return calc

        attach_atomic_num = constants.ATOMIC_NUMS[self.attach_site.element]
        possible_attach_sites = [atom.xyz for atom in calc.atoms_L if atom.atomic_num == attach_atomic_num]
        possible_attach_sites.sort(key=cmp_to_key(functools.partial(
            Adsorbate.attach_site_comparison, use_angle=self.attach_site.use_angle)))
        chosen_site = possible_attach_sites[self.attach_site.z_depth_index]

        R_inv = np.linalg.inv(calc.lattice_A.R_A)
        ad_atoms_L = Adsorbate.compute_atoms(self.group, self.attach_site)
        adsorbed_xyzs = [chosen_site + R_inv@atom.xyz for atom in ad_atoms_L]
        final_atoms_L = Atoms(calc.atoms_L + ad_atoms_L.repositioned(adsorbed_xyzs))
        return Calc(calc.lattice_A, final_atoms_L)


@dataclass(frozen=True, order=True)
class Adsorbates(AdsorbateInterface):
    adsorbates: tuple[Adsorbate]

    def __post_init__(self):
        if len(self.adsorbates) < 2:
            raise ValueError('Adsorbates should be multiple elements. Otherwise use Adsorbate')
        object.__setattr__(self, 'adsorbates', tuple(sorted(self.adsorbates, key=str)))

    def __str__(self):
        return '_'.join(map(str, self.adsorbates))

    def attach(self, calc: Calc) -> Calc:
        for adsorbate in self.adsorbates:
            calc = adsorbate.attach(calc)
        return calc


def adsorbates_from_site_groups(disjoint_site_groups: list[list[str]], material: Material) -> list[AdsorbateInterface]:
    '''
    Site groups is a list of adsorbate strings not yet assigned a site on the surface.
    Each of these strings should have its own disjoint set of sites on the surface for this to work.
    This function determines all possible ways to attach each adsorbate string (site group) to the surface,
    which forms an adsorbate, and then comes up with all possible ways to to combine the adsorbates from
    the disjoint sets. For a single item in site_groups, this returns an adsorbate. For multiple item,
    it combines them into an Adsorbates object.
    '''
    all_ads = [ads_from_site_group(material.sites, site_group) for site_group in disjoint_site_groups]

    out = []
    for ad_combo in itertools.product(*all_ads):
        flat_ads = []
        for ad_or_ads in itertools.chain(ad_combo):
            flat_ads += ad_or_ads if isinstance(ad_or_ads, list) else [ad_or_ads]
        out.append(flat_ads[0] if len(flat_ads) == 1 else Adsorbates(flat_ads))
    return out


def ads_from_site_group(sites: Sites, site_group: list[str]) -> list[Adsorbate | list[Adsorbate]]:
    # We have multiple adsorbates on one site
    if len(site_group) > 1:
        return [[Adsorbate(group, site) for group, site in zip(site_group, multi_site)]
                for multi_site in sites.matching_sites(len(site_group))]

    # There is just a single adsorbate at this site
    [group] = site_group
    # Check if it is a spectator hydrogen
    if group.startswith('*H') and not group.startswith('*HOH') and sites.H_spec(0) is not None:
        # Check if it follows the format "*H#", where # is a single digit number
        if str.isdigit(group[2:]):
            i = int(group[2:])
        else:
            raise ValueError(f'Invalid spectator species: {group}')
        return [Adsorbate('*H', sites.H_spec(i))]
    else:
        if group.count('*') == 1:
            index = 1
        elif group.count('*') == 2:
            index = '2Bridges'
        else:
            raise ValueError(f'Invalid adsorbate group: {group}')
        return [Adsorbate(group, site) for site in sites.matching_sites(index)]
