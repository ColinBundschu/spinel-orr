from dataclasses import dataclass

from .adsorbate import Adsorbate, AdsorbateInterface
from .free_species import FreeSpecies
from .step import Step


@dataclass(frozen=True, order=True)
class Reaction:
    steps: tuple[Step, ...]
    _one_cycle_free_species: tuple[tuple[FreeSpecies, int], ...] | None

    def __post_init__(self):
        if all(isinstance(step.adsorbate, str) for step in self.steps):
            object.__setattr__(self, 'steps', tuple(sorted(self.steps)))
        else:
            clean_step = []
            adsorbate_steps = []
            adsorbates_steps = []
            for step in self.steps:
                if isinstance(step.adsorbate, Adsorbate):
                    if str(step.adsorbate) == 'clean':
                        clean_step.append(step)
                    else:
                        adsorbate_steps.append(step)
                elif isinstance(step.adsorbate, AdsorbateInterface):
                    adsorbates_steps.append(step)
                else:
                    raise ValueError(f'Invalid adsorbate type: {step.adsorbate}')
            object.__setattr__(self, 'steps', tuple(clean_step + sorted(adsorbate_steps) + sorted(adsorbates_steps)))

        if len(set(step.adsorbate for step in self.steps)) != len(self.steps):
            raise NotImplementedError(f'Not able to handle duplicate adsorbates in {self.steps}')

        for i in range(self.num_steps):
            # Build a set of the number of electrons for each step of a given index.
            # They should have the same value (so the set should be length 1)
            if len({step.free_species.get(FreeSpecies('e-', None), 0) for step in self.steps if step.index == i}) > 1:
                raise ValueError(f'Electron count is not consistent in step {i}')

        if self._one_cycle_free_species is not None:
            object.__setattr__(self, '_one_cycle_free_species', tuple(sorted(self.one_cycle_free_species.items())))

    @property
    def is_cyclic(self):
        return self._one_cycle_free_species is not None

    @property
    def one_cycle_free_species(self) -> dict[FreeSpecies, int]:
        if not self.is_cyclic:
            raise ValueError('Non cyclic reactions do not have a cycle of free species')
        return dict(self._one_cycle_free_species)

    @property
    def num_steps(self):
        return max(step.index for step in self.steps) + 1

    @property
    def all_steps_cyclic(self) -> list[Step]:
        if not self.is_cyclic:
            raise ValueError('Non cyclic reactions do not have cyclic steps')
        all_steps = []
        for step in self.steps:
            all_steps.append(step)
            if step.index == 0:
                new_step_fs = step.free_species
                for species, count in self.one_cycle_free_species.items():
                    species.merge_into_dict(count, new_step_fs)
                all_steps.append(Step(self.num_steps, step.adsorbate, new_step_fs))
        return all_steps


def reaction_from_list(step_list: list[tuple[int, list[str], list[tuple[str, str, int]]]],
                       one_cycle_fs_list: list[tuple[str, str, int]]) -> Reaction:
    '''Accepts a list of arguments from which to make FreeSpecies and Step objects, for compactness'''
    steps = []
    for index, ad_strs, free_species in step_list:
        fs_dict = {FreeSpecies(species, phase): count for species, phase, count in free_species}
        for ad_str in ad_strs:
            steps.append(Step(index, ad_str, fs_dict))
    one_cycle_fs = {FreeSpecies(species, phase): count for species, phase, count in one_cycle_fs_list}
    return Reaction(tuple(steps), one_cycle_fs)
