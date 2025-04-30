
import dataclasses
import itertools
from dataclasses import dataclass

import ecat_types.adsorbate
from ecat_types import Adsorbate, AdsorbateInterface, Material, Reaction, Spectator, Step

@dataclass(frozen=True)
class Filter:
    '''If guest list is set, then only ads "on the guest list" are allowed.
    this eventually will support forces convergence and run status'''
    _: dataclasses.KW_ONLY
    guest_list: list[str] = None

    def is_allowed(self, ad: AdsorbateInterface) -> bool:
        if self.guest_list:
            return str(ad) in self.guest_list
        return True


def compute_steps(reaction: Reaction,
                  spectators: tuple[None | Spectator, ...],
                  material: Material,
                  ad_filter: Filter) -> Reaction:
    '''
    Each item in the reaction list is a compact way to specify
    Step objects for multiple adsorbates who share the same step index
    and free species: (step index, list of adsorbates, free species)
    e.g. (1, ['*OO','O*O','(*O)(*O)','*OO*'], [('H2O','aq',2),('OH-','aq',1),('e-',None,3)])
    The spectators are a second set of adsorbates and free species,
    but without a step index. All combinations of reaction steps and
    spectators are taken to make each final step.
    This allows us to represent a large number of surface states
    returns a list of tuples: (step_index, ad_names, fs_dict)
    '''

    if len(set(spectators)) != len(spectators):
        raise ValueError(f'Duplicate spectator detected in {spectators}')

    steps = []
    for spectator, base_step in itertools.product(spectators, reaction.steps):
        for step in steps_from_spectator_and_str_step(material, base_step, spectator, reaction):
            if ad_filter.is_allowed(step.adsorbate):
                steps.append(step)
    return Reaction(tuple(steps), reaction.one_cycle_free_species)


def steps_from_spectator_and_str_step(material: Material, step: Step, spectator: Spectator | None,
                                  reaction: Reaction) -> list[Step]:
    if step.adsorbate == 'clean' and spectator is None:
        return [Step(step.index, Adsorbate(None, None), step.free_species)]

    ad_names: list[list[str]] = []
    if step.adsorbate is not None and step.adsorbate != 'clean':
        ad_names.append(split_ad_name(step.adsorbate))

    new_index = step.index
    new_fs_dict = step.free_species
    if spectator:
        if len(split_ad_name(spectator.adsorbate)) != 1 or split_ad_name(spectator.adsorbate)[0] != spectator.adsorbate:
            raise NotImplementedError(f'Split spectators not yet supported: {spectator.adsorbate}')

        for fs, count in spectator.free_species.items():
            fs.merge_into_dict(count, new_fs_dict)
            if fs.name == 'e-':
                new_index -= count
                while new_index < 0:
                    new_index += reaction.num_steps
                    for cycle_fs, cycle_fs_count in reaction.one_cycle_free_species.items():
                        cycle_fs.merge_into_dict(cycle_fs_count, new_fs_dict)
                    if new_index >= reaction.num_steps:
                        raise ValueError('Reaction cycle causes step to overshoot, check reaction')

                while new_index >= reaction.num_steps:
                    new_index -= reaction.num_steps
                    for cycle_fs, cycle_fs_count in reaction.one_cycle_free_species.items():
                        cycle_fs.merge_into_dict(-cycle_fs_count, new_fs_dict)
                    if new_index < 0:
                        raise ValueError('Reaction cycle causes step to overshoot, check reaction')

        ad_names += [[f'{spectator.adsorbate}{site}'] for site in spectator.sites]

    steps = [Step(new_index, adsorbate, new_fs_dict) for adsorbate in
            ecat_types.adsorbate.adsorbates_from_site_groups(ad_names, material)]
    return steps


def split_ad_name(ad_name: str) -> list[str] | None:
    split_name = []
    if not ')(' in ad_name:
        split_name = [ad_name]
    else:
        if not (ad_name.startswith('(') and ad_name.endswith(')')):
            raise ValueError(f'Invalid adsorbate name: {ad_name}')

        split_name = [ad for ad in ad_name[1:-1].split(')(')]

    if any('(' in ad or ')' in ad or '*' not in ad for ad in split_name):
        raise ValueError(f'Invalid adsorbate name: {ad_name}')

    return split_name
