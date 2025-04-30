from dataclasses import dataclass

from .free_species import FreeSpecies


@dataclass(frozen=True, order=True)
class Spectator:
    """
    A class to represent a Spectator in a catalytic system.
        adsorbate : str
            The adsorbate associated with the spectator.
        sites : tuple[int, ...]
            A tuple of integers representing the sites occupied by the spectator.
        _free_species : tuple[tuple[FreeSpecies, int], ...]
            A tuple of tuples where each inner tuple contains a FreeSpecies object and an integer count.
    """
    adsorbate: str
    sites: tuple[int, ...]
    _free_species: tuple[tuple[FreeSpecies, int], ...]

    @classmethod
    def make_spectator(cls, adsorbate: str, sites: list[int], free_species: list[tuple[str, str, int]]) -> 'Spectator':
        """
        Create a Spectator instance.
            adsorbate (str): The adsorbate species.
            sites (list[int]): A list of site indices where the adsorbate is located.
            free_species (list[tuple[str, str, int]]): A list of tuples representing free species, 
                where each tuple contains the species name, phase, and count.
        """
        free_species = [(FreeSpecies(species, phase), count) for species, phase, count in free_species]
        return cls(adsorbate, tuple(sorted(sites)), tuple(sorted(free_species)))

    @property
    def free_species(self) -> dict[FreeSpecies, int]:
        """
        This method retrieves the free species from the internal storage and 
        returns them as a dictionary.
        """
        return dict(self._free_species)
