
import asyncio
import dataclasses
from dataclasses import dataclass

from google.cloud.firestore import AsyncDocumentReference

import jdftx.jdftx_factory
import jdftx.launch_type
import remote.database
import remote.jobs
from ecat_types import AdsorbateInterface, FreeSpecies, Material, Reaction
from jdftx.jdftx import Jdftx
from jdftx.jdftx_factory import JdftxFactory
from jdftx.launch_type import LaunchType
from remote.node import Node

from . import geo
from .mn4c.bi_mn4c import BiMN4C
from .mn4c.mn4c import MN4C


@dataclass(frozen=True)
class Study:
    node: Node
    cores: int
    fluid_dft_kwargs: dict
    reaction: Reaction
    slab_material: Material
    _: dataclasses.KW_ONLY
    _base_slab_dft: Jdftx | None = None
    _bulk_dft: Jdftx | None = None
    _slab_dfts: dict[AdsorbateInterface, Jdftx] | None = None
    _fs_dfts: dict[FreeSpecies, Jdftx] | None = None
    base_elec_smearing_eV: float | None = None
    slab_elec_smearing_eV: float | None = None

    def __post_init__(self):
        if self.slab_material.geo != 'slab':
            raise NotImplementedError(f'Only slab supported, not {self.slab_material.geo}')

    def unique_free_species(self) -> set[FreeSpecies]:
        free_species = set()
        for step in self.reaction.all_steps_cyclic:
            for fs in step.free_species:
                free_species |= Study.free_species_deps(fs)
        return free_species

    def __str__(self):
        return str(self.slab_material)

    @property
    def fluid(self) -> str | None:
        return self.fluid_dft_kwargs.get('fluid', None)

    @property
    def base_mat(self) -> Material:
        if (isinstance(self.slab_material, MN4C) or isinstance(self.slab_material, BiMN4C)):
            return self.slab_material
        return dataclasses.replace(self.slab_material, geo='bulk', facet=None, layers=None, frozen_layers=0)

    @staticmethod
    def free_species_deps(fs: FreeSpecies) -> set[FreeSpecies]:
        '''
        We use the Reversible Hydrogen Electrode
        Note that a pH correction is only needed if we want to compare our electrode voltage to
        directly to another experiment/calculation using the SHE. The differences between the RHE
        and the working electrode will be otherwise consistent
        Standard and Reversible Hydrogen Electrodes: Theory, Design, Operation, and Applications
        https://doi.org/10.1021/acscatal.0c02046
        '''
        deps = {fs}
        if fs.name == 'H+':
            deps |= Study.free_species_deps(FreeSpecies('H2', 'gas'))

        # From Eq 13 in the RHE paper, which holds for both alkaline and acidic environments
        if fs.name == 'OH-':
            deps |= Study.free_species_deps(FreeSpecies('H2O', 'aq'))
            deps |= Study.free_species_deps(FreeSpecies('H+', 'aq'))

        return deps

    async def fs_dft(self, fs: FreeSpecies) -> Jdftx:
        if self._fs_dfts is None:
            object.__setattr__(self, '_fs_dfts', {})

        if fs not in self._fs_dfts:
            material = Material('molecule', basename=fs.name)
            if fs.is_charged:
                self._fs_dfts[fs] = None
            else:
                jdftx_kwargs = {}
                if fs.phase == 'aq':
                    jdftx_kwargs |= self.fluid_dft_kwargs
                self._fs_dfts[fs] = await JdftxFactory.create_molecule(self.node, material, jdftx_kwargs)

        return self._fs_dfts[fs]

    def slab_doc_ref(self, adsorbate: AdsorbateInterface | None, mu_eV: float | None) -> AsyncDocumentReference:
        filename = Jdftx.make_base_filename(self.slab_material.KPOINT_DEFAULT,
                                            fluid=self.fluid, elec_smearing_eV=self.slab_elec_smearing_eV, mu_eV=mu_eV)
        ad_str = str(adsorbate) if adsorbate else 'clean'
        return remote.database.dft_doc(self.node.db, self.slab_material.basename, self.slab_material.geo_str,
                                       ad_str, filename)

    async def slab_dft(self, ad: AdsorbateInterface | None, mu_eV: float | None) -> Jdftx | None:
        if self._slab_dfts is None:
            object.__setattr__(self, '_slab_dfts', {})

        key = (ad, mu_eV)
        if key not in self._slab_dfts:
            fluid = self.fluid_dft_kwargs['fluid'] if self.fluid_dft_kwargs else None
            job_id = Jdftx.make_job_id(self.slab_material, self.slab_material.KPOINT_DEFAULT, fluid=fluid, elec_smearing_eV=self.slab_elec_smearing_eV, adsorbate=ad, mu_eV=mu_eV)
            if await self.node.job_is_running(job_id, use_cache=True):
                self.node.log.info(f'{job_id} already running', material=self.slab_material)
                return None
            
            slab_calc = await jdftx.jdftx_factory.load_from_previous(
                self.node, self.slab_material, ad, self.slab_material.KPOINT_DEFAULT,
                self.fluid, self.slab_elec_smearing_eV, mu_eV=mu_eV)
            if slab_calc is None:
                base_dft = await self.base_dft()
                slab_calc = geo.compute_slab_geometry(self.slab_material, ad, base_dft.calc, self.base_mat.geo)

            jdftx_kwargs = self.fluid_dft_kwargs | self.slab_material.dft_kwargs | {'mu_eV': mu_eV}
            self._slab_dfts[key] = Jdftx(self.slab_material, slab_calc, self.slab_material.KPOINT_DEFAULT,
                                        adsorbate=ad, elec_smearing_eV=self.slab_elec_smearing_eV, **jdftx_kwargs)
        return self._slab_dfts[key]

    def base_doc_ref(self) -> AsyncDocumentReference:
        if self.base_mat.geo == 'slab':
            return self.slab_doc_ref(None, None)
        if self.base_mat.geo == 'bulk':
            filename = Jdftx.make_base_filename(
                self.base_mat.KPOINT_DEFAULT, fluid=None, elec_smearing_eV=self.base_elec_smearing_eV)
            return remote.database.dft_doc(self.node.db, self.base_mat.basename, 'bulk', 'clean', filename)
        raise NotImplementedError(f'Only slab and bulk supported, not {self.base_mat.geo}')

    async def base_dft(self) -> Jdftx | None:
        if self.base_mat.geo == 'slab':
            return await self.slab_dft(None, None)

        if self.base_mat.geo != 'bulk':
            raise ValueError(f'Expected bulk or slab material, got {self.base_mat.geo}')

        if self._bulk_dft is None:
            job_id = Jdftx.make_job_id(self.base_mat, self.base_mat.KPOINT_DEFAULT,
                                       elec_smearing_eV=self.base_elec_smearing_eV)
            if await self.node.job_is_running(job_id, use_cache=True):
                self.node.log.info(f'{job_id} already running', material=self.base_mat)
                return None

            base_dft = await JdftxFactory.create_bulk(self.node, self.base_mat, self.base_mat.KPOINT_DEFAULT,
                                                      self.base_elec_smearing_eV)
            object.__setattr__(self, '_bulk_dft', base_dft)
        return self._bulk_dft

    async def launch_slabs(self, launch_type: LaunchType, skip_converged: bool, nodes_per_job: int,
                           initial_start_index: int, *, mu_eV: float | None = None) -> tuple[LaunchType, int, list[str]]:
        sorted_steps = sorted(self.reaction.steps, key=lambda s: str(s.adsorbate))
        dfts = await asyncio.gather(*[self.slab_dft(step.adsorbate, mu_eV) for step in sorted_steps])
        unfinished_dfts = [dft for dft in dfts if dft and not (skip_converged and dft.calc.is_converged)]

        # We want the small queue on Polaris, which starts at 10 nodes and caps at 24
        if launch_type.batch == 'parallel' and 10 <= len(unfinished_dfts):
            unfinished_dfts = unfinished_dfts[:10]
        else:
            batch = 'serial' if launch_type.batch == 'parallel' else launch_type.batch
            launch_type = LaunchType(dry=launch_type.dry, wrapped=launch_type.wrapped, gpu=launch_type.gpu,
                                     format=launch_type.format, batch=batch)

        lines = []
        for i, dft in enumerate(unfinished_dfts):
            start_index = initial_start_index + (i * nodes_per_job) if launch_type.batch == 'parallel' else 0
            lines += await dft.launch(self.node, launch_type, self.cores, nodes_per_job, start_index) or []
        return launch_type, len(unfinished_dfts), lines or None
