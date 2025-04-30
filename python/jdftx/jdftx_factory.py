

import dataclasses
import os

import remote.database
from ecat_types import AdsorbateInterface, Calc, Material
from ecat_types.kpoints import KPoints
from jdftx.jdftx import VAC_A_DEFAULT, Jdftx
from remote.node import Node
from studies import geo


async def load_from_previous(node: Node, material: Material, adsorbate: AdsorbateInterface | None,
                             kpoints: KPoints, fluid: str | None, elec_smearing_eV: float | None,
                             mu_eV: float | None) -> Calc | None:
    # If using fluid, check other fluid models as well, in order of quality
    fluid_models = [fluid] + [model for model in ['candle', 'nlpcm', None] if model != fluid] if fluid else [None]
    elec_smearings_eV = [elec_smearing_eV]
    # We do not load the smearing except for bulk because of different lattice vectors
    if material.geo == 'bulk' and elec_smearing_eV is not None:
        elec_smearings_eV += [smearing for smearing in [None, 0.025, 0.05] if elec_smearing_eV != smearing]
    folder = Jdftx.make_folder(node.dft_out_folder, material, adsorbate=adsorbate)
    exact_job_id = None
    for load_fluid in fluid_models:
        for elec_smear_eV in elec_smearings_eV:
            filename = Jdftx.make_base_filename(kpoints, fluid=load_fluid, elec_smearing_eV=elec_smear_eV, mu_eV=mu_eV)
            job_id = Jdftx.make_job_id(material, kpoints, fluid=fluid, elec_smearing_eV=elec_smear_eV, adsorbate=adsorbate, mu_eV=mu_eV)
            if await node.job_is_running(job_id, use_cache=True):
                raise ValueError(f'Job {job_id} is running and should not be loaded')
            calc = await remote.database.sync_calculation(node, os.path.join(folder, filename), job_is_running=False)

            # We only load from calc states that have completed at least one full ionic iteration
            if calc is not None and calc.forces_max_L2_eVpA is not None:
                if exact_job_id:
                    node.log.info(f'{exact_job_id}  Loaded from {filename}')
                    calc = dataclasses.replace(calc, origin=filename)
                else:
                    node.log.debug(f'{job_id}  Loaded')
                return calc

            exact_job_id = exact_job_id or job_id
    return None


class JdftxFactory():
    @staticmethod
    async def create_bulk(node: Node, material: Material, kpoints: KPoints, elec_smearing_eV: float | None, *,
                          lattice_tol_frac: float = 0.2) -> Jdftx:
        if material.geo != 'bulk':
            raise ValueError(f'Expected bulk material, got {material.geo}')

        origin_calc = geo.make_default_calc(material)
        calc = await load_from_previous(node, material, adsorbate=None, kpoints=kpoints, fluid=None, elec_smearing_eV=elec_smearing_eV, mu_eV=None)

        if calc:
            # Check that the bulk and default lattice vectors are somewhat close to the same.
            for a, b in zip(origin_calc.lattice_A.R_A.ravel(), calc.lattice_A.R_A.ravel()):
                if a == b:
                    continue
                if a == 0 or b == 0:
                    raise ValueError(f'0s mismatch: loaded {calc.lattice_A} vs. origin {origin_calc.lattice_A}')
                if abs((a-b)/a) > lattice_tol_frac:
                    raise ValueError(f'Lattice mismatch: loaded {calc.lattice_A} vs. origin {origin_calc.lattice_A}')

            # Check that the number of atoms match
            if len(origin_calc.atoms_L) != len(calc.atoms_L):
                raise ValueError(f'atoms mismatch: loaded {calc.atoms_L} vs. origin {origin_calc.atoms_L}')
        else:
            calc = origin_calc

        bulk_calc = geo.compute_bulk_geometry(material, calc)
        return Jdftx(material, bulk_calc, kpoints, elec_smearing_eV=elec_smearing_eV, **material.dft_kwargs)

    @staticmethod
    async def create_molecule(node: Node, material: Material, jdftx_kwargs: dict, *,
                              vacuum_A=(VAC_A_DEFAULT, VAC_A_DEFAULT, VAC_A_DEFAULT)) -> Jdftx:
        default_calc = geo.make_default_calc(material)
        molecule_calc = await load_from_previous(node, material, adsorbate=None, kpoints=KPoints(1, 1, 1),
                                                 fluid=jdftx_kwargs.get('fluid', None), elec_smearing_eV=None, mu_eV=None)
        if molecule_calc:
            # Check that the number of atoms and the atomic numbers match
            if any(a.atomic_num != b.atomic_num for a, b in zip(default_calc.atoms_L, molecule_calc.atoms_L)):
                raise ValueError(
                    f'{material} atoms mismatch: {default_calc.atoms_L} molecule vs. {molecule_calc.atoms_L} default')
        else:
            molecule_calc = geo.adjust_vacuum_and_recenter(default_calc, vacuum_A)

        return Jdftx(material, molecule_calc, KPoints(1, 1, 1), **(jdftx_kwargs | material.dft_kwargs))
