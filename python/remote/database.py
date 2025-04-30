
import asyncio
from dataclasses import dataclass
import datetime
import math
import os
import pickle
import re
from enum import Enum

import numpy as np
from google.cloud.firestore import (AsyncDocumentReference, DocumentSnapshot,
                                    FieldFilter)
from structlog import BoundLogger

import ecat_types.atom
import ecat_types.minimization_point
import jdftx.parse
from constants import (ANGSTROM2_GRAM_PER_EV_SEC2, AVOGADRO, MNT_PATH,
                       MAX_FORCE_LATTICE_eVpA, h2_A2eVu, hbar_eVs, hc_eVm,
                       kB_eVpK)
from ecat_types import (Atom, Atoms, Calc, FreeSpecies, Lattice, LatticeType,
                        Material, Status, Step)
from jdftx.jdftx import Jdftx
from remote.async_batch_database import AsyncBatchDatabase
from remote.node import Node, StatResult
from studies import Study


def dft_doc(db: AsyncBatchDatabase, material_name: str, geo_str: str, ad_str: str, filename: str
            ) -> AsyncDocumentReference:
    material_doc_ref = db.client.collection(
        'materials').document(material_name)
    ad_doc_ref = material_doc_ref.collection(geo_str).document(ad_str)
    return ad_doc_ref.collection('calcs').document(filename)


async def update_study(node: Node, study: Study, delete_first: bool) -> None:
    study_log = node.log.bind(study=str(study))
    study_doc_ref = node.db.client.collection('studies').document(str(study))
    doc = await study_doc_ref.get()
    if doc.exists:
        if delete_first:
            study_log.info(event='deleting_old_study')
            await node.db.delete_doc(study_doc_ref)
        else:
            study_log.info(event='study_exists')
            return

    study_log.info(event='adding_study')
    # Make the study document
    data = {
        'Material': study.slab_material.basename,
        'Facet': study.slab_material.facet,
        'Layers': study.slab_material.layers,
        'Frozen_layers': study.slab_material.frozen_layers,
        'Base_dft_ref': study.base_doc_ref(),
        'free_species_name': [],
        'free_species_phase': [],
        'free_species_molar_mass_gpMol': [],
        'free_species_dft_ref': [],
        'step_index': [],
        'step_dft_ref': [],
        'step_free_species_counts': [],
    }

    # Add the free species
    free_species = list(study.unique_free_species())
    for fs in free_species:
        data['free_species_name'].append(fs.name)
        data['free_species_phase'].append(fs.phase)
        if fs.is_charged:
            data['free_species_molar_mass_gpMol'].append(None)
            data['free_species_dft_ref'].append(None)
        else:
            fluid = study.fluid if fs.phase == 'aq' else None
            filename = Jdftx.make_base_filename('111', fluid=fluid)
            data['free_species_molar_mass_gpMol'].append(fs.molar_mass_gpMol)
            data['free_species_dft_ref'].append(
                dft_doc(node.db, fs.name, 'molecule', 'clean', filename))

    # Add the reaction steps
    for step in study.reaction.all_steps_cyclic:
        data['step_index'].append(step.index)
        # TODO: mu_eV
        data['step_dft_ref'].append(study.slab_doc_ref(step.adsorbate, None))
        for fs in free_species:
            for fs_step, count in step.free_species.items():
                if fs_step == fs:
                    data['step_free_species_counts'].append(count)
                    break
            else:
                data['step_free_species_counts'].append(0)

    await node.db.set(study_doc_ref, data)


class UpdateResult(Enum):
    DBBetter = 'DB Better'
    OverwriteDB = 'Overwrite DB'
    LocalBetter = 'Local Better'
    CalcInSync = 'Synced'
    NewLocalCalc = 'New Local'


def calc_firestore_size_KB(path: str, data: dict) -> int:
    def field_size(item) -> int:
        if isinstance(item, str):
            return len(item) + 1
        if isinstance(item, datetime.datetime):
            return 8
        if isinstance(item, float):
            return 8
        if isinstance(item, int):
            return 8
        if isinstance(item, bool):
            return 1
        if item is None:
            return 1
        raise NotImplementedError

    size = 16 + 32 + sum(len(name) + 1 for name in path.split('/'))
    for k, v in data.items():
        size += len(k) + 1
        if isinstance(v, list):
            size += sum(field_size(elem) for elem in v)
        else:
            size += field_size(v)
    return size / 1000


def calc_from_data(node: Node, data: dict, local: bool) -> Calc | None:
    if 'Lattice_type' not in data or 'atoms_num' not in data:
        node.log.info(event='dft_data_incomplete')
        return None

    kwargs = {}

    lattice_type = LatticeType[data['Lattice_type']]
    a, b, c = data['Lattice_abc']
    lattice_A = Lattice(lattice_type, a=a, b=b, c=c)

    if 'Lattice_type_start' in data:
        lattice_type_start = LatticeType[data['Lattice_type_start']]
        a_start, b_start, c_start = data['Lattice_abc_start']
        kwargs['lattice_A_start'] = Lattice(
            lattice_type_start, a=a_start, b=b_start, c=c_start)

    atom_args = zip(data['atoms_num'], data['atoms_xyz'][0::3], data['atoms_xyz'][1::3], data['atoms_xyz'][2::3],
                    data['atoms_mag'], data['atoms_ox'], data['atoms_frozen'])
    atoms_L = Atoms([Atom(num, (x, y, z), magnetic_moment=mag, oxidation_state=ox, frozen=frozen)
                     for num, x, y, z, mag, ox, frozen in atom_args])

    return Calc(lattice_A, atoms_L,
                status=Status[data['Status']],
                forces_max_L2_eVpA=data.get('Forces_max_L2_eVpA', None),
                min_data=ecat_types.minimization_point.min_points_from_data(
                    data),
                origin='local' if local else 'db',
                **kwargs)


def parse_parallelization(lines: list[str]) -> tuple[int, int, int]:
    pattern = r'Run totals: (\d+) processes, (\d+) threads, (\d+) GPUs'
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            processes = int(match.group(1))
            threads = int(match.group(2))
            gpus = int(match.group(3))
            return processes, threads, gpus

    raise ValueError('Input does not match expected format.')


def parse_progress(log: BoundLogger, out_lines: list[str]) -> dict:
    data = {}

    # Threads
    procs, threads, gpus = parse_parallelization(out_lines[:30])
    if procs != 1:
        data['Processes'] = 1
    data['Threads'] = threads
    if gpus != 0:
        data['GPUs'] = gpus

    # Energy components
    ecomp_dict_eV = jdftx.parse.parse_ecomp_eV(out_lines)
    if ecomp_dict_eV:
        data['Etot_eV'] = ecomp_dict_eV['Etot_eV']

    mu_eV = jdftx.parse.parse_mu_eV(out_lines)
    if mu_eV is not None:
        data['mu_eV'] = mu_eV

    # Atoms
    atoms_L = jdftx.parse.parse_atoms_L(out_lines)
    if atoms_L is not None:
        data |= atoms_L.to_firestore_dict()

    # Lattice, Forces, Bond length
    lattice_start = jdftx.parse.parse_outfile_lattice_A(
        log, out_lines, last=False)
    lattice = jdftx.parse.parse_outfile_lattice_A(log, out_lines, last=True)
    if lattice is not None:
        data |= lattice_start.to_firestore_dict(last=False)
        data |= lattice.to_firestore_dict(last=True)
        data |= jdftx.parse.parse_minimization(out_lines, lattice)
        # Bond length (if 2 atoms and lattice vectors are present)
        if atoms_L is not None and len(atoms_L) == 2:
            data['Bond_length_A'] = ecat_types.atom.bond_length_A(
                *atoms_L, lattice.R_A)

    # Stress
    stress_eVpA3 = jdftx.parse.parse_stress_eVpA3(out_lines)
    if stress_eVpA3 is not None:
        data['stress_eVpA3'] = list(np.array(stress_eVpA3).ravel())

    return data


async def scan_node(node: Node) -> None:
    try:
        await scan_folder(node, node.dft_out_folder, 2)
    except Exception:
        node.log.exception('Exception')
        raise


async def scan_folder(node: Node, folder: str, depth: int) -> None:

    sub_folders, files = await node.listdir(folder)
    if files:
        raise ValueError(f'Should not be files inside of a nested folder series: {folder} {files}')

    tasks = []
    for sub_folder in sub_folders:
        next_folder = os.path.join(folder, sub_folder)
        if depth > 0:
            await scan_folder(node, next_folder, depth-1)
        else:
            tasks.append(scan_adsorbate(node, next_folder))

    await asyncio.gather(*tasks)


async def scan_adsorbate(node: Node, ad_folder: str) -> None:
    tasks = []
    file_sets = await index_calc_files(node, ad_folder)
    for ext_map in file_sets.values():
        if '.out' in ext_map:
            material, geo_str, adsorbate = ext_map['.out'].folderpath.split(
                '/')[-3:]
            job_id = Jdftx.make_job_id_from_filename(
                material, geo_str, adsorbate, ext_map['.out'].basename)
            job_is_running = await node.job_is_running(job_id, use_cache=True)
            if not job_is_running:
                tasks.append(sync_calculation(
                    node, ext_map['.out'].base_filepath, job_is_running))

    await asyncio.gather(*tasks)


async def sync_calculation(node: Node, base_path: str, job_is_running: bool, *,
                           python_error: str | None = None) -> Calc | None:
    doc_ref = dft_doc(node.db, *base_path.split('/')[-4:])
    log = node.log.bind(path=doc_ref.path)

    # Download the database calculation if it exists
    doc = await doc_ref.get()
    db_data = doc.to_dict() if doc.exists else None

    # Read local data if it exists - local data is authoritative
    log, local_data = await read_local_data(node, base_path, job_is_running, doc_ref, log)
    if local_data and np.isfinite(local_data.get('Etot_eV', np.inf)):
        if db_data:
            log.debug(event=UpdateResult.OverwriteDB.value)
        else:
            log.info(event=UpdateResult.NewLocalCalc.value)
            await node.db.make_parents(doc_ref)

        if python_error is not None:
            local_data['Python_error'] = python_error

        await node.db.set(doc_ref, local_data)
        return calc_from_data(node, local_data, local=True)

    # Use database data if local data does not exist
    if db_data and np.isfinite(db_data.get('Etot_eV', np.inf)):
        return calc_from_data(node, db_data, local=False)


async def read_local_data(node: Node, base_filepath: str, job_is_running: bool,
                          doc_ref: AsyncDocumentReference, log: BoundLogger) -> tuple[BoundLogger, dict]:
    local_data = {}
    out_stat = await node.stat_file(base_filepath + '.out')
    if out_stat:
        out_lines = await node.readlines(out_stat.filepath)
        local_data |= compile_metadata(out_stat)
        status, status_error = await node.run_maybe_in_executor(jdftx.parse.parse_status, out_lines)
        if job_is_running:
            if status in [None, Status.Ionic_Minimize, Status.Lattice_Minimize] and status_error is None:
                status = Status.Running
            else:
                raise ValueError(f'{status.name}: {status_error}')
        elif status in [Status.Ionic_Minimize, Status.Lattice_Minimize, None]:
            status = Status.Stopped

        local_data['Status'] = status.name
        if status_error is not None:
            local_data['Status_error'] = status_error
        try:
            local_data |= await node.run_maybe_in_executor(parse_progress, log, out_lines)
        except ValueError:
            return log, {}
        # To ensure we count this field in the size
        local_data['Size_KB'] = 0.0
        local_data['Size_KB'] = calc_firestore_size_KB(
            doc_ref.path, local_data)
        log_keys = ['Last_modified_utc_ts', 'Forces_max_L2_eVpA',
                    'Etot_eV', 'Status', 'Runtime_s']
        log = log.bind(**{key: local_data[key]
                       for key in log_keys if key in local_data})
    return log, local_data


async def index_calc_files(node: Node, ad_folder: str) -> dict[str, dict[str, StatResult]]:
    folders, calc_files = await node.listdir(ad_folder)
    if folders:
        raise ValueError(f'Should not be folders inside of an adsorbate folder: {ad_folder} {folders}')

    file_sets = {}
    for calc_file in calc_files:
        for tail in ['_dry.out', '_start.xsf']:
            if calc_file.name.endswith(tail):
                basename = calc_file.name[:-len(tail)]
                break
        else:
            basename, tail = os.path.splitext(calc_file.name)

        if not basename in file_sets:
            file_sets[basename] = {}

        file_sets[basename][tail] = calc_file

    return file_sets


def compile_metadata(out_stat: StatResult) -> dict:
    material, geo_str, adsorbate = out_stat.folderpath.split('/')[-3:]
    geo, facet, layers, frozen_layers = Material.parse_geo_str(geo_str)
    kpoint, *fluid_smearing_mu_parts = out_stat.basename.split('_')

    # Elec smearing
    for part in fluid_smearing_mu_parts:
        if part.startswith('fs'):
            elec_smearing_eV = part
            break
    else:
        elec_smearing_eV = None

    # Fluid
    for part in fluid_smearing_mu_parts:
        if part in ['nlpcm', 'candle']:
            fluid = part
            break
    else:
        fluid = None

    # mu
    for part in fluid_smearing_mu_parts:
        if part.startswith('mu'):
            mu_V = int(part[2:]) / 1000
            break
    else:
        mu_V = None

    data = {
        'Adsorbate': adsorbate,
        'Geo': geo,
        'Kpoints': kpoint,
        'Last_modified_utc_ts': out_stat.last_modified,
        'Material': material,
    }

    if layers is not None:
        data['Layers'] = layers

    if frozen_layers is not None:
        data['Frozen_layers'] = frozen_layers

    if facet is not None:
        data['Facet'] = facet

    if fluid is not None:
        data['Fluid'] = fluid

    if elec_smearing_eV is not None:
        data['Elec_smearing_eV'] = float('0.' + elec_smearing_eV[2:])

    if mu_V is not None:
        data['mu_V'] = mu_V

    return data


async def query_minimization(db: AsyncBatchDatabase, geo: str) -> list[tuple[tuple[float, ...], float, float]]:
    query = db.client.collection_group('calcs').where(
        filter=FieldFilter('Geo', '==', geo))
    results = await query.get()
    docs_data = [doc.to_dict() for doc in results]
    filtered_data = [
        data for data in docs_data if 'min_forces_max_L2_eVpA' in data and 'one_hot_encoding' in data]
    xyts = []
    for data in filtered_data:
        min_points = ecat_types.minimization_point.min_points_from_data(data)
        for i in range(1, len(min_points)):
            last = min_points[i-1]
            current = min_points[i]
            full_encoding = tuple(
                data['one_hot_encoding'] + [current.forces_max_L2_eVpA])
            xyts.append((full_encoding, current.Etot_eV -
                        last.Etot_eV, current.time_s - last.time_s))
    return xyts


# async def query_converged(db: AsyncBatchDatabase) -> dict[str, list[dict]]:
#     query = db.client.collection_group('calcs').where(filter=FieldFilter('Status', '==', 'Ionic_Converged'))
#     results = await query.get()
#     docs_data = [doc.to_dict() for doc in results]
#     material_dict = {}
#     for data in docs_data:
#         if data.get('Forces_max_L2_eVpA', np.inf) < MAX_FORCE_eVpA:
#             if data['Material'] not in material_dict:
#                 material_dict[data['Material']] = []
#             material_dict[data['Material']].append(data)
#     return material_dict


async def make_fs_map(study_docs: list[DocumentSnapshot]) -> dict[str, FreeSpecies]:
    fs_map = {}
    for study_doc in study_docs:
        refs: list[AsyncDocumentReference] = study_doc.get(
            'free_species_dft_ref')
        names: list[str] = study_doc.get('free_species_name')
        phases: list[str] = study_doc.get('free_species_phase')
        for ref, name, phase in zip(refs, names, phases):
            fs_str = f'{name}_{phase}' if phase else name
            if fs_str in fs_map:
                continue

            Etot_eV = None
            bond_length_A = None
            if ref:
                fs_doc = await ref.get()
                Etot_eV = fs_doc.get('Etot_eV')
                forces_max_L2_eVpA = fs_doc.get('Forces_max_L2_eVpA')
                if forces_max_L2_eVpA >= MAX_FORCE_LATTICE_eVpA:
                    raise ValueError(f'Free species {fs_str} has forces_max_L2_eVpA of {forces_max_L2_eVpA}')
                try:
                    bond_length_A = fs_doc.get('Bond_length_A')
                except KeyError:
                    pass

            fs = FreeSpecies(name, phase, Etot_eV=Etot_eV,
                             bond_length_A=bond_length_A)
            if str(fs) != fs_str:
                raise ValueError(f'Free species name mismatch: {fs} vs {fs_str}')
            fs_map[fs_str] = fs
    return fs_map


@dataclass(frozen=True)
class StudyData:
    material: str
    adsorbates: dict[str, Step]
    lattice_abc: tuple[float, float, float]
    magnetic_moments: tuple[float, ...]
    oxidation_states: tuple[float, ...]


async def process_study_doc(node: Node, fs_map: dict[str, FreeSpecies], study_doc: DocumentSnapshot,
                            T_K: float, n_steps: int, Ecycle_eV: float) -> StudyData:
    # Fetch the base dft
    base_dft_ref: AsyncDocumentReference = study_doc.get('Base_dft_ref')
    base_dft_doc = await base_dft_ref.get()
    if not base_dft_doc.exists:
        return None

    # Fetch the base DFT lattice
    lattice_type: str = base_dft_doc.get('Lattice_type')
    lattice_abc: tuple[float, float | None, float |
                       None] = base_dft_doc.get('Lattice_abc')
    if lattice_type == 'FCC':
        lattice_abc = (lattice_abc[0], lattice_abc[0], lattice_abc[0])
    elif lattice_type != 'FCO':
        raise ValueError(f'Unsupported lattice type: {lattice_type}')

    # Fetch the magnetic moments and oxidation states
    atoms_mag = tuple(base_dft_doc.get('atoms_mag'))
    atoms_ox = tuple(base_dft_doc.get('atoms_ox'))

    # Fetch the free species
    step_refs: list[AsyncDocumentReference] = study_doc.get('step_dft_ref')
    fs_names: list[str] = study_doc.get('free_species_name')
    fs_phases: list[str] = study_doc.get('free_species_phase')
    steps_fs_counts = np.array(study_doc.get('step_free_species_counts')).reshape(len(step_refs), len(fs_names))
    step_indices = study_doc.get('step_index')
    ads_map = {}
    for step_ref, step_index, step_fs_counts in zip(step_refs, step_indices, steps_fs_counts):
        step_doc = await step_ref.get()
        if not step_doc.exists:
            continue

        # Exit early if the step is a duplicate of the start step (has the last step index) or the forces are high
        step_calc = calc_from_data(node, step_doc.to_dict(), local=False)
        if step_index == n_steps or not step_calc.is_converged:
            continue

        ad: str = step_doc.get('Adsorbate')
        if ad in ads_map:
            raise ValueError(f'Non-unique adsorbate name {ad}')

        Efree0V_eV: float = step_doc.get('Etot_eV') - Ecycle_eV * step_index / n_steps
        step_fs: list[tuple[FreeSpecies, int]] = []
        for fs_name, fs_phase, fs_count in zip(fs_names, fs_phases, step_fs_counts):
            if fs_count:
                fs_str = f'{fs_name}_{fs_phase}' if fs_phase else fs_name
                fs_Efree_eV = compute_or_fetch_fs_free_energy_eV(
                    fs_str, fs_map, T_K)
                Efree0V_eV += fs_count * fs_Efree_eV
                step_fs.append((fs_map[fs_str], fs_count))

        ads_map[ad] = Step(step_index, ad, step_fs, Efree0V_eV=Efree0V_eV)
    return StudyData(study_doc.get('Material'), ads_map, lattice_abc, atoms_mag, atoms_ox)


async def fetch_steps(node: Node, T_K: float, n_steps: int, *, use_pickle: bool = True) -> list[StudyData]:
    # If use_pickle is True, load data from pickle file if it exists
    pickle_file = os.path.join(MNT_PATH, 'pickles', 'training_data.pkl')
    if use_pickle and os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        return data

    study_docs: list[DocumentSnapshot] = await node.db.client.collection('studies').get()
    docs = [doc for doc in study_docs if '100o_5x3' in doc.id]
    fs_map = await make_fs_map(docs)
    Ecycle_eV = one_cycle_energy(fs_map, T_K)
    study_tasks = [process_study_doc(node, fs_map, doc,  T_K, n_steps, Ecycle_eV) for doc in docs]
    study_data = await asyncio.gather(*study_tasks)

    # Save the generated data to a pickle file
    full_data = [data for data in study_data if data is not None and len(data.adsorbates) > 0]
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(full_data, f)
    return full_data


def one_cycle_energy(fs_map: dict[str, FreeSpecies], T_K: float) -> float:
    energy_eV = 0
    for fs_str, count in [('O2_gas', -1), ('e-', -4), ('H2O_aq', -2), ('OH-_aq', 4)]:
        energy_eV += count * compute_or_fetch_fs_free_energy_eV(fs_str, fs_map, T_K)
    return energy_eV


def compute_or_fetch_fs_free_energy_eV(fs_str, fs_map: dict[str, FreeSpecies], T_K: float) -> float:
    if fs_str == 'e-':
        return 0

    fs = fs_map[fs_str]
    if fs.Efree_eV is not None:
        return fs.Efree_eV

    if fs_str == 'H+_aq':
        return compute_or_fetch_fs_free_energy_eV('H2_gas', fs_map, T_K) / 2
    elif fs_str == 'OH-_aq':
        h2O_Efree_eV = compute_or_fetch_fs_free_energy_eV(
            'H2O_aq', fs_map, T_K)
        proton_Efree_eV = compute_or_fetch_fs_free_energy_eV(
            'H+_aq', fs_map, T_K)
        return h2O_Efree_eV - proton_Efree_eV
    elif fs.molar_mass_gpMol is not None:
        if fs.Etot_eV is None:
            raise ValueError(f'Cannot compute free energy for {fs.name} because it has no Etot_eV')

        kT_eV = kB_eVpK * T_K
        molecular_mass_g = fs.molar_mass_gpMol / AVOGADRO
        if fs.is_diatomic:
            if fs.bond_length_A is None:
                raise ValueError(f'Cannot compute free energy for {fs.name} because it has no bond_length_A')

            sigma = 2 if fs.is_homodiatomic else 1
            I_gA2 = (molecular_mass_g * (fs.bond_length_A ** 2)) / 4
            B_eV = (ANGSTROM2_GRAM_PER_EV_SEC2 * (hbar_eVs ** 2)) / (2 * I_gA2)

            sum_terms = [(2 * j + 1) * math.exp(-(B_eV * (j * (j + 1))) / kT_eV)
                         for j in range(101)]
            z_rot = sum(sum_terms) / sigma
        elif fs.name == 'H2O':
            sigma = 2
            A_pm, B_pm, C_pm = 2787.61, 1450.74, 928.77  # m^-1 values for H2O
            z_rot = math.sqrt((math.pi * (kT_eV ** 3)) /
                              (A_pm * B_pm * C_pm * (hc_eVm ** 3))) / sigma
            z_rot /= 6  # Ad-hoc correction

        lambda_thermal_A = math.sqrt(
            h2_A2eVu / (2 * math.pi * fs.molar_mass_gpMol * kT_eV))
        number_density_pA3 = fs.number_density_pA3(T_K)
        z_trans = 1 / (number_density_pA3 * (lambda_thermal_A ** 3))
        E_trans_eV = -kT_eV * math.log(z_trans)
        E_rot_eV = -kT_eV * math.log(z_rot)

        fs_Efree_eV = fs.Etot_eV + E_trans_eV + E_rot_eV
        fs_map[fs_str] = FreeSpecies(fs.name, fs.phase, bond_length_A=fs.bond_length_A,
                                     Etot_eV=fs.Etot_eV, Efree_eV=fs_Efree_eV)
        return fs_Efree_eV

    else:
        raise NotImplementedError(
            f'Not Implemented: Cannot compute free energy for {fs.name}')
