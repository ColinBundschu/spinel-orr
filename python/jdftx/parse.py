
import dataclasses
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
from structlog import BoundLogger

import constants
import ecat_types.minimization_point
from ecat_types import (Atom, Atoms, Lattice, LatticeType, MinimizationPoint,
                        Status)


def lines_after_last_instance(lines: Iterable[str], line_start: str) -> list[str] | None:
    '''
    Return the lines after the last instance of a line that starts with line_start.
    Returns an empty list if line_start matches the last line of the file.
    This means the result is not properly "truthy" as to whether any match was found.
    '''
    for i, line in reversed(list(enumerate(lines))):
        if line.startswith(line_start):
            return lines[i+1:]

    return None


def lines_at_and_after_first_instance(lines: Iterable[str], line_start: str) -> list[str] | None:
    '''
    Return the lines at and after the first instance of a line that starts with line_start.
    Returns an empty list if line_start matches the last line of the file.
    '''
    for i, line in enumerate(lines):
        if line.startswith(line_start):
            return lines[i:]

    return None


def dry_run_succeeded(out_lines: Iterable[str]) -> bool:
    line_start = 'Dry run successful: commands are valid and initialization succeeded.'
    lines = lines_after_last_instance(out_lines, line_start)
    return lines is not None


def parse_ionpos_L(out_lines: Iterable[str]) -> Atoms | None:
    '''lines are from either the .out or .ionpos file'''
    line_start = '# Ionic positions in lattice coordinates:'
    lines = lines_after_last_instance(out_lines, line_start)
    if lines is None:
        lines = lines_at_and_after_first_instance(out_lines, 'ion ')
        if lines is None:
            return None

    atoms = []
    for line in lines:
        if not line.startswith('ion '):
            break
        _, atomic_symbol, x, y, z, can_move = line.split()
        frozen = not bool(int(can_move))
        atoms.append(Atom(constants.ATOMIC_NUMS[atomic_symbol], (float(x), float(y), float(z)), frozen=frozen))
    return Atoms(atoms)


def parse_ecomp_eV(lines: Iterable[str]) -> dict[str, float] | None:
    '''lines are from either the .out or .Ecomponents file'''
    line_start = '# Energy components:'
    lines = lines_after_last_instance(lines, line_start)
    if lines is None:
        return None

    energies_eV = {}
    for line in lines:
        # Delimiter
        if line.startswith('-------------------------------------'):
            continue

        parsed_name, equal_sign, energy_H = line.split()
        if equal_sign != '=':
            raise ValueError('Unable to parse Ecomponents')
        name_with_units = parsed_name + '_eV'
        energies_eV[name_with_units] = float(energy_H) * constants.EV_PER_HARTREE
        if name_with_units == 'Etot_eV':
            return energies_eV
    raise ValueError('Never found Etot, despite having "# Energy components:"')


def parse_mu_eV(lines: list[str]) -> float | None:
    line_start = 'FillingsUpdate:  mu:'
    for line in lines[::-1]:
        if line.lstrip().startswith(line_start):
            mu_eV = float(line.split()[2]) * constants.EV_PER_HARTREE
            return mu_eV
    return None


def parse_outfile_lattice_A(log: BoundLogger, out_lines: Iterable[str], last: bool) -> Lattice | None:
    ''' Reciprocal lattice vectors in Angstroms from the .out file from lattice minimization'''

    # Determine the specified lattice symmetry
    for line in out_lines:
        if line.startswith('lattice Cubic'):
            lattice_type = LatticeType.Cubic
            break
        if line.startswith('lattice Orthorhombic'):
            lattice_type = LatticeType.Orthorhombic
            break
        if line.startswith('lattice Face-Centered Cubic'):
            lattice_type = LatticeType.FCC
            break
        if line.startswith('lattice Face-Centered Orthorhombic'):
            lattice_type = LatticeType.FCO
            break
    else:
        return None

    # Parse the lattice vectors
    if last:
        lines = lines_after_last_instance(out_lines, '# Lattice vectors:')
    else:
        for i, line in enumerate(out_lines):
            if line.startswith('# Lattice vectors:'):
                lines = out_lines[i+1:]
                break
        else:
            lines = None

    if not lines:
        lines = lines_after_last_instance(out_lines, '---------- Initializing the Grid ----------')
        if not lines:
            return None

    R_A = np.array([[float(x)*constants.ANGSTROM_PER_BOHR for x in line.split()[1:4]] for line in lines[1:4]])
    lattice = lattice_from_parsed_R_A(R_A, lattice_type)
    return lattice


def lattice_from_parsed_R_A(R_A: np.ndarray, lattice_type: LatticeType, *, abs_tol_A: float = 0.2) -> Lattice:
    if R_A.shape != (3, 3):
        raise ValueError(f'Expected 3x3 array for R_A, got {R_A.shape}')

    # Construct a lattice for Cubic and Orthorhombic
    if lattice_type in [LatticeType.Cubic, LatticeType.Orthorhombic]:
        if any(abs(R_A[i][j]) > abs_tol_A for i in range(3) for j in range(3) if i != j):
            raise ValueError(f'Non diagonal element of lattice vector is too far above zero cutoff: {R_A}')

        a, b, c = np.diag(R_A)
        if lattice_type == LatticeType.Cubic:
            if abs(a, b) > abs_tol_A or abs(a, c) > abs_tol_A:
                raise ValueError(f'Cubic should have lattice constants of the same length: {R_A}')
            return Lattice(LatticeType.Cubic, a=a)
        return Lattice(LatticeType.Orthorhombic, a=a, b=b, c=c)

    # Construct a lattice for FCC and FCO
    if lattice_type in [LatticeType.FCC, LatticeType.FCO]:
        for i in range(3):
            if abs(R_A[i][i]) > abs_tol_A:
                raise ValueError(f'Element {i},{i} of lattice vectors is too far above zero cutoff for FC: {R_A[i][i]}')
            x0, x1 = [R_A[i][j] for j in range(3) if j != i]
            if abs(x0 - x1) > abs_tol_A:
                raise ValueError(f'Non-zero elements of lattice vector row are not equal for FC: {R_A[i]}')
            if x0 < 0 or x1 < 0:
                raise ValueError(f'Negative elements of lattice vector row for FC: {R_A[i]}')

        a, b, c = [np.sqrt(2 * sum(R_A[i]**2)) for i in range(3)]

        if abs(a - b) < abs_tol_A and abs(a - c) < abs_tol_A and abs(b - c) < abs_tol_A:
            return Lattice(LatticeType.FCC, a=(a + b + c)/3)
        return Lattice(LatticeType.FCO, a=a, b=b, c=c)

    raise NotImplementedError(f'Lattice type {lattice_type} not implemented')


def parse_status(out_lines: tuple[str]) -> tuple[Status | None, str | None]:
    '''Check for an end status'''
    for line in out_lines:
        if line.startswith('IonicMinimize:  Step failed: resetting history.'):
            return Status.Ionic_Step_Failure, None

    i = len(out_lines)
    status, error = None, None
    while i > 0 and status is None:
        i -= 1
        line = out_lines[i]
        if line.startswith("Dumping '") and ".wfns' ..." in line and 'done' not in line:
            error = line
            status = Status.Wfns_Save_Failure
        if line.startswith('Failed.'):
            error = '\n'.join(out_lines[i-5:i+1])
            if out_lines[i-3].startswith('Hint: Did you specify the correct nBandsOld, EcutOld and kdepOld?'):
                status = Status.Wfns_Mismatch
            elif 'Insufficient atomic orbitals' in out_lines[i-2]:
                status = Status.Insuf_Atomic_Orbitals
            else:
                status = Status.Failed

        elif line.startswith('Dry run successful: commands are valid and initialization succeeded.'):
            status = Status.Dry_Run_Succeeded

        elif line.startswith('Input parsing failed with'):
            error = '\n'.join(out_lines)
            status = Status.Input_Parsing_Failed

        elif line.startswith('LatticeMinimize: Converged'):
            status = Status.Lattice_Converged

        elif line.startswith('LatticeMinimize: None of the convergence criteria satisfied after'):
            status = Status.Lattice_Not_Converged

        elif line.startswith('IonicMinimize: Converged'):
            status = Status.Ionic_Converged

        elif line.startswith('IonicMinimize: None of the convergence criteria satisfied after'):
            status = Status.Ionic_Not_Converged

        elif line.startswith('IonicMinimize: Probably at roundoff error limit. (Stopping)'):
            status = Status.Ionic_Step_Failure

        elif line.startswith('IonicMinimize: Iter:'):
            if 'nan' in line:
                error = line
                status = Status.NaN_Failure
            else:
                status = Status.Ionic_Minimize

        elif line.startswith('LatticeMinimize: Iter:'):
            status = Status.Lattice_Minimize

        # These lines will match as errors spuriously, so we want to catch them before the error check
        elif 'Probably at roundoff' in line or 'Verifying exactness' in line:
            continue

        elif 'error' in line.lower() or 'exception' in line.lower():
            error = out_lines[i].rstrip()
            status = Status.Error

    return status, error


def parse_min_step(line: str, next_line: str, forces: dict | None, unit_cell_volume_A3: float | None, dft_ts: int,
                   last_time_s: float) -> MinimizationPoint | None:
    '''If this line is a minimization step, returns the minimization type and iteration.
    Otherwise, returns None'''

    def helper(iter_name: str, min_type: str) -> MinimizationPoint:
        words = line.split()
        iteration = int(words[words.index(iter_name)+1])
        energy_index = words.index('Etot:') if 'Etot:' in words else words.index('F:')
        Etot_eV = float(words[energy_index+1]) * constants.EV_PER_HARTREE

        # Not all minimization steps add a timestamp
        time_s = float(words[words.index('t[s]:')+1]) if 't[s]:' in words else last_time_s
        if time_s < last_time_s:
            raise ValueError(f'New time {time_s} is less than previous time {last_time_s}')

        if min_type in ['LatticeMinimize', 'IonicMinimize']:
            if forces is None:
                return None
            return MinimizationPoint(Etot_eV, time_s, iteration, min_type, forces['Forces_sum_L2_eVpA'],
                                     forces['Forces_max_L2_eVpA'], unit_cell_volume_A3, dft_ts)
        if forces:
            raise ValueError(f'Forces present for non-lattice/ionic step: {min_type}')

        return MinimizationPoint(Etot_eV, time_s, iteration, min_type, None, None, None, dft_ts)

    if line.lstrip().startswith('SCF: Cycle:'):
        return helper('Cycle:', 'scf')

    for min_type in ['LatticeMinimize', 'IonicMinimize', 'ElecMinimize', 'FluidMinimize']:
        if line.lstrip().startswith(f'{min_type}: Iter:'):
            if min_type in ['ElecMinimize', 'FluidMinimize']:
                if next_line.lstrip().startswith(f'{min_type}: None of the convergence criteria'):
                    min_type = min_type.lower()
                elif not next_line.lstrip().startswith(f'{min_type}: Converged'):
                    return None
            return helper('Iter:', min_type)

    return None


def parse_timestamp(date_str, tz_name='Etc/UTC'):
    date_format = "Start date and time: %a %b %d %H:%M:%S %Y"
    dt_naive = datetime.strptime(date_str.strip(), date_format)
    dt_aware = dt_naive.replace(tzinfo=zoneinfo.ZoneInfo(tz_name))
    timestamp = dt_aware.timestamp()
    return int(timestamp)


def parse_minimization(out_lines: Iterable[str], lattice: Lattice) -> dict[str, list]:
    '''All fields of min_data are lazily initialized when the first value is added'''

    n_jdftx_runs = np.count_nonzero(['*************** JDFTx ' in line for line in out_lines])
    if n_jdftx_runs == 0 or len(out_lines) < 4:
        return {}

    if n_jdftx_runs > 1:
        raise ValueError(f'{n_jdftx_runs} JDFTx runs found in output file, expected no more than 1.')

    dft_ts = parse_timestamp(out_lines[3])

    min_points: list[MinimizationPoint] = []
    lationic_min_points: list[MinimizationPoint] = []
    forces = None
    unit_cell_volume_A3 = None
    data = {'Dft_utc_timestamp_s': dft_ts}
    last_time_s = 0
    for i in range(len(out_lines) - 1):
        if out_lines[i].startswith('# Forces in Lattice coordinates:'):
            if forces:
                raise ValueError('Forces already present when parsing new forces')
            forces = parse_and_process_forces(out_lines[i+1:], lattice)
            continue

        if out_lines[i].startswith('unit cell volume ='):
            unit_cell_volume_A3 = float(out_lines[i].split()[-1])*constants.ANGSTROM_PER_BOHR**3
            continue

        # This will error if the forces are not present for lattice or ionic steps
        parsed_min_point = parse_min_step(out_lines[i], out_lines[i+1], forces,
                                          unit_cell_volume_A3, dft_ts, last_time_s)
        if parsed_min_point is not None:
            if parsed_min_point.type in ['LatticeMinimize', 'IonicMinimize']:
                lationic_min_points.append(parsed_min_point)
                data |= forces  # Use the last known good forces for the overall data
                forces = None  # We consume the forces when we use them
                unit_cell_volume_A3 = None  # We consume the volume when we use it
            last_time_s = parsed_min_point.time_s
            min_points.append(parsed_min_point)
    data |= ecat_types.minimization_point.serialize_min_points(min_points)
    return data


def parse_and_process_forces(lines: list[str], lattice: Lattice) -> dict:
    force_vecs_eVpL = []
    for line in lines:
        # Forces are in Hartree per Lattice coordinate
        if line.startswith('force'):
            _, _, x, y, z, can_move = line.split()
            if int(can_move):
                force_vecs_eVpL.append([constants.EV_PER_HARTREE*float(coord) for coord in (x, y, z)])
            else:
                force_vecs_eVpL.append([0, 0, 0])
        else:
            break  # We've reached the end of the force lines

    force_vecs_eVpA = np.array(force_vecs_eVpL) @ np.linalg.inv(lattice.R_A).T
    force_norms_eVpA = np.linalg.norm(force_vecs_eVpA, axis=1)
    return {
        'forces_eVpA': list(force_vecs_eVpA.flat),
        'Forces_sum_L2_eVpA': np.sum(force_norms_eVpA),
        'Forces_max_L2_eVpA': np.max(force_norms_eVpA),
    }


def parse_stress_eVpA3(lines: Iterable[str]) -> list[list[float]] | None:
    '''
    Positive stress means the forces are compressive: the lattice wants to contract
    lines are from either the .out or .stress file
    '''
    line_start = '# Stress tensor'
    lines = lines_after_last_instance(lines, line_start)
    if lines is None:
        return None

    eVpA3_per_HpB3 = constants.EV_PER_HARTREE/constants.ANGSTROM_PER_BOHR**3
    return [[float(x)*eVpA3_per_HpB3 for x in line.split()[1:4]] for line in lines[:3]]


@dataclass(frozen=True, order=True)
class OxMag():
    oxidation_state: float
    magnetic_moment: float


def parse_oxidation_magnetic_moments(out_lines: Iterable[str]) -> dict[str, list[OxMag]] | None:
    '''We will always have oxidation states, but not always magnetic moments'''
    line_start = '#--- Lowdin population analysis ---'
    lines = lines_after_last_instance(out_lines, line_start)
    if lines is None:
        return None

    ox_mag_states = {}
    i = 0
    while i < len(lines) and lines[i].startswith('# oxidation-state'):
        oxidation_parts = lines[i].split()
        atomic_symbol = oxidation_parts[2]
        ox_values = [float(x) for x in oxidation_parts[3:]]
        i += 1

        if lines[i].startswith('# magnetic-moments'):
            magnetic_parts = lines[i].split()
            if atomic_symbol != magnetic_parts[2]:
                raise ValueError('Atomic symbol mismatch in oxidation state and magnetic moments:\n'
                                 f'Oxidation state: {oxidation_parts}\n'f'Magnetic moments: {magnetic_parts}')

            mag_values = [float(x) for x in magnetic_parts[3:]]
            i += 1
        else:
            mag_values = [None]*len(ox_values)
        ox_mag_states[atomic_symbol] = [OxMag(ox, mag) for ox, mag in zip(ox_values, mag_values)]

    return ox_mag_states


def parse_atoms_L(out_lines: Iterable[str]) -> Atoms | None:
    atoms = parse_ionpos_L(out_lines)
    if atoms is None:
        return None

    ox_mag_states = parse_oxidation_magnetic_moments(out_lines)
    if ox_mag_states is None:
        return atoms

    new_atoms = []
    for atom in atoms:
        ox_mag = ox_mag_states[atom.symbol].pop(0)
        new_atom = dataclasses.replace(atom, oxidation_state=ox_mag.oxidation_state,
                                       magnetic_moment=ox_mag.magnetic_moment)
        new_atoms.append(new_atom)

    if any(ox_mag_states.values()):
        raise ValueError('Not all oxidation states / magnetic moments were consumed')

    return Atoms(new_atoms)
