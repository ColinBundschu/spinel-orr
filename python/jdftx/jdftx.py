
import dataclasses
import glob
import os
from dataclasses import dataclass

import numpy as np

import constants
import ecat_types.atom
import jdftx.launch_type
import jdftx.parse
from ecat_types import (Adsorbate, AdsorbateInterface, Calc, LatticeType,
                        Material)
from ecat_types.kpoints import KPoints
from jdftx.launch_type import LaunchType
import remote.node
from remote.node import Node

VAC_A_DEFAULT = 15


@dataclass(frozen=True, order=True)
class Jdftx():
    material: Material
    calc: Calc
    kpoints: KPoints
    _: dataclasses.KW_ONLY
    adsorbate: Adsorbate | None = None
    coulomb_truncation_embed: str | None = None
    elec_smearing_eV: float | None = None
    fluid: str | None = None
    fluid_anion: str | None = None
    fluid_cation: str | None = None
    elec_alphaTstart: float = 0.5
    elec_nIterations: int = 100
    ionic_nIterations: int = 300
    knormThreshold: float = 1e-4
    lattice_iter: int = 300
    mu_eV: float | None = None

    def __post_init__(self):
        if self.material.geo == 'bulk':
            if any(int(kpoint) < 4 for kpoint in self.kpoints.tuple):
                raise ValueError('Bulk calculations require at least 4 kpoints in each direction')
            if self.fluid:
                raise ValueError('Cannot have a fluid in a bulk calculation')

        if self.material.geo == 'molecule' and self.kpoints != KPoints(1, 1, 1):
            raise ValueError('Cannot specify kpoints for a molecule calculation')

    @staticmethod
    def make_base_filename(kpoints: KPoints, *, fluid: str | None = None, elec_smearing_eV: float | None = None, mu_eV: float | None = None) -> str:
        name = str(kpoints)
        if fluid is not None:
            name += f'_{fluid}'
        if elec_smearing_eV is not None:
            if elec_smearing_eV == 0.05:
                name += '_fs05'
            elif elec_smearing_eV == 0.025:
                name += '_fs025'
            elif elec_smearing_eV == 0.2:
                name += '_fs2'
            elif elec_smearing_eV == 0.25:
                name += '_fs25'
            elif elec_smearing_eV == 0.1:
                name += '_fs1'
            else:
                raise NotImplementedError(f'elec_smearing {elec_smearing_eV} not implemented')
        if mu_eV is not None:
            name += f'_mu{int(mu_eV * 1000)}'
        return name

    @staticmethod
    def make_folder(dft_out_folder: str | None, material: Material, *,
                    adsorbate: AdsorbateInterface | None = None) -> str:
        folders = [material.basename, material.geo_str, str(adsorbate or 'clean')]
        if dft_out_folder is not None:
            folders = [dft_out_folder] + folders
        return os.path.join(*folders)

    def folder(self, dft_out_folder: str | None) -> str:
        return Jdftx.make_folder(dft_out_folder, self.material, adsorbate=self.adsorbate)

    @staticmethod
    def make_job_id(material: Material, kpoints: KPoints, *, fluid: str | None = None,
                 elec_smearing_eV: float | None = None, adsorbate: AdsorbateInterface | None = None,
                 mu_eV: float | None = None) -> str:
        '''Bulk calculations do not specify fluid or adsorbate, so default to none'''
        filename = Jdftx.make_base_filename(kpoints, fluid=fluid, elec_smearing_eV=elec_smearing_eV, mu_eV=mu_eV)
        ad_str = str(adsorbate) if adsorbate else 'clean'
        return Jdftx.make_job_id_from_filename(material.basename, material.geo_str, ad_str, filename)

    @staticmethod
    def make_job_id_from_filename(material_name: str, geo_str: str, adsorbate: str, filename: str) -> str:
        '''Bulk calculations do not specify adsorbate, but when deserializing it will have the value of "clean"'''
        parts = [material_name, geo_str, filename]
        if geo_str != 'bulk':
            parts.append(adsorbate)
        elif adsorbate != 'clean':
            raise ValueError(f'Unexpected bulk adsorbate: {adsorbate} on {material_name}')

        return '__'.join(parts)

    @property
    def job_id(self) -> str:
        return Jdftx.make_job_id(self.material, self.kpoints, fluid=self.fluid,
                                 elec_smearing_eV=self.elec_smearing_eV, adsorbate=self.adsorbate, mu_eV=self.mu_eV)

    @property
    def filename(self) -> str:
        return Jdftx.make_base_filename(self.kpoints, fluid=self.fluid, elec_smearing_eV=self.elec_smearing_eV, mu_eV=self.mu_eV)

    def filepath(self, dft_out_folder: str | None) -> str:
        return os.path.join(self.folder(dft_out_folder), self.filename)

    @property
    def do_lattice_minimization(self) -> bool:
        return bool((self.material.geo == 'bulk' or self.material.do_lattice_min) and self.lattice_iter)

    def xsf_command(self, node: Node, launch_type: LaunchType) -> list[str]:
        filepath = self.filepath(node.dft_out_folder)
        out_path = f'{filepath}{launch_type.dry_str}.out'
        xsf_path = f'{filepath}_end.xsf'
        return [node.createXSF_path, out_path, xsf_path] + (['nbound'] if self.fluid else [])

    def format_magnetic_moments(self):
        magnetic_moments: list[str] = []
        for element in set(self.calc.atoms_L.symbols):
            element_indices = [i for i, atom in enumerate(self.calc.atoms_L) if atom.symbol == element]
            if any(np.array(self.calc.atoms_L.magnetic_moments)[element_indices]):
                magnetic_moments.append(element)
                for i in element_indices:
                    magnetic_moments.append(str(self.calc.atoms_L.magnetic_moments[i] or 0))
        return ' '.join(magnetic_moments)

    def format_oxidation_state(self):
        oxidation_states: list[str] = []
        for element in set(self.calc.atoms_L.symbols):
            element_indices = [i for i, atom in enumerate(self.calc.atoms_L) if atom.symbol == element]
            if any(np.array(self.calc.atoms_L.oxidation_states)[element_indices]):
                oxidation_states.append(element)
                for i in element_indices:
                    if self.calc.atoms_L.oxidation_states[i]:
                        oxidation_states.append(str(self.calc.atoms_L.oxidation_states[i]))
                        break
        return ' '.join(oxidation_states)

    async def write_start_xsf(self, node: Node) -> None:
        start_xsf_path = self.filepath(node.dft_out_folder) + '_start.xsf'
        if await node.exists(start_xsf_path):
            raise ValueError('Start xsf file already exists.')

        await node.writelines(start_xsf_path, self.calc.as_xsf_lines())

    async def write_in_file(self, node: Node):
        in_path = self.filepath(node.dft_out_folder) + '.in'
        if await node.exists(in_path):
            raise ValueError('In file already exists.')

        lines: list[str] = []
        lines.append(f'include {self.filename}.lattice')
        lines.append(f'include {self.filename}.ionpos')
        lines.append('ion-species GBRV/$ID_pbe.uspp')
        lines.append('elec-cutoff 20 100 #Recommended cutoffs for GBRV pseudopotentials')
        lines.append('spintype z-spin')

        if self.coulomb_truncation_embed:
            lines.append(f'coulomb-truncation-embed {self.coulomb_truncation_embed}')

        if any(self.calc.atoms_L.magnetic_moments):
            lines.append(f'initial-magnetic-moments {self.format_magnetic_moments()}')

        if any(self.calc.atoms_L.oxidation_states):
            lines.append(f'initial-oxidation-state {self.format_oxidation_state()}')

        if self.material.dft_U is not None:
            lines.append(f'add-U {self.material.dft_U}')

        # H2 is the only system where we want to disable the core overlap check
        if (len(self.calc.atoms_L) == 2
            and self.calc.atoms_L[0].symbol == 'H'
            and self.calc.atoms_L[1].symbol == 'H'
                and ecat_types.atom.bond_length_A(*self.calc.atoms_L, self.calc.lattice_A.R_A) < 1.0):
            lines.append('core-overlap-check None')

        # Dump variables
        lines.append('dump Ionic Stress')
        lines.append('dump End Stress' + ' BoundCharge' if self.fluid else '')

        # Electronic Minimization
        lines.append(f'electronic-minimize nIterations {self.elec_nIterations}'
                     f' alphaTstart {self.elec_alphaTstart}'
                     )

        # Fermi Smearing
        if self.elec_smearing_eV:
            lines.append(f'elec-smearing Fermi {self.elec_smearing_eV / constants.EV_PER_HARTREE}')

        # Target mu
        if self.mu_eV is not None:
            lines.append(f'target-mu {self.mu_eV / constants.EV_PER_HARTREE}')

        # Ionic Minimization
        if self.ionic_nIterations > 0 and not self.do_lattice_minimization:
            lines.append(
                f'ionic-minimize nIterations {self.ionic_nIterations}'
                f' knormThreshold {self.knormThreshold}'
                ' maxThreshold yes')
            # lines.append('lcao-params 50') # http://jdftx.org/OxideSurfaces.html

        # kpoints (for periodic structures)
        if self.material.geo == 'bulk' or self.material.geo == 'slab':
            lines.append(f'kpoint 0.5 0.5 {0.5 if self.material.geo == "bulk" else 0} 1')
            lines.append(f'kpoint-folding {self.kpoints.x} {self.kpoints.y} {self.kpoints.z}')

        # Fluid Minimization
        if self.fluid:
            if self.fluid == 'nlpcm':
                fluid = 'NonlinearPCM'
            elif self.fluid == 'candle':
                fluid = 'LinearPCM'
            else:
                raise NotImplementedError(f'Fluid {self.fluid} not implemented')
            lines.append(f'fluid {fluid}')
            lines.append('fluid-solvent H2O')

            if self.fluid == 'candle':
                lines.append('pcm-variant CANDLE')
            elif self.fluid == 'nlpcm':
                lines.append('pcm-variant GLSSA13')

            if self.fluid_cation and self.fluid_anion:
                lines.append(f'fluid-cation {self.fluid_cation} 1.')
                lines.append(f'fluid-anion {self.fluid_anion} 1.')

        # Lattice Minimization
        if self.do_lattice_minimization:
            lines.append(
                f'lattice-minimize nIterations {self.lattice_iter}'
                f' maxThreshold yes')
            lines.append('lcao-params 50') # http://jdftx.org/OxideSurfaces.html

        if self.material.geo == 'molecule':
            lines.append('coulomb-interaction Isolated')
            lines.append('coulomb-truncation-embed 0.5 0.5 0.5')

        await node.writelines(in_path, lines)

    async def write_ionpos_file(self, node: Node, launch_type: LaunchType) -> None:
        ionpos_path = self.filepath(node.dft_out_folder) + '.ionpos'
        if await node.exists(ionpos_path):
            if launch_type.dry:
                return
            raise ValueError('Ionpos file already exists.')

        lines: list[str] = []
        for atom in self.calc.atoms_L:
            lines.append(f'ion {atom.symbol:5} {atom.x:20.16} {atom.y:20.16} {atom.z:20.16} {0 if atom.frozen else 1}')
        await node.writelines(ionpos_path, lines)

    async def write_lattice_file(self, node: Node, launch_type: LaunchType) -> None:
        lattice_path = self.filepath(node.dft_out_folder) + '.lattice'
        if await node.exists(lattice_path):
            if launch_type.dry:
                return
            raise ValueError('Lattice file already exists.')

        lines: list[str] = []
        match self.calc.lattice_A.lattice_type:
            case LatticeType.Cubic | LatticeType.FCC:
                a_B = constants.BOHR_PER_ANGSTROM * self.calc.lattice_A.a
                lines.append(f'lattice {self.calc.lattice_A.lattice_type.value} {a_B}')
            case LatticeType.Orthorhombic | LatticeType.FCO:
                a_B = constants.BOHR_PER_ANGSTROM * self.calc.lattice_A.a
                b_B = constants.BOHR_PER_ANGSTROM * self.calc.lattice_A.b
                c_B = constants.BOHR_PER_ANGSTROM * self.calc.lattice_A.c
                lines.append(f'lattice {self.calc.lattice_A.lattice_type.value} {a_B} {b_B} {c_B}')
            case _:
                raise NotImplementedError(f'Lattice type {self.calc.lattice_A.lattice_type} not implemented')

        await node.writelines(lattice_path, lines)

    async def launch(self, node: Node, launch_type: LaunchType, cores: int, N_nodes: int,
                     node_start_index: int | None) -> None | list[str]:
        folder = self.folder(node.dft_out_folder)
        out_path = self.filepath(node.dft_out_folder) + launch_type.dry_str + '.out'
        log = node.log.bind(path=out_path)

        # Prep for launch
        await node.mkdir(folder)
        await self.delete_old_files(node, launch_type)
        await self.write_start_xsf(node)
        await self.write_in_file(node)
        await self.write_ionpos_file(node, launch_type)
        await self.write_lattice_file(node, launch_type)

        jdftx_cmd = [node.jdftx_path(launch_type.gpu), '-i', f'{self.filename}.in', '-o', os.path.basename(out_path)]
        if cores:
            jdftx_cmd += ['-c', str(cores)]

        if launch_type in [jdftx.launch_type.SLURM_WRAPPED, jdftx.launch_type.SLURM_WRAPPED_GPU]:
            await self.write_slurm_script(node, launch_type, cores)
            log.info('|--> Wrapped SBATCH')
            sbatch_cmd = ['sbatch', f'{self.filepath(node.dft_out_folder)}.sh']
            await node.run(sbatch_cmd, cwd=folder)

        elif launch_type in [jdftx.launch_type.DRY_LOCAL, jdftx.launch_type.DRY_LOCAL_GPU]:
            log.debug('|--> Dry Run')
            await node.run(jdftx_cmd + ['-n'], cwd=folder)
            out_lines = await node.readlines(out_path)
            log.info(event=('Dry Run ✅' if jdftx.parse.dry_run_succeeded(out_lines) else 'Dry Run 💀'))

        elif launch_type in [jdftx.launch_type.LOCAL, jdftx.launch_type.LOCAL_GPU]:
            log.info('|--> Local')
            await node.run(jdftx_cmd, cwd=folder)
            await node.run(self.xsf_command(node, launch_type), cwd=folder)

        elif launch_type in [jdftx.launch_type.LOCAL_WRAPPED, jdftx.launch_type.LOCAL_WRAPPED_GPU]:
            log.info('|--> Wrapped Local')
            stdout_path = self.filepath(node.dft_out_folder) + launch_type.dry_str + '.stdout'
            cmd = self.wrapped_cmd(node, launch_type, cores)
            await node.exec_nohup_command(cmd, cwd=folder, stdout_path=stdout_path)

        elif launch_type in [jdftx.launch_type.PBS_WRAPPED_GPU]:
            await self.write_pbs_wrapped_gpu_script(node, N_nodes)
            log.info('|--> Wrapped PBS GPU')
            pbs_cmd = ['qsub', f'{self.filepath(node.dft_out_folder)}.sh']
            await node.run(pbs_cmd, cwd=folder)

        elif launch_type in [jdftx.launch_type.PBS_PARALLEL_WRAPPED_GPU]:
            log.debug('|--> Batch Wrapped PBS GPU (Run Parallel)')
            if node_start_index is None:
                raise ValueError('node_start_index must be provided for batch jobs')
            return self.get_pbs_batch_gpu_lines(launch_type, node, node_start_index, N_nodes, run_parallel=True)

        elif launch_type in [jdftx.launch_type.PBS_SERIAL_WRAPPED_GPU]:
            log.debug('|--> Batch Wrapped PBS GPU (Run Serial)')
            if node_start_index is None:
                raise ValueError('node_start_index must be provided for batch jobs')
            return self.get_pbs_batch_gpu_lines(launch_type, node, node_start_index, N_nodes, run_parallel=False)

        else:
            raise NotImplementedError(f'launch_type {launch_type} not implemented')

    async def delete_old_files(self, node: Node, launch_type: LaunchType) -> None:
        '''Cleanup old files so as to not mix them up with the new results
           We do not delete old files needed to start from a previous calculation'''
        extensions = ['_start.xsf', '.in', '_dry.out', '.ionpos', '.lattice', '.stress', '-output.log', '-error.log']
        if not launch_type.dry:
            extensions += ['.sh', '_end.xsf', '.out', '.wfns', '.fluidState', '.nbound']

        filepaths = [self.filepath(node.dft_out_folder) + ext for ext in extensions]
        slurm_wildcard = self.filepath(node.dft_out_folder) + '_slurm*'
        filepaths += glob.glob(slurm_wildcard) if node.is_local else [slurm_wildcard]
        for filename in [f'hostfile_{self.filename}', 'jdftx-stacktrace']:
            filepaths.append(os.path.join(self.folder(node.dft_out_folder), filename))
        await node.delete_files(filepaths, missing_ok=True)

    async def write_pbs_wrapped_gpu_script(self, node: Node, N_nodes: int) -> None:
        hostfilename = f'hostfile_{self.filename}'
        lines = [
            '#!/bin/bash -l',
            f'#PBS -N {remote.node.shorten_job_id(self.job_id)}',
            f'#PBS -l select={N_nodes}',
            '#PBS -l walltime=72:00:00',
            '#PBS -q preemptable',
            # '#PBS -r y', # This restarts preempt killed jobs, but wrapped jobs don't clear the output file and die
            '#PBS -l filesystems=home',
            '#PBS -A von',
            f'#PBS -o {self.filepath(node.dft_out_folder)}-output.log',
            f'#PBS -e {self.filepath(node.dft_out_folder)}-error.log',
            '',
            'NODES=(`cat $PBS_NODEFILE`) # Extract the hostnames of the nodes',
            'export http_proxy="http://proxy.alcf.anl.gov:3128"',
            'export https_proxy="http://proxy.alcf.anl.gov:3128"',
            'export ftp_proxy="http://proxy.alcf.anl.gov:3128"',
            'export MPICH_GPU_SUPPORT_ENABLED=1',
            'module load craype-accel-nvidia80',
            '',
            f'cd {self.folder(node.dft_out_folder)}',
            'echo ${NODES[0]} > ' + hostfilename,
        ]
        for i in range(1, N_nodes):
            lines += ['echo ${NODES['f'{i}'']} >> ' + hostfilename]

        lines.append('export JDFTX_MEMPOOL_SIZE=38000')
        lines.append(' '.join(self.wrapped_cmd(node, jdftx.launch_type.PBS_WRAPPED_GPU, cores=None, N_nodes=N_nodes)))
        # lines.append(' '.join(self.xsf_command(node, jdftx.launch_type.PBS_WRAPPED_GPU)))
        await node.writelines(self.filepath(node.dft_out_folder) + '.sh', lines)

    def get_pbs_batch_gpu_lines(self, launch_type: LaunchType, node: Node, node_start_index: int, N_nodes: int,
                                        run_parallel: bool) -> list[str]:
        hostfilename = f'hostfile_{self.filename}'
        lines = [
            f'cd {self.folder(node.dft_out_folder)}',
            'echo ${NODES['f'{node_start_index}'']} > ' + hostfilename,
        ]
        for i in range(1, N_nodes):
            lines += ['echo ${NODES['f'{node_start_index + i}'']} >> ' + hostfilename]

        cmd_parts = self.wrapped_cmd(node, launch_type, cores=None, N_nodes=N_nodes)
        if run_parallel:
            cmd_parts += ['&']
        lines.append(' '.join(cmd_parts))
        lines.append('sleep 1s')
        return lines


    async def write_slurm_script(self, node: Node, launch_type: LaunchType, cores: int) -> None:
        lines = [
            '#!/bin/bash',
            '#SBATCH --nodes=1',
            f'#SBATCH --job-name={remote.node.shorten_job_id(self.job_id)}',
            f'#SBATCH --output={self.filename}_slurm%j.log'
        ]

        if launch_type.gpu:
            match cores:
                case 15:
                    lines += [
                        f'#SBATCH --mem={85*node.gpu_count}G',
                        '#SBATCH --nodes=1',
                        f'#SBATCH --ntasks-per-node={node.gpu_count}',
                        '#SBATCH --qos=standby',
                        '#SBATCH --time=12:00:00',
                        '#SBATCH --account=vonset',
                        f'#SBATCH --gpus={node.gpu_count}',
                        'module load jdftx', # Sets the mempool to 70000
                    ]
                case 16:
                    lines += [
                        f'#SBATCH --mem={110*node.gpu_count}g',
                        '#SBATCH --nodes=1',
                        f'#SBATCH --cpus-per-task={cores}',
                        f'#SBATCH --ntasks-per-node={node.gpu_count}',
                        '#SBATCH --partition=ghx4',
                        '#SBATCH --time=12:00:00',
                        '#SBATCH --account=bdtx-dtai-gh',
                        f'#SBATCH --gpus-per-node={node.gpu_count}',
                        'export JDFTX_MEMPOOL_SIZE=90000',
                    ]
                case 32:
                    lines += [
                        f'#SBATCH --gres=gpu:{node.gpu_count}',
                        '#SBATCH --ntasks=1',
                        f'#SBATCH --cpus-per-task={cores}',
                        '#SBATCH --partition=gpu-xl',
                        '#SBATCH --time=48:00:00',
                        'export JDFTX_MEMPOOL_SIZE=38000',
                    ]
                case _:
                    raise ValueError(f'Unexpected number of cores per job: {cores}')
        else:
            lines += [
                '#SBATCH --partition=dft',
                '#SBATCH --time=14-00:00:00',
                '#SBATCH --ntasks=1',
                f'#SBATCH --cpus-per-task={cores}',
                f'#SBATCH --mem={1750 * cores}',
            ]

        if launch_type.wrapped:
            if launch_type.dry:
                raise NotImplementedError('Cannot wrap a dry run yet with slurm')
            lines.append(' '.join(self.wrapped_cmd(node, launch_type, cores)))
        else:
            lines.append(f'{node.jdftx_path(launch_type.gpu)} -i {self.filename}.in'
                         f' -o {self.filename}{launch_type.dry_str}.out' + (' -n' if launch_type.dry else ''))

        # lines.append(' '.join(self.xsf_command(node, launch_type)))

        await node.writelines(self.filepath(node.dft_out_folder) + '.sh', lines)

    def wrapped_cmd(self, node: Node, launch_type: LaunchType, cores: int | None, N_nodes: int = 1) -> list[str]:
        base_filepath = self.filepath(node.dft_out_folder)
        cmd = [node.python_path, node.wrapper_script_path, base_filepath]
        cmd += ['--N_nodes', str(N_nodes)]
        if launch_type.gpu:
            cmd += ['-g', str(node.gpu_count)]
        if launch_type.dry:
            cmd += ['-n']
        if cores:
            cmd += ['-c', str(cores)]
        cmd += ['-j', node.job_type]
        return cmd
