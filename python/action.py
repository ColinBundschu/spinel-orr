
# pylint: disable=unused-variable,unused-import,multiple-statements,line-too-long

import asyncio
import itertools
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum

import ecat_types.reaction
import jdftx.launch_type
import log_format
import remote.database
import remote.jobs
from studies.Pd_Pt_core_shell.PdPt import Pd4Pt
from studies.Pd_Pt_core_shell.Pt import Pt
import studies.study_factory
from constants import BATCH_SCRIPTS_PATH, PBS_USER
from ecat_types import Reaction, Spectator
from jdftx.launch_type import LaunchType
from remote.async_batch_database import AsyncBatchDatabase
from remote.node import Node, RsyncType
from studies import MN4C, BiMN4C, Filter, Spinel, Study
from studies.Ni_graphene.Ni import Ni


class ActionType(Enum):
    RSYNC_SMALL = 'Rsync small files from node to the local machine'
    WIPE_DB_GROUP = 'Delete all materials from the database'
    UPDATE_FIELD = 'Update a field in the database'
    RUN_ML = 'Run a run machine learning training'
    SCAN_NODE = 'Scan the node to update all calcs'
    ECHEM = 'Run the electrochemistry visualization'
    LAUNCH_BATCH_PBS_WRAPPED_GPU = 'Launch a batch of jobs on pbs, using gpus that are wrapped to upload'


class StudyType(Enum):
    MNC = 'MNC'
    NiC = 'Ni on graphene'
    PdPt = 'PdPt'
    Pt = 'Pt'
    SPINEL_100s = 'Spinel 100s'
    SPINEL_100o = 'Spinel 100o'
    SPINEL_111s = 'Spinel 111s'
    SPINEL_111b = 'Spinel 111b'


@asynccontextmanager
async def conditional_db(use_db, log):
    if use_db:
        async with AsyncBatchDatabase(log) as cm:
            yield cm
    else:
        yield None


@contextmanager
def conditional_executor(use_executor):
    if use_executor:
        with ProcessPoolExecutor() as cm:
            yield cm
    else:
        yield None


#     0     1     2     3     4     5     6     7
X = ('Zn', 'Ni', 'Co', 'Fe', 'Mn', 'Cr', 'Al', 'Ga')
#     A     AB    AB    AB    AB    B     B     B
def x(a, b, c): return (X[a], X[a], X[b], X[c], X[b], X[c])
def s(a, b, c=None): return (X[a], X[a], X[b], X[b], X[c if c is not None else b], X[c if c is not None else b])
def r(a, b, c, d, e, f): return (X[a], X[b], X[c], X[d], X[e], X[f])


def S(ABC):
    A, B, C = ABC.split()
    return (A, A, B, B, C, C)


async def main():
    N_batch_jobs = 0
    batch_lines = tuple()
    log = log_format.initialize_logging(logging.INFO)
    with conditional_executor(False) as executor:
        async with conditional_db(True, log) as db:
            if os.environ.get('USER') == PBS_USER:
                localhost = Node(log, 'localhost', executor=executor, db=db, job_type='slurm', gpu_count=1, python_path='/u/cbu/py/bin/python3.13')
                # localhost = Node(log, 'localhost', executor=executor, db=db, job_type='slurm', gpu_count=1, python_path='/home/cbu/.conda-envs/py/bin/python')
                # localhost = Node(log, 'localhost', executor=executor, db=db, job_type='pbs', gpu_count=4, python_path='/usr/bin/python3.11')
            elif os.environ.get('USER') == 'crb273':
                localhost = Node(log, 'localhost', executor=executor, db=db, job_type='slurm', gpu_count=1, python_path='/home/crb273/py/bin/python3.12')
            else:
                localhost = Node(log, 'localhost', executor=executor, db=db, job_type='slurm', gpu_count=1, python_path='python')

    ####################################################################################################################
            action_args = [
                # [ActionType.ECHEM],
                # [ActionType.SCAN_NODE],
                # [ActionType.RSYNC_SMALL],
                # [ActionType.RUN_ML],
                # [jdftx.launch_type.PBS_WRAPPED_GPU, StudyType.NiC, 'ads'],
                
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Co Co Co'), 16],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Fe Co Co'), 16], # No preference for sites
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', ('Fe', 'Co', 'Fe', 'Co', 'Co', 'Co'), 16], # No preference for sites
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Mn Co Co'), 16],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Co Co Mn'), 16],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Ni Co Co'), 16],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Co Mn Mn'), 16],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Mn Mn Mn'), 16],

                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Mn Mn Fe'), 16], # Moderate preferece for Mn at A sites, Fe at B sites
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Ni Mn Mn'), 16],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Fe Fe Co'), 16], # Strong preference for Co at B sites
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Fe Fe Fe'), 16],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Mn Fe Fe'), 16], # Moderate preferece for Mn at A sites, Fe at B sites
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_111s, 'ads', S('Ni Fe Fe'), 16], # 

                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.PdPt, 'ads', None, 15],
                # [jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.Pt, 'ads', None, 15],

            ]

            # random.seed(123)
            # heo_combos = pick_combos(200, with_subsurface_Al=False)
            # for combo in heo_combos[10:80] + heo_combos[150:]:
            #     action_args.append([jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_100o, 'ads', r(*combo), 16])
            # random.seed(1234)
            # heo_al_combos = pick_combos(20, with_subsurface_Al=True)
            # for combo in heo_al_combos:
            #     action_args.append([jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_100o, 'ads', r(*combo), 16])

            # all_reg = [(i, j, k) for i in range(5) for j in range(1, 8) for k in range(1, 8)]
            # for a, b, c in all_reg:
            #     action_args.append([jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_100o, 'ads', s(a, b, c), 16])

            # all_striped = [(i, j, k) if X[j] < X[k] else (i, k, j) for i in range(5) for j in range(1, 7) for k in range(j+1, 7)]
            # for a, b, c in all_striped:
            #     action_args.append([jdftx.launch_type.SLURM_WRAPPED_GPU, StudyType.SPINEL_100o, 'base', x(a, b, c), 16])

            # for i, (a, b, c) in list(enumerate(all_combs))[110:]:
            #     name = f'{i:03}{X[a]}{X[b]}{X[c]}'
            #     if await localhost.job_is_running(name, use_cache=True):
            #         localhost.log.info(f'{name} already running')
            #         continue

            #     action_args.append([jdftx.launch_type.PBS_PARALLEL_WRAPPED_GPU, StudyType.SPINEL_100s, 'ads', s(a, b, c)])
            #     action_args.append([ActionType.LAUNCH_BATCH_PBS_WRAPPED_GPU, name])

            # for i, (a, b, c) in list(enumerate(all_combs))[15:100]:
            #     name = f'{i:03}{X[a]}{X[b]}{X[c]}'
            #     if await localhost.job_is_running(name, use_cache=True):
            #         localhost.log.info(f'{name} already running')
            #         continue

            #     action_args.append([jdftx.launch_type.PBS_SERIAL_WRAPPED_GPU, StudyType.SPINEL_100s, 'ads', s(a, b, c)])
            #     action_args.append([ActionType.LAUNCH_BATCH_PBS_WRAPPED_GPU, name])
    ####################################################################################################################

            for i, args in enumerate(action_args):
                localhost.log = log.bind(action=i)

                action = Action(localhost, N_batch_jobs, batch_lines, localhost, *args)
                result = await action.execute()
                if result:
                    actual_launch_type, N_jobs, lines = result
                    if actual_launch_type == jdftx.launch_type.PBS_PARALLEL_WRAPPED_GPU:
                        N_batch_jobs += N_jobs
                    elif actual_launch_type == jdftx.launch_type.PBS_SERIAL_WRAPPED_GPU:
                        N_batch_jobs = 1
                    else:
                        raise ValueError(f'Did not expect a return value from action type: {action.action_type}')
                    batch_lines += tuple(lines)
                if action.action_type == ActionType.LAUNCH_BATCH_PBS_WRAPPED_GPU:
                    N_batch_jobs = 0
                    batch_lines = tuple()


@dataclass
class Action:
    node: Node
    N_batch_jobs: int
    batch_lines: tuple[str]
    local_node: Node
    action_type: ActionType | LaunchType
    study_type: StudyType | str | None = None
    geos: str | None = None
    stoich: tuple[str] | str | None = None
    cores: int | None = None
    skip_converged: bool = True

    @property
    def fluid_dft_kwargs(self) -> dict:
        return {
            'fluid': 'candle',
            'fluid_cation': 'Na+',
            'fluid_anion': 'F-',
        }

    @property
    def orr_reaction(self) -> Reaction:
        return ecat_types.reaction.reaction_from_list([
            (0, ['*OH'],                                               [('O2', 'gas', 2), ('e-', None, 4), ('H2O', 'aq', 3)]),
            (0, ['(*O)(*OH)(*OH)(*OH)', '(*OOH)(*OH)(*OH)'],           [('O2', 'gas', 1), ('e-', None, 4), ('H2O', 'aq', 2)]),
            (0, ['(*OH)(*OH)(*OH)(*OH)(*OH)'],                         [('O2', 'gas', 1), ('e-', None, 4), ('H2O', 'aq', 1)]),
            (0, ['(*O)(*O)(*OH)'],                                     [('O2', 'gas', 1), ('e-', None, 4), ('H2O', 'aq', 3)]),
            (0, ['(*O)(*O)(*O)(*O)(*OH)'],                             [('e-', None, 4), ('H2O', 'aq', 3)]),
            (0, ['(*O)(*O)(*O)(*O)(*O)(*O)(*OH)'],                     [('O2', 'gas', -1),('e-', None, 4), ('H2O', 'aq', 3)]),
            (1, ['clean'],                                             [('O2', 'gas', 2), ('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 3)]),
            (1, ['(*O)(*O)', '*OO*', '*OO'],                           [('O2', 'gas', 1), ('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 3)]),
            (1, ['(*O)(*O)(*O)(*O)'],                                  [('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 3)]),
            (1, ['(*O)(*O)(*O)(*O)(*O)(*O)'],                          [('O2', 'gas', -1), ('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 3)]),
            (1, ['(*O)(*O)(*O)(*O)(*O)(*O)(*O)(*O)'],                  [('O2', 'gas', -2), ('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 3)]),
            (1, ['(*O)(*OH)(*OH)'],                                    [('O2', 'gas', 1), ('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 2)]),
            (1, ['(*OH)(*OH)(*OH)(*OH)'],                              [('O2', 'gas', 1), ('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 1)]),
            (1, ['(*OH)(*OH)(*OH)(*OH)(*OH)(*OH)(*OH)(*OH)'],          [('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', -1)]),
            (1, ['(*OO)(*OH)(*OH)(*OH)(*OH)', '(*O)(*O)(*OH)(*OH)(*OH)(*OH)'], [('OH-', 'aq', 1), ('e-', None, 3), ('H2O', 'aq', 1)]),
            (2, ['(*O)(*OH)', '(*OH)(*O)', '*OOH'],                    [('O2', 'gas', 1), ('OH-', 'aq', 2), ('e-', None, 2), ('H2O', 'aq', 2)]),
            (2, ['(*O)(*O)(*O)(*OH)'],                                 [('OH-', 'aq', 2), ('e-', None, 2), ('H2O', 'aq', 2)]),
            (2, ['(*O)(*O)(*O)(*O)(*O)(*OH)'],                         [('O2', 'gas', -1), ('OH-', 'aq', 2), ('e-', None, 2), ('H2O', 'aq', 2)]),
            (2, ['(*O)(*O)(*O)(*O)(*O)(*O)(*O)(*OH)'],                 [('O2', 'gas', -2), ('OH-', 'aq', 2), ('e-', None, 2), ('H2O', 'aq', 2)]),
            (2, ['(*OH)(*OH)(*OH)'],                                   [('O2', 'gas', 1), ('OH-', 'aq', 2), ('e-', None, 2), ('H2O', 'aq', 1)]),
            (2, ['(*OH)(*OH)(*OH)(*OH)(*OH)(*OH)(*OH)'],               [('OH-', 'aq', 2), ('e-', None, 2), ('H2O', 'aq', -1)]),
            (2, ['(*OO)(*OH)(*OH)(*OH)', '(*O)(*O)(*OH)(*OH)(*OH)'],   [('OH-', 'aq', 2), ('e-', None, 2), ('H2O', 'aq', 1)]),
            (2, ['(*OOH)(*OH)(*OH)(*OH)(*OH)'],                        [('OH-', 'aq', 2), ('e-', None, 2)]),
            (3, ['*O'],                                                [('O2', 'gas', 1), ('OH-', 'aq', 3), ('e-', None, 1), ('H2O', 'aq', 2)]),
            (3, ['(*O)(*O)(*O)'],                                      [('OH-', 'aq', 3), ('e-', None, 1), ('H2O', 'aq', 2)]),
            (3, ['(*O)(*O)(*O)(*O)(*O)'],                              [('O2', 'gas', -1), ('OH-', 'aq', 3), ('e-', None, 1), ('H2O', 'aq', 2)]),
            (3, ['(*O)(*O)(*O)(*O)(*O)(*O)(*O)'],                      [('O2', 'gas', -2), ('OH-', 'aq', 3), ('e-', None, 1), ('H2O', 'aq', 2)]),
            (3, ['(*O)(*OH)(*OH)(*OH)(*OH)', '(*OOH)(*OH)(*OH)(*OH)'], [('OH-', 'aq', 3), ('e-', None, 1)]),
            (3, ['(*OO)(*OH)(*OH)'],                                   [('OH-', 'aq', 3), ('e-', None, 1), ('H2O', 'aq', 1)]),
            (3, ['(*OH)(*OH)'],                                        [('O2', 'gas', 1), ('OH-', 'aq', 3), ('e-', None, 1), ('H2O', 'aq', 1)]),
            (3, ['(*OH)(*OH)(*OH)(*OH)(*OH)(*OH)'],                    [('OH-', 'aq', 3), ('e-', None, 1), ('H2O', 'aq', -1)]),
        ], [('O2', 'gas', -1), ('e-', None, -4), ('H2O', 'aq', -2), ('OH-', 'aq', 4)])

    @property
    def hor_reaction(self) -> Reaction:
        return ecat_types.reaction.reaction_from_list([
            (0, ['(*H)(*H)'], [('OH-', 'aq', 2)]),
            (1, ['*H'], [('H2O', 'aq', 1), ('OH-', 'aq', 1), ('e-', None, 1)]),
            (2, ['clean'], [('H2O', 'aq', 2), ('e-', None, 2)]),
        ], [('H2', 'gas', -1), ('H2O', 'aq', 2), ('OH-', 'aq', -2), ('e-', None, 2)])


    @property
    def graphene_reaction(self) -> Reaction:
        return ecat_types.reaction.reaction_from_list([
            (0, ['clean', '*', '*OO', '*OH', '*HOH'], []),
        ], [])

    async def execute_launch(self, *, nodes_per_job: int = 1):
        if not isinstance(self.action_type, LaunchType):
            raise ValueError('Expected LaunchType')

        match self.study_type:
            case StudyType.NiC:
                study = await self.make_NiC_study()
            case StudyType.MNC:
                study = await self.make_MNC_study()
            case StudyType.Pt:
                study = await self.make_Pt_study(use_pd_base=False)
            case StudyType.PdPt:
                study = await self.make_Pt_study(use_pd_base=True)
            case StudyType.SPINEL_100s:
                study = await self.make_spinel_study('100s', layers=5, frozen_layers=3)
            case StudyType.SPINEL_100o:
                study = await self.make_spinel_study('100o', layers=5, frozen_layers=3)
            case StudyType.SPINEL_111s:
                study = await self.make_spinel_study('111s', layers=8, frozen_layers=5)
            case StudyType.SPINEL_111b:
                study = await self.make_spinel_study('111b', layers=9, frozen_layers=4)
            case _:
                raise ValueError(f'Unexpected study type: {self.study_type}')

        await remote.database.update_study(self.node, study, delete_first=True)
        base_dft = await study.base_dft()
        if not base_dft:
            return  # base dft is already running

        if self.geos == 'fs':
            return await self.launch_fs_calcs(nodes_per_job, study)
        
        if self.geos == 'base' or not base_dft.calc.is_converged:
            return await self.launch_base_calc(nodes_per_job, study, base_dft)

        start_index = self.N_batch_jobs if self.action_type.batch == 'parallel' else 0
        if self.geos == 'ads':
            actual_launch_type, N_jobs, lines = await study.launch_slabs(self.action_type, self.skip_converged, nodes_per_job, start_index, mu_eV=None)
        elif self.geos == 'mu1VRHEpH14':
            SHE_candle_eV = -4.66
            pH14_RHE_adjustment_eV = 0.059 * 14
            mu_VRHE = 1
            mu_eV =  SHE_candle_eV + pH14_RHE_adjustment_eV + mu_VRHE
            actual_launch_type, N_jobs, lines = await study.launch_slabs(self.action_type, self.skip_converged, nodes_per_job, start_index, mu_eV=mu_eV)
        else:
            raise ValueError(f'Unexpected geos: {self.geos}')
        if lines:
            return actual_launch_type, N_jobs, lines

    async def launch_base_calc(self, nodes_per_job, study, base_dft):
        if self.action_type.batch:
            raise ValueError('Expected non-batch action type for base dft')
        if not base_dft.calc.is_converged or not self.skip_converged:
            await base_dft.launch(self.node, self.action_type, study.cores, nodes_per_job, node_start_index=0)
        else:
            self.local_node.log.info('Already converged', material=base_dft.material)

    async def launch_fs_calcs(self, nodes_per_job, study):
        for fs in study.unique_free_species():
            fs_dft = await study.fs_dft(fs)
            if not fs_dft:
                self.local_node.log.info(f'No calculation for {fs}')
            elif fs_dft.calc.is_converged and self.skip_converged:
                self.local_node.log.info(f'Already converged {fs}')
            else:
                await fs_dft.launch(self.node, self.action_type, study.cores, nodes_per_job, node_start_index=0)

    async def execute(self):
        if isinstance(self.action_type, LaunchType):
            return await self.execute_launch()

        match self.action_type:
            case ActionType.ECHEM:
                import visualization.echem.figures as figures  # pylint: disable=import-outside-toplevel
                figures.create_rde_fig()

            case ActionType.SCAN_NODE:
                await remote.database.scan_node(self.node)

            case ActionType.RSYNC_SMALL:
                await self.node.rsync(RsyncType.SMALL, self.local_node)

            case ActionType.WIPE_DB_GROUP:
                query = self.local_node.db.client.collection_group('materials')
                # results = await query.get()
                # filtered_results = [doc for doc in results if 'Fe3O4' in doc.id]
                # print([doc.id for doc in filtered_results])
                # tasks = [self.local_node.db.delete_doc(doc.reference) for doc in filtered_results]
                # await asyncio.gather(*tasks)

            case ActionType.UPDATE_FIELD:
                raise NotImplementedError('This action is not yet implemented')
                # await self.local_node.db.update_field('calcs', 'Threads', 4)

            case ActionType.RUN_ML:
                import ml.run_ml  # pylint: disable=import-outside-toplevel
                await ml.run_ml.train_dft_mep_prediction(self.local_node, 298)
                # await ml.composite_bo.train_dft_mep_partial_eval(self.local_node, 298)

            case ActionType.LAUNCH_BATCH_PBS_WRAPPED_GPU:
                if self.N_batch_jobs == 0:
                    self.node.log.info(f'{self.study_type}  All jobs are completed')
                    return
                self.node.log.info(f'{self.study_type}  Launching')
                job_sh_path = os.path.join(BATCH_SCRIPTS_PATH, self.study_type + '.sh')
                await remote.jobs.write_pbs_wrapped_gpu_script(self.node, self.study_type, job_sh_path,
                                                               self.N_batch_jobs, self.batch_lines)
                await self.node.run(['qsub', job_sh_path])

    async def make_NiC_study(self) -> Study:
        slab_mat = Ni('slab', facet='111s', layers=3, frozen_layers=2)
        spectators = (None,)
        ad_filter = Filter(guest_list=['y1-x', 's1-x', 'xs2-x', 't4-x'])
        full_reaction = studies.study_factory.compute_steps(self.graphene_reaction, spectators, slab_mat, ad_filter)
        return Study(self.node, self.cores, self.fluid_dft_kwargs, full_reaction, slab_mat,
                     base_elec_smearing_eV=0.025, slab_elec_smearing_eV=0.025)

    async def make_Pt_study(self, use_pd_base: bool) -> Study:
        if use_pd_base:
            slab_mat = Pd4Pt('slab', facet='100s', layers=6, frozen_layers=2)
        else:
            slab_mat = Pt('slab', facet='100s', layers=6, frozen_layers=2)
        spectators = (None,)
        # guest_list = ['clean'] + [f'{site}0-{ad}' for site in ('a', 'b', 'h') for ad in ('xO', 'xOH', 'xOO', 'xOOH')]
        # guest_list += [
        #     'a0-xO_a1-xO', 'a0-xO_a2-xO',
        #     'a0-xO_a1-xOH', 'a0-xO_a2-xOH', 'a0-xO_a1-xOH_a2-xOH', 'a0-xO_a1-xOH_a2-xOH_a3-xOH',
        #     'a0-xOO_a1-xOH', 'a0-xOO_a2-xOH', 'a0-xOO_a1-xOH_a2-xOH', 'a0-xOO_a1-xOH_a2-xOH_a3-xOH',
        #     'a0-xOH_a1-xOH', 'a0-xOH_a2-xOH', 'a0-xOH_a1-xOH_a2-xOH', 'a0-xOH_a1-xOH_a2-xOH_a3-xOH', 'a0-xOH_a2-xOH_a4-xOH_a6-xOH', 
        #     'a0-xOOH_a1-xOH', 'a0-xOOH_a2-xOH', 'a0-xOOH_a1-xOH_a2-xOH', 'a0-xOOH_a1-xOH_a2-xOH_a3-xOH',
        #     ]
        # guest_list += [
        #     'b0-xO_b1-xO', 'b0-xO_b2-xO',
        #     'b0-xO_b1-xOH', 'b0-xO_b2-xOH', 'b0-xO_b1-xOH_b2-xOH', 'b0-xO_b1-xOH_b2-xOH_b3-xOH',
        #     'b0-xOO_b1-xOH', 'b0-xOO_b2-xOH', 'b0-xOO_b1-xOH_b2-xOH', 'b0-xOO_b1-xOH_b2-xOH_b3-xOH',
        #     'b0-xOH_b1-xOH', 'b0-xOH_b2-xOH', 'b0-xOH_b1-xOH_b2-xOH', 'b0-xOH_b1-xOH_b2-xOH_b3-xOH', 'b0-xOH_b2-xOH_b4-xOH_b6-xOH', 
        #     'b0-xOOH_b1-xOH', 'b0-xOOH_b2-xOH', 'b0-xOOH_b1-xOH_b2-xOH', 'b0-xOOH_b1-xOH_b2-xOH_b3-xOH',
        #     ]
        # guest_list += ['_'.join([f'a{i}-xOH' for i in range(k)]) for k in range(5,9)]
        # guest_list += ['_'.join([f'b{i}-xOH' for i in range(k)]) for k in range(5,9)]
        # ad_filter = Filter(guest_list=guest_list)
        ad_filter = Filter()
        full_reaction = studies.study_factory.compute_steps(self.orr_reaction, spectators, slab_mat, ad_filter)
        return Study(self.node, self.cores, self.fluid_dft_kwargs, full_reaction, slab_mat,
                     base_elec_smearing_eV=0.25, slab_elec_smearing_eV=0.25)

    async def make_MNC_study(self) -> Study:
        if isinstance(self.stoich, str):
            mnc_mat = MN4C('slab', self.stoich)
        elif len(self.stoich) == 3:
            mnc_mat = BiMN4C('slab', *self.stoich)
        else:
            raise ValueError(f'Unsupported MNC stoich: {self.stoich}')

        spectators = (None, H_spectator([0]))
        no_spec = [
            'clean',
            'N-xH',
            'MA-xO_MB-xO',
        ]
        base_ads = [
            'MA-xOH_MB-xO',
            'MA-xOH_MB-xOH',
            'M-xO',
            'M-xOH',
            'M-xOO',
            'M-xOOH',
            'M-xOH',
            'MN-xO',
            'MN-xOH',
            'MN-xOO',
            'MN-xOOH',
            'MN-xOH',
        ]

        ads = [ad + '_N-xH' for ad in base_ads] + base_ads + no_spec
        ad_filter = Filter(guest_list=ads)

        full_reaction = studies.study_factory.compute_steps(self.orr_reaction, spectators, mnc_mat, ad_filter)
        return Study(self.node, self.cores, self.fluid_dft_kwargs, full_reaction, mnc_mat,
                     base_elec_smearing_eV=0.25, slab_elec_smearing_eV=0.25)

    async def make_spinel_study(self, facet: str, layers: int, frozen_layers: int) -> Study:
        slab_mat = Spinel('slab', self.stoich, facet=facet, layers=layers, frozen_layers=frozen_layers)
        if facet in ['100o', '111s', '111b']:
            elec_smearing_eV = 0.025
            match facet:
                case '100o':
                    spectator_indices, adsorbates = self.ads_100o_striped(slab_mat) if slab_mat.is_striped else self.ads_100o_reg()
                case '111s':
                    spectator_indices, adsorbates = self.ads_111s()
                case '111b':
                    spectator_indices, adsorbates = self.ads_111b()
                case _:
                    raise ValueError(f'Unexpected facet: {facet}')

            spectators = [H_spectator(indices) if indices else None for indices in spectator_indices]
            spec_strs = [''.join([f'O{i}-xH_' for i in indices]) for indices in spectator_indices]
            reg_spinel_ads = []
            for spec, ad in itertools.product(spec_strs, adsorbates):
                if ad:
                    reg_spinel_ads.append(f'{spec}{ad}')
                elif spec:
                    if spec[-1] == '_':
                        reg_spinel_ads.append(spec[:-1])
                    else:
                        raise ValueError(f'Unexpected spectator: {spec}')
                else:
                    reg_spinel_ads.append('clean')
        else:
            reg_spinel_ads = ['clean']
            elec_smearing_eV = 0.05
            spectators = [None]
            if facet == '100s':
                spectators += [
                    H_spectator([0]),
                    H_spectator([3]),
                    H_spectator([4]),
                    H_spectator([0,1]),
                    H_spectator([0,4]),
                    H_spectator([4,5]),
                    H_spectator([1,5]),
                    H_spectator([3,7]),
                    H_spectator([0,1,4]),
                    H_spectator([0,3,7]),
                    H_spectator([0,4,5]),
                    H_spectator([0,1,3,7]),
                    H_spectator([0,1,4,5]),
                    H_spectator([0,3,4,7]),
                    H_spectator([0,1,3,4,5]),
                    H_spectator([0,1,3,4,7]),
                    H_spectator([0,1,3,4,5,7]),
                    ]
            elif facet == '111s':
                spectators += [H_spectator([2,4,8,12]), H_spectator([4,8,12]), H_spectator([8,12]), H_spectator([12])]

            no_spec = [
                'O-xH',
                'TetA-xO_TetB-xO',
            ]
            always_ads = [
                'OctOctT2B-xOOx',
                'OctOctO2B-xOOx',
                'TetA-xOH_TetB-xO',
                'TetA-xOH_TetB-xOH',
                'Tet-xO',
                'Tet-xOH',
                'Tet-xOO',
                'Tet-xOOH',
                'TetO-xOH',
            ]
            basic_octs = [
                'Oct-xO',
                'Oct-xOH',
                'Oct-xOO',
                'Oct-xOOH',
                'TetOct2B-xOOx',
            ]

            if slab_mat.is_striped:
                basic_ads = no_spec + always_ads + [f'O-xH_{ad}' for ad in always_ads]
                striped_oct0s = [ad.replace('Oct', f'Oct{slab_mat.octs[0]}') for ad in basic_octs]
                striped_oct0s += [f'O-xH_{ad}' for ad in striped_oct0s]
                striped_oct1s = [ad.replace('Oct', f'Oct{slab_mat.octs[1]}') for ad in basic_octs]
                striped_oct1s += [f'O4-xH_{ad}' for ad in striped_oct1s]
                doubles = [
                    f'Oct{slab_mat.octs[0]}-xO_Oct{slab_mat.octs[1]}-xO',
                    f'Oct{slab_mat.octs[0]}-xOH_Oct{slab_mat.octs[1]}-xOH',
                    f'Oct{slab_mat.octs[0]}-xO_Oct{slab_mat.octs[1]}1-xO',
                    f'Oct{slab_mat.octs[0]}-xOH_Oct{slab_mat.octs[1]}1-xOH',
                    f'Oct{slab_mat.octs[0]}-xO_Tet-xO',
                    f'Oct{slab_mat.octs[0]}-xOH_Tet-xOH',
                    f'Oct{slab_mat.octs[1]}-xO_Tet-xO',
                    f'Oct{slab_mat.octs[1]}-xOH_Tet-xOH',
                ]
                doubles += [f'O-xH_{ad}' for ad in doubles]
                striped_ads = basic_ads + striped_oct0s + striped_oct1s + doubles
                ad_filter = Filter(guest_list=striped_ads)
            else:
                no_spec += ['Tet1-xOH_Tet2-xOH_Tet3-xOH_TetA-xO_TetB-xO']
                multi_100_2 = [
                    'Oct-xO_Tet-xO',
                    'Oct-xOH_Tet-xOH',
                    'Oct-xO_Oct2-xO',
                    'Oct-xOH_Oct2-xOH',
                    'Oct-xO_Oct3-xO',
                    'Oct-xOH_Oct3-xOH',
                ]
                multi_100_3 = [
                    'Oct-xO_Oct2-xOH_Oct3-xOH',
                    'Oct-xOO_Oct2-xOH_Oct3-xOH',
                    'Oct-xOH_Oct2-xOH_Oct3-xOH',
                    'Oct-xOOH_Oct2-xOH_Oct3-xOH',
                ]
                multi_100_4 = [
                    'Oct-xO_Oct1-xOH_Oct2-xOH_Oct3-xOH',
                    'Oct-xOO_Oct1-xOH_Oct2-xOH_Oct3-xOH',
                    'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH',
                    'Oct-xOOH_Oct1-xOH_Oct2-xOH_Oct3-xOH',

                    'Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xO',
                    'Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOO',
                    # 'Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH', # Covered below in OH_4_5_100
                    'Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOOH',
                ]

                multi_111_2 = [
                    'Tet-xO_Tet3-xOH',
                    'Tet-xOO_Tet3-xOH',
                    'Tet-xOH_Tet3-xOH',
                    'Tet-xOOH_Tet3-xOH',
                    'TetB-xOH_TetC3-xOH',
                ]

                multi_111_3 = [
                    'Tet-xO_Tet1-xOH_Tet3-xOH',
                    'Tet-xOO_Tet1-xOH_Tet3-xOH',
                    'Tet-xOH_Tet1-xOH_Tet3-xOH',
                    'Tet-xOOH_Tet1-xOH_Tet3-xOH',
                ]

                multi_111_4 = [
                    'Tet-xO_Tet1-xOH_Tet2-xOH_Tet3-xOH',
                    'Tet-xOO_Tet1-xOH_Tet2-xOH_Tet3-xOH',
                    'Tet-xOH_Tet1-xOH_Tet2-xOH_Tet3-xOH',
                    'Tet-xOOH_Tet1-xOH_Tet2-xOH_Tet3-xOH',
                ]

                h_ads = [f'O-xH_{ad}' for ad in (always_ads + basic_octs + multi_100_2)]
                # h_ads = [f'O-xH_{ad}' for ad in (always_ads + basic_octs + multi_100_2 + multi_100_3  + multi_100_4  + multi_100_5)]
                h_ads += [f'O12-xH_{ad}' for ad in (always_ads + basic_octs)]
                reg_spinel_ads += [
                    no_spec + always_ads + basic_octs + h_ads + multi_100_2
                    # + multi_100_3  + multi_100_4
                    # + multi_111_2 + multi_111_3 + multi_111_4
                ]
                for spec in ['O12-xH_O2-xH_O4-xH_O8-xH', 'O12-xH_O2-xH_O4-xH_O8-xH', 'O12-xH_O4-xH_O8-xH', 'O12-xH_O8-xH', 'O12-xH']:
                    reg_spinel_ads += [spec]
                    for ad in (always_ads + multi_111_2 + multi_111_3 + multi_111_4):
                        reg_spinel_ads += [f'{spec}_{ad}']


                # OH_4_5_100 = [
                #     'Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                #     'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xO',
                #     'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOO',
                #     'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                #     'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOOH',
                #     'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetA-xO',
                #     'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetB-xO',
                #     'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetA-xO_TetB-xO',
                # ]
                # multi_spec_H_100 = [
                #     'O-xH',
                #     'O3-xH',                
                #     'O4-xH',
                #     'O-xH_O1-xH',
                #     'O-xH_O4-xH',
                #     'O4-xH_O5-xH',
                #     'O-xH_O1-xH_O4-xH',
                #     'O-xH_O4-xH_O5-xH',
                #     'O-xH_O1-xH_O4-xH_O5-xH',
                #     'O-xH_O1-xH_O3-xH_O4-xH_O5-xH',
                #     'O-xH_O1-xH_O3-xH_O4-xH_O5-xH_O7-xH',
                # ]
                # reg_spinel_ads += OH_4_5_100 + [f'{spec}_{ad}' for spec, ad in itertools.product(multi_spec_H_100, OH_4_5_100)]
                mep_100 = [
                    # Step 1
                    'O-xH_O1-xH_O3-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetA-xO',
                    'O-xH_O1-xH_O3-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetB-xO',
                    'O-xH_O1-xH_O3-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xO',
                    'O-xH_O1-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_O1-xH_O4-xH_Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xO',
                    # Step 2
                    'O-xH_O1-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetA-xO_TetB-xO',
                    'O-xH_O1-xH_O3-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_O1-xH_O4-xH_O5-xH_Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    # Step 3
                    'O-xH_O4-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_O1-xH_O3-xH_O4-xH_O7-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetA-xO_TetB-xO',
                    'O-xH_O1-xH_O4-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xO',
                    'O-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH',
                    # Step 4
                    'O-xH_O4-xH_Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O4-xH_O5-xH_Oct-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_O1-xH_O4-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xOH',
                    'O-xH_O1-xH_O3-xH_O4-xH_O5-xH_O7-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_TetA-xO_TetB-xO',
                    'O-xH_O1-xH_O4-xH_O5-xH_Oct-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH_Tet-xO',
                    'O-xH_Oct-xOH_Oct2-xOH_Oct3-xOH',
                ]
                reg_spinel_ads += mep_100

        ad_filter = Filter(guest_list=reg_spinel_ads)
        full_reaction = studies.study_factory.compute_steps(self.orr_reaction, spectators, slab_mat, ad_filter)
        return Study(self.node, self.cores, self.fluid_dft_kwargs, full_reaction, slab_mat,
                     base_elec_smearing_eV=elec_smearing_eV, slab_elec_smearing_eV=elec_smearing_eV)

    def ads_111s(self):
        spectator_indices = []
        O_sites = [12,2,4,8] # Out of order on purpose to match string sorting in Adsorbates constructor
        for count in range(len(O_sites) + 1):
            for combo in itertools.combinations(O_sites, count):
                spectator_indices.append(list(combo))
        adsorbates = [
            '',
            't0-xOH',
            't0-xOH_t1-xOH',
            't0-xOH_t2-xOH',
            't0-xOH_t1-xOH_t2-xOH',
            't0-xOH_t1-xOH_t2-xOH_t3-xOH',
        ]
        return spectator_indices, adsorbates
    
    def ads_111b(self):
        spectator_indices = [[]]
        adsorbates = ['']
        for intermediate in ['OO', 'O', 'OH', 'OOH']:
            adsorbates += [
                f'o0-x{intermediate}',
                f'o0-x{intermediate}_o1-xOH',
                f'o0-x{intermediate}_o2-xOH',
                f'o0-x{intermediate}_o4-xOH',
                f'o0-x{intermediate}_o1-xOH_o2-xOH',
                f'o0-x{intermediate}_o1-xOH_o3-xOH',
                f'o0-x{intermediate}_o1-xOH_o5-xOH',
                f'o0-x{intermediate}_o1-xOH_o2-xOH_o3-xOH',
                f'o0-x{intermediate}_o1-xOH_o3-xOH_o4-xOH',
                f'o0-x{intermediate}_o1-xOH_o3-xOH_o5-xOH',
                f'o0-x{intermediate}_o1-xOH_o2-xOH_o3-xOH_o4-xOH',
                f'o0-x{intermediate}_o1-xOH_o2-xOH_o3-xOH_o4-xOH_o5-xOH',
            ]
        return spectator_indices, adsorbates

    def ads_100o_reg(self):
        spectator_indices = [
            [],
            [0],
            [2],
            [7],
            [0,1],
            [0,2],
            [0,5],
            [0,7],
            [1,7],
            [2,3],
            [2,4],
            [2,6],
            [0,1,5],
            [0,1,7],
            [0,2,4],
            [0,2,6],
            [0,3,5],
            [2,4,6],
            # [0,3,5,6], # Co3O4
            # [2,4,5,7], # Co3O4
            # [0,1,2,3,7], # Co3O4
            # [0,1,3,6,7], # Co3O4
        ]
        # for r in range(9):  # from 0 to 8 inclusive
        #     for combo in itertools.combinations(range(8), r):
        #         spectator_indices.append(list(combo))
        adsorbates = [
            '',
            'Oct0-xOH',
            'Oct0-xOH_Oct1-xOH',
            'Oct0-xOH_Oct2-xOH',
            'Oct0-xOH_Oct1-xOH_Oct2-xOH',
            'Oct0-xOH_Oct1-xOH_Oct2-xOH_Oct3-xOH',
        ]
    
        return spectator_indices, adsorbates

    def ads_100o_striped(self, slab_mat: Spinel):
        spectator_indices = [
            [],
            [0],
            [2],
            [7],
            [0,5],
            [0,7],
            [1,7],
            [2,3],
            [0,1,5],
            [0,1,7],
            [0,2,4],
            [0,2,6],
            [0,3,5],
            [2,4,6],
        ]
        # for i in range(8):
        #     spectator_indices += [[i]]
        #     for j in range(i+1, 8):
        #         spectator_indices += [[i, j]]
        #         for k in range(j+1, 8):
        #             spectator_indices += [[i, j, k]]
        adsorbates = [
            '',
            f'Oct{slab_mat.octs[0]}0-xOH',
            f'Oct{slab_mat.octs[1]}0-xOH',
            f'Oct{slab_mat.octs[0]}0-xOH_Oct{slab_mat.octs[0]}1-xOH',
            f'Oct{slab_mat.octs[0]}0-xOH_Oct{slab_mat.octs[1]}0-xOH',
            f'Oct{slab_mat.octs[1]}0-xOH_Oct{slab_mat.octs[1]}1-xOH',
            f'Oct{slab_mat.octs[0]}0-xOH_Oct{slab_mat.octs[0]}1-xOH_Oct{slab_mat.octs[1]}0-xOH',
            f'Oct{slab_mat.octs[0]}0-xOH_Oct{slab_mat.octs[1]}0-xOH_Oct{slab_mat.octs[1]}1-xOH',
            f'Oct{slab_mat.octs[0]}0-xOH_Oct{slab_mat.octs[0]}1-xOH_Oct{slab_mat.octs[1]}0-xOH_Oct{slab_mat.octs[1]}1-xOH',
        ]
        return spectator_indices, adsorbates


def H_spectator(indices: list[int]) -> Spectator:
    return Spectator.make_spectator('*H', indices, [('H2O', 'aq', -len(indices)), ('OH-', 'aq', len(indices)), ('e-', None, -len(indices))])


def pick_combos(count: int, with_subsurface_Al: bool) -> tuple[tuple[int]]:
    indices = [range(5), range(5), range(2,7), range(2,7), range(2,7), range(2,7)]
    min_unique = 4
    combos = []
    i = 0
    while i < count:
        combo = tuple([random.choice(index_list) for index_list in indices])
        if len(set(combo)) < min_unique or combo in combos or X[combo[2]] > X[combo[3]] or X[combo[4]] > X[combo[5]]:
            continue
        # We do not allow any configurations with Al on the surface due to its high reactivity with Oxygen
        if combo[2] == 6 or combo[3] == 6:
            continue
        if with_subsurface_Al and 6 not in combo:
            continue
        if not with_subsurface_Al and 6 in combo:
            continue
        i += 1
        combos.append(combo)
    return combos