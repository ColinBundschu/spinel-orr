import argparse
import asyncio
import logging
import os

from google.api_core.exceptions import RetryError

import log_format
import remote.database
from remote.async_batch_database import AsyncBatchDatabase
from remote.node import Node


async def monitor_jdftx(node: Node, cores: int | None, base_filepath: str, dry_run: bool, gpus_per_node: int,
                        N_nodes: int, upload_interval_s: float, job_type: str | None) -> None:
    '''
    Monitor the jdftx process, upload results periodically, and upload final results upon completion.
    '''

    base_filename = os.path.basename(base_filepath)
    folder = os.path.dirname(base_filepath)
    in_filename = base_filename + '.in'
    out_filename = base_filename + ('_dry.out' if dry_run else '.out')
    cmd = []
    if job_type == 'pbs':
        cmd += ['mpiexec', '-n', str(N_nodes * gpus_per_node), '--hostfile', f'hostfile_{base_filename}','--ppn', '4',
               '--depth=8', '--cpu-bind', 'depth', '--env', 'OMP_NUM_THREADS=1', '-env', 'OMP_PLACES=thread', '--']
    else:
        cmd += ['srun']
    cmd += [node.jdftx_path(gpus_per_node > 0), '--input', in_filename, '--output', out_filename]
    if cores:
        cmd += ['--cores', str(cores)]
    if dry_run:
        cmd.append('--dry-run')

    if await node.exists(os.path.join(folder, out_filename)):
        raise ValueError('Out file already exists.')

    if not await node.exists(os.path.join(folder, in_filename)):
        raise ValueError('In file already exists.')

    node.log.info(f'Starting jdftx process: {" ".join(cmd)}')
    jdftx_process = await node.create_subprocess_exec(cmd, folder)
    node.log.info(f'jdftx process launched with PID {jdftx_process.pid}')

    while True:
        # Sleep first, to avoid a race condition with jdftx not having created the output file yet
        await asyncio.sleep(upload_interval_s)
        job_is_running = jdftx_process.returncode is None

        try:
            await remote.database.sync_calculation(node, base_filepath, job_is_running)
        except RetryError as e:
            node.log.error(event='Firebase RetryError', error=str(e))

        if not job_is_running:
            node.log.info(f'jdftx process exitted with return code {jdftx_process.returncode}')
            break

    if jdftx_process.returncode != 0:
        message = f'jdftx returned nonzero return code: {jdftx_process.returncode} with\n'
        node.log.error(event='nonzero_return_code', error=message)

    node.log.info('python exitting')


async def main(base_filepath: str, dry_run: bool, gpus_per_node: int, N_nodes: int, cores: int | None,
               upload_interval: float, job_type: str | None) -> None:
    log = log_format.initialize_logging(logging.INFO)
    async with AsyncBatchDatabase(log, batch_size=1) as db:
        node = Node(log, 'localhost', db=db, job_type=job_type)
        try:
            await monitor_jdftx(node, cores, base_filepath, dry_run, gpus_per_node, N_nodes, upload_interval, job_type)
        except Exception as e:
            node.log.exception(event='monitor_jdftx_exception')
            await remote.database.sync_calculation(node, base_filepath, job_is_running=False, python_error=str(e))
            raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wrapper script for managing jdftx processes and uploads.')

    parser.add_argument('base_filepath', type=str, help='Base path to the files for the calculation')
    parser.add_argument('-n', dest='dry_run', action='store_true', help='Set to make the calculation a dry run')
    parser.add_argument('-c', '--cores', type=int, default=None, help='number of cores for jdftx to use')
    parser.add_argument('--N_nodes', type=int, default=1, help='number of nodes for jdftx to use')
    parser.add_argument('-g', '--gpus_per_node', type=int, default=0, help='Number of gpus per node')
    parser.add_argument('-u', '--upload_interval', type=int, default=120, help='Interval in seconds between uploads')
    parser.add_argument('-j', '--job_type', type=str, default=None, help='slurm or pbs or None')

    args = parser.parse_args()

    asyncio.run(main(args.base_filepath, args.dry_run, args.gpus_per_node, args.N_nodes,
                     args.cores, args.upload_interval, args.job_type))
