
from pathlib import Path

import remote.node
from remote.node import Node


async def write_pbs_wrapped_gpu_script(node: Node, job_id: str, job_sh_path: str, N_nodes: int,
                                       job_lines: list[str]) -> None:

    job_id = remote.node.shorten_job_id(job_id)
    if N_nodes == 1:
        queue = 'preemptable'
        walltime = '72:00:00'
    elif 10 <= N_nodes < 25:
        queue = 'prod'
        walltime = '01:00:00'
    elif 400 <= N_nodes <= 450:
        queue = 'prod'
        walltime = '08:00:00'
    else:
        raise ValueError(f'Unsupported number of nodes: {N_nodes}')

    script_folder = Path(job_sh_path).parent
    lines = [
        '#!/bin/bash -l',
        f'#PBS -N {job_id}',
        f'#PBS -l select={N_nodes}',
        f'#PBS -l walltime={walltime}',
        f'#PBS -q {queue}',
        '#PBS -l filesystems=home',
        '#PBS -A von',
        f'#PBS -o {script_folder / (job_id + "-output.log")}',
        f'#PBS -e {script_folder / (job_id + "-error.log")}',
        '',
        'NODES=(`cat $PBS_NODEFILE`) # Extract the hostnames of the nodes',
        'export http_proxy="http://proxy.alcf.anl.gov:3128"',
        'export https_proxy="http://proxy.alcf.anl.gov:3128"',
        'export ftp_proxy="http://proxy.alcf.anl.gov:3128"',
        'export MPICH_GPU_SUPPORT_ENABLED=1',
        'module load craype-accel-nvidia80',
        'export JDFTX_MEMPOOL_SIZE=38000',
    ]
    lines.append('')
    lines += job_lines
    lines.append('')
    lines.append('wait')
    await node.writelines(job_sh_path, lines)
