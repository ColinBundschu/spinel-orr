
import asyncio
import datetime
import functools
import getpass
import itertools
import os
import re
import subprocess
from asyncio import Lock
from asyncio.subprocess import Process
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from stat import S_ISDIR, S_ISREG
from typing import Callable

import paramiko
from structlog.stdlib import BoundLogger
from constants import CREATEXSF_PATH, DFT_OUT_FOLDER, JDFTX_GPU_PATH, JDFTX_PATH, PBS_USER, WRAPPER_SCRIPT_PATH

from remote.async_batch_database import AsyncBatchDatabase


@dataclass(frozen=True, order=True)
class StatResult:
    folderpath: str
    name: str
    st_mtime: str

    @property
    def filepath(self) -> str:
        return os.path.join(self.folderpath, self.name)

    @property
    def basename(self) -> str:
        return os.path.splitext(self.name)[0]

    @property
    def base_filepath(self) -> str:
        return os.path.splitext(self.filepath)[0]

    @property
    def last_modified(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.st_mtime, tz=datetime.timezone.utc)


@dataclass(frozen=True, order=True)
class CommandResult:
    node_name: str
    cmd: list[str]
    cwd: str | None
    returncode: int
    stdout: str
    stderr: str

    def __str__(self):
        lines = [f'\nCommandResult from {self.node_name} (returncode {self.returncode})',
                 '\nCOMMAND', ' '.join(self.cmd)]
        if self.cwd:
            lines += ['\nCWD', self.cwd]
        if self.stderr:
            lines += ['\nSTD ERROR', self.stderr.lstrip('\n')]
        if self.stdout:
            lines += ['\nSTD OUT', self.stdout.lstrip('\n')]
        return '\n'.join(lines)


class RsyncType(Enum):
    SMALL = 'small'
    ALL = 'all'


class Node:
    def __init__(self, log: BoundLogger, hostname: str, *,
                 executor: ProcessPoolExecutor | None = None,
                 db: AsyncBatchDatabase | None = None,
                 username: str | None = None,
                 display_name: str | None = None,
                 jump_host: str | None = None,
                 job_type: str | None = None,
                 gpu_count: int = 0,
                 python_path: str | None = None,
                 ):

        self.log = log.bind(hostname=hostname)
        self.display_name = display_name
        self.client = None
        self.sftp = None
        self._lock = Lock()
        self.connection_initialized = False
        self.executor = executor
        self.db = db
        self.job_type = job_type
        self.gpu_count = gpu_count
        self.python_path = python_path
        self._job_cache = None

        if hostname == 'localhost':
            self.hostname = hostname
            self.username = username or getpass.getuser()
            self.jump_host = None
            self.pkey = None
        else:
            config = paramiko.SSHConfig.from_path(os.path.expanduser('~/.ssh/config'))
            host_config = config.lookup(hostname)
            self.hostname = host_config.get('hostname', hostname)
            self.username = host_config.get('user', username)
            identity_file = host_config.get('identityfile', [None])[0]
            jump_host = host_config.get('proxyjump', jump_host)
            self.jump_host = config.lookup(jump_host).get('hostname', jump_host) if jump_host else None
            self.pkey = paramiko.RSAKey.from_private_key_file(identity_file) if identity_file else None

    def _init_connection(self) -> None:
        '''This allows us to lazily instantiate connections to slow nodes'''
        if self.connection_initialized or self.hostname == 'localhost':
            return

        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        sock = paramiko.ProxyCommand(f'ssh {self.jump_host} -W {self.hostname}:22') if self.jump_host else None
        self.client.connect(self.hostname, username=self.username, disabled_algorithms={
                            'pubkeys': ['rsa-sha2-256']}, sock=sock, pkey=self.pkey)
        self.sftp = self.client.open_sftp()
        self.connection_initialized = True

    def __del__(self):
        if self.connection_initialized:
            self.sftp.close()
            self.client.close()

    def __str__(self):
        return self.display_name or self.hostname

    @property
    def dft_out_folder(self) -> str:
        ''' We may not use the same out path in the future for all machines'''
        return DFT_OUT_FOLDER

    @property
    def wrapper_script_path(self) -> str:
        ''' We may not use the same out path in the future for all machines'''
        return WRAPPER_SCRIPT_PATH

    @property
    def createXSF_path(self) -> str:
        ''' We may not use the same out path in the future for all machines'''
        return CREATEXSF_PATH

    @property
    def is_local(self) -> bool:
        return self.hostname == 'localhost'

    def jdftx_path(self, gpu: bool) -> str:
        ''' We may not use the same out path in the future for all machines'''
        return JDFTX_GPU_PATH if gpu else JDFTX_PATH

    async def run_maybe_in_executor(self, func: Callable, *args):
        if self.executor:
            return await asyncio.get_running_loop().run_in_executor(self.executor, func, *args)
        return func(*args)

    async def listdir(self, folderpath: str) -> tuple[list[str], list[StatResult]]:
        self._init_connection()
        loop = asyncio.get_running_loop()
        async with self._lock:
            # return await loop.run_in_executor(None, self._listdir_sync, folderpath)
            folders, files = [], []
            if self.is_local:
                entries = await loop.run_in_executor(None, os.scandir, folderpath)
                for entry in entries:
                    if entry.is_dir():
                        folders.append(entry.name)
                    elif entry.is_file():
                        entry_stat = entry.stat()
                        files.append(StatResult(folderpath, entry.name, entry_stat.st_mtime))
            else:
                entries = await loop.run_in_executor(None, self.sftp.listdir_attr, folderpath)
                for entry in entries:
                    if S_ISDIR(entry.st_mode):
                        folders.append(entry.filename)
                    elif S_ISREG(entry.st_mode):
                        files.append(StatResult(folderpath, entry.filename, entry.st_mtime))
            return folders, files

    def _stat_file_sync(self, filepath: str) -> StatResult | None:
        if self.is_local:
            if os.path.isfile(filepath):
                entry_stat = os.stat(filepath)
                return StatResult(os.path.dirname(filepath), os.path.basename(filepath), entry_stat.st_mtime)
        else:
            try:
                entry = self.sftp.stat(filepath)
                if S_ISREG(entry.st_mode):
                    return StatResult(os.path.dirname(filepath), os.path.basename(filepath), entry.st_mtime)
            except FileNotFoundError:
                pass
        return None

    async def stat_file(self, filepath: str) -> StatResult | None:
        self._init_connection()
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(None, self._stat_file_sync, filepath)

    async def rmdir(self, folderpath: str) -> None:
        self._init_connection()
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self.is_local:
                await loop.run_in_executor(None, os.rmdir, folderpath)
            else:
                await loop.run_in_executor(None, self.sftp.rmdir, folderpath)

    async def mkdir(self, folderpath: str) -> None:
        self._init_connection()
        if self.is_local:
            loop = asyncio.get_running_loop()
            async with self._lock:
                await loop.run_in_executor(None, functools.partial(os.makedirs, folderpath, exist_ok=True))
        else:
            await self.run(['mkdir', '-p', folderpath])

    async def writelines(self, filepath: str, lines: list[str], *, add_newlines: bool = True) -> None:
        self._init_connection()
        loop = asyncio.get_running_loop()
        data = '\n'.join(lines) + '\n' if add_newlines else ''.join(lines)
        async with self._lock:
            await loop.run_in_executor(None, self._writelines_sync, filepath, data)

    def _writelines_sync(self, filepath: str, data: str) -> None:
        if self.is_local:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            with self.sftp.open(filepath, 'w') as f:
                f.write(data.encode())

    async def delete_files(self, filepaths: list[str], *, missing_ok: bool = False) -> None:
        self._init_connection()
        cmd = ['rm'] + (['-f'] if missing_ok else []) + filepaths
        result = await self.run(cmd)
        if result.returncode or result.stderr:
            self.log.error(event='delete_files_failed', filepaths=filepaths, force=missing_ok,
                           stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        else:
            self.log.debug(event='delete_files_succeeded', filepaths=filepaths, force=missing_ok)

    async def run(self, cmd: list[str], *, cwd: str | None = None) -> CommandResult:
        self._init_connection()
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self.is_local:
                raw_result = await loop.run_in_executor(None, functools.partial(
                    subprocess.run, cmd, capture_output=True, text=True, check=False, cwd=cwd))
                return CommandResult(str(self), cmd, cwd, raw_result.returncode, raw_result.stdout, raw_result.stderr)

            if cwd:
                cmd = [f'cd "{cwd}" &&'] + cmd
            _, f_stdout, f_stderr = self.client.exec_command(' '.join(cmd))
            returncode = f_stdout.channel.recv_exit_status()
            stdout = ''.join(f_stdout.readlines())
            stderr = ''.join(f_stderr.readlines())
            return CommandResult(str(self), cmd, cwd, returncode, stdout, stderr)

    async def create_subprocess_exec(self, cmd: list[str], cwd: str) -> Process:
        self._init_connection()
        if self.is_local:
            return await asyncio.create_subprocess_exec(*cmd, cwd=cwd)
        raise NotImplementedError('exec_command not implemented for remote nodes yet')

    async def exec_nohup_command(self, cmd: list[str], cwd: str, stdout_path: str) -> None:
        self._init_connection()
        async with self._lock:
            cmd_str = ' '.join(cmd)
            ts_now = datetime.datetime.now().isoformat()
            full_cmd_str = f'echo "\n{ts_now}\n" >> "{stdout_path}" && nohup {cmd_str} >> "{stdout_path}" 2>&1 &'
            if self.is_local:
                process = await asyncio.create_subprocess_shell(full_cmd_str, cwd=cwd)
                self.log.info(f'executed nohup command {cmd_str} in {cwd} with PID {process.pid}')
            else:
                channel = self.client.get_transport().open_session()
                channel.exec_command(f'cd "{cwd}" && {full_cmd_str}')

    async def exists(self, path: str) -> bool:
        self._init_connection()
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(None, self._exists_sync, path)

    def _exists_sync(self, path: str) -> bool:
        self._init_connection()
        if self.is_local:
            return os.path.exists(path)
        else:
            try:
                self.sftp.stat(path)
                return True
            except IOError as e:
                if 'No such file' in str(e):
                    return False
                else:
                    raise

    async def readlines(self, filepath: str) -> tuple[str]:
        self._init_connection()
        loop = asyncio.get_running_loop()
        async with self._lock:
            def read_local_file():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.readlines()

            def read_remote_file():
                with self.sftp.open(filepath, 'r') as f:
                    return f.read().decode('utf-8').splitlines()

            if self.is_local:
                lines = await loop.run_in_executor(None, read_local_file)
            else:
                lines = await loop.run_in_executor(None, read_remote_file)

            return tuple(lines)

    async def rsync(self, rsync_type: RsyncType, target_node: 'Node') -> None:
        self._init_connection()
        match rsync_type:
            case RsyncType.ALL:
                exclude_list = []
            case RsyncType.SMALL:
                exclude_list = itertools.chain(*[('--exclude', '*' + x) for x in ['.wfns', '.fluidState']])

        remote_path = os.path.join(f'{self.username}@{self.hostname}:{self.dft_out_folder}', "*")
        self.log.info(f'rsyncing {rsync_type} files from {remote_path} to {target_node.dft_out_folder}...')
        command = ['rsync', '-v', '--archive', '--update', '--inplace',
                   *exclude_list, remote_path, target_node.dft_out_folder]

        result = await self.run(command)
        self.log.info(result.stdout)
        if result.returncode or result.stderr:
            raise RuntimeError(result)

    async def job_is_running(self, job_id: str, *, use_cache: bool = False) -> bool:
        job_id = shorten_job_id(job_id)
        try:
            match self.job_type:
                case 'slurm':
                    if not use_cache or self._job_cache is None:
                        self._job_cache = await self.run(['squeue', '-o', '%j'])
                    return job_id in self._job_cache.stdout.splitlines()[1:]
                case 'pbs':
                    if not use_cache or self._job_cache is None:
                        result = await self.run(['qstat', '-f'])
                        lines = result.stdout.splitlines()
                        self._job_cache = []
                        for i, line in enumerate(lines):
                            # Find the start of a job by its Job_Name
                            name_parts = line.split()
                            if len(name_parts) != 3 or name_parts[0] != 'Job_Name':
                                continue

                            # Check if the job is owned by the current user
                            for j in range(i, i + 10):
                                owner_parts = lines[j].split()
                                if len(owner_parts) == 3 and owner_parts[0] == 'Job_Owner':
                                    owner_matches = owner_parts[2].startswith(PBS_USER + '@')
                                    break
                            else:
                                raise ValueError(f'Unable to find Job_Owner: {lines[i:i+20]}')

                            # Find the job_state
                            for j in range(i, i + 10):
                                state_parts = lines[j].split()
                                if len(state_parts) == 3 and state_parts[0] == 'job_state':
                                    state_is_valid = state_parts[2] != 'F'
                                    break
                            else:
                                raise ValueError(f'Unable to find job_state: {lines[i:i+20]}')

                            if owner_matches and state_is_valid:
                                self._job_cache.append(name_parts[2])

                    return job_id in self._job_cache
                case _:
                    raise ValueError(f'Unknown job format {self.job_type}')
        except FileNotFoundError:
            return False

def shorten_job_id(job_id: str) -> str:
    if job_id.count('-xH_') > 1:
        job_id = job_id.replace('O-xH_', 'O0-xH_')
        oxygen_parts = re.findall(r'O(\d+)-xH', job_id)
        oxygen_numbers = ''.join(oxygen_parts)
        remaining_part = re.sub(r'O\d+-xH_', '', job_id)
        job_id = f'{remaining_part}_O{oxygen_numbers}-xH'

    job_id = job_id.replace('_', '').replace('Tet', 't').replace('Oct', 'o').replace('-', '')
    return job_id
