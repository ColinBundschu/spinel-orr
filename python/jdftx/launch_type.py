
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class LaunchType:
    dry: bool
    wrapped: bool
    gpu: bool
    format: str
    batch: str | None

    @property
    def dry_str(self):
        return '_dry' if self.dry else ''

LOCAL =                    LaunchType(dry=False, wrapped=False, gpu=False, format='local', batch=None)
LOCAL_GPU =                LaunchType(dry=False, wrapped=False, gpu=True,  format='local', batch=None)
SLURM =                    LaunchType(dry=False, wrapped=False, gpu=False, format='slurm', batch=None)
PBS_PARALLEL_WRAPPED_GPU = LaunchType(dry=False, wrapped=True,  gpu=True,  format='pbs'  , batch='parallel' )
PBS_SERIAL_WRAPPED_GPU =   LaunchType(dry=False, wrapped=True,  gpu=True,  format='pbs'  , batch='serial')
PBS_WRAPPED_GPU =          LaunchType(dry=False, wrapped=True,  gpu=True,  format='pbs'  , batch=None)
LOCAL_WRAPPED =            LaunchType(dry=False, wrapped=True,  gpu=False, format='local', batch=None)
LOCAL_WRAPPED_GPU =        LaunchType(dry=False, wrapped=True,  gpu=True,  format='local', batch=None)
SLURM_WRAPPED =            LaunchType(dry=False, wrapped=True,  gpu=False, format='slurm', batch=None)
SLURM_WRAPPED_GPU =        LaunchType(dry=False, wrapped=True,  gpu=True,  format='slurm', batch=None)
DRY_LOCAL =                LaunchType(dry=True,  wrapped=False, gpu=False, format='local', batch=None)
DRY_LOCAL_GPU =            LaunchType(dry=True,  wrapped=False, gpu=True,  format='local', batch=None)
