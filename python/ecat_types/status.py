

from enum import Enum


class Status(Enum):
    Fresh =                 '🆕 Fresh  '
    Failed =                '💀 Failed '
    Insuf_Atomic_Orbitals = '💀 AtomOrb'
    Wfns_Mismatch =         '💀 Wfns   '
    Wfns_Save_Failure =     '💀 WfnSave'
    Dry_Run_Succeeded =     '✅ DryRun '
    Input_Parsing_Failed =  '💀 InFile '
    Lattice_Minimize =      '🏃 Lattice'
    Lattice_Converged =     '✅ Lattice'
    Lattice_Not_Converged = '❌ Lattice'
    Ionic_Minimize =        '🏃 Ionic  '
    Ionic_Converged =       '✅ Ionic  '
    Ionic_Not_Converged =   '❌ Ionic  '
    Ionic_Step_Failure =    '💀 IonStep'
    NaN_Failure =           '💀 NaN    '
    Running =               '🏃 Running'
    Stopped =               '✋ Stopped'
    Error =                 '⚠️  Error  '
    PythonError =           '⚠️  PyError'
