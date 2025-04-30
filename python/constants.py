import os
from pathlib import Path

import numpy as np

ANGSTROM2_GRAM_PER_EV_SEC2 = 16021.76634
AVOGADRO = 6.02214076E23
BOHR_PER_ANGSTROM = 1.889726124279
ANGSTROM_PER_BOHR = 1/BOHR_PER_ANGSTROM
h2_A2eVu = 0.1650260739
kB_eVpK = 8.61733326E-5  # Boltzmann constant in eV/K
hbar_eVs = 6.582119569E-16
hc_eVm = 1.2398419739E-6
P_ATM_eVpA3 = 6.24150907446076E-7
EV_PER_HARTREE = 27.211386246

ATOMIC_SYMBOLS = np.array([
    None, 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
    'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
    'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
    'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
    'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
    'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo',
])
ATOMIC_NUMS = {symbol: num for num, symbol in enumerate(ATOMIC_SYMBOLS)}

UNIT_CELL_FRAC_CONV = 0.036
MAX_ENERGY_CHANGE_eV = 0.85
MAX_LATTICE_CHANGE_A = 0.06
MAX_FORCE_LATTICE_eVpA = 0.02
MAX_FORCE_SLAB_eVpA = 0.02

PBS_USER = 'cbu'
MNT_PATH = '/mnt/z/' if os.environ.get('USER') in ['exouser', 'colin'] else Path.home()
BATCH_SCRIPTS_PATH = os.path.join(MNT_PATH, 'batch_scripts')
FIREBASE_CERT_PATH = os.path.join(MNT_PATH, 'von', 'python', 'PLACEHOLDER.json')
WRAPPER_SCRIPT_PATH = os.path.join(MNT_PATH, 'von', 'python', 'monitor_jdftx.py')
JDFTX_PATH = os.path.join(MNT_PATH, 'jdftx', 'build', 'jdftx')
# Locally built to match hardware configuration
JDFTX_GPU_PATH = os.path.join(Path.home(), 'jdftx', 'build', 'jdftx_gpu')
# JDFTX_GPU_PATH = 'jdftx_gpu'
CREATEXSF_PATH = os.path.join(MNT_PATH, 'jdftx', 'jdftx-git', 'jdftx', 'scripts', 'createXSF')
DFT_OUT_FOLDER = os.path.join(MNT_PATH, 'dft_out')
# DFT_OUT_FOLDER = '/work/nvme/bdtx/cbu/dft_out'

ELECTRONEGATIVITIES_NORMALIZED = {
    'Mg': 0.0000,  # 1.31
    'Ni': 1.0000,  # 1.91
    'Zn': 0.5667,  # 1.65
    'Co': 0.9500,  # 1.88
    'Fe': 0.8667,  # 1.83
    'Mn': 0.4000,  # 1.55
    'Al': 0.5000,  # 1.61
    'Ga': 0.8333,  # 1.81
}

IONIC_RADII_NORMALIZED = {
    'Mg': 0.7037,  # 0.72 A
    'Ni': 0.5926,  # 0.69 A
    'Zn': 0.7778,  # 0.74 A
    'Co': 0.4444,  # 0.65 A
    'Fe': 0.9259,  # 0.78 A
    'Mn': 1.0000,  # 0.80 A
    'Al': 0.0000,  # 0.53 A
    'Ga': 0.3333,  # 0.62 A
}

VALENCE_ELECTRON_COUNTS = {
    'Mg': 2,
    'Ni': 10,  # 4s²3d⁸
    'Zn': 2,   # 4s²
    'Co': 9,   # 4s²3d⁷
    'Fe': 8,   # 4s²3d⁶
    'Mn': 7,   # 4s²3d⁵
    'Al': 3,   # 3s²3p¹
    'Ga': 3,   # 4s²4p¹
}

FIRST_IONIZATION_ENERGIES_eV = {
    'Mg': 7.646,
    'Ni': 7.6398,
    'Zn': 9.3942,
    'Co': 7.881,
    'Fe': 7.9024,
    'Mn': 7.434,
    'Al': 5.9858,
    'Ga': 5.9993,
}

PERIODIC_TABLE_POSITIONS = {
    'Mg': (3, 2),  # Row 3, Group 2 (Alkaline Earth Metals)
    'Ni': (4, 10),  # Row 4, Group 10 (Transition Metals)
    'Zn': (4, 12),  # Row 4, Group 12 (Transition Metals)
    'Co': (4, 9),  # Row 4, Group 9 (Transition Metals)
    'Fe': (4, 8),  # Row 4, Group 8 (Transition Metals)
    'Mn': (4, 7),  # Row 4, Group 7 (Transition Metals)
    'Al': (3, 13),  # Row 3, Group 13 (Post-Transition Metals)
    'Ga': (4, 13),  # Row 4, Group 13 (Post-Transition Metals)
}
