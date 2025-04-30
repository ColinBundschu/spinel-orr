
from ecat_types import Atom, Atoms, Calc, Lattice, LatticeType


def Pt() -> Calc:
    lattice_A = Lattice(LatticeType.FCC, a=3.94)
    atoms = Atoms([Atom(78, (0, 0, 0))])
    return Calc(lattice_A, atoms)

def Si() -> Calc:
    lattice_A = Lattice(LatticeType.FCC, a=5.6)
    atoms = Atoms([Atom(14, (0, 0, 0)), Atom(14, (0.25, 0.25, 0.25))])
    return Calc(lattice_A, atoms)


# def Co3O4() -> Calc:
#     # Spinel with Magnetic Moments from 10.1039/c6cp05554k
#     lattice_A = Lattice(LatticeType.FCC, a=8.084) # Experimental from Table S1 10.1021/acs.jpcc.7b11869
#     atoms = Atoms([
#         Atom(27, (0.0000000, 0.0000000, 0.0000000), magnetic_moment=-3),  # Tetrahedral (A-Site)
#         Atom(27, (0.2500000, 0.2500000, 0.2500000), magnetic_moment=3),  # Tetrahedral (A-Site)
#         Atom(27, (0.1250000, 0.6250000, 0.6250000)),  # Octahedral (B-Site)
#         Atom(27, (0.6250000, 0.6250000, 0.6250000)),  # Octahedral (B-Site)
#         Atom(27, (0.6250000, 0.6250000, 0.1250000)),  # Octahedral (B-Site)
#         Atom(27, (0.6250000, 0.1250000, 0.6250000)),  # Octahedral (B-Site)
#         # Relaxed at 441 bulk. Cobalt moved so little kept at theoretical
#         # Oxygen copied from output
#         Atom(8,  (0.8623523, 0.8623523, 0.4129428)),
#         Atom(8,  (0.3876874, 0.3876874, 0.3876874)),
#         Atom(8,  (0.4129428, 0.8623523, 0.8623523)),
#         Atom(8,  (0.8623523, 0.8623523, 0.8623523)),
#         Atom(8,  (0.8623523, 0.4129428, 0.8623523)),
#         Atom(8,  (0.8369377, 0.3876874, 0.3876874)),
#         Atom(8,  (0.3876874, 0.3876874, 0.8369377)),
#         Atom(8,  (0.3876874, 0.8369377, 0.3876874))])
#     return Calc(lattice_A, atoms)


def H2() -> Calc:
    lattice_A = Lattice(LatticeType.Cubic, a=1)
    atoms = Atoms([
        Atom(1, (0.37, 0, 0)),
        Atom(1, (-0.37, 0, 0))])
    return Calc(lattice_A, atoms)


def Au() -> Calc:
    lattice_A = Lattice(LatticeType.Cubic, a=1)
    atoms = Atoms([Atom(79, (0, 0, 0))])
    return Calc(lattice_A, atoms)


def H2O() -> Calc:
    lattice_A = Lattice(LatticeType.Cubic, a=1)
    atoms = Atoms([
        Atom(8, (0, 0, 0)),
        Atom(1, (0.7493682, 0.6077836, 0)),
        Atom(1, (-0.7493682, 0.6077836, 0))])
    return Calc(lattice_A, atoms)


def O2() -> Calc:
    lattice_A = Lattice(LatticeType.Cubic, a=1)
    atoms = Atoms([
        Atom(8, (0, 0, 0), magnetic_moment=1),
        Atom(8, (1.2075, 0, 0), magnetic_moment=1)])
    return Calc(lattice_A, atoms)


def NO() -> Calc:
    lattice_A = Lattice(LatticeType.Cubic, a=1)
    atoms = Atoms([
        Atom(7, (0, 0, 0), magnetic_moment=1),
        Atom(8, (1.152, 0, 0), magnetic_moment=1)])
    return Calc(lattice_A, atoms)


def HNO3() -> Calc:
    lattice_A = Lattice(LatticeType.Cubic, a=1)
    atoms = Atoms([
        Atom(7, (0, 0, 0)),
        Atom(8, (0, 1.46, 0)),
        Atom(1, (1, 1.66, 0)),
        Atom(8, (-1.05, -0.5, 0)),
        Atom(8, (1.05, -0.5, 0))])
    return Calc(lattice_A, atoms)
