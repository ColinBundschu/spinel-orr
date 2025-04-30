import { scaleLinear } from 'd3-scale';

export const AVOGADRO = 6.02214076E23;

export const ANGSTROM2_GRAM_PER_EV_SEC2 = 16021.76634;
export const h2_A2eVu = 0.1650260739; // Planck constant squared, Angstroms^2 * electronvolts * atomic mass units
export const kB_eVpK = 8.61733326E-5; // Boltzmann constant, electronvolts / kelvin
export const hbar_eVs = 6.582119569E-16; // Reduced Planck constant electronvolts * seconds
export const hc_eVm = 1.2398419739E-6; // Planck constant by speed of light, electronvolts meters
export const P_ATM_eVpA3 = 6.24150907446076E-7; // electronvolts / Angstrom^3
export const ATOMIC_SYMBOLS = [
  '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
  'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
  'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
  'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
  'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
  'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl',
  'Uup', 'Lv', 'Uus', 'Uuo'];
export const ATOMIC_SYMBOLS_SET = new Set<string>(ATOMIC_SYMBOLS.filter((symbol): symbol is string => symbol != null));
export const ATOMIC_RADII_A = [
  0, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
  2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.16, 1.20, 2.44,
  2.15, 2.07, 2.04, 1.92, 1.84, 1.83, 1.79, 1.77, 1.74, 1.72, 1.58, 1.93, 1.83, 1.81, 2.06, 1.92, 2.20, 2.45, 2.23,
  2.18, 2.17, 2.08, 2.07, 2.05, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.93, 1.92, 1.92, 1.87, 1.82, 1.78,
  1.77, 1.75, 1.73, 1.71, 1.66, 1.55, 1.96, 1.90, 1.90, 2.02, 2.00, 2.20, 2.23, 2.01, 2.01, 1.99, 1.98, 1.96, 1.90,
  1.87, 1.80, 1.69, 1.65, 1.65, 1.64, 1.63, 1.61, 1.60, 1.60];
export const MAX_RELAX_dE_eV = 0.85;
export const MAX_FORCE_eVpA = 0.02;

export const CONVERGED_COLOR = 'violet'
export function convergenceColor(force_meVpA: number | null): string {
  if (force_meVpA === null) return 'blue';
  if (force_meVpA < 0 || !Number.isFinite(force_meVpA) || Number.isNaN(force_meVpA)) return 'purple';
  const colorScale = scaleLinear<string>()
    .domain([MAX_FORCE_eVpA * 1000, MAX_FORCE_eVpA * 1000 + 1, 100, 500, 2000])
    .range(['aquamarine', 'limegreen', 'yellow', 'orange', 'red'])
    .clamp(true);
  return colorScale(force_meVpA);
}

export enum MinType {
  ElecConverged,
  ElecNotConverged,
  FluidConverged,
  FluidNotConverged,
  SCFConverged,
  SCFNotConverged,
  Ionic,
  Lattice,
}
