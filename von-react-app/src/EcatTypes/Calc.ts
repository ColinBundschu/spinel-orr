import {
  Matrix, matrix, multiply, reshape,
} from 'mathjs';
import { DocumentSnapshot, Timestamp } from 'firebase/firestore';
import Lattice, { LatticeType } from './Lattice.ts';
import SiteLabel from '../Plots/SiteLabel.ts';
import {
  ATOMIC_SYMBOLS, MAX_RELAX_dE_eV, MAX_FORCE_eVpA, MinType,
} from '../constants.ts';
import { scaleLinear } from 'd3-scale';

function mapCharToMinType(char: string): MinType {
  switch (char) {
    case 'e':
      return MinType.ElecNotConverged;
    case 'E':
      return MinType.ElecConverged;
    case 's':
      return MinType.SCFNotConverged;
    case 'S':
      return MinType.SCFConverged;
    case 'f':
      return MinType.FluidNotConverged;
    case 'F':
      return MinType.FluidConverged;
    case 'I':
      return MinType.Ionic;
    case 'L':
      return MinType.Lattice;
    default:
      throw new Error(`Unknown MinType character: ${char}`);
  }
}

export interface Atom {
  elem: string;
  x: number;
  y: number;
  z: number;
}

export function distanceBetweenAtoms(atom1: Atom, atom2: Atom): number {
  const dx = atom1.x - atom2.x;
  const dy = atom1.y - atom2.y;
  const dz = atom1.z - atom2.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export interface MinimizationPoint {
  Etot_eV: number;
  time_s: number;
  iter: number;
  type: MinType;
  forces_sum_L2_eVpA?: number;
  forces_max_L2_eVpA?: number;
}

export interface CalcData {
  // Required fields
  Status: string;
  Geo: string;
  Last_modified_utc_ts: Timestamp;
  atoms_num: number[];
  atoms_xyz: number[];

  // Fields which may not always be present
  Lattice_abc?: [number, number, number];
  Lattice_type?: string;
  Etot_eV?: number;
  Forces_max_L2_eVpA?: number;
  stress_eVpA3?: number[];
  Bond_length_A?: number;
  Status_error?: string;

  // Fields that are either all present or all absent
  min_Etots_eV?: number[];
  min_iters?: number[];
  min_times_s?: number[];
  min_types?: string;
  // New fields that may not always be present
  min_forces_sum_L2_eVpA?: number[];
  min_forces_max_L2_eVpA?: number[];
  min_dft_utc_timestamps_s?: (number | null)[];
}

export default class Calc {
  // Required fields
  readonly snapshot: DocumentSnapshot;
  readonly adsorbate: string;
  readonly material: string;

  // Required if the document exists
  readonly status: string | null;
  readonly geo: string | null;
  readonly lastModified: Date | null;

  // Required if the document exists and is not in an error state
  readonly atoms: Atom[] | null;
  readonly lattice: Lattice | null;

  // Optional fields if the document exists
  readonly Etot_eV: number | null;
  readonly L2MaxForce_eVpA: number | null;
  readonly minData: MinimizationPoint[] | null;
  readonly stress_eVpA3: Matrix | null;
  readonly statusError: string | null;

  // Fields that don't exist on all calcs
  readonly bond_length_A?: number;

  constructor(snapshot: DocumentSnapshot) {
    this.lastModified = null;
    this.status = null;
    this.geo = null;
    this.atoms = null;
    this.lattice = null;
    this.minData = null;

    this.snapshot = snapshot;
    [, this.material, , this.adsorbate] = snapshot.ref.path.split('/');
    let data: CalcData | null = null;
    let tOffset_s = 0;
    if (snapshot.exists()) {
      data = snapshot.data() as CalcData;

      if (data.Last_modified_utc_ts == null) throw new Error(`Calc ${snapshot.ref.path} missing Last_modified_utc_ts`);
      this.lastModified = data.Last_modified_utc_ts.toDate();

      if (data.Status == null) throw new Error(`Calc for ${snapshot.ref.path} is missing Status`);
      this.status = data.Status;

      if (data.Geo == null) throw new Error(`Calc for ${snapshot.ref.path} is missing Geo`);
      this.geo = data.Geo;

      if (!['PythonError', 'Error'].includes(this.status) && data.Lattice_abc != null && data.Lattice_type != null) {
        const [a, b, c] = data.Lattice_abc;
        const latticeType = LatticeType[data.Lattice_type as keyof typeof LatticeType];
        this.lattice = new Lattice(latticeType, a, b, c);

        if (data.atoms_num == null || data.atoms_xyz == null) {
          throw new Error(`Calc for ${snapshot.ref.path} is missing atom data`);
        }
        // data.atoms_xyz is a flat array of x, y, z coordinates, going x0, y0, z0,  x1, y1, z1, ...
        this.atoms = data.atoms_num.map((elem_num, index) => {
          const xyz_frac = data!.atoms_xyz.slice(index * 3, index * 3 + 3);
          const xyz = multiply(this.lattice!.R_A, xyz_frac).toArray();
          return {
            elem: ATOMIC_SYMBOLS[elem_num],
            x: xyz[0] as number,
            y: xyz[1] as number,
            z: xyz[2] as number,
          };
        });

        // Either all minimization data should be present and the same length, or none should be present
        if (data.min_Etots_eV && data.min_iters && data.min_times_s && data.min_types
            && data.min_Etots_eV.length === data.min_iters.length
            && data.min_iters.length === data.min_times_s.length
            && data.min_times_s.length === data.min_types.length) {
          let tLast_s = 0;
          let lastDftTimestamp: number | null = null;
          this.minData = data.min_Etots_eV.map((Etot_eV, index) => {
            if (data!.min_dft_utc_timestamps_s?.[index] != null
                && data!.min_dft_utc_timestamps_s?.[index] !== lastDftTimestamp) {
              lastDftTimestamp = data!.min_dft_utc_timestamps_s![index];
              tOffset_s = tLast_s;
            }
            tLast_s = data!.min_times_s![index] + tOffset_s;
            return {
              Etot_eV,
              iter: data!.min_iters![index],
              time_s: tLast_s,
              type: mapCharToMinType(data!.min_types!.charAt(index)),
              forces_sum_L2_eVpA: data!.min_forces_sum_L2_eVpA?.[index],
              forces_max_L2_eVpA: data!.min_forces_max_L2_eVpA?.[index],
            };
          });
        } else if ([data.min_Etots_eV, data.min_iters, data.min_times_s, data.min_types].some(Boolean)) {
          throw new Error(`Minimization data in inconsistent state for ${this.material}`);
        }
      }
    }

    this.Etot_eV = data?.Etot_eV ?? null;
    this.L2MaxForce_eVpA = data?.Forces_max_L2_eVpA ?? null;
    this.stress_eVpA3 = data?.stress_eVpA3 ? matrix(reshape(data.stress_eVpA3, [3, 3])) : null;
    this.statusError = data?.Status_error ?? null;

    // Bond length only makes sense on diatomic calcs, so we leave it undefined if it's not present.
    if (data?.Bond_length_A) this.bond_length_A = data.Bond_length_A;
  }

  get localPath(): string {
    return `dft_out/${this.snapshot.ref.path.split('/').slice(1, 4).join('/')}`;
  }

  get localBasenamePath(): string {
    return `${this.localPath}/${this.snapshot.ref.path.split('/').slice(5)}`;
  }

  get jobname(): string {
    const [, material, facet, adsorbate, , basename] = this.snapshot.ref.path.split('/');
    if (facet === 'bulk') return [material, facet, basename].join('__');
    return [material, facet, basename, adsorbate].join('__');
  }

  get statusIcon(): string {
    const oneDay_ms = 24 * 60 * 60 * 1000;
    switch (this.status) {
      case 'Running':
      case 'Ionic_Minimize':
      case 'Lattice_Minimize':
        if (this.lastModified == null || this.lastModified < new Date(Date.now() - oneDay_ms)) {
          return '❓';
        }
        return '🏃';
      case 'Stopped':
        return '✋';
      case 'Dry_Run_Succeeded':
        return '✅';
      case 'Lattice_Converged':
        if (!this.lastHasLowestEtot(MinType.Lattice)) return '☢️';
        if (this.isConverged) return '☑️';
        return '✅';
      case 'Ionic_Converged':
        if (!this.lastHasLowestEtot(MinType.Ionic)) return '☢️';
        if (this.isConverged) return '☑️';
        return '✅';
      case 'Lattice_Not_Converged':
      case 'Ionic_Not_Converged':
        return '❌';
      case 'Failed':
      case 'Insuf_Atomic_Orbitals':
      case 'Wfns_Mismatch':
      case 'Wfns_Save_Failure':
      case 'Input_Parsing_Failed':
      case 'Ionic_Step_Failure':
        return '💀';
      case 'PythonError':
      case 'Error':
        return '⚠️';
      default:
        return '❓';
    }
  }

  get siteLabel(): SiteLabel {
    return new SiteLabel(this.adsorbate);
  }

  get adLabel(): string {
    let fullLabel = this.adsorbate
      .replaceAll('O-xH_', 'O0-xH_')
      .replaceAll('Oct-', 'Oct0-');

    function captureAndRemove(
      input: string,
      pattern: RegExp
    ): [captures: string, cleansed: string] {
      let capturedDigits = '';
      const cleansed = input.replace(pattern, (_, digits: string) => {
        capturedDigits += digits;
        return ''; // remove the matched substring
      });
      return [capturedDigits, cleansed];
    }

    interface TransformationRule {
      pattern: RegExp;
      symbol: string; 
      suffix: string;
    }

    // Put any pattern → “symbol + suffix” rules into a single array:
    const transformations: TransformationRule[] = [
      { pattern: /O(\d+)-xH/g,   symbol: '◌', suffix: '-xH_' },
      { pattern: /Oct(\d+)-xOH/g, symbol: '◈', suffix: '-xOH_' },
      { pattern: /a(\d+)-xOH/g,   symbol: '⛰️', suffix: '-xOH_' },
      { pattern: /a(\d+)-xO/g,   symbol: '⛰️', suffix: '-xO_' },
      { pattern: /b(\d+)-xOH/g,   symbol: '🌉', suffix: '-xOH_' },
      { pattern: /b(\d+)-xO/g,    symbol: '🌉', suffix: '-xO_' },
      { pattern: /h(\d+)-xOH/g,    symbol: '⛳️', suffix: '-xOH_' },
      { pattern: /h(\d+)-xO/g,    symbol: '⛳️', suffix: '-xO_' },
      { pattern: /t(\d+)-xOH/g,    symbol: '▲', suffix: '-xOH_' },
      { pattern: /t(\d+)-xO/g,    symbol: '▲', suffix: '-xO_' },
      { pattern: /o(\d+)-xOH/g,    symbol: '🐙', suffix: '-xOH_' },
      { pattern: /o(\d+)-xO/g,    symbol: '🐙', suffix: '-xO_' },
    ];

    // Start from the original “fullLabel”
    let runningLabel = fullLabel;
    let cleansedString = '';

    // Apply each rule in sequence
    for (const { pattern, symbol, suffix } of transformations) {
      const [digits, updated] = captureAndRemove(runningLabel, pattern);
      runningLabel = updated;
      if (digits.length > 0) {
        cleansedString += ` ${symbol}${digits}${suffix}`;
      }
    }

    // Finally, append the leftover string
    cleansedString += runningLabel;


    cleansedString = cleansedString
      .replaceAll('__', '_')
      .replaceAll('__', '_')
      .replaceAll('__', '_')
      .replaceAll('__', '_')
      .replaceAll('__', '_')
      .replaceAll('__', '_')
      .replaceAll('__', '_');


    let label = cleansedString
      .replaceAll('_', '')
      .replaceAll('Tet', ' ◬')
      .replaceAll('Oct', ' ◈')
      .replaceAll('-', '')
      .replaceAll('x', '*');

    if (label.length === 0) return 'clean';
    return label;
  }

  get roundedMaxForce_meVpA(): number | null {
    return this.L2MaxForce_eVpA != null ? Math.round(1000 * this.L2MaxForce_eVpA) : null;
  }

  get roundedInitForce_meVpA(): number | null {
    const minDataWithForces = this.minData?.filter((point) => point.forces_max_L2_eVpA != null);
    return minDataWithForces?.[0]?.forces_max_L2_eVpA != null
      ? Math.round(1000 * minDataWithForces[0].forces_max_L2_eVpA)
      : null;
  }

  get isRelaxed(): boolean {
    return this.status === 'Lattice_Converged' || this.status === 'Ionic_Converged';
  }
  
  get isConverged(): boolean {
    if (this.L2MaxForce_eVpA == null) return false;
    return (this.L2MaxForce_eVpA < MAX_FORCE_eVpA &&
      ((this.status === 'Lattice_Converged' && this.minimizationEnergyDrop_eV(MinType.Lattice) < MAX_RELAX_dE_eV) ||
      (this.status === 'Ionic_Converged' && this.minimizationEnergyDrop_eV(MinType.Ionic) < MAX_RELAX_dE_eV)));
  }

  get formattedlastModifiedAgo(): string {
    if (this.lastModified == null) return '';
    const milliseconds = new Date().getTime() - this.lastModified.getTime();
    const seconds = milliseconds / 1000;
    const minutes = seconds / 60;
    const hours = minutes / 60;
    const days = hours / 24;
    const months = days / 30; // Rough approximation
    const years = days / 365;
  
    if (years >= 1) {
      return `${years.toFixed(1)}y`;
    } if (months >= 1) {
      return `${months.toFixed(1)}mo`;
    } if (days >= 10) {
      return `${days.toFixed(0)}d`;
    } if (days >= 1) {
      return `${days.toFixed(1)}d`;
    } if (hours >= 10) {
      return `${hours.toFixed(0)}h`;
    } if (hours >= 1) {
      return `${hours.toFixed(1)}h`;
    } if (minutes >= 1) {
      return `${minutes.toFixed(0)}m`;
    }
    return `${seconds.toFixed(0)}s`;
  }

  get lastModifiedColor(): string {
    if (this.lastModified == null) return 'cyan';
    const hours = (new Date().getTime() - this.lastModified.getTime()) / (1000 * 60 * 60);
    const colorScale = scaleLinear<string>()
      .domain([0.025, 1, 72])
      .range(['limegreen', 'aquamarine', 'MediumPurple'])
      .clamp(true);
    return colorScale(hours);
  }

  minimizationEnergyDrop_eV(targetType: MinType): number {
    const dataOfType = this.minData?.filter((point) => point.type === targetType);
    if (!dataOfType) {
      return Infinity;
    } 
    const maxEtot_eV = Math.max(...dataOfType.map((point) => point.Etot_eV));
    return maxEtot_eV - dataOfType.slice(-1)[0].Etot_eV;
  }

  isLowestEtot(targetType: MinType): boolean {
    const lastInstanceOfType = this.minData?.findLast((point) => point.type === targetType);
    if (!lastInstanceOfType) {
      return false;
    }
    const minEtot_eV = Math.min(...this.minData!.map((point) => point.Etot_eV));
    return lastInstanceOfType.Etot_eV === minEtot_eV;
  }

  lastHasLowestEtot(targetType: MinType): boolean {
    const lastInstanceOfType = this.minData?.findLast((point) => point.type === targetType);
    if (!lastInstanceOfType) {
      return false;
    }
    const minEtot_eV = Math.min(...this.minData!
      .filter((point) => point.type === targetType)
      .map((point) => point.Etot_eV));
    return lastInstanceOfType.Etot_eV === minEtot_eV;
  }
}
