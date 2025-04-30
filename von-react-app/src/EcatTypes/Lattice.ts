import { Matrix, matrix, det } from 'mathjs';

export enum LatticeType { Cubic, Orthorhombic, FCC, FCO }

export default class Lattice {
  readonly lattice_type: LatticeType;
  readonly a: number;
  readonly b: number | null;
  readonly c: number | null;

  constructor(lattice_type: LatticeType, a: number, b: number | null = null, c: number | null = null) {
    this.lattice_type = lattice_type;
    this.a = a;
    this.b = b;
    this.c = c;

    const lattice_str = `a=${this.a}, b=${this.b}, c=${this.c}`;
    switch (this.lattice_type) {
      case LatticeType.Cubic:
      case LatticeType.FCC:
        if (this.a === null || this.b !== null || this.c !== null) {
          throw new Error(`FCC/Cubic should specify exactly "a": ${lattice_str}`);
        }
        break;
      case LatticeType.FCO:
      case LatticeType.Orthorhombic:
        if (this.a === null || this.b === null || this.c === null) {
          throw new Error(`FCO/Orthorhombic should specify exactly "a", "b", "c": ${lattice_str}`);
        }
        break;
      default:
        throw new Error(`Lattice type ${this.lattice_type} not implemented`);
    }
  }

  get R_A(): Matrix {
    switch (this.lattice_type) {
      case LatticeType.Cubic:
        return matrix([[this.a, 0, 0], [0, this.a, 0], [0, 0, this.a]]);
      case LatticeType.Orthorhombic:
        return matrix([[this.a, 0, 0], [0, this.b!, 0], [0, 0, this.c!]]);
      case LatticeType.FCC:
        return matrix([[0, this.a / 2, this.a / 2], [this.a / 2, 0, this.a / 2], [this.a / 2, this.a / 2, 0]]);
      case LatticeType.FCO:
        return matrix([[0, this.a / 2, this.a / 2], [this.b! / 2, 0, this.b! / 2], [this.c! / 2, this.c! / 2, 0]]);
      default:
        throw new Error(`No lattice vectors defined for ${this.lattice_type}`);
    }
  }

  get volume_A3(): number {
    return det(this.R_A);
  }
}
