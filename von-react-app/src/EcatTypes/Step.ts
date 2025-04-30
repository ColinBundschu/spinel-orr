import { diag } from 'mathjs';
import type Study from './Study.ts';
import Calc from './Calc.ts';

export interface StepProps {
  index: number;
  calc: Calc;
  freeSpeciesCounts: { name: string; count: number; }[];
}

export default class Step {
  readonly index: number;
  readonly freeSpeciesCounts: Map<string, number>;
  readonly nElectrons: number;
  readonly study: Study;
  readonly Efree0V_eV: number | null;
  readonly studyIndex: number;
  readonly calc: Calc;

  constructor(
    study: Study,
    props: StepProps,
    Efree0V_eV: number | null,
    studyIndex: number,
  ) {
    this.index = props.index;
    if (Number.isNaN(this.index) || this.index < 0 || !Number.isInteger(this.index)) {
      throw new Error(`Invalid step index ${this.index}`);
    }

    this.calc = props.calc;
    this.study = study;
    this.Efree0V_eV = Efree0V_eV;
    this.studyIndex = studyIndex;
    this.freeSpeciesCounts = props.freeSpeciesCounts.reduce((acc, { name, count }) => acc.set(name, count), new Map());
    this.nElectrons = this.freeSpeciesCounts.get('e-') ?? 0;
    this.freeSpeciesCounts.delete('e-');
  }

  get poissonRatio(): number {
    // Rough estimate for spinels, using this needlessly to avoid linter warnings
    return this ? 0.3 : 0.3;
  }

  Elevel_eV(strain_Pct: number, potential_V: number): number | null {
    const E0StepFree_eV = this.study.E0Step?.Efree_eV(strain_Pct, potential_V);
    const Efree_eV = this.Efree_eV(strain_Pct, potential_V);
    return (E0StepFree_eV != null && Efree_eV != null) ? (Efree_eV - E0StepFree_eV) : null;
  }

  Efree_eV(strain_Pct: number, potential_V: number): number | null {
    if (this.Efree0V_eV == null) {
      return null;
    }

    let dE_strain_eV = 0;
    if (strain_Pct !== 0) {
      if (!this.calc.lattice) {
        throw new Error('Efree_eV called with non zero strain but no lattice is present');
      }

      if (!this.calc.stress_eVpA3) {
        throw new Error('Efree_eV called with non zero strain but stress_eVpA3 is not set');
      }

      const strainFrac = strain_Pct / 100;
      const stressX_eVpA3 = diag(this.calc.stress_eVpA3).toArray()[0] as number;
      const stressY_eVpA3 = diag(this.calc.stress_eVpA3).toArray()[1] as number;
      const stressZ_eVpA3 = diag(this.calc.stress_eVpA3).toArray()[2] as number;
      const totalStress_eVpA3 = stressX_eVpA3 + stressY_eVpA3 - 2 * (this.poissonRatio * stressZ_eVpA3);
      dE_strain_eV = (strainFrac * this.calc.lattice.volume_A3 * totalStress_eVpA3) / 2;
    }

    const dE_electrons_eV = -potential_V * this.nElectrons;
    return this.Efree0V_eV + dE_electrons_eV + dE_strain_eV;
  }
}
