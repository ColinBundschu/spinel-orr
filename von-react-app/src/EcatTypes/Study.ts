import {
  DocumentData, DocumentReference,
  QueryDocumentSnapshot,
  getDoc,
} from 'firebase/firestore';
import * as math from 'mathjs';
import {
  ANGSTROM2_GRAM_PER_EV_SEC2,
  AVOGADRO,
  h2_A2eVu,
  hbar_eVs, hc_eVm, kB_eVpK,
} from '../constants.ts';
import Calc from './Calc.ts';
import FreeSpecies from './FreeSpecies.ts';
import Step, { StepProps } from './Step.ts';

interface StudyProps {
  Base_dft_ref: DocumentReference;
  Layers: number;
  Facet: string;
  Material: string;
  free_species_dft_ref: (DocumentReference | null)[];
  free_species_molar_mass_gpMol: (number | null)[];
  free_species_name: string[];
  free_species_phase: (string | null)[];
  step_dft_ref: DocumentReference[];
  step_free_species_counts: number[];
  step_index: number[];
}

function calculateIsCyclicAndVeq(steps: Array<Step>, VeqThreshold_V: number = 0.001):
{ isCyclic: boolean; Veq_V: number | undefined } {
  let Veq_V: number | undefined;
  let isCyclic = false;

  const maxStepIndex = Math.max(...steps.map((step) => step.index));
  const startSteps = steps.filter((step) => step.index === 0);
  const endSteps = steps.filter((step) => step.index === maxStepIndex);
  startSteps.forEach((startStep) => {
    endSteps.forEach((endStep) => {
      if (startStep.calc.snapshot.ref.path === endStep.calc.snapshot.ref.path) {
        // We only need to find one cyclic pair to know that the reaction is cyclic, so we could exit here.
        // However, we want to make sure all cyclic pairs produce the same onset voltage and throw an error
        // if they don't.
        isCyclic = true;
        if (startStep.Efree0V_eV != null && endStep.Efree0V_eV != null) {
          const currVeq_V = (endStep.Efree0V_eV - startStep.Efree0V_eV) / (endStep.nElectrons - startStep.nElectrons);
          if (Veq_V === undefined) {
            Veq_V = currVeq_V;
          } else if (Math.abs(Veq_V - currVeq_V) > VeqThreshold_V) {
            throw new Error('Veq_V is not well defined because it is not constant across all cyclic pairs');
          }
        }
      }
    });
  });

  return { isCyclic, Veq_V };
}

export default class Study {
  readonly id: string;
  readonly baseCalc: Calc;
  readonly layers: number;
  readonly temperature_K: number;
  readonly facet: string;
  readonly material: string;

  readonly freeSpecies: Map<string, FreeSpecies>;
  readonly fsFreeEnergies_eV: Map<string, number | null>; // free energy of each free species in Hartree

  readonly maxStepIndex: number;
  readonly steps: Array<Step>;
  readonly isCyclic: boolean; // whether the reaction end state is the same as the start state
  readonly Veq_V: number | undefined; // equilibrium potential, only defined if reaction is cyclic
  readonly E0Step: Step | null; // lowest energy step at index 0 at Von (assuming no barrier, no strain)

  constructor(
    id: string,
    props: StudyProps,
    baseCalc: Calc,
    freeSpecies: FreeSpecies[],
    allStepProps: StepProps[],
  ) {
    this.id = id;
    this.baseCalc = baseCalc;
    this.layers = props.Layers;
    this.facet = props.Facet;
    this.material = props.Material;
    this.temperature_K = 298;

    this.freeSpecies = freeSpecies.reduce((acc, fs) => acc.set(String(fs), fs), new Map());
    this.fsFreeEnergies_eV = new Map<string, number>();

    // Compute the free energy of each step in Hartree at 0 V by adding its Etot_eV to the
    // free energy of the free species in the step
    this.steps = allStepProps.map((stepProps, index) => {
      const Efree0V_eV = Array.from(stepProps.freeSpeciesCounts)
        .reduce<number | null>((acc, { name, count }) => {
          // Electrons by definition have 0 energy when the potential is 0
          const Efree0VFS_eV = (name === 'e-') ? 0 : this.computeOrFetchFsFreeEnergy_eV(name);
          return (acc == null || Efree0VFS_eV == null) ? null : acc + (count * Efree0VFS_eV);
        }, stepProps.calc.Etot_eV);
      return new Step(this, stepProps, Efree0V_eV, index);
    });

    this.maxStepIndex = Math.max(...this.steps.map((step) => step.index));
    // Determine if the reaction is cyclic. If it is, try to compute the equilibrium potential
    ({ isCyclic: this.isCyclic, Veq_V: this.Veq_V } = calculateIsCyclicAndVeq(this.steps));

    // Set E0Step to the lowest energy step at index 0 at Veq with no barrier or strain
    const index0_steps = this.stepsWithEnergyAtIndex(0, 0);
    this.E0Step = index0_steps ? index0_steps[0] : null;
  }

  get allEnergyStepsHaveStress(): boolean {
    // We don't plot steps without energy, so those are fine to be missing stress
    // But we need all steps with energy to have stress to use the stress slider
    return this.steps.every((step) => step.Efree0V_eV == null || step.calc.stress_eVpA3 != null);
  }

  get numRelaxedOnly(): number {
    return this.uniqueStepsRelaxed().length - this.uniqueStepsConverged().length;
  }

  get numPartialOnly(): number {
    return this.uniqueStepsWithEnergy().length - this.uniqueStepsRelaxed().length;
  }

  get numNoProgress(): number {
    return this.uniqueDftStepsSortedByForces().length - this.uniqueStepsWithEnergy().length;
  }

  stepFromAdsorbateName(adsorbateName: string): Step | null {
    return this.steps.find((step) => step.calc.adsorbate === adsorbateName) ?? null;
  }

  stepsWithEnergyAtIndex(index: number, strain_Pct: number): Array<Step> {
    // Sorted from lowest to highest energy
    return this.steps
      .filter((step) => (
        step.Efree0V_eV != null
        && Number.isFinite(step.Efree0V_eV)
        && !Number.isNaN(step.Efree0V_eV)
        && step.index === index))
      .sort((a, b) => a.Efree_eV(strain_Pct, 0)! - b.Efree_eV(strain_Pct, 0)!);
  }

  stepsWithEnergy(): Array<Step> {
    return this.steps.filter((step) => (
      step.Efree0V_eV != null && Number.isFinite(step.Efree0V_eV) && !Number.isNaN(step.Efree0V_eV)
    ));
  }

  uniqueStepsWithEnergy(): Array<Step> {
    return this.uniqueDftStepsSortedByForces().filter((step) => (
      step.Efree0V_eV != null && Number.isFinite(step.Efree0V_eV) && !Number.isNaN(step.Efree0V_eV)
    ));
  }

  uniqueStepsConverged(): Array<Step> {
    return this.uniqueDftStepsSortedByForces().filter((step) => step.calc.isConverged);
  }

  uniqueStepsRelaxed(): Array<Step> {
    return this.uniqueDftStepsSortedByForces().filter((step) => step.calc.isRelaxed);
  }

  uniqueDftStepsSortedByForces(): Array<Step> {
    const uniqueStepsMap = this.steps.reduce<Map<string, Step>>((acc, step) => {
      if (!acc.has(step.calc.snapshot.ref.path)) {
        acc.set(step.calc.snapshot.ref.path, step);
      }
      return acc;
    }, new Map<string, Step>());
    return Array.from(uniqueStepsMap.values()).sort((a, b) => {
      if (b.calc?.L2MaxForce_eVpA == null) return -1;
      if (a.calc?.L2MaxForce_eVpA == null) return 1;
      if (b.calc.isConverged && !a.calc.isConverged) return -1;
      if (a.calc.isConverged && !b.calc.isConverged) return 1;
      if ((b.calc.status === 'Lattice_Converged' || b.calc.status === 'Ionic_Converged') && !(a.calc.status === 'Lattice_Converged' || a.calc.status === 'Ionic_Converged')) return -1;
      if ((a.calc.status === 'Lattice_Converged' || a.calc.status === 'Ionic_Converged') && !(b.calc.status === 'Lattice_Converged' || b.calc.status === 'Ionic_Converged')) return 1;
      return b.calc.L2MaxForce_eVpA - a.calc.L2MaxForce_eVpA;
    });
  }

  computeOrFetchFsFreeEnergy_eV(fsName: string): number | null {
    const fs: FreeSpecies | undefined = this.freeSpecies.get(fsName);
    if (fs == null) {
      throw new Error(`Cannot compute free energy for ${fsName} because it is not in the study`);
    }

    // Check for a cached value and return it if it exists
    const cached_value = this.fsFreeEnergies_eV.get(fsName);
    if (cached_value !== undefined) {
      return cached_value;
    }

    let fsFreeEnergy_eV: number | null = null;
    if (fs.name === 'H+' && fs.phase === 'aq') {
      const h2FreeEnergy_eV = this.computeOrFetchFsFreeEnergy_eV('H2_gas');
      if (h2FreeEnergy_eV != null) {
        fsFreeEnergy_eV = h2FreeEnergy_eV / 2;
      }
    } else if (fs.name === 'OH-' && fs.phase === 'aq') {
      const h2OFreeEnergy_eV = this.computeOrFetchFsFreeEnergy_eV('H2O_aq');
      const protonFreeEnergy_eV = this.computeOrFetchFsFreeEnergy_eV('H+_aq');
      if (h2OFreeEnergy_eV != null && protonFreeEnergy_eV != null) {
        fsFreeEnergy_eV = h2OFreeEnergy_eV - protonFreeEnergy_eV;
      }
    } else if (fs.molar_mass_gpMol) {
      if (fs?.calc?.Etot_eV == null) {
        throw new Error(`Cannot compute free energy for ${String(fs)} because it has no Etot_eV`);
      }

      const kT_eV = kB_eVpK * this.temperature_K;
      const molecularMass_g = fs.molar_mass_gpMol / AVOGADRO;
      let z_rot = null;
      if (fs.isDiatomic) {
        if (fs.calc.bond_length_A == null) {
          throw new Error(`Cannot compute free energy for ${String(this)} because it has no bond_length_A`);
        }

        // sigma: rotational symmetry number
        const sigma = fs.isHomoDiatomic ? 2 : 1;
        // I_gA2: moment of inertia
        const I_gA2 = (molecularMass_g * (fs.calc.bond_length_A ** 2)) / 4;
        // B_eV: rotational constant (in energy units)
        const B_eV = (ANGSTROM2_GRAM_PER_EV_SEC2 * (hbar_eVs ** 2)) / (2 * I_gA2);

        // Explicitly sum over the angular momentum J states (since H2 is very light)
        const sum = Array.from({ length: 101 }, (_, j) => 100 - j) // Generate an array [100, 99, ..., 0]
          .reduce((acc, j) => acc + (2 * j + 1) * Math.exp(-(B_eV * (j * (j + 1))) / kT_eV), 0);

        z_rot = sum / sigma;
      } else if (fs.name === 'H2O') {
        const sigma = 2; // rotational symmetry number of the Cv2 space group
        // https://ui.adsabs.harvard.edu/abs/1967JChPh..47.2454H/abstract
        const A_pm = 2787.61; // m^-1
        const B_pm = 1450.74; // m^-1
        const C_pm = 928.77; // m^-1
        z_rot = Math.sqrt((Math.PI * (kT_eV ** 3)) / (A_pm * B_pm * C_pm * hc_eVm ** 3)) / sigma;
        // ad-hoc correction of 6 from total entropy of 19 from Table 7 https://doi.org/10.1246/bcsj.50.65
        // Took difference of 300K value gas phase compututation here and then determined this using the difference
        // In energy from the S value in the table (dE = S * T) to back compute the partition function
        z_rot /= 6;
      } else {
        throw new Error(`Not Implemented: Cannot compute free energy for ${String(fs)}`);
      }
      // atomic mass unit = grams per mole
      const lambdaThermal_A = Math.sqrt(h2_A2eVu / (2 * Math.PI * fs.molar_mass_gpMol * kT_eV));
      const numberDensity_pA3 = fs.numberDensity_pA3(this.temperature_K);
      const z_trans = 1 / (numberDensity_pA3 * lambdaThermal_A ** 3);
      const E_trans_eV = -kT_eV * Math.log(z_trans);
      const E_rot_eV = -kT_eV * Math.log(z_rot);

      if (fs?.calc?.Etot_eV != null) {
        // Free energy = Dft energy + translational energy + rotational energy
        fsFreeEnergy_eV = fs.calc.Etot_eV + E_trans_eV + E_rot_eV;
      }
    } else {
      throw new Error(`Not Implemented: Cannot compute free energy for ${String(fs)}`);
    }

    this.fsFreeEnergies_eV.set(fsName, fsFreeEnergy_eV);
    return fsFreeEnergy_eV;
  }

  Von_V(strain_Pct: number, forward: boolean): number | null {
    const turnOffs_V = Array.from({ length: this.maxStepIndex }, (_, i) => {
      const iSteps = this.stepsWithEnergyAtIndex(i, strain_Pct);
      const iNextSteps = this.stepsWithEnergyAtIndex(i + 1, strain_Pct);
      if (iSteps.length === 0 || iNextSteps.length === 0) {
        return forward ? -Infinity : Infinity;
      }
      return iSteps[0].Efree_eV(strain_Pct, 0)! - iNextSteps[0].Efree_eV(strain_Pct, 0)!;
    });

    return forward ? Math.min(...turnOffs_V) : Math.max(...turnOffs_V);
  }

  calculateSteadyStatePopulation(minEs_eV: number[]): number[] {
    const kT_eV = kB_eVpK * this.temperature_K;
    const n = minEs_eV.length - 1;
    const rateMatrix = math.zeros(n, n) as math.Matrix;

    // Populate the rate matrix
    minEs_eV.slice(0, -1).forEach((eI, i) => {
      const iPrev = i === 0 ? n - 1 : i - 1;
      const iNext = i === n - 1 ? 0 : i + 1;
      const dEPrev = Math.max(minEs_eV[iPrev] - minEs_eV[iPrev + 1], -0.1);
      const dENext = Math.max(minEs_eV[i + 1] - eI, -0.1);
      rateMatrix.subset(math.index(i, iPrev), Math.exp(-dEPrev / kT_eV));
      rateMatrix.subset(math.index(i, iNext), Math.exp(-dENext / kT_eV));
      const rowSum = math.sum(rateMatrix.subset(math.index(i, math.range(0, n))));
      rateMatrix.subset(math.index(i, i), -rowSum);
    });

    // Set the last column of the rate matrix to ones to enforce probabilities sum to 1
    // The rate matrix is not full rank, so we can do this without losing information
    // This is the "standard" way to solve for the steady state population in CTMCs
    rateMatrix.subset(math.index(math.range(0, n), n - 1), Array(n).fill(1));
    const b = math.zeros(n, 1) as math.Matrix;
    b.subset(math.index(n - 1, 0), 1);
    // Let pi be the steady state population vector, Q be the rate matrix, and b be the right hand side,
    // and Q* be Q with the last row replaced with 1s
    // Solve pi * Q = 0, pi * 1 = 1 as a linear system Q*^T pi^T = b
    const steadyStatePopulation = math.lusolve(math.transpose(rateMatrix), b).toArray().flat() as number[];
    return steadyStatePopulation;
  }
}

export async function fetchStudy(studySnapshot: QueryDocumentSnapshot<DocumentData>): Promise<Study> {
  const data = studySnapshot.data() as StudyProps;
  const baseCalcPromise = getDoc(data.Base_dft_ref);

  const freeSpeciesPromise = Promise.all(data.free_species_dft_ref.map(async (dft_ref, i) => {
    const calc = dft_ref ? new Calc(await getDoc(dft_ref)) : null;
    const name = data.free_species_name[i];
    const phase = data.free_species_phase[i];
    const molar_mass_gpMol = data.free_species_molar_mass_gpMol[i];
    return new FreeSpecies(name, phase, molar_mass_gpMol, calc);
  }));

  const allStepPropsPromise: Promise<StepProps[]> = Promise.all(data.step_dft_ref.map(async (dft_ref, i) => {
    const freeSpeciesCounts = data.step_free_species_counts
      .slice(i * data.free_species_name.length, (i + 1) * data.free_species_name.length)
      .map((count, j) => {
        const name = data.free_species_name[j];
        const phase = data.free_species_phase[j];
        return { name: phase ? `${name}_${phase}` : name, count };
      })
      .filter((fsc) => fsc.count !== 0);

    return {
      index: data.step_index[i],
      calc: new Calc(await getDoc(dft_ref)),
      freeSpeciesCounts,
    };
  }));

  const baseCalc = new Calc(await baseCalcPromise);
  const freeSpecies = await freeSpeciesPromise;
  const allStepProps = await allStepPropsPromise;
  return new Study(studySnapshot.id, data, baseCalc, freeSpecies, allStepProps);
}
