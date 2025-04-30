import {
  AVOGADRO, ATOMIC_SYMBOLS_SET, P_ATM_eVpA3, kB_eVpK,
} from '../constants.ts';
import Calc from './Calc.ts';

interface Token {
  symbol: string,
  count: number,
}

function makeToken(symbol: string, countStr: string): Token {
  if (!ATOMIC_SYMBOLS_SET.has(symbol)) {
    throw new Error(`Unknown atomic symbol ${symbol}`);
  }

  const count = countStr ? parseInt(countStr, 10) : 1;
  if (Number.isNaN(count) || count < 1) {
    throw new Error(`Invalid count ${countStr} for atomic symbol ${symbol}`);
  }

  return { symbol, count };
}

/**
 * Tokenizes a molecule into its atomic components.
 * @param {string} molecule - The molecule string.
 * @returns {Token[]} An array of tokens.
 */
function tokenizeMolecule(molecule: string): Token[] {
  const tokens: Token[] = [];
  let symbol = '';
  let countStr = '';
  [...molecule].forEach((char) => {
    // If character is uppercase, it's the start of a new atomic symbol
    if (char >= 'A' && char <= 'Z') {
      if (symbol) {
        tokens.push(makeToken(symbol, countStr));
      }
      symbol = char;
      countStr = '';
    } else if (!symbol) {
      throw new Error(`Atomic symbol must start with uppercase letter, but got ${char} in molecule ${molecule}`);
    } else if (char >= 'a' && char <= 'z') {
      // If character is lowercase, it's part of an atomic symbol
      symbol += char;
    } else if (char >= '0' && char <= '9') {
      // If character is a digit, it's part of a count
      countStr += char;
    } else {
      throw new Error(`Unexpected character ${char} in molecule ${molecule}`);
    }
  });

  // Add the last token
  if (symbol) {
    tokens.push(makeToken(symbol, countStr));
  }

  return tokens;
}

export default class FreeSpecies {
  readonly name: string;
  readonly calc: Calc | null;
  readonly phase: string | null;
  readonly molar_mass_gpMol: number | null;

  constructor(name: string, phase: string | null, molar_mass_gpMol: number | null, calc: Calc | null) {
    this.name = name;
    this.calc = calc;
    this.phase = phase;
    this.molar_mass_gpMol = molar_mass_gpMol;
  }

  toString(): string {
    return this.phase ? `${this.name}_${this.phase}` : this.name;
  }

  /**
   * @return {boolean} Whether the species is a diatomic molecule.
   */
  get isDiatomic(): boolean {
    return this.isHomoDiatomic || this.isHeteroDiatomic;
  }

  /**
   * @return {boolean} Whether the species is a diatomic molecule of a single element.
   */
  get isHomoDiatomic(): boolean {
    const tokens = tokenizeMolecule(this.name);
    return tokens.length === 1 && tokens[0].count === 2;
  }

  /**
   * @return {boolean} Whether the species is a diatomic molecule of two different elements.
   */
  get isHeteroDiatomic(): boolean {
    const tokens = tokenizeMolecule(this.name);
    return tokens.length === 2 && tokens[0].count === 1 && tokens[1].count === 1;
  }

  /**
   * @param {number} temperature_K - The temperature in Kelvin.
   * @return {number} The number density in Angstrom^-3.
   */
  numberDensity_pA3(temperature_K: number): number {
    if (this.phase === 'aq') {
      if (!this.molar_mass_gpMol) {
        throw new Error(`Molar mass of ${String(this)} is not defined.`);
      }
      if (this.name === 'H2O') {
        const density_gpcm3 = 0.997048; // www.wolframalpha.com/input?i=water+density
        const density_gpA3 = density_gpcm3 * 1E-24;
        return (density_gpA3 * AVOGADRO) / this.molar_mass_gpMol;
      }
    }

    if (this.phase === 'gas') {
      let pressure_eVpA3;
      let vapor_pressure_atm = 1;
      if (this.name === 'O2') {
        vapor_pressure_atm = 0.207; // 21 kPa, en.wikipedia.org/wiki/Vapor_pressure
        pressure_eVpA3 = vapor_pressure_atm * P_ATM_eVpA3;
      } else if (this.name === 'H2') {
        // The Standard Hydrogen Electrode (SHE) defines the pressure of H2 to be 1 atm
        pressure_eVpA3 = P_ATM_eVpA3;
      } else if (this.name === 'H2O') {
        vapor_pressure_atm = 0.0227; // 2.3 kPa, en.wikipedia.org/wiki/Vapor_pressure
        pressure_eVpA3 = vapor_pressure_atm * P_ATM_eVpA3;
      } else {
        throw new Error(`Vapor Pressure of ${String(this)} has not yet been implemented.`);
      }

      return pressure_eVpA3 / (kB_eVpK * temperature_K); // from ideal gas law: V/N = P/kT
    }

    throw new Error(`Number Density of ${String(this)} has not yet been implemented.`);
  }
}
