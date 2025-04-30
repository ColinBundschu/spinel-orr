export interface Cmap {
  name: string;
  fill: string;
}

export default class SiteLabel {
  readonly adsorbate: string;

  constructor(adsorbate: string) {
    this.adsorbate = adsorbate;
  }

  get colorMap(): Cmap {
    // Sorted in the order that we want to use them, from most specific to least
    const cmap = new Map([
      ['-xH_TetOct', 'Plum'],
      ['-xH_Oct', 'SkyBlue'],
      ['-xH_Tet', 'LightGreen'],
      ['-xH', 'LightCoral'],
      ['TetOct', 'MediumOrchid'],
      ['Oct', 'DodgerBlue'],
      ['Tet', 'ForestGreen'],
      ['clean', 'FireBrick'],
    ]);

    // Find the first match in the list and use that
    const foundEntry = Array.from(cmap).find(([key]) => this.adsorbate.includes(key));
    const [rawName, fill] = foundEntry ?? ['Default', 'darkred'];
    const name = rawName.includes('-xH') ? rawName.replaceAll('-xH', '*H ').replaceAll('_', '') : rawName;
    return { name, fill };
  }
}
