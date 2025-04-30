import {
  useEffect, useMemo, useRef, useState,
} from 'react';
import * as $3Dmol from '3dmol';
import { Box, Slider } from '@mui/material';
import Calc, { Atom, distanceBetweenAtoms } from '../EcatTypes/Calc.ts';
import { ATOMIC_RADII_A, ATOMIC_SYMBOLS } from '../constants.ts';
import Step from '../EcatTypes/Step.ts';

function matchAtomsToRef(atoms: Atom[], referenceAtoms: Atom[]): [Atom[], Atom[]] {
  const unmatchedAtoms = [...atoms];
  const matchedAtoms: Atom[] = [];

  referenceAtoms.forEach((refAtom) => {
    // Find the closest atom of the same type in the calc
    let closestAtomIndex: number | null = null;
    let minDistance = Infinity;

    unmatchedAtoms.forEach((atom, index) => {
      if (atom.elem === refAtom.elem) {
        const distance = Math.sqrt((atom.x - refAtom.x) ** 2 + (atom.y - refAtom.y) ** 2 + (atom.z - refAtom.z) ** 2);
        if (distance < minDistance) {
          minDistance = distance;
          closestAtomIndex = index;
        }
      }
    });

    if (closestAtomIndex == null) throw new Error('Could not match all atoms to the reference atoms.');
    const [matchedAtom] = unmatchedAtoms.splice(closestAtomIndex, 1);
    matchedAtoms.push(matchedAtom);
  });

  return [matchedAtoms, unmatchedAtoms];
}

function matchAtomsToEachOther(
  atoms0: Atom[],
  atoms1: Atom[],
): [Atom[], Atom[], Atom[], Atom[]] {
  const matchedAtoms0: Atom[] = [];
  const matchedAtoms1: Atom[] = [];
  const unmatchedAtoms0: Atom[] = [...atoms0];
  const unmatchedAtoms1: Atom[] = [...atoms1];
  const movementRange_A = 4;
  
  while (true) {
    // Find the closest match, then match them, then loop again for the next closest match
    let closestIndex0: number | null = null;
    let closestIndex1: number | null = null;
    let shortestDistance = Infinity;
    unmatchedAtoms0.forEach((atom0, index0) => {
      unmatchedAtoms1.forEach((atom1, index1) => {
        const distance = distanceBetweenAtoms(atom0, atom1);
        if (distance <= movementRange_A && distance < shortestDistance && atom0.elem === atom1.elem) {
          closestIndex0 = index0;
          closestIndex1 = index1;
          shortestDistance = distance;
        }
      });
    });

    if (shortestDistance !== Infinity) {
      matchedAtoms0.push(unmatchedAtoms0.splice(closestIndex0!, 1)[0]);
      matchedAtoms1.push(unmatchedAtoms1.splice(closestIndex1!, 1)[0]);
    } else {
      // Return when no more matches are found
      return [matchedAtoms0, unmatchedAtoms0, matchedAtoms1, unmatchedAtoms1];
    }
  }
}

function optimizeAtomMatches(
  atoms0: Atom[],
  atoms1: Atom[],
) {
  let swapFound = true;
  while (swapFound) {
    swapFound = false;
    atoms0.forEach((atom0A, indexA) => {
      atoms0.forEach((atom0B, indexB) => {
        const currentDist = distanceBetweenAtoms(atom0A, atoms1[indexA]) + distanceBetweenAtoms(atom0B, atoms1[indexB]);
        const swappedDist = distanceBetweenAtoms(atom0A, atoms1[indexB]) + distanceBetweenAtoms(atom0B, atoms1[indexA]);
        if (swappedDist < currentDist && !swapFound) {
          atoms0[indexA] = atom0B;
          atoms0[indexB] = atom0A;
          swapFound = true;
        }
      })
    })
  }
}

function makeStepCalcsComparable(start: Step, end: Step, reference: Step): [Calc, Calc] {
  const startAtoms = [...start.calc.atoms!];
  const endAtoms = [...end.calc.atoms!];
  const referenceAtoms = [...reference.calc.atoms!];

  // Sort atoms by z-coordinate, this ensures we are comparing atoms from bottom to top
  startAtoms.sort((a, b) => a.z - b.z);
  endAtoms.sort((a, b) => a.z - b.z);
  referenceAtoms.sort((a, b) => a.z - b.z);

  const [baseStartAtoms, baseUnmatchedStartAtoms] = matchAtomsToRef(startAtoms, referenceAtoms);
  const [baseEndAtoms, baseUnmatchedEndAtoms] = matchAtomsToRef(endAtoms, referenceAtoms);
  const [matchedStartAtoms, unmatchedStartAtoms, matchedEndAtoms, unmatchedEndAtoms] = (
    matchAtomsToEachOther(baseUnmatchedStartAtoms, baseUnmatchedEndAtoms));
  optimizeAtomMatches(matchedStartAtoms, matchedEndAtoms);
  baseStartAtoms.push(...matchedStartAtoms);
  baseEndAtoms.push(...matchedEndAtoms);

  // Add the missing atoms to the base lists at consistent positions
  unmatchedStartAtoms.forEach((atom) => {
    // Add missing atoms to end base, at same x and y, but z + 5
    const adjustedAtom = { ...atom, z: atom.z + 5 };
    baseEndAtoms.push(adjustedAtom);
    baseStartAtoms.push(atom);
  });

  unmatchedEndAtoms.forEach((atom) => {
    // Add missing atoms to start base, at same x and y, but z + 5
    const adjustedAtom = { ...atom, z: atom.z + 5 };
    baseStartAtoms.push(adjustedAtom);
    baseEndAtoms.push(atom);
  });

  // Create new `Calc` objects with the updated atom lists
  const newStartCalc = { ...start.calc, atoms: baseStartAtoms } as Calc;
  const newEndCalc = { ...end.calc, atoms: baseEndAtoms } as Calc;

  return [newStartCalc, newEndCalc];
}

interface PathwayViewerProps {
  mepSteps: Step[];
  reference: Step;
  width: number;
}

export default function PathwayViewer({
  mepSteps, reference, width,
}: PathwayViewerProps): JSX.Element {
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const modelRef = useRef<$3Dmol.GLModel | null>(null);
  const viewerInstanceRef = useRef<$3Dmol.GLViewer | null>(null);
  const [interpolationProgress, setInterpolationProgress] = useState(0);

  const calcPairs: [Calc, Calc][] = useMemo(() => mepSteps.reduce<[Calc, Calc][]>((acc, _, index, arr) => {
    if (index < arr.length - 1) acc.push(makeStepCalcsComparable(arr[index], arr[index + 1], reference));
    return acc;
  }, []), [mepSteps, reference]);

  const frames_per_pair = 10;
  const frames = calcPairs.length * frames_per_pair + 1;

  const interpolatedAtoms = useMemo(() => Array.from({ length: frames }, (_, i) => {
    const progress = (i % frames_per_pair) / frames_per_pair;
    const pairIndex = Math.floor(i / frames_per_pair) % calcPairs.length;
    const [calc1, calc2] = calcPairs[pairIndex];
    return calc1.atoms!.map((atom1, j) => {
      const atom2 = calc2.atoms![j];
      return {
        elem: atom1.elem,
        x: atom1.x + progress * (atom2.x - atom1.x),
        y: atom1.y + progress * (atom2.y - atom1.y),
        z: atom1.z + progress * (atom2.z - atom1.z),
      };
    });
  }), [calcPairs]);

  useEffect(() => {
    if (viewerRef.current) {
      if (!viewerInstanceRef.current) {
        viewerInstanceRef.current = $3Dmol.createViewer(viewerRef.current, {
          backgroundColor: 'white',
        });
      }

      const viewer = viewerInstanceRef.current;

      if (!modelRef.current) {
        const initialAtoms = interpolatedAtoms[0];
        const moleculeString = `${initialAtoms.length}\n\n${initialAtoms.map(
          (atom) => `${atom.elem} ${atom.x} ${atom.y} ${atom.z}`,
        ).join('\n')}`;

        modelRef.current = viewer.addModel(moleculeString, 'xyz');
        modelRef.current.setStyle({}, { stick: { radius: 0.2, colorscheme: 'default' } });

        initialAtoms.forEach((atom, i) => {
            modelRef.current!.setStyle(
              { serial: i },
              {
                sphere: { radius: 0.4 * ATOMIC_RADII_A[ATOMIC_SYMBOLS.indexOf(atom.elem)] },
                stick: { radius: 0.1, colorscheme: 'default' },
              },
            );
        });

        viewer.zoomTo();
        viewer.rotate(-75, 'x');
        viewer.rotate(30, 'z');
        viewer.translate(0, -50);
      }

      viewer.render();
    }
  }, [interpolatedAtoms]);

  useEffect(() => {
    if (modelRef.current && viewerInstanceRef.current) {
      const viewer = viewerInstanceRef.current;
      const currentAtoms = interpolatedAtoms[interpolationProgress];

      // Remove the previous model
      viewer.removeAllModels();

      // Add the updated coordinates as a new model
      const moleculeString = `${currentAtoms.length}\n\n${currentAtoms.map(
        (atom) => `${atom.elem} ${atom.x} ${atom.y} ${atom.z}`,
      ).join('\n')}`;

      modelRef.current = viewer.addModel(moleculeString, 'xyz');
      modelRef.current.setStyle({}, { stick: { radius: 0.2, colorscheme: 'default' } });

      currentAtoms.forEach((atom, i) => {
          modelRef.current!.setStyle(
            { serial: i },
            {
              sphere: { radius: 0.4 * ATOMIC_RADII_A[ATOMIC_SYMBOLS.indexOf(atom.elem)] },
              stick: { radius: 0.1, colorscheme: 'default' },
            },
          );
      });

      viewer.render();
    }
  }, [interpolationProgress, interpolatedAtoms]);

  // Generate marks for slider with ticks only at numbered steps
  const marks = useMemo(() => (
    Array.from({ length: calcPairs.length + 1 }, (_, i) => ({
      value: i * frames_per_pair,
      label: (i + 1).toString(),
    }))
  ), [calcPairs.length, frames_per_pair]);

  return (
    <Box
      sx={{
        width: `${width}px`,
        height: `${width}px`,
        position: 'relative',
        marginTop: '20px',
        marginBottom: '20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      <div ref={viewerRef} style={{ width: '100%', height: '100%' }} />

      <Slider
        value={interpolationProgress}
        onChange={(_event, newValue) => setInterpolationProgress(newValue as number)}
        min={0}
        max={frames - 1}
        step={1}
        marks={marks}
        sx={{ width: '80%', marginTop: '10px' }}
      />
    </Box>
  );
}
