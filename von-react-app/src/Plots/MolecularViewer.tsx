import { useEffect, useRef } from 'react';
import * as $3Dmol from '3dmol';
import { Box } from '@mui/material';
import Calc from '../EcatTypes/Calc.ts';
import { ATOMIC_RADII_A, ATOMIC_SYMBOLS } from '../constants.ts';

interface MolecularViewerProps {
  calc: Calc;
  width: number;
}

export default function MolecularViewer({ calc, width }: MolecularViewerProps): JSX.Element {
  const viewerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (viewerRef.current && calc.atoms != null) {
      // Clear previous viewer content (optional, good for dynamic updates)
      viewerRef.current.innerHTML = '';

      // Set up the viewer on the target element
      const viewer = $3Dmol.createViewer(viewerRef.current, {
        backgroundColor: 'white', // Set background color for viewer
      });

      // Create the molecule string in XYZ format
      let moleculeString = `${calc.atoms.length}\n\n`;
      calc.atoms.forEach((atom) => {
        moleculeString += `${atom.elem} ${atom.x} ${atom.y} ${atom.z}\n`;
      });

      // Add the model to the viewer using the XYZ format
      const model = viewer.addModel(moleculeString, 'xyz');
      model.setStyle({}, { stick: { radius: 0.2, colorscheme: 'default' } });
      calc.atoms.forEach((atom, i) => {
        model.setStyle(
          { serial: i }, // select the atom by serial number
          {
            sphere: { radius: 0.4 * ATOMIC_RADII_A[ATOMIC_SYMBOLS.indexOf(atom.elem)] },
            stick: { radius: 0.1, colorscheme: 'default' },
          },
        );
      });

      // Zoom to the molecule and render the viewer
      viewer.zoomTo();
      if (calc.geo !== 'bulk') {
        viewer.rotate(-75, 'x');
        viewer.rotate(30, 'z');
        viewer.translate(0, -50);
      }
      viewer.render();
    }
  }, [calc]);

  return (
    <Box
      sx={{
        width: `${width}px`,
        height: '275px',
        position: 'relative',
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {/* The div that will hold the 3Dmol viewer */}
      <div ref={viewerRef} style={{ width: '100%', height: '100%' }} />
    </Box>
  );
}
