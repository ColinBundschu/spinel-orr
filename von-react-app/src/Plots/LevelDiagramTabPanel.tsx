import {
  useState, useEffect, useMemo, useRef,
} from 'react';
import { Mark } from '@mui/base/useSlider';
import Box from '@mui/material/Box';
import { MathJax, MathJaxContext } from 'better-react-mathjax';
import Study from '../EcatTypes/Study.ts';
import LevelDiagram from './LevelDiagram.tsx';
import SharedSlider from './SharedSlider.tsx';
import Step from '../EcatTypes/Step.ts';
import { Cmap } from './SiteLabel.ts';

const markYOffset = -11;

function StrainMark(markStrain_Pct: number): Mark {
  const translateX = markStrain_Pct < 0 ? '-10px' : '-6px';
  return {
    value: markStrain_Pct,
    label: (
      <div style={{ transform: `translate(${translateX}, 0%)`, position: 'absolute', top: markYOffset }}>
        {`${markStrain_Pct}%`}
      </div>
    ),
  };
}

function potentialMark([markLabel, potential_V]: [string | null, number]): Mark {
  const label = markLabel ? `${potential_V} (\\(${markLabel}\\))` : potential_V.toFixed(1);
  const translateX = markLabel ? '-15px' : '-10px';
  return {
    value: potential_V,
    label: (
      <div style={{ transform: `translate(${translateX}, 0%)`, position: 'absolute', top: markYOffset }}>
        <MathJax inline dynamic>{label}</MathJax>
      </div>
    ),
  };
}

interface LevelDiagramTabPanelProps {
  study: Study;
  toolbarHeight: number;
  Esteps: Step[];
  cmaps: Cmap[];
  levelsLegendToggle: {[key: string]: boolean};
  setLevelsLegendToggle: React.Dispatch<React.SetStateAction<{[key: string]: boolean}>>;
  potential_V: number;
  setPotential_V: React.Dispatch<React.SetStateAction<number>>;
  strain_Pct: number;
  setStrain_Pct: React.Dispatch<React.SetStateAction<number>>;
  VonORR_V: number | null;
  VonOER_V: number | null;
  hideConverged: boolean;
  mepSteps: (Step | null)[];
}

const diagramMinHeight = 400;
const diagramWidthRatio = 4 / 3;

export default function LevelDiagramTabPanel({
  study, toolbarHeight, Esteps, cmaps, levelsLegendToggle, setLevelsLegendToggle, potential_V, setPotential_V,
  strain_Pct, setStrain_Pct, VonORR_V, VonOER_V, hideConverged, mepSteps,
}: LevelDiagramTabPanelProps): JSX.Element {
  const slidersRef = useRef<HTMLDivElement>(null);
  const [levelDiagramDimensions, setLevelDiagramDimensions] = useState({
    height: diagramMinHeight, width: diagramMinHeight * diagramWidthRatio,
  });

  // Set the size of the level diagram based on the window size, the toolbar height, and the slider heights
  useEffect(() => {
    const updateDimensions = () => {
      const slidersHeight = slidersRef.current?.offsetHeight ?? 0;
      const emptySpace = window.innerHeight - toolbarHeight - slidersHeight - 70;
      const heightLimitFromWidth = (window.innerWidth - 40) / diagramWidthRatio;
      const diagramHeight = Math.max(diagramMinHeight, Math.min(emptySpace, heightLimitFromWidth));
      if (diagramHeight === levelDiagramDimensions.height) return;
      setLevelDiagramDimensions({ height: diagramHeight, width: diagramHeight * diagramWidthRatio });
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => {
      // Cleanup: remove event listener when component unmounts
      window.removeEventListener('resize', updateDimensions);
    };
  }, [toolbarHeight, slidersRef.current]);

  // Compile the VonORR_V, Veq_V, and VonOER_V values, and create slider marks for the ones that are not null
  const Veq_V = Number.isFinite(study?.Veq_V) ? (Math.round(100 * study.Veq_V!) / 100) : null;
  const Vmid_V = Math.round(100 * (Veq_V ?? 1)) / 100;
  const maxDiff_V = Math.max(Math.abs(Vmid_V - (VonORR_V ?? 0.3)), Math.abs(Vmid_V - (VonOER_V ?? 1.7)));
  const min_V = Math.round(100 * (Vmid_V - maxDiff_V)) / 100 - 0.3;
  const max_V = Math.round(100 * (Vmid_V + maxDiff_V)) / 100 + 0.3;
  const potentialMarks = useMemo(() => (([
    VonORR_V != null ? ['ORR', VonORR_V] : null,
    Veq_V != null ? ['Eq', Veq_V] : [null, Vmid_V],
    VonOER_V != null ? ['OER', VonOER_V] : null,
  ].filter(Boolean) as [string, number][]).map(potentialMark)), [VonORR_V, Veq_V, VonOER_V]);

  return (
    <Box>
      <LevelDiagram
        study={study}
        strain_Pct={strain_Pct}
        potential_V={potential_V}
        dimensions={levelDiagramDimensions}
        mepSteps={mepSteps}
        steps={Esteps}
        cmaps={cmaps}
        levelsLegendToggle={levelsLegendToggle}
        setLevelsLegendToggle={setLevelsLegendToggle}
        hideConverged={hideConverged}
      />
      <Box ref={slidersRef} sx={{ maxWidth: levelDiagramDimensions.width }}>
        <MathJaxContext config={{ tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] } }}>
          <SharedSlider
            label="Potential (V)"
            value={potential_V}
            callback={setPotential_V}
            disabled={study == null}
            min={min_V}
            max={max_V}
            step={0.01}
            marks={potentialMarks}
          />
        </MathJaxContext>
        <SharedSlider
          label="Strain (Ep. %)"
          value={strain_Pct}
          callback={setStrain_Pct}
          disabled={study == null || !study.allEnergyStepsHaveStress}
          min={-3}
          max={3}
          step={0.05}
          marks={[-3, 0, 3].map(StrainMark)}
        />
      </Box>
    </Box>
  );
}
