import TabContext from '@mui/lab/TabContext';
import TabList from '@mui/lab/TabList';
import TabPanel from '@mui/lab/TabPanel';
import Box from '@mui/material/Box';
import CssBaseline from '@mui/material/CssBaseline';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import Tab from '@mui/material/Tab';
import {
  collection, DocumentData, getDocs, QueryDocumentSnapshot,
} from 'firebase/firestore';
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef, useState,
} from 'react';
import { useSearchParams } from 'react-router-dom';
import Calc from './EcatTypes/Calc.ts';
import Step from './EcatTypes/Step.ts';
import Study, { fetchStudy } from './EcatTypes/Study.ts';
import { db } from './FirestoreData.tsx';
import ConvergenceTabPanel from './Plots/ConvergenceTabPanel.tsx';
import LevelDiagramTabPanel from './Plots/LevelDiagramTabPanel.tsx';
import StudyDropdown from './StudyDropdown.tsx';
import DashboardTabPanel from './Plots/DashboardTabPanel.tsx';
import PathwayViewer from './Plots/PathwayViewer.tsx';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

export default function App(): JSX.Element {
  const [searchParams, setSearchParams] = useSearchParams();
  const studyIdFromUrl = searchParams.get('studyId'); // Extract snapshot ID from the URL
  const tabFromUrl = searchParams.get('tab') || '1'; // Default to '1' if no tab is in the URL
  const exactMatchFromUrl = searchParams.get('exactMatch') === 'true'; // Implicitly defaults to false
  const hideConvergedFromUrl = searchParams.get('hideConverged') === 'true'; // Implicitly defaults to false
  const facetFromUrl = searchParams.get('facet') || '100o';

  const maxConcurrentFetches = 1;
  const concurrentFetchesRef = useRef(0);
  const pendingStudiesQueue = useRef<QueryDocumentSnapshot<DocumentData>[]>([]);
  const fetchedStudies = useRef<{ [key: string]: Study }>({});
  const [studyId, setStudyId] = useState<string | null>(studyIdFromUrl);
  const [study, setStudyRaw] = useState<Study | null>(null);
  const [studySnapshots, setStudySnapshots] = useState<Record<string, QueryDocumentSnapshot<DocumentData>>>({});
  const [filteredStudies, setFilteredStudies] = useState<QueryDocumentSnapshot<DocumentData, DocumentData>[]>([]);
  const [fetchedStudiesVersion, setFetchedStudiesVersion] = useState<number>(0);

  const toolbarRef = useRef<HTMLDivElement>(null);
  const [tabValue, setTabValue] = useState(tabFromUrl);
  const [levelsLegendToggle, setLevelsLegendToggle] = useState<{ [key: string]: boolean }>({});
  const [selectedCalc, setSelectedCalc] = useState<Calc | null>(null);
  const convStartState = {
    Elec: true, Lattice: true, Ionic: true,
  };
  const [convLegendToggle, setConvLegendToggle] = useState<{ [key: string]: boolean }>(convStartState);
  const [potential_V, setPotential_V] = useState<number>(0);
  const [strain_Pct, setStrain_Pct] = useState<number>(0);
  const [exactMatch, setExactMatch] = useState<boolean>(exactMatchFromUrl);
  const [hideConverged, setHideConverged] = useState<boolean>(hideConvergedFromUrl);
  const [facet, setFacet] = useState<string>(facetFromUrl);

  const getStudy = useCallback(async (studySnapshot: QueryDocumentSnapshot<DocumentData>) => {
    if (!(studySnapshot.id in fetchedStudies.current)) {
      fetchedStudies.current[studySnapshot.id] = await fetchStudy(studySnapshot);
      setFetchedStudiesVersion((prevVersion) => prevVersion + 1);
    }
    return fetchedStudies.current[studySnapshot.id];
  }, []);

  const processNextInQueue = useCallback(() => {
    if (pendingStudiesQueue.current.length > 0 && concurrentFetchesRef.current < maxConcurrentFetches) {
      const nextStudySnapshot = pendingStudiesQueue.current.shift();
      if (nextStudySnapshot) {
        concurrentFetchesRef.current += 1;
        getStudy(nextStudySnapshot).finally(() => {
          concurrentFetchesRef.current -= 1;
          processNextInQueue();
        });
      }
    }
  }, []);

  const queueFetchStudy = useCallback((studySnapshot: QueryDocumentSnapshot<DocumentData>) => {
    pendingStudiesQueue.current.push(studySnapshot);
    processNextInQueue();
  }, [processNextInQueue]);

  const handleTabChange = useCallback((_event: React.SyntheticEvent, newTabValue: string) => {
    setTabValue(newTabValue);
  }, []);

  useEffect(() => {
    setSearchParams({
      studyId: studyId || '',
      adsorbate: selectedCalc?.geo === 'bulk' ? 'bulk' : selectedCalc?.adsorbate || searchParams.get('adsorbate') || '',
      tab: tabValue,
      exactMatch: exactMatch.toString(),
      hideConverged: hideConverged.toString(),
      facet: facet,
    });
  }, [studyId, selectedCalc, tabValue, exactMatch, hideConverged, facet]);

  const setStudy = useCallback(async (
    studySnapshot: QueryDocumentSnapshot<DocumentData> | null,
    adsorbate: string | null,
  ) => {
    setStudyId(studySnapshot?.id || '');
    const newStudy = studySnapshot === null ? null : await getStudy(studySnapshot);
    setStudyRaw(newStudy);
    if (newStudy != null) {
      if (adsorbate === 'bulk') {
        setSelectedCalc(newStudy.baseCalc);
      } else {
        const matchingCalc = newStudy.steps.find((step) => step.calc.adsorbate === adsorbate)?.calc;
        if (matchingCalc != null) setSelectedCalc(matchingCalc);
      }
    }
  }, []);

  // Set the selected study based on studyId from the URL after studies are fetched
  useEffect(() => {
    const fetchStudies = async () => {
      const studiesSnapshot = await getDocs(collection(db, 'studies'));
      const studySnapshotsDict = Object.fromEntries(studiesSnapshot.docs.map((doc) => [doc.id, doc]));
      setStudySnapshots(studySnapshotsDict);

      // If there's a snapshot ID in the URL, set the study to the one matching that ID
      if (studyId && studySnapshotsDict[studyId]) {
        await setStudy(studySnapshotsDict[studyId], searchParams.get('adsorbate'));
      } else if (studiesSnapshot.docs.length > 0) {
        const firstStudyId = studiesSnapshot.docs[0].id;
        await setStudy(studySnapshotsDict[firstStudyId], searchParams.get('adsorbate'));
      }
    };

    fetchStudies();
  }, []);

  function Von_V(strain: number, forward: boolean): number | null {
    const rawVon_V = study?.Von_V(strain, forward);
    return Number.isFinite(rawVon_V) ? (Math.round(100 * rawVon_V!) / 100) : null;
  }

  const Esteps = useMemo<Step[]>(() => (study?.E0Step ? study.stepsWithEnergy() : []), [study]);
  const cmaps = useMemo(
    () => Esteps
      .map((step) => step.calc.siteLabel.colorMap)
      .filter((({ name }, index, array) => array.findIndex((cmap) => cmap.name === name) === index)),
    [Esteps],
  );

  // Calculate the energies for the steps at the current strain and potential
  const mepSteps = useMemo(() => study ? Array.from({ length: study.maxStepIndex + 1 }, (_, i) => {
    const iSteps = Esteps.filter((step) => step.index === i);
    if (iSteps.length === 0) return null;
    return iSteps.reduce((mepStep, step) => (
      mepStep.Elevel_eV(strain_Pct, potential_V)! < step.Elevel_eV(strain_Pct, potential_V)! ? mepStep : step
    ));
  }) : [], [Esteps, study, potential_V, strain_Pct]);

  useEffect(() => {
    setLevelsLegendToggle(cmaps.reduce((acc, cmap) => ({ ...acc, [cmap.name]: true }), {}));
    setConvLegendToggle(convStartState);
    setStrain_Pct(0);
    setPotential_V(study?.Veq_V ?? 1);
  }, [cmaps, study]);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <TabContext value={tabValue}>
        <Box
          ref={toolbarRef}
          style={{
            zIndex: darkTheme.zIndex.drawer + 1,
            position: 'relative',
            backgroundColor: '#282828',
          }}
        >
          <StudyDropdown
            study={study}
            setStudy={setStudy}
            selectedCalc={selectedCalc}
            exactMatch={exactMatch}
            setExactMatch={setExactMatch}
            hideConverged={hideConverged}
            setHideConverged={setHideConverged}
            studySnapshots={studySnapshots}
            filteredStudies={filteredStudies}
            setFilteredStudies={setFilteredStudies}
            facet={facet}
            setFacet={setFacet}
          />
          <TabList onChange={handleTabChange}>
            <Tab label="Level Diagram" value="1" />
            <Tab label="MEP" value="2" />
            <Tab label="Convergence" value="3" />
            <Tab label="Dashboard" value="4" />
          </TabList>
        </Box>

        <TabPanel value="1">
          {study !== null && (
            <LevelDiagramTabPanel
              study={study}
              toolbarHeight={toolbarRef.current?.offsetHeight ?? 0}
              Esteps={Esteps}
              cmaps={cmaps}
              levelsLegendToggle={levelsLegendToggle}
              setLevelsLegendToggle={setLevelsLegendToggle}
              potential_V={potential_V}
              setPotential_V={setPotential_V}
              strain_Pct={strain_Pct}
              setStrain_Pct={setStrain_Pct}
              VonORR_V={Von_V(strain_Pct, true)}
              VonOER_V={Von_V(strain_Pct, false)}
              hideConverged={hideConverged}
              mepSteps={mepSteps}
            />
          )}
        </TabPanel>

        <TabPanel value="2">
        {study !== null && mepSteps.every((step) => (step != null) && (step.calc.status !== 'Error'))
          && (study.stepFromAdsorbateName('clean')?.calc.atoms != null) && (
          <PathwayViewer
            mepSteps={mepSteps as Step[]}
            reference={study.stepFromAdsorbateName('clean')!}
            width={800}
          />
        )}
        </TabPanel>

        <TabPanel value="3">
          {study !== null && (
            <ConvergenceTabPanel
              study={study}
              toolbarHeight={toolbarRef.current?.offsetHeight ?? 0}
              selectedCalc={selectedCalc}
              setSelectedCalc={setSelectedCalc}
              convLegendToggle={convLegendToggle}
              setConvLegendToggle={setConvLegendToggle}
              hideConverged={hideConverged}
            />
          )}
        </TabPanel>

        <TabPanel value="4">
          <DashboardTabPanel
            fetchedStudiesVersion={fetchedStudiesVersion}
            fetchedStudies={fetchedStudies.current}
            filteredStudies={filteredStudies}
            setStudy={setStudy}
            queueFetchStudy={queueFetchStudy}
          />
        </TabPanel>

      </TabContext>
    </ThemeProvider>
  );
}
