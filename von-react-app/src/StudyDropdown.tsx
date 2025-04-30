import {
  useState, useEffect,
} from 'react';
import {
  QueryDocumentSnapshot, DocumentData,
} from 'firebase/firestore';
import {
  ButtonGroup, Button, FormControlLabel, Checkbox, Select, MenuItem, InputLabel, FormControl, SelectChangeEvent,
} from '@mui/material';
import Study from './EcatTypes/Study.ts';
import Calc from './EcatTypes/Calc.ts';

interface Props {
  study: Study | null;
  setStudy: (studySnapshot: QueryDocumentSnapshot<DocumentData> | null, adsorbate: string | null) => Promise<void>;
  selectedCalc: Calc | null;
  exactMatch: boolean;
  setExactMatch: (exactMatch: boolean) => void;
  hideConverged: boolean;
  setHideConverged: (hideConverged: boolean) => void;
  studySnapshots: Record<string, QueryDocumentSnapshot<DocumentData>>;
  filteredStudies: QueryDocumentSnapshot<DocumentData, DocumentData>[];
  setFilteredStudies: (studySnapshots: QueryDocumentSnapshot<DocumentData, DocumentData>[]) => void;
  facet: string;
  setFacet: (facet: string) => void;
}

const filterLabels = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ga', 'Mg', 'Mn', 'Ni', 'Zn'];
const facetOptions = ['111s', '111b', '100s', '100o'];

export default function StudyDropdown({
  study, setStudy, selectedCalc, exactMatch, setExactMatch, hideConverged, setHideConverged, studySnapshots,
  filteredStudies, setFilteredStudies, facet, setFacet,
}: Props): JSX.Element {
  const [selectedFilters, setSelectedFilters] = useState<{[key: string]: boolean}>(
    Object.fromEntries(filterLabels.map((filter) => [filter, false])),
  );

  useEffect(() => {
    const localFilteredStudies = Object.values(studySnapshots).filter((snapshot) => {
      // Check toggle match
      const facetMatches = facetOptions.filter((option) => snapshot.id.includes(option));
      if (facetMatches.length > 1) {
        console.error(`Error: Multiple matches (${facetMatches.join(', ')}) for study: ${snapshot.id}`);
        return false; // Exclude if multiple matches found
      }
      const hasToggleMatch = facetMatches.length === 1 && facetMatches[0] === facet;
  
      // Check other filters
      const matchesFilters = Object.entries(selectedFilters).every(([key, value]) => {
        if (exactMatch) {
          return (!value && !snapshot.id.includes(key)) || (value && snapshot.id.includes(key));
        }
        return !value || snapshot.id.includes(key);
      });
  
      return hasToggleMatch && matchesFilters;
    });
  
    // Update filteredStudies state
    setFilteredStudies(localFilteredStudies);
  }, [studySnapshots, selectedFilters, exactMatch, facet, setFilteredStudies]);
  

  const handleStudySelect = async (event: SelectChangeEvent<string>) => {
    const newStudyId = event.target.value;
    setStudy(newStudyId === '' ? null : studySnapshots[newStudyId], selectedCalc?.adsorbate ?? null);
  };

  const handleFacetSelect = async (event: SelectChangeEvent<string>) => {
    const newFacet = event.target.value;
    setFacet(newFacet);
    if (study == null) return;

    // Replace the existing facet with the new facet
    const newStudyId = study.id.replace(new RegExp(facetOptions.join('|')), newFacet);
  
    // Update the study with the new ID
    if (newStudyId in studySnapshots) {
      await setStudy(studySnapshots[newStudyId], selectedCalc?.adsorbate ?? null);
    } else {
      console.error(`Study with ID ${newStudyId} does not exist in snapshots`);
    }
  };

  return (
    <>
      <ButtonGroup style={{ margin: '10px' }}>
        <FormControlLabel
          control={
            <Checkbox checked={hideConverged} onChange={(e) => setHideConverged(e.target.checked)} color="primary" />
          }
          label="Hide Converged"
        />
        <FormControlLabel
          control={
            <Checkbox checked={exactMatch} onChange={(e) => setExactMatch(e.target.checked)} color="primary" />
          }
          label="Exact Match"
        />

        {filterLabels.map((filter) => (
          <Button
            key={filter}
            variant={selectedFilters[filter] ? 'contained' : 'outlined'}
            color="primary"
            onClick={() => setSelectedFilters({ ...selectedFilters, [filter]: !selectedFilters[filter] })}
            sx={{ textTransform: 'none' }}
          >
            {filter}
          </Button>
        ))}
      </ButtonGroup>

      <ButtonGroup style={{ margin: '10px' }}>
        {facetOptions.map((option) => (
          <Button
            key={option}
            variant={facet === option ? 'contained' : 'outlined'}
            color="primary"
            onClick={() => handleFacetSelect({ target: { value: option } } as SelectChangeEvent<string>)}
            sx={{ textTransform: 'none' }}
          >
            {option}
          </Button>
        ))}
      </ButtonGroup>

      <FormControl style={{ marginTop: '10px', minWidth: 150 }}>
        <InputLabel id="study-select-label">Select a Study</InputLabel>
        <Select
          labelId="study-select-label"
          value={study?.id && filteredStudies.map((s) => s.id).includes(study?.id) ? study?.id : ''}
          label="Select a Study"
          onChange={handleStudySelect}
          sx={{ marginBottom: '20px' }}
        >
          {filteredStudies.length === 0
          && (
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
          )}
          {filteredStudies.map((studySnapshot) => (
            <MenuItem key={studySnapshot.id} value={studySnapshot.id}>
              {studySnapshot.id.replace('_100s_5x3', '').replace('_100o_5x3', '').replace('_111s_8x5', '').replace('_111b_9x4', '')}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
    </>
  );
}
