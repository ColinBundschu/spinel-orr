import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import {
  DataGrid, GridAlignment, GridColDef, GridRenderCellParams,
} from '@mui/x-data-grid';
import { DocumentData, QueryDocumentSnapshot } from 'firebase/firestore';
import { useEffect, useState, useMemo } from 'react';
import { CONVERGED_COLOR, convergenceColor } from '../constants.ts';
import Study from '../EcatTypes/Study.ts';
import Calc from '../EcatTypes/Calc.ts';
import PieChart from './PieChart'; // Update the path if needed

function customSortComparator(v1: Calc | null, v2: Calc | null): number {
  if (v1?.roundedMaxForce_meVpA == null) return 1;
  if (v2?.roundedMaxForce_meVpA == null) return -1;
  if (v1.roundedMaxForce_meVpA !== v2.roundedMaxForce_meVpA) {
    return v2.roundedMaxForce_meVpA - v1.roundedMaxForce_meVpA;
  }
  if (v1?.roundedInitForce_meVpA == null) return 1;
  if (v2?.roundedInitForce_meVpA == null) return -1;
  if (v1.roundedInitForce_meVpA !== v2.roundedInitForce_meVpA) {
    return v2.roundedMaxForce_meVpA - v1.roundedMaxForce_meVpA;
  }
  if (v1.status == null) return 1;
  if (v2.status == null) return -1;
  return v1.status.localeCompare(v2.status);
}

export interface StudyMetadata {
  studySnapshot: QueryDocumentSnapshot<DocumentData>;
  study: Study;
  Von_V: number;
}

interface DashboardTabPanelProps {
  fetchedStudiesVersion: number;
  fetchedStudies: { [key: string]: Study };
  filteredStudies: QueryDocumentSnapshot<DocumentData, DocumentData>[];
  setStudy: (studySnapshot: QueryDocumentSnapshot<DocumentData> | null, adsorbate: string | null) => Promise<void>;
  queueFetchStudy: (studySnapshot: QueryDocumentSnapshot<DocumentData>) => void;
}

export default function DashboardTabPanel({
  fetchedStudiesVersion, fetchedStudies, filteredStudies, setStudy, queueFetchStudy,
}: DashboardTabPanelProps): JSX.Element {
  const [studies, setStudies] = useState<StudyMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(0);

  const fetchStudies = async () => {
    filteredStudies.forEach((studySnapshot) => {
      if (!(studySnapshot.id in fetchedStudies)) {
        queueFetchStudy(studySnapshot);
        setIsLoading(1);
      }
    });
  };

  const pieData = useMemo(() => {
    const totals = { converged: 0, relaxed: 0, partial: 0, noProgress: 0 };

    studies.forEach(({ study }) => {
      totals.converged += study.uniqueStepsConverged().length;
      totals.relaxed += study.numRelaxedOnly;
      totals.partial += study.numPartialOnly;
      totals.noProgress += study.numNoProgress;
    });

    return [
      { label: 'Converged', value: totals.converged },
      { label: 'Relaxed', value: totals.relaxed },
      { label: 'Partial', value: totals.partial },
      { label: 'No Progress', value: totals.noProgress },
    ];
  }, [studies]);

  function customRenderCell(params: GridRenderCellParams) {
    const calc: Calc | null = params.value;
    if (calc?.roundedMaxForce_meVpA == null) return null;

    const handleClick = () => {
      if (params.row?.studySnapshot) {
        setStudy(params.row.studySnapshot, params.colDef.headerName === 'base' ? 'bulk' : calc.adsorbate);
      }
    };

    return (
      <Button
        onClick={handleClick}
        variant="contained"
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          textTransform: 'none',
          borderRadius: 0,
          padding: 0,
          backgroundColor: '#191919',
          cursor: 'pointer',
          color: 'inherit',
        }}
      >
        <span>{calc.statusIcon}</span>
        <span style={{ color: (calc.isConverged ? CONVERGED_COLOR : convergenceColor(Number(calc.roundedMaxForce_meVpA))) }}>
          {calc.roundedMaxForce_meVpA}
        </span>
        (
        <span style={{ color: convergenceColor(Number(calc.roundedInitForce_meVpA)) }}>
          {calc.roundedInitForce_meVpA}
        </span>
        )
      </Button>
    );
  }

  useEffect(() => {
    setIsLoading(0);
  }, [filteredStudies]);

  useEffect(() => {
    let waitingOnStudies = false;
    const loadedStudies = filteredStudies.reduce((metadata, studySnapshot) => {
      if (studySnapshot.id in fetchedStudies) {
        if (studySnapshot.id.endsWith('_100o_5x3')) {
          const study = fetchedStudies[studySnapshot.id];
          const loadedStudy = {
            studySnapshot,
            study,
            Von_V: study.Von_V(0, true),
          } as StudyMetadata;
          metadata.push(loadedStudy);
        }
      } else {
        waitingOnStudies = true;
      }
      return metadata;
    }, [] as StudyMetadata[]);
    if (!waitingOnStudies) {
      setIsLoading(2);
    }
    setStudies(loadedStudies.sort((a, b) => b.Von_V - a.Von_V));
  }, [filteredStudies, fetchedStudiesVersion]);

  const getButtonText = () => {
    switch (isLoading) {
      case 1:
        return 'Fetching...';
      case 2:
        return 'All Studies Loaded';
      default:
        return 'Fetch Studies';
    }
  };

  const adsorbates = ['clean'];
  const columns: GridColDef[] = [
    {
      field: 'studyId',
      headerName: 'Study ID',
      width: 150,
      align: 'left',
      headerAlign: 'left',
    },
    ...['base'].concat(adsorbates).map((name) => ({
      field: name,
      headerName: name,
      align: 'left' as GridAlignment,
      headerAlign: 'left' as GridAlignment,
      width: 110,
      sortComparator: customSortComparator,
      renderCell: customRenderCell,
    })),
    {
      field: 'converged',
      headerName: 'Converged',
      width: 80,
      align: 'center',
      headerAlign: 'center',
    },
    {
      field: 'relaxed',
      headerName: 'Relaxed',
      width: 80,
      align: 'center',
      headerAlign: 'center',
    },
    {
      field: 'partial',
      headerName: 'Partial',
      width: 80,
      align: 'center',
      headerAlign: 'center',
    },
    {
      field: 'noProgress',
      headerName: 'No Progress',
      width: 80,
      align: 'center',
      headerAlign: 'center',
    },
    {
      field: 'total',
      headerName: 'Total',
      width: 80,
      align: 'center',
      headerAlign: 'center',
    },
    {
      field: 'Von_V',
      headerName: 'Von (V)',
      width: 80,
      align: 'center',
      headerAlign: 'center',
      renderCell: (params) => (
        <span>
          {params.value?.toFixed(2) ?? 'N/A'}
        </span>
      ),
    },
  ];

  const rows = studies.map(({
    studySnapshot, study, Von_V,
  }) => ({
    id: studySnapshot.id,
    studySnapshot,
    study,
    converged: study.uniqueStepsConverged().length,
    relaxed: study.numRelaxedOnly,
    partial: study.numPartialOnly,
    noProgress: study.numNoProgress,
    total: study.uniqueDftStepsSortedByForces().length,
    Von_V,
    studyId: studySnapshot.id.replace('_100o_5x3', ''),
    base: study.baseCalc,
    ...adsorbates.reduce((acc, adsorbate) => ({
      ...acc,
      [adsorbate]: study.steps.find((step) => step.calc.adsorbate === adsorbate)?.calc,
    }), {}),
  }));

  return (
    <Box sx={{
      display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100vh',
    }}
    >
      <PieChart data={pieData} width={200} height={200}/>

      <Button
        variant="contained"
        color="primary"
        onClick={fetchStudies}
        disabled={isLoading !== 0}
        sx={{ marginTop: 1 }}
      >
        {getButtonText()}
      </Button>

      {isLoading === 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', marginTop: 2 }}>
          <CircularProgress />
        </Box>
      )}

      <Box sx={{ marginTop: 2, overflowX: 'auto', width: '100%' }}>
        <DataGrid
          rows={rows}
          columns={columns}
          initialState={{
            pagination: {
              paginationModel: { pageSize: 15 },
            },
          }}
          pageSizeOptions={[15, 50]}
        />
      </Box>
    </Box>
  );
}
