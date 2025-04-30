import { ListItemButton, TextField, Switch, Box } from '@mui/material';
import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import React, { useState } from 'react';
import Calc from '../EcatTypes/Calc.ts';
import Study from '../EcatTypes/Study.ts';
import { CONVERGED_COLOR, convergenceColor } from '../constants.ts';
import ConvergencePlot from './ConvergencePlot.tsx';
import CopyTextBox from './CopyTextBox.tsx';
import MolecularViewer from './MolecularViewer.tsx';

interface ConvergenceTabPanelProps {
  study: Study;
  toolbarHeight: number;
  selectedCalc: Calc | null;
  setSelectedCalc: React.Dispatch<React.SetStateAction<Calc | null>>;
  convLegendToggle: { [key: string]: boolean };
  setConvLegendToggle: React.Dispatch<React.SetStateAction<{ [key: string]: boolean }>>;
  hideConverged: boolean;
}

export default function ConvergenceTabPanel({
  study, toolbarHeight, selectedCalc, setSelectedCalc, convLegendToggle, setConvLegendToggle, hideConverged,
}: ConvergenceTabPanelProps): JSX.Element {
  const [filterText, setFilterText] = useState('');
  const [useSlurm, setUseSlurm] = useState(false);
  const steps = study.uniqueDftStepsSortedByForces();

  const drawerWidth = 400;
  const plotWidth = 500;

  const convergenceMenuItem = (calc: Calc, name: string) => (
    <ListItem key={calc.snapshot.ref.path} disablePadding>
      <ListItemButton
        selected={selectedCalc?.snapshot.ref.path === calc.snapshot.ref.path}
        onClick={() => setSelectedCalc(calc)}
        sx={{
          '&.Mui-selected': {
            backgroundColor: 'primary.hover',
            color: 'white',
            '&:hover': {
              backgroundColor: 'primary.dark',
            },
          },
        }}
      >
        {(calc.lastModified && (new Date().getTime() - calc.lastModified.getTime()) / (1000 * 60 * 60 * 24) < 3) && (
          <Box component="span" sx={{color: calc.lastModifiedColor}}>
            {calc.formattedlastModifiedAgo}
          </Box>
        )}
        <Box component="span">{calc.statusIcon}</Box>
        <Box
          component="span"
          sx={{ textAlign: 'right', marginRight: '1rem' }}
          style={{ color: (calc.isConverged ? CONVERGED_COLOR : convergenceColor(calc.roundedMaxForce_meVpA)) }}
        >
          {calc.roundedMaxForce_meVpA}
        </Box>
        <Box component="span" sx={{ textAlign: 'left' }}>{name}</Box>
      </ListItemButton>
    </ListItem>
  );

  const slurmCommands = {
    cdLocalFolder: selectedCalc ? `cd /mnt/z/${selectedCalc.localPath}` : '',
    catLog: selectedCalc ? `cat /mnt/z/${selectedCalc.localPath}/*slurm*` : '',
    watchOut: selectedCalc ? `watch -n 0.5 "grep -v -e Nonlinear -e SubspaceRotationAdjust -e FillingsUpdate /mnt/z/${selectedCalc.localBasenamePath}.out | tail -n 70"` : '',
    jobInQueue: selectedCalc ? `sq | grep "${selectedCalc.jobname} "` : '',
    delJob: selectedCalc ? `scancel $(sq | grep "${selectedCalc.jobname} " | awk '{print $1}')` : '',
  };
  const pbsCommands = {
    cdLocalFolder: selectedCalc ? `cd ~/${selectedCalc.localPath}` : '',
    catLog: selectedCalc ? `cat ~/${selectedCalc.localPath}/*slurm*` : '',
    watchOut: selectedCalc ? `watch -n 0.5 "grep -v -e Nonlinear -e SubspaceRotationAdjust -e FillingsUpdate ~/${selectedCalc.localBasenamePath}.out | tail -n 70"` : '',
    jobInQueue: selectedCalc ? `qstat | grep "${selectedCalc.jobname}"` : '',
    delJob: selectedCalc ? `qdel $(qstat | grep "${selectedCalc.jobname}" | awk '{print $1}')` : '',
  };

  const commands = useSlurm ? slurmCommands : pbsCommands;

  const matchesFilter = (name: string, filter: string) => {
    const tokens = filter.toLowerCase().split(' ').filter(Boolean);
    return tokens.every((token) => name.toLowerCase().includes(token));
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'row',
        height: `calc(100vh - ${toolbarHeight}px - 50px)`,
        margin: 0,
        padding: 0,
        boxSizing: 'border-box',
      }}
    >
      <Drawer
        variant="permanent"
        open
        ModalProps={{
          keepMounted: true,
        }}
        sx={{
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            overflowY: 'auto',
            height: '100%',
          },
        }}
      >
        <List>
          <Box style={{ height: toolbarHeight }} />
          <Box sx={{ padding: '10px' }}>
            <TextField
              label="Filter Calculations"
              variant="outlined"
              size="small"
              fullWidth
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
          </Box>
          {matchesFilter('bulk', filterText) && convergenceMenuItem(study.baseCalc, 'Bulk')}
          {steps
            .filter((step) => (!hideConverged || !step.calc.isConverged)
              && matchesFilter(step.calc.adsorbate, filterText))
            .map((step) => convergenceMenuItem(step.calc, `${step.studyIndex} ${step.calc.adsorbate}`))}
        </List>
      </Drawer>
      <Box
        sx={{
          flexGrow: 1,
          marginLeft: `${drawerWidth}px`,
          overflowY: 'auto',
          height: '100%',
        }}
      >
        <ConvergencePlot
          calc={selectedCalc}
          dimensions={{ width: plotWidth, height: plotWidth * 0.85 }}
          convLegendToggle={convLegendToggle}
          setConvLegendToggle={setConvLegendToggle}
        />
        {selectedCalc && (
          <>
            {selectedCalc.atoms != null && (
              <MolecularViewer calc={selectedCalc} width={plotWidth - 100} />
            )}
            {selectedCalc.statusError && (
              <Box sx={{ marginLeft: 2, marginBottom: 3 }}>{`Status Error: ${selectedCalc.statusError}`}</Box>
            )}
            <Box sx={{ display: 'flex', alignItems: 'center', marginBottom: 2, marginLeft: 2 }}>
              <Box sx={{ marginRight: 1 }}>PBS</Box>
              <Switch
                checked={useSlurm}
                onChange={() => setUseSlurm(!useSlurm)}
                color="primary"
              />
              <Box sx={{ marginLeft: 1 }}>SLURM</Box>
            </Box>
            <CopyTextBox label="watch tail 70 .out" textToCopy={commands.watchOut} maxWidth={`${plotWidth + 20}px`} />
            <CopyTextBox label="cat .log" textToCopy={commands.catLog} maxWidth={`${plotWidth + 20}px`} />
            <CopyTextBox label="check job in queue" textToCopy={commands.jobInQueue} maxWidth={`${plotWidth + 20}px`} />
            <CopyTextBox label="delete job" textToCopy={commands.delJob} maxWidth={`${plotWidth + 20}px`} />
            <CopyTextBox label="cd out folder" textToCopy={commands.cdLocalFolder} maxWidth={`${plotWidth + 20}px`} />
          </>
        )}
      </Box>
    </Box>
  );
}
