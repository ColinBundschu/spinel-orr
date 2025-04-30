import React, { ChangeEvent, ReactNode } from 'react';
import Box from '@mui/material/Box';
import Slider from '@mui/material/Slider';
import { Mark } from '@mui/base/useSlider';
import MuiInput from '@mui/material/Input';
import Stack from '@mui/material/Stack';

interface SharedSliderProps {
  label: ReactNode;
  value: number;
  callback: (value: number) => void;
  disabled: boolean;
  min: number;
  max: number;
  step: number;
  marks: Mark[] | boolean;
}

export default function SharedSlider({
  label, value, callback, disabled, min, max, step, marks = false,
}: SharedSliderProps) {
  const onSliderChange = (_event: Event | React.SyntheticEvent, newValue: number | number[]): void => {
    if (typeof newValue === 'number') {
      callback(newValue);
    }
  };

  const onInputChange = (event: ChangeEvent<HTMLInputElement>): void => {
    callback(Number(event.target.value));
  };

  return (
    <Box sx={{
      marginTop: 1,
      backgroundColor: '#222',
      paddingBottom: 2,
      paddingRight: 3,
      paddingLeft: 2,
      paddingTop: 0,
      borderRadius: 1,
    }}
    >
      <Stack direction="row" alignItems="center" spacing={3}>
        <Box sx={{ width: 102, textAlign: 'left' }}>
          {label}
        </Box>
        <Box sx={{ paddingBottom: 0.2, paddingRight: 1 }}>
          <MuiInput
            disabled={disabled}
            sx={{
              width: 40,
              '& input::-webkit-outer-spin-button, & input::-webkit-inner-spin-button': {
                WebkitAppearance: 'none',
                margin: 0,
              },
              '& input[type=number]': {
                MozAppearance: 'textfield',
              },
            }}
            value={value}
            margin="dense"
            onChange={onInputChange}
            inputProps={{
              step,
              min,
              max,
              type: 'number',
              'aria-labelledby': 'input-slider',
            }}
          />
        </Box>
        <Box sx={{ paddingTop: 2.5, flex: 1 }}>
          <Slider
            disabled={disabled}
            value={value}
            min={min}
            step={step}
            max={max}
            onChange={onSliderChange}
            valueLabelDisplay="auto"
            sx={{
              '&.MuiSlider-root': {
                padding: '0px',
              },
            }}
            marks={marks || [{ value: 0, label: <div /> }]}
          />
        </Box>
      </Stack>
    </Box>
  );
}
