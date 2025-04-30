import Box from '@mui/material/Box';
import IconButton from '@mui/material/IconButton';
import TextField from '@mui/material/TextField';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { useSnackbar } from 'notistack';

interface CopyTextBoxProps {
  label: string;
  textToCopy: string;
  maxWidth: string;
}

export default function CopyTextBox({ label, textToCopy, maxWidth }: CopyTextBoxProps) {
  const { enqueueSnackbar } = useSnackbar();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(textToCopy);
      enqueueSnackbar('Text copied to clipboard!', {
        variant: 'success',
        autoHideDuration: 800,
        anchorOrigin: {
          vertical: 'bottom',
          horizontal: 'right',
        },
      });
    } catch (err) {
      enqueueSnackbar('Failed to copy text.', {
        variant: 'error',
        autoHideDuration: 800,
        anchorOrigin: {
          vertical: 'bottom',
          horizontal: 'right',
        },
      });
    }
  };

  return (
    <Box display="flex" alignItems="center" marginBottom={1} sx={{ maxWidth }}>
      <Box
        sx={{
          width: '100%', // Ensure the hover target fills the container
          '.MuiTextField-root:hover': {
            bgcolor: 'action.hover', // Apply hover effect here
          },
          cursor: 'pointer',
        }}
        onClick={handleCopy}
      >
        <TextField
          variant="outlined"
          label={label}
          value={textToCopy}
          InputProps={{
            readOnly: true,
            sx: {
              fontSize: '.8rem',
            },
          }}
          InputLabelProps={{
            sx: {
              fontSize: '1.2rem',
              color: 'orange',
            },
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              pointerEvents: 'none', // Disable pointer events on the input itself
            },
          }}
          fullWidth
        />
      </Box>
      <IconButton onClick={handleCopy} aria-label="copy">
        <ContentCopyIcon />
      </IconButton>
    </Box>
  );
}
