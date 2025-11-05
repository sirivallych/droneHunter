import { Backdrop, CircularProgress, Typography, Stack } from '@mui/material'

export default function LoadingOverlay({ open, message }: { open: boolean, message?: string }) {
  return (
    <Backdrop open={open} sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Stack alignItems="center" spacing={2}>
        <CircularProgress color="inherit" />
        {message && <Typography>{message}</Typography>}
      </Stack>
    </Backdrop>
  )
}


