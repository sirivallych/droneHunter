import { Container, Box, Typography, Button, Stack, Paper, CssBaseline, Snackbar, Alert } from '@mui/material'
import { useMemo, useRef, useState, useEffect } from 'react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import axios from 'axios'
import UploadCard from './components/UploadCard'
import LoadingOverlay from './components/LoadingOverlay'
import Dashboard from './components/Dashboard'
import VideoPlayer from './components/videoPlayer'

// cast to any to avoid type issues if vite types aren't picked up by tooling
const API_BASE = (import.meta as any).env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadedName, setUploadedName] = useState<string | null>(null)
  const [inputId, setInputId] = useState<string | null>(null)
  const [processing, setProcessing] = useState(false)
  const [progressMsg, setProgressMsg] = useState('')
  const [outputName, setOutputName] = useState<string | null>(null)
  const [outputId, setOutputId] = useState<string | null>(null)
  const [metricsPath, setMetricsPath] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<any>(null)
  const [alertOpen, setAlertOpen] = useState(false)
  const [alertMessage, setAlertMessage] = useState('')
  const [alertSeverity, setAlertSeverity] = useState<'success' | 'info' | 'warning' | 'error'>('info')
  const audioCtxRef = useRef<AudioContext | null>(null)
  const oscillatorRef = useRef<OscillatorNode | null>(null)
  const gainNodeRef = useRef<GainNode | null>(null)

  const ensureAudioContext = () => {
    try {
      if (!audioCtxRef.current) {
        // @ts-ignore - Safari prefix
        const Ctx = window.AudioContext || (window as any).webkitAudioContext
        audioCtxRef.current = new Ctx()
      }
      if (audioCtxRef.current?.state === 'suspended') {
        audioCtxRef.current.resume()
      }
    } catch {}
  }

  // Stop beep when alert is closed
  useEffect(() => {
    if (!alertOpen) {
      // Stop the continuous beep when alert closes
      if (oscillatorRef.current) {
        try {
          oscillatorRef.current.stop()
          oscillatorRef.current = null
        } catch {}
      }
      if (gainNodeRef.current) {
        try {
          gainNodeRef.current.disconnect()
          gainNodeRef.current = null
        } catch {}
      }
    }
  }, [alertOpen])

  const canStart = useMemo(() => !!uploadedName && !processing, [uploadedName, processing])

  const onUpload = async (file: File) => {
    ensureAudioContext()
    setSelectedFile(file)
    setProcessing(true)
    setProgressMsg('Uploading...')
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post(`${API_BASE}/upload`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setUploadedName(res.data.filename)
      setInputId(res.data.input_id || null)
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Upload failed')
    } finally {
      setProcessing(false)
      setProgressMsg('')
    }
  }

  const startProcessing = async () => {
    ensureAudioContext()
    if (!uploadedName) return
    setProcessing(true)
    setProgressMsg('Analyzing Video...')
    try {
      const form = new FormData()
      form.append('filename', uploadedName)
      if (inputId) form.append('input_id', inputId)
      const res = await axios.post(`${API_BASE}/process`, form)
      setOutputName(res.data.output)
      setOutputId(res.data.output_id || null)
      
      // Always show alert with detection status
      const detectionProb = res.data.detection_probability ?? 0
      const droneDetected = res.data.drone_detected ?? false
      const videoType = res.data.video_type || 'Unknown'
      
      // Set alert message and severity based on detection status
      if (detectionProb > 20) {
        setAlertMessage(`Drone Alert: Detected with ${detectionProb}% probability (${videoType} video)`)
        setAlertSeverity('warning')
        setAlertOpen(true)
        // Play continuous beep for high probability detections - will stop when alert closes
        try {
          const ctx = audioCtxRef.current
          if (!ctx) throw new Error('no audio context')
          
          // Clean up any existing beep
          if (oscillatorRef.current) {
            try {
              oscillatorRef.current.stop()
            } catch {}
          }
          if (gainNodeRef.current) {
            try {
              gainNodeRef.current.disconnect()
            } catch {}
          }
          
          const o = ctx.createOscillator()
          const g = ctx.createGain()
          o.type = 'sine'
          o.frequency.value = 880
          o.connect(g)
          g.connect(ctx.destination)
          g.gain.setValueAtTime(0.001, ctx.currentTime)
          g.gain.exponentialRampToValueAtTime(0.2, ctx.currentTime + 0.01)
          o.start()
          
          // Store references so we can stop it when alert closes
          oscillatorRef.current = o
          gainNodeRef.current = g
        } catch {}
      } else if (droneDetected) {
        setAlertMessage(`Detection Status: Drone detected with ${detectionProb}% probability (${videoType} video)`)
        setAlertSeverity('info')
        setAlertOpen(true)
      } else {
        setAlertMessage(`Detection Status: No drone detected (${videoType} video)`)
        setAlertSeverity('success')
        setAlertOpen(true)
      }
      if (res.data.metrics) {
        // backend now returns just the metrics filename (basename)
        setMetricsPath(res.data.metrics)
        const metricsRes = await axios.get(`${API_BASE}/download/metrics/${res.data.metrics}`)
        setMetrics({
          ...metricsRes.data,
          video_type: res.data.video_type,   // include video_type from backend response
          drone_detected: res.data.drone_detected ?? metricsRes.data.drone_detected ?? false,
          detection_probability: res.data.detection_probability ?? metricsRes.data.detection_probability ?? 0
        })
      } else {
        setMetrics(null)
      }
      console.log(metrics)
      // Refresh library
      try {
        const list = await axios.get(`${API_BASE}/videos`)
        setLibrary(list.data || [])
      } catch {}
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Processing failed')
    } finally {
      setProcessing(false)
      setProgressMsg('')
    }
  }
  const [library, setLibrary] = useState<any[]>([])

  const loadLibrary = async () => {
    try {
      const res = await axios.get(`${API_BASE}/videos`)
      setLibrary(res.data || [])
    } catch {}
  }


  const download = async () => {
    const triggerDownload = (blob: Blob, suggestedName: string) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = suggestedName
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    }

    try {
      if (outputId) {
        const res = await axios.get(`${API_BASE}/download/video/byid/${outputId}` as string, { responseType: 'blob' })
        const name = outputName || 'processed.mp4'
        triggerDownload(res.data, name)
        return
      }
      if (!outputName) return
      const res = await axios.get(`${API_BASE}/download/video/${outputName}` as string, { responseType: 'blob' })
      triggerDownload(res.data, outputName)
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Download failed')
    }
  }

  const theme = createTheme({
    palette: {
      mode: 'light',
      background: {
        default: '#e8ecf1'
      },
      primary: {
        main: '#2563eb',
        dark: '#1d4ed8'
      },
      text: {
        primary: '#1e293b'
      }
    },
    typography: {
      fontFamily: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif'
    },
    shape: {
      borderRadius: 16
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: '1rem',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
            backgroundColor: '#ffffff'
          }
        }
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: '0.75rem',
            textTransform: 'none',
            fontWeight: 600
          }
        }
      }
    }
  })

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 6, minHeight: '100dvh', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <Typography variant="h4" align="center" gutterBottom sx={{ color: 'text.primary' }}>
          Drone Detection & Tracking
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" gutterBottom>
          Upload a video and start detection. We’ll process it and show final results.
        </Typography>

        <Box sx={{ my: 4 }}>
          <UploadCard onUpload={onUpload} disabled={processing} />
        </Box>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center" alignItems="center">
          <Button variant="contained" color="primary" disabled={!canStart} onClick={startProcessing}>
            Start Detection
          </Button>
          <Button variant="outlined" disabled={!outputName} onClick={download}>
            Download Result
          </Button>
        </Stack>

        {/* No video preview while uploading or processing; show only final results */}
        {processing && (
          <Box sx={{ mt: 4 }}>
            <Paper sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body1" color="text.secondary">Processing video, please wait…</Typography>
            </Paper>
          </Box>
        )}

        {!processing && (outputId || outputName) && (
          <Box sx={{ mt: 4 }}>
            <Paper sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>Processing Complete</Typography>
              <Typography variant="body2" color="text.secondary">
                Use the Download button to get the processed video.
              </Typography>
            </Paper>
          </Box>
        )}

        {/* Library */}
        {!processing && (
          <Box sx={{ mt: 4 }}>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
              <Typography variant="h6">Stored Videos</Typography>
              <Button size="small" variant="text" onClick={loadLibrary}>Refresh</Button>
            </Stack>
            {library.length === 0 ? (
              <Paper sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">No stored videos yet.</Typography>
              </Paper>
            ) : (
              <Stack spacing={1}>
                {library.map((item) => (
                  <Paper key={item.id} sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box>
                      <Typography variant="subtitle2">{item.output?.filename || 'processed.mp4'}</Typography>
                      <Typography variant="caption" color="text.secondary">{item.createdAt}</Typography>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                        Type: {item.video_type || 'Unknown'}
                      </Typography>
                    </Box>
                    <Stack direction="row" spacing={1}>
                      
                      {item.output?.id && (
                        <Button size="small" variant="outlined" onClick={async () => {
                          try {
                            const res = await axios.get(`${API_BASE}/download/video/byid/${item.output.id}` as string, { responseType: 'blob' })
                            const name = item.output?.filename || 'processed.mp4'
                            const url = URL.createObjectURL(res.data)
                            const a = document.createElement('a')
                            a.href = url
                            a.download = name
                            document.body.appendChild(a)
                            a.click()
                            a.remove()
                            URL.revokeObjectURL(url)
                          } catch (e: any) {
                            alert(e?.response?.data?.detail || 'Download failed')
                          }
                        }}>Download</Button>
                      )}
                    </Stack>
                  </Paper>
                ))}
              </Stack>
            )}
          </Box>
        )}

{!processing && metrics && (
  <Box sx={{ my: 2 }}>
    <Dashboard metrics={metrics} />

    {/* Video player */}
    {metrics.output_id && (
      <Box sx={{ mt: 3 }}>
        <VideoPlayer videoId={metrics.output_id} />
        <Typography variant="body2" sx={{ mt: 1 }}>
          Video Type: {metrics.video_type || "Unknown"}
        </Typography>
      </Box>
    )}
  </Box>
)}


        <LoadingOverlay open={processing} message={progressMsg || 'Processing video, please wait…'} />
        <Snackbar 
          open={alertOpen} 
          onClose={() => setAlertOpen(false)} 
          anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        >
          <Alert 
            onClose={() => setAlertOpen(false)} 
            severity={alertSeverity} 
            variant="filled" 
            sx={{ width: '100%' }}
          >
            {alertMessage || 'Detection Status'}
          </Alert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  )
}


