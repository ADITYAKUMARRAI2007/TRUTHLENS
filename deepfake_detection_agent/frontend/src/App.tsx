// deepfake_detection_agent/frontend/src/App.tsx
import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import AnimatedBackground from './components/AnimatedBackground';
import UploadSection from './components/UploadSection';
import ResultsSection from './components/ResultsSection';

export interface DetectionResult {
  status: 'REAL' | 'FAKE' | 'UNCERTAIN';
  confidence: number;          // 0..100
  fileName: string;
  fileType: 'video' | 'audio' | 'document' | 'code';
  explanation: string;
  approved?: boolean;
  originalUrl?: string;
  replayUrl?: string;
}

// DO NOT put "VITE_API_BASE=..." inside code.
// Set it via Netlify env or .env files.
// Fallback to production Render URL.
const API_BASE = import.meta.env.VITE_API_BASE || 'https://truthlens-ispk.onrender.com';

// simple type detection
function detectFileType(file: File): DetectionResult['fileType'] {
  const mt = (file.type || '').toLowerCase();
  const name = (file.name || '').toLowerCase();
  if (mt.startsWith('video/')) return 'video';
  if (mt.startsWith('audio/')) return 'audio';
  if (mt.startsWith('image/')) return 'document';
  if (mt === 'application/pdf' || name.endsWith('.pdf')) return 'document';
  return 'code';
}

function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [isAnimationMode, setIsAnimationMode] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);

  // ping backend to wake Render cold start & confirm CORS
  const prePing = async () => {
    try {
      const r = await fetch(`${API_BASE}/health`, { method: 'GET', mode: 'cors' });
      // helpful during debugging
      console.log('Health:', r.status, await r.text());
    } catch (e) {
      console.warn('Health ping failed:', e);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsDetecting(true);
    await prePing(); // wake Render before big upload

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
        // no credentials needed; CORS is handled server-side
      });

      if (!response.ok) {
        const t = await response.text().catch(() => '');
        console.error('Analyze failed:', response.status, t);
        throw new Error(`Analyze failed: ${response.status} ${t}`);
      }

      const data = await response.json();

      const jid = (data && data.job_id) ? String(data.job_id) : '';
      if (!jid) throw new Error('Missing job_id in analyze response');
      setJobId(jid);

      const prelimStatus: DetectionResult['status'] =
        (data?.result as any) ||
        (data?.prelim_result as any) ||
        'UNCERTAIN';

      const scores = [data?.ai_probability, data?.final_score, data?.video_score, data?.image_score]
        .filter((x: any) => typeof x === 'number') as number[];
      const conf = scores.length ? Math.round(scores[0] * 100) : 0;

      const baseResult: DetectionResult = {
        status: prelimStatus,
        confidence: conf,
        fileName: file.name,
        fileType: detectFileType(file),
        explanation: 'Awaiting admin approval. Once approved, full report will unlock.',
        approved: false,
      };

      setDetectionResult(baseResult);
      setIsAnimationMode(true);
    } catch (err: any) {
      console.error('File upload failed:', err);
      alert(`Failed to analyze file.\nAPI_BASE=${API_BASE}\n${err?.message ?? err}`);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleNewDetection = () => {
    setDetectionResult(null);
    setJobId(null);
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      <AnimatedBackground />
      <div className="relative z-10 flex">
        <Sidebar
          currentPage={currentPage}
          onPageChange={setCurrentPage}
          isAnimationMode={isAnimationMode}
          onAnimationModeToggle={() => setIsAnimationMode(!isAnimationMode)}
        />

        <main className="flex-1 p-8">
          <div className="max-w-6xl mx-auto">
            <header className="mb-12">
              <div className="flex items-center gap-4 mb-4">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
                  TruthLens
                </h1>
                <span className="px-3 py-1 bg-yellow-500/20 border border-yellow-500/30 rounded-full text-yellow-400 text-sm font-medium">
                  Beta v2.2
                </span>
              </div>
              <p className="text-gray-300 text-lg">
                Spot the fakes before they fool you. Built by humans, powered by AI 🤖
              </p>
              <p className="text-xs text-gray-400 mt-2">API_BASE: {API_BASE}</p>
            </header>

            {!detectionResult ? (
              <UploadSection onFileUpload={handleFileUpload} isDetecting={isDetecting} />
            ) : (
              <ResultsSection
                result={detectionResult}
                isAnimationMode={isAnimationMode}
                onNewDetection={handleNewDetection}
                jobId={jobId ?? ''}
                apiBase={API_BASE}
                onApprovedRefresh={(merged) => setDetectionResult(merged)}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;