// src/App.tsx
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
  approved?: boolean;          // gate UI until admin approves
  originalUrl?: string;        // filled after approval from server
  replayUrl?: string;          // filled after approval from server
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8001';

// robust file-type detection (handles PDFs and images reliably)
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

  // store job id from /analyze so ResultsSection can poll /jobs/{id}
  const [jobId, setJobId] = useState<string | null>(null);

  const handleFileUpload = async (file: File) => {
    setIsDetecting(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error(`Failed to analyze file: ${response.statusText}`);

      const data = await response.json();

      // Backend returns: { ok, job_id, status:"PENDING", prelim_result, ... }
      const jid = (data && data.job_id) ? String(data.job_id) : '';
      if (!jid) throw new Error('Missing job_id in analyze response');
      setJobId(jid);

      // prefer explicit result if present; otherwise prelim
      const prelimStatus: DetectionResult['status'] =
        (data?.result as any) ||
        (data?.prelim_result as any) ||
        'UNCERTAIN';

      // confidence: avoid NaN contamination from Number(undefined)
      const rawScores = [
        data?.ai_probability,
        data?.final_score,
        data?.video_score,
        data?.image_score,
      ].map(x => (typeof x === 'number' ? x : undefined));

      const firstScore = rawScores.find(v => typeof v === 'number');
      const confidencePct = Number.isFinite(firstScore as number)
        ? Math.round((firstScore as number) * 100)
        : 0;

      const explanation =
        'Awaiting admin approval. Once approved, the full report (links, replay, logs) will unlock here.';

      const baseResult: DetectionResult = {
        status: prelimStatus,
        confidence: confidencePct,
        fileName: file.name,
        fileType: detectFileType(file),
        explanation,
        approved: false,
        originalUrl: undefined, // server will provide after approval
        replayUrl: undefined,   // server will provide after approval
      };

      setDetectionResult(baseResult);
      setIsAnimationMode(true);
    } catch (err) {
      console.error('File upload failed:', err);
      alert('Failed to analyze file. Check backend is running.');
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
            </header>

            {!detectionResult ? (
              <UploadSection onFileUpload={handleFileUpload} isDetecting={isDetecting} />
            ) : (
              <ResultsSection
                result={detectionResult}
                isAnimationMode={isAnimationMode}
                onNewDetection={handleNewDetection}
                // pass job id + api base so ResultsSection can poll /jobs/{id}
                jobId={jobId ?? ''}                 // prevents `/jobs/undefined`
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