// src/components/ResultsSection.tsx
import React, { useState, useEffect, useMemo } from 'react';
import {
  CheckCircle,
  XCircle,
  HelpCircle,
  RotateCcw,
  FileText,
  Download,
  Shield,
  Clock
} from 'lucide-react';
import { DetectionResult } from '../App';

interface ResultsSectionProps {
  result: DetectionResult;
  isAnimationMode: boolean;
  onNewDetection: () => void;
  jobId: string;
  apiBase?: string; // default http://127.0.0.1:8001
  onApprovedRefresh?: (merged: DetectionResult) => void;
}

const isPlayableVideoUrl = (u?: string | null) => {
  if (!u) return false;
  if (u.startsWith('blob:')) return true;
  if (u.startsWith('data:video')) return true;
  return /\.mp4($|\?)/i.test(u);
};

const ResultsSection: React.FC<ResultsSectionProps> = ({
  result,
  isAnimationMode,
  onNewDetection,
  jobId,
  apiBase = 'http://127.0.0.1:8001',
  onApprovedRefresh,
}) => {
  const [showAnimation, setShowAnimation] = useState(true);
  const [animationComplete, setAnimationComplete] = useState(false);

  const [approved, setApproved] = useState<boolean>(Boolean(result.approved));
  const [serverData, setServerData] = useState<Partial<DetectionResult>>({});

  const merged: DetectionResult = useMemo(
    () => ({ ...result, ...serverData, approved }),
    [result, serverData, approved]
  );

  // --- Poll job until approved ---
  useEffect(() => {
    if (!jobId || approved) return;

    let timer: number | undefined;

    const poll = async () => {
      try {
        const resp = await fetch(`${apiBase}/jobs/${encodeURIComponent(jobId)}`);
        if (!resp.ok) return; // 404 while pending is expected

        const data = await resp.json();

        const isApproved = data?.status === 'APPROVED' || Boolean(data?.approved);
        if (!isApproved) return;

        setApproved(true);

        const serverOriginal =
          data.original_link || data.originalUrl || (data.result && data.result.originalUrl);
        const nextOriginal = isPlayableVideoUrl(serverOriginal)
          ? serverOriginal
          : merged.originalUrl;

        const serverReplay =
          data.replay_link || data.replayUrl || (data.result && data.result.replayUrl);
        const nextReplay = isPlayableVideoUrl(serverReplay)
          ? serverReplay
          : merged.replayUrl;

        const nextStatus =
          (data.result && data.result.result) || data.result || merged.status;

        const nextConfidence =
          typeof data.ai_probability === 'number'
            ? Math.round(data.ai_probability * 100)
            : merged.confidence;

        const next: Partial<DetectionResult> = {
          status: nextStatus,
          originalUrl: nextOriginal,
          replayUrl: nextReplay,
          confidence: nextConfidence,
          approved: true,
        };

        setServerData(next);
        onApprovedRefresh?.({ ...merged, ...next } as DetectionResult);
      } catch {
        /* ignore transient network errors */
      }
    };

    poll();
    timer = window.setInterval(poll, 3000) as unknown as number;
    return () => timer && window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, approved, apiBase]);

  // Animation timing
  useEffect(() => {
    if (isAnimationMode) {
      setShowAnimation(true);
      setAnimationComplete(false);
      const t = window.setTimeout(() => setAnimationComplete(true), 3000);
      return () => window.clearTimeout(t);
    } else {
      setShowAnimation(false);
      setAnimationComplete(true);
    }
  }, [isAnimationMode, merged.status]);

  const CircularProgress: React.FC<{ percentage: number; color: string }> = ({ percentage, color }) => {
    const circumference = 2 * Math.PI * 45;
    const safe = Math.max(0, Math.min(100, Number.isFinite(percentage) ? percentage : 0));
    const strokeDashoffset = circumference - (safe / 100) * circumference;

    return (
      <div className="relative w-32 h-32">
        <svg className="transform -rotate-90 w-32 h-32" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="45" stroke="currentColor" strokeWidth="6" fill="transparent" className="text-gray-700" />
          <circle
            cx="50" cy="50" r="45"
            stroke={color}
            strokeWidth="6"
            fill="transparent"
            strokeDasharray={2 * Math.PI * 45}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-in-out"
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold" style={{ color }}>
            {safe}%
          </span>
        </div>
      </div>
    );
  };

  const handleDownload = () => {
    const srcUrl = merged.replayUrl ?? merged.originalUrl;
    if (!srcUrl) {
      alert('Video not available yet. Please try again in a moment.');
      return;
    }
    const a = document.createElement('a');
    a.href = srcUrl;
    const baseName = (merged.fileName || 'video').replace(/\.[^/.]+$/, '');
    const isReplay = Boolean(merged.replayUrl);
    a.download = isReplay ? `${baseName}_truthlens_replay.mp4` : `${baseName}_original.mp4`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  // ---------------- PENDING REVIEW (still show left original) ----------------
  if (!approved) {
    return (
      <div className="space-y-8">
        <div className="max-w-xl mx-auto bg-gray-900 border border-gray-700 rounded-2xl p-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Shield className="w-8 h-8 text-indigo-400" />
            <h2 className="text-2xl font-bold text-white">Awaiting Admin Approval</h2>
          </div>
          <p className="text-gray-300 mb-6">
            Your submission is currently in review. You’ll see the full report here once an admin approves it.
          </p>
          <div className="flex items-center justify-center gap-2 text-sm text-gray-400 mb-6">
            <Clock className="w-4 h-4" />
            <span>Auto-refreshing every 3 seconds…</span>
          </div>

          <div className="grid grid-cols-2 gap-4 text-left mb-6">
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-gray-400 text-sm">File</div>
              <div className="text-white font-medium break-all">{merged.fileName}</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="text-gray-400 text-sm">Type</div>
              <div className="text-white font-medium capitalize">{merged.fileType}</div>
            </div>
          </div>

          {merged.fileType === 'video' && (
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="p-3 text-sm text-gray-400">Original Submission (local preview)</div>
              <video controls className="w-full h-64 object-cover bg-black">
                <source src={merged.originalUrl ?? ''} type="video/mp4" />
              </video>
            </div>
          )}

          <div className="mt-8">
            <button
              onClick={onNewDetection}
              className="px-5 py-3 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold"
            >
              Submit Another
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---------------- APPROVED → normal flow ----------------
  if (isAnimationMode && showAnimation && !animationComplete) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="relative mb-8">
            {merged.status === 'REAL' ? (
              <div className="w-48 h-48 mx-auto relative">
                <CheckCircle className="w-48 h-48 text-green-400 animate-bounce" />
              </div>
            ) : merged.status === 'FAKE' ? (
              <div className="w-48 h-48 mx-auto relative">
                <XCircle className="w-48 h-48 text-red-400 animate-pulse" />
              </div>
            ) : (
              <div className="w-48 h-48 mx-auto relative">
                <HelpCircle className="w-48 h-48 text-yellow-400 animate-pulse" />
              </div>
            )}
          </div>
          <p className="text-gray-300 text-xl">
            We're {merged.confidence ?? 0}% sure about this
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-4 mb-4">
          {merged.status === 'REAL' ? (
            <CheckCircle className="w-12 h-12 text-green-400" />
          ) : merged.status === 'FAKE' ? (
            <XCircle className="w-12 h-12 text-red-400" />
          ) : (
            <HelpCircle className="w-12 h-12 text-yellow-400" />
          )}
          <h2
            className={`text-3xl font-bold ${
              merged.status === 'REAL'
                ? 'text-green-400'
                : merged.status === 'FAKE'
                ? 'text-red-400'
                : 'text-yellow-400'
            }`}
          >
            {merged.status === 'REAL'
              ? 'This Looks Real to Us!'
              : merged.status === 'FAKE'
              ? 'Red Flags Everywhere 🚩'
              : 'Uncertain Outcome 🤷‍♂️'}
          </h2>
        </div>
        <p className="text-gray-400">
          Just finished analyzing:{' '}
          <span className="text-white font-mono">{merged.fileName}</span>
        </p>
      </div>

      {/* Confidence + File Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Confidence */}
        <div className="p-8 rounded-2xl bg-gray-900 border border-gray-700">
          <h3 className="text-xl font-semibold mb-6 text-center text-gray-200">How Sure Are We?</h3>
          <div className="flex justify-center">
            <CircularProgress
              percentage={merged.confidence}
              color={merged.status === 'REAL' ? '#22c55e' : merged.status === 'FAKE' ? '#ef4444' : '#fbbf24'}
            />
          </div>
        </div>
        {/* File Info */}
        <div className="p-8 rounded-2xl bg-gray-900 border border-gray-700">
          <h3 className="text-xl font-semibold mb-6 text-gray-200">What We Found</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-800">
              <span className="text-gray-400">File Name:</span>
              <span className="text-white font-medium">{merged.fileName}</span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-800">
              <span className="text-gray-400">File Type:</span>
              <span className="text-white font-medium capitalize">{merged.fileType}</span>
            </div>
            <div className="flex justify-between items-center p-3 rounded-lg bg-gray-800">
              <span className="text-gray-400">Status:</span>
              <span
                className={`font-medium ${
                  merged.status === 'REAL'
                    ? 'text-green-400'
                    : merged.status === 'FAKE'
                    ? 'text-red-400'
                    : 'text-yellow-400'
                }`}
              >
                {merged.status}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Nerdy Details */}
      <div className="p-8 rounded-2xl bg-gray-900 border border-gray-700">
        <h3 className="text-xl font-semibold mb-6 text-gray-200 flex items-center gap-2">
          <FileText className="w-5 h-5 text-cyan-400" />
          The Nerdy Details
        </h3>
        <div className="p-6 rounded-lg bg-black/30 border border-cyan-500/20 font-mono text-sm">
          {merged.explanation}
        </div>
      </div>

      {/* UNIVERSAL ACTIONS */}
      <div className="flex justify-center">
        <button
          onClick={onNewDetection}
          className="flex items-center gap-2 px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-xl text-white font-semibold"
        >
          <RotateCcw className="w-5 h-5" />
          Try Another One!
        </button>
      </div>

      {/* Replay Section */}
      {merged.fileType === 'video' && (
        <div className="mt-12 space-y-10">
          <h3 className="text-2xl font-bold text-center text-white">TruthLens Replay</h3>
          <p className="text-center text-gray-400">
            {merged.status === 'REAL'
              ? 'Stabilized/cleaned preview of the original.'
              : 'AI-reconstructed version showing what the real content might look like.'}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="p-3 text-sm text-gray-400">Original Submission</div>
              <video controls className="w-full h-64 object-cover bg-black">
                <source src={merged.originalUrl ?? ''} type="video/mp4" />
              </video>
            </div>

            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="p-3 text-sm text-gray-400">
                {merged.status === 'REAL' ? 'Reality Reconstruction (stabilized)' : 'Reality Reconstruction'}
              </div>
              {merged.replayUrl ? (
                <video controls className="w-full h-64 object-cover bg-black">
                  <source src={merged.replayUrl} type="video/mp4" />
                </video>
              ) : (
                <div className="w-full h-64 flex items-center justify-center bg-black/60 text-gray-400">
                  Replay not available.
                </div>
              )}
            </div>
          </div>

          <div className="flex gap-4">
            <button
              onClick={handleDownload}
              disabled={!merged.replayUrl && !merged.originalUrl}
              className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium text-white ${
                merged.replayUrl || merged.originalUrl
                  ? 'bg-indigo-600 hover:bg-indigo-500'
                  : 'bg-gray-700 cursor-not-allowed'
              }`}
              title={
                merged.replayUrl
                  ? 'Download Replay MP4'
                  : merged.originalUrl
                  ? 'Download Original MP4'
                  : 'No video available yet'
              }
            >
              <Download className="w-4 h-4" />
              Download MP4
            </button>

            <button className="px-4 py-2 bg-gray-700 rounded-md font-medium text-white">
              Reprocess
            </button>
            <button className="px-4 py-2 bg-gray-700 rounded-md font-medium text-white">
              Export Report
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsSection;