// frontend/src/api.ts

// Resolve API base once, with a safe production fallback.
// Netlify/Preview/Branch env should set VITE_API_BASE, but if not,
// we default to your Render URL (never localhost in prod).
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string)?.trim() ||
  'https://truthlens-ispk.onrender.com';

export const analyze = async (file: File, apiBase?: string) => {
  const base = (apiBase ?? API_BASE).trim();
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${base}/analyze`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Analyze failed: ${res.status} ${res.statusText}`);
  return res.json();
};

// (Optional helper you already use elsewhere)
export async function fetchJob(jobId: string, apiBase?: string) {
  const base = (apiBase ?? API_BASE).trim();
  const res = await fetch(`${base}/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error(`Get job failed: ${res.status} ${res.statusText}`);
  return res.json();
}