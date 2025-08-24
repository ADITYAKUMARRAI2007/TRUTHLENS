// frontend/src/api.ts
declare global {
  interface ImportMeta {
    env: {
      VITE_API_BASE?: string;
    };
  }
}

export const API_BASE =
  (import.meta.env.VITE_API_BASE as string) ?? 'http://127.0.0.1:8001';

export async function analyze(file: File) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Analyze failed: ${res.status} ${res.statusText}`);
  return res.json();
}