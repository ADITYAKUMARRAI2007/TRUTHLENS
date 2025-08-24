// frontend/src/api.ts
export const analyze = async (file: File, apiBase: string) => {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${apiBase}/analyze`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Analyze failed: ${res.statusText}`);
  return res.json();
};