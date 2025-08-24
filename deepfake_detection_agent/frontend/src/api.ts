export async function analyzeFile(file: File) {
    const formData = new FormData();
    formData.append("file", file);
  
    const res = await fetch("http://127.0.0.1:8001/analyze", {
      method: "POST",
      body: formData,
    });
  
    if (!res.ok) throw new Error("Failed to analyze file");
    return res.json(); // { result, video_score, audio_score, ai_probability }
  }
  