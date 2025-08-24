/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{ts,tsx,js,jsx}"],
    darkMode: "class",
    theme: {
      extend: {
        colors: {
          bg: "#08080b",
          panel: "#101116",
          border: "#1f2230",
          neonBlue: "#00f7ff",
          neonPink: "#ff00ff",
          neonGreen: "#00ffa3",
        },
        boxShadow: {
          glowPink: "0 0 24px rgba(255, 0, 255, 0.35)",
          glowBlue: "0 0 24px rgba(0, 247, 255, 0.35)",
          glowRed: "0 0 24px rgba(255, 77, 77, 0.45)",
          glowGreen: "0 0 24px rgba(0, 255, 163, 0.45)"
        }
      },
    },
    plugins: [],
  };
  