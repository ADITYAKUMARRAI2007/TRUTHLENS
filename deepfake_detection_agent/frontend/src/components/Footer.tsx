import React from 'react';

export default function Footer() {
  return (
    <footer className="mt-12 text-center text-gray-400 text-sm">
      <p>
        © {new Date().getFullYear()} TruthLens — Built by humans, powered by AI 🤖
      </p>
    </footer>
  );
}