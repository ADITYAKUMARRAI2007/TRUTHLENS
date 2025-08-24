import React, { useCallback, useState } from 'react';
import { Upload, FileVideo, FileAudio, FileText, Code, Loader2 } from 'lucide-react';

interface UploadSectionProps {
  onFileUpload: (file: File) => void;
  isDetecting: boolean;
}

const UploadSection: React.FC<UploadSectionProps> = ({ onFileUpload, isDetecting }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      onFileUpload(files[0]);
    }
  }, [onFileUpload]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileUpload(files[0]);
    }
  }, [onFileUpload]);

  const fileTypes = [
    { icon: FileVideo, label: 'Video Files', desc: 'MP4, AVI, MOV' },
    { icon: FileAudio, label: 'Audio Files', desc: 'MP3, WAV, M4A' },
    { icon: FileText, label: 'Documents', desc: 'PDF, DOC, TXT' },
    { icon: Code, label: 'Code Projects', desc: 'ZIP, TAR, Folders' },
  ];

  if (isDetecting) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <div className="relative mb-8">
            <div className="w-32 h-32 mx-auto rounded-full bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 flex items-center justify-center animate-pulse">
              <Loader2 className="w-12 h-12 text-cyan-400 animate-spin" />
            </div>
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-cyan-500/10 to-purple-500/10 animate-ping" />
          </div>
          <h3 className="text-2xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            AI Analysis in Progress
          </h3>
          <p className="text-gray-400 mb-2">Deep learning algorithms are scanning your file...</p>
          <p className="text-sm text-gray-500">This may take a few moments</p>
          
          <div className="mt-8 space-y-2">
            <div className="text-sm text-cyan-400">► Analyzing metadata patterns</div>
            <div className="text-sm text-purple-400">► Detecting compression artifacts</div>
            <div className="text-sm text-pink-400">► Verifying authenticity signatures</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Main Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative p-12 rounded-2xl border-2 border-dashed transition-all duration-300 ${
          isDragOver
            ? 'border-cyan-400 bg-cyan-500/5 shadow-lg shadow-cyan-500/20'
            : 'border-gray-600 hover:border-purple-500 hover:bg-purple-500/5'
        }`}
      >
        {/* Background Glow Effect */}
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-cyan-500/5 via-purple-500/5 to-pink-500/5 backdrop-blur-sm" />
        
        <div className="relative text-center">
          <div className="mb-6">
            <div className="w-20 h-20 mx-auto rounded-full bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 flex items-center justify-center mb-4">
              <Upload className="w-10 h-10 text-cyan-400" />
            </div>
            <h2 className="text-3xl font-bold mb-2 bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Drop Your Files Here
            </h2>
            <p className="text-sm text-gray-500 mb-4">
              Don't worry, we don't store anything permanently 🔒
            </p>
            <p className="text-gray-400 text-lg mb-6">
              Just drag & drop, or click if you're old school
            </p>
          </div>

          <input
            type="file"
            onChange={handleFileSelect}
            accept="video/*,audio/*,image/*,.pdf,.doc,.docx,.txt,.zip,.tar,.gz,.js,.ts,.py,.cpp,.java"
            className="hidden"
            id="file-upload"
          />
          
          <label
            htmlFor="file-upload"
            className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-cyan-500 to-purple-600 rounded-xl text-white font-semibold text-lg cursor-pointer transition-all duration-300 hover:from-cyan-400 hover:to-purple-500 hover:scale-105 hover:shadow-lg hover:shadow-cyan-500/25 active:scale-95"
          >
            <Upload className="w-6 h-6" />
            Let's Find Out! 🕵️
          </label>
        </div>
      </div>

      {/* File Type Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {fileTypes.map((type, index) => {
          const Icon = type.icon;
          return (
            <div
              key={index}
              className="p-6 rounded-xl bg-gradient-to-br from-gray-800/50 to-gray-900/50 border border-gray-700/50 backdrop-blur-sm hover:border-purple-500/50 transition-all duration-300 group"
            >
              <div className="w-12 h-12 rounded-lg bg-gradient-to-r from-purple-500/20 to-pink-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                <Icon className="w-6 h-6 text-purple-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">{type.label}</h3>
              <p className="text-sm text-gray-400">{type.desc}</p>
            </div>
          );
        })}
      </div>

      {/* Quick Tips */}
      <div className="bg-gradient-to-r from-gray-800/30 to-gray-900/30 border border-gray-700/30 rounded-xl p-6 backdrop-blur-sm">
        <h3 className="text-lg font-semibold text-yellow-400 mb-3 flex items-center gap-2">
          💡 Pro Tips from Our Team
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
          <div>• Higher resolution videos = better detection accuracy</div>
          <div>• Audio deepfakes are trickier - we're getting better at those!</div>
          <div>• Code analysis works best with complete project folders</div>
          <div>• Suspicious? Run it twice - we don't mind 😊</div>
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-cyan-900/20 to-cyan-800/20 border border-cyan-500/20">
          <div className="w-12 h-12 rounded-full bg-cyan-500/20 flex items-center justify-center mx-auto mb-4">
            <span className="text-cyan-400 font-bold">AI</span>
          </div>
          <h3 className="text-lg font-semibold text-cyan-400 mb-2">Smart AI Detection</h3>
          <p className="text-gray-400 text-sm">Our neural networks have seen it all - trained on millions of samples</p>
        </div>

        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-purple-900/20 to-purple-800/20 border border-purple-500/20">
          <div className="w-12 h-12 rounded-full bg-purple-500/20 flex items-center justify-center mx-auto mb-4">
            <span className="text-purple-400 font-bold">94%</span>
          </div>
          <h3 className="text-lg font-semibold text-purple-400 mb-2">Pretty Darn Accurate</h3>
          <p className="text-gray-400 text-sm">We're honest about our limits - still improving every day!</p>
        </div>

        <div className="text-center p-6 rounded-xl bg-gradient-to-br from-pink-900/20 to-pink-800/20 border border-pink-500/20">
          <div className="w-12 h-12 rounded-full bg-pink-500/20 flex items-center justify-center mx-auto mb-4">
            <span className="text-pink-400 font-bold">⚡</span>
          </div>
          <h3 className="text-lg font-semibold text-pink-400 mb-2">Lightning Fast</h3>
          <p className="text-gray-400 text-sm">Usually takes 2-5 seconds, but complex files might need a coffee break</p>
        </div>
      </div>
    </div>
  );
};

export default UploadSection;