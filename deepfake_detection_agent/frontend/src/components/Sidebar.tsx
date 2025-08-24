import React from 'react';
import { Home, Upload, FileText, Zap, ToggleLeft, ToggleRight } from 'lucide-react';

interface SidebarProps {
  currentPage: string;
  onPageChange: (page: string) => void;
  isAnimationMode: boolean;
  onAnimationModeToggle: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  currentPage, 
  onPageChange, 
  isAnimationMode, 
  onAnimationModeToggle 
}) => {
  const menuItems = [
    { id: 'home', icon: Home, label: 'Home' }
  ];

  return (
    <div className="w-64 min-h-screen relative">
      {/* Sidebar Background */}
      <div className="absolute inset-0 backdrop-blur-md bg-black/30 border-r border-cyan-500/20" />
      
      <div className="relative z-10 p-6">
        {/* Logo */}
        <div className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-purple-600 flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
            TruthLens
          </span>
        </div>

        {/* Navigation Menu */}
        <nav className="space-y-2 mb-8">
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentPage === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => onPageChange(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-300 group ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 shadow-lg shadow-cyan-500/10'
                    : 'hover:bg-white/5 hover:border hover:border-white/10'
                }`}
              >
                <Icon className={`w-5 h-5 transition-colors duration-300 ${
                  isActive ? 'text-cyan-400' : 'text-gray-400 group-hover:text-white'
                }`} />
                <span className={`transition-colors duration-300 ${
                  isActive ? 'text-cyan-400' : 'text-gray-400 group-hover:text-white'
                }`}>
                  {item.label}
                </span>
              </button>
            );
          })}
        </nav>

        {/* Animation Mode Toggle */}
        <div className="p-4 rounded-lg bg-gradient-to-r from-purple-900/20 to-pink-900/20 border border-purple-500/20">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Display Mode</h3>
          <button
            onClick={onAnimationModeToggle}
            className="flex items-center gap-3 w-full"
          >
            {isAnimationMode ? (
              <ToggleRight className="w-6 h-6 text-cyan-400" />
            ) : (
              <ToggleLeft className="w-6 h-6 text-gray-400" />
            )}
            <span className="text-sm text-gray-300">
              {isAnimationMode ? 'Animation Mode' : 'Static Results'}
            </span>
          </button>
        </div>

        {/* Status Indicator */}
        <div className="mt-8 p-4 rounded-lg bg-green-900/20 border border-green-500/20">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-sm font-medium text-green-400">All Systems Go</span>
          </div>
          <p className="text-xs text-gray-400">Last updated: 2 mins ago</p>
        </div>

        {/* Fun Stats */}
        <div className="mt-4 p-3 rounded-lg bg-purple-900/10 border border-purple-500/10">
          <div className="text-xs text-gray-400 space-y-1">
            <div className="flex justify-between">
              <span>Files analyzed today:</span>
              <span className="text-purple-400">1,247</span>
            </div>
            <div className="flex justify-between">
              <span>Deepfakes caught:</span>
              <span className="text-red-400">23</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;