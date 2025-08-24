import React from 'react';

const AnimatedBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 z-0">
      {/* Animated Grid Lines */}
      <div className="absolute inset-0 opacity-20">
        <div className="grid-container">
          {/* Vertical Lines */}
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={`v-${i}`}
              className="absolute h-full w-px animate-pulse"
              style={{
                left: `${(i + 1) * 5}%`,
                background: `linear-gradient(180deg, 
                  transparent, 
                  #00f7ff ${Math.random() * 100}%, 
                  #ff00ff ${Math.random() * 100}%, 
                  transparent)`,
                animationDelay: `${i * 0.1}s`,
                animationDuration: `${3 + Math.random() * 2}s`
              }}
            />
          ))}
          
          {/* Horizontal Lines */}
          {Array.from({ length: 15 }).map((_, i) => (
            <div
              key={`h-${i}`}
              className="absolute w-full h-px animate-pulse"
              style={{
                top: `${(i + 1) * 6.67}%`,
                background: `linear-gradient(90deg, 
                  transparent, 
                  #9d4edd ${Math.random() * 100}%, 
                  #00f7ff ${Math.random() * 100}%, 
                  transparent)`,
                animationDelay: `${i * 0.15}s`,
                animationDuration: `${4 + Math.random() * 2}s`
              }}
            />
          ))}
        </div>
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0 overflow-hidden">
        {Array.from({ length: 15 }).map((_, i) => (
          <div
            key={`particle-${i}`}
            className="absolute w-1 h-1 rounded-full animate-float"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              backgroundColor: ['#00f7ff', '#ff00ff', '#9d4edd'][i % 3],
              boxShadow: `0 0 10px ${['#00f7ff', '#ff00ff', '#9d4edd'][i % 3]}`,
              animationDelay: `${i * 0.2}s`,
              animationDuration: `${10 + Math.random() * 10}s`
            }}
          />
        ))}
      </div>

      {/* Gradient Overlays */}
      <div className="absolute inset-0 bg-gradient-to-br from-purple-900/10 via-transparent to-cyan-900/10" />
      <div className="absolute inset-0 bg-gradient-to-tr from-pink-900/5 via-transparent to-purple-900/5" />
    </div>
  );
};

export default AnimatedBackground;