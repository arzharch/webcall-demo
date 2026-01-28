/**
 * Animated Waveform Component
 * Shows audio activity visualization
 */

import React from 'react';

interface WaveformProps {
  isActive: boolean;
  isSpeaking?: boolean;
  barCount?: number;
  className?: string;
}

export function Waveform({ isActive, isSpeaking = false, barCount = 9, className = '' }: WaveformProps) {
  const showActive = isActive || isSpeaking;
  
  return (
    <div className={`flex items-center justify-center gap-1 h-8 ${className}`}>
      {Array.from({ length: barCount }).map((_, i) => (
        <div
          key={i}
          className={`w-1 rounded-full transition-all duration-300 ${
            showActive
              ? isSpeaking 
                ? 'bg-gradient-to-t from-amber-500 to-orange-400 waveform-bar'
                : 'bg-gradient-to-t from-indigo-500 to-cyan-400 waveform-bar'
              : 'bg-slate-600 h-1'
          }`}
          style={{
            height: showActive ? undefined : '4px',
          }}
        />
      ))}
    </div>
  );
}

export default Waveform;
