/**
 * Transcript Display Component
 * Shows conversation history with smooth animations
 */

import React, { useEffect, useRef } from 'react';
import type { TranscriptEntry } from '../lib/types';

interface TranscriptDisplayProps {
  transcripts: TranscriptEntry[];
  maxHeight?: string;
  className?: string;
}

export function TranscriptDisplay({ transcripts, maxHeight = '300px', className = '' }: TranscriptDisplayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [transcripts]);
  
  if (transcripts.length === 0) {
    return (
      <div className={`flex flex-col items-center justify-center h-full text-slate-500 ${className}`}>
        <svg className="w-12 h-12 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={1.5} 
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" 
          />
        </svg>
        <p className="text-sm">Conversation will appear here</p>
      </div>
    );
  }
  
  return (
    <div 
      ref={containerRef}
      className={`glass-effect rounded-2xl p-4 flex flex-col gap-3 overflow-y-auto scroll-smooth ${className}`}
      style={{ maxHeight }}
    >
      {transcripts.map((entry, index) => (
        <div
          key={entry.id}
          className={`flex ${entry.role === 'user' ? 'justify-end' : 'justify-start'}`}
          style={{ animationDelay: `${index * 0.05}s` }}
        >
          <div className={`transcript-bubble ${entry.role}`}>
            {/* Speaker label */}
            <div className={`text-xs mb-1 ${
              entry.role === 'user' ? 'text-indigo-200' : 'text-slate-400'
            }`}>
              {entry.role === 'user' ? 'You' : 'Bella'}
            </div>
            
            {/* Message content */}
            <p className="text-sm leading-relaxed">{entry.content}</p>
            
            {/* Timestamp */}
            <div className={`text-xs mt-1 opacity-60`}>
              {entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default TranscriptDisplay;
