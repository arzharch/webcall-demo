/**
 * Call Button Component
 * Animated button for starting/ending calls
 */

import React from 'react';
import type { CallStatus } from '../lib/types';

interface CallButtonProps {
  status: CallStatus;
  onClick: () => void;
  disabled?: boolean;
}

export function CallButton({ status, onClick, disabled }: CallButtonProps) {
  const isActive = status === 'active' || status === 'connected' || status === 'processing' || status === 'speaking';
  const isConnecting = status === 'connecting';
  
  return (
    <div className="relative">
      {/* Pulse rings for active state */}
      {isActive && (
        <>
          <div className="absolute inset-0 rounded-full bg-red-500/30 animate-pulse-ring" />
          <div 
            className="absolute inset-0 rounded-full bg-red-500/20 animate-pulse-ring" 
            style={{ animationDelay: '0.5s' }}
          />
        </>
      )}
      
      {/* Main button */}
      <button
        onClick={onClick}
        disabled={disabled || isConnecting}
        className={`
          relative z-10 w-20 h-20 rounded-full flex items-center justify-center
          transition-all duration-300 transform
          ${isActive 
            ? 'bg-rose-500 hover:bg-rose-600 shadow-xl shadow-rose-200 ring-4 ring-rose-100' // Elegant Rose for End
            : 'bg-emerald-600 hover:bg-emerald-700 shadow-xl shadow-emerald-200 ring-4 ring-emerald-100' // Elegant Emerald for Start
          }
          ${disabled || isConnecting ? 'opacity-50 cursor-not-allowed grayscale' : 'hover:-translate-y-1 active:scale-95'}
        `}
        aria-label={isActive ? 'End call' : 'Start call'}
      >
        {isConnecting ? (
          // Loading spinner
          <svg className="w-8 h-8 animate-spin text-white" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path 
              className="opacity-75" 
              fill="currentColor" 
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" 
            />
          </svg>
        ) : isActive ? (
          // End call icon
          <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M16 8l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2M3 12c0-1.654 1.346-3 3-3h12c1.654 0 3 1.346 3 3v0c0 1.654-1.346 3-3 3H6c-1.654 0-3-1.346-3-3v0z"
            />
          </svg>
        ) : (
          // Phone icon
          <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" 
            />
          </svg>
        )}
      </button>
    </div>
  );
}

export default CallButton;
