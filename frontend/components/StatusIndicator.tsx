/**
 * Status Indicator Component
 * Shows current call status with animation
 */

import React from 'react';
import type { CallStatus } from '../lib/types';

interface StatusIndicatorProps {
  status: CallStatus;
  duration?: number;
  sessionId?: string | null;
  className?: string;
}

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: string }> = {
  idle: { label: 'Ready to call', color: 'slate', icon: 'ready' },
  connecting: { label: 'Connecting...', color: 'yellow', icon: 'connecting' },
  connected: { label: 'Connected', color: 'green', icon: 'connected' },
  active: { label: 'In Call', color: 'green', icon: 'active' },
  processing: { label: 'Processing...', color: 'blue', icon: 'processing' },
  speaking: { label: 'Bella is speaking', color: 'cyan', icon: 'speaking' },
  ended: { label: 'Call ended', color: 'slate', icon: 'ended' },
  error: { label: 'Connection error', color: 'red', icon: 'error' },
};

const DEFAULT_CONFIG = { label: 'Unknown', color: 'slate', icon: 'unknown' };

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export function StatusIndicator({ status, duration, sessionId, className = '' }: StatusIndicatorProps) {
  const config = STATUS_CONFIG[status] || DEFAULT_CONFIG;
  const isActive = ['active', 'connected', 'processing', 'speaking'].includes(status);
  
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      {/* Status dot */}
      <div className="relative">
        <div className={`
          status-dot
          ${status === 'error' ? 'error' : isActive ? 'active' : 'inactive'}
        `} />
        {isActive && (
          <div className="absolute inset-0 status-dot active animate-ping opacity-75" />
        )}
      </div>
      
      {/* Status text */}
      <div className="flex flex-col">
        <span className={`text-sm font-medium ${
          status === 'error' ? 'text-red-400' : 
          isActive ? 'text-green-400' : 
          'text-slate-400'
        }`}>
          {config.label}
        </span>
        
        {/* Duration */}
        {isActive && duration !== undefined && (
          <span className="text-xs text-slate-500 font-mono">
            {formatDuration(duration)}
          </span>
        )}
      </div>
    </div>
  );
}

export default StatusIndicator;
