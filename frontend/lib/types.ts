/**
 * Type definitions for Bella Voice AI Frontend
 */

// Call status enum
export type CallStatus = 
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'active'
  | 'processing'
  | 'speaking'
  | 'ended'
  | 'error';

// Transcript entry
export interface TranscriptEntry {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  intent?: string;
}

// WebSocket message types from server
export interface WSStatusMessage {
  type: 'status';
  status: CallStatus;
  message?: string;
  session_id?: string;
}

export interface WSTranscriptMessage {
  type: 'transcript';
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  intent?: string;
}

export interface WSErrorMessage {
  type: 'error';
  message: string;
  code?: string;
}

export type WSMessage = WSStatusMessage | WSTranscriptMessage | WSErrorMessage;

// API response types
export interface CallSession {
  id: string;
  caller_name: string;
  status: CallStatus;
  started_at: string;
  ended_at?: string;
  duration_seconds?: number;
  turn_count?: number;
}

export interface BookingInfo {
  id: number;
  name: string;
  party_size: number;
  booking_date: string;
  booking_time: string;
  status: string;
  notes?: string;
}

export interface HealthCheck {
  status: 'healthy' | 'unhealthy';
  checks: Record<string, boolean>;
}

export interface DailyStats {
  date: string;
  total_calls: number;
  completed_calls: number;
  avg_duration?: number;
  total_cost?: number;
  total_bookings: number;
}

// Audio config
export interface AudioConfig {
  sampleRate: number;
  channels: number;
  bitsPerSample: number;
}

export const AUDIO_CONFIG: AudioConfig = {
  sampleRate: 16000,
  channels: 1,
  bitsPerSample: 16,
};

// Playback audio config (TTS uses 24kHz)
export const PLAYBACK_AUDIO_CONFIG: AudioConfig = {
  sampleRate: 24000,
  channels: 1,
  bitsPerSample: 16,
};
