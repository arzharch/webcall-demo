/**
 * useVoiceCall Hook
 * Manages WebSocket connection for voice calls
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { createLogger } from '../lib/logger';
import { api, sanitizeInput } from '../lib/api';
import type { CallStatus, TranscriptEntry, WSMessage, AUDIO_CONFIG, PLAYBACK_AUDIO_CONFIG } from '../lib/types';

const logger = createLogger('VoiceCall');

// Audio constants
const CAPTURE_SAMPLE_RATE = 16000;
const PLAYBACK_SAMPLE_RATE = 24000;

interface UseVoiceCallOptions {
  onTranscript?: (entry: TranscriptEntry) => void;
  onStatusChange?: (status: CallStatus) => void;
  onError?: (error: Error) => void;
}

interface UseVoiceCallReturn {
  status: CallStatus;
  sessionId: string | null;
  isConnected: boolean;
  isSpeaking: boolean;
  transcripts: TranscriptEntry[];
  startCall: (callerName: string) => Promise<void>;
  endCall: () => void;
  error: string | null;
  callDuration: number;
}

export function useVoiceCall(options: UseVoiceCallOptions = {}): UseVoiceCallReturn {
  const { onTranscript, onStatusChange, onError } = options;
  
  // State
  const [status, setStatus] = useState<CallStatus>('idle');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioQueueRef = useRef<ArrayBuffer[]>([]);
  const isPlayingRef = useRef(false);
  const startTimeRef = useRef<number | null>(null);
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Update status with callback
  const updateStatus = useCallback((newStatus: CallStatus) => {
    logger.info(`Status changed: ${newStatus}`);
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);

  // Add transcript entry
  const addTranscript = useCallback((entry: TranscriptEntry) => {
    logger.debug('New transcript', { role: entry.role, content: entry.content.slice(0, 50) });
    setTranscripts(prev => [...prev, entry]);
    onTranscript?.(entry);
  }, [onTranscript]);

  // Play audio from queue
  const playNextAudio = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;
    if (!audioContextRef.current) return;
    
    isPlayingRef.current = true;
    setIsSpeaking(true);
    
    const audioData = audioQueueRef.current.shift();
    if (!audioData) {
      isPlayingRef.current = false;
      setIsSpeaking(false);
      return;
    }
    
    try {
      const ctx = audioContextRef.current;
      
      // Convert LINEAR16 to Float32
      const int16Array = new Int16Array(audioData);
      const float32Array = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768;
      }
      
      // Create audio buffer
      const audioBuffer = ctx.createBuffer(1, float32Array.length, PLAYBACK_SAMPLE_RATE);
      audioBuffer.getChannelData(0).set(float32Array);
      
      // Play buffer
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      
      source.onended = () => {
        isPlayingRef.current = false;
        if (audioQueueRef.current.length > 0) {
          playNextAudio();
        } else {
          setIsSpeaking(false);
        }
      };
      
      source.start();
      logger.debug(`Playing audio: ${float32Array.length} samples`);
    } catch (err) {
      logger.error('Audio playback error', err);
      isPlayingRef.current = false;
      setIsSpeaking(false);
      // Try next audio
      if (audioQueueRef.current.length > 0) {
        setTimeout(playNextAudio, 100);
      }
    }
  }, []);

  // Handle WebSocket message
  const handleMessage = useCallback((event: MessageEvent) => {
    if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
      // Binary audio data
      const handleBinary = async () => {
        const buffer = event.data instanceof Blob 
          ? await event.data.arrayBuffer() 
          : event.data;
        audioQueueRef.current.push(buffer);
        logger.debug(`Audio received: ${buffer.byteLength} bytes`);
        playNextAudio();
      };
      handleBinary();
    } else {
      // JSON message
      try {
        const msg: WSMessage = JSON.parse(event.data);
        logger.debug('WS message received', msg);
        
        switch (msg.type) {
          case 'status':
            updateStatus(msg.status);
            if (msg.session_id) {
              setSessionId(msg.session_id);
            }
            break;
            
          case 'transcript':
            addTranscript({
              id: crypto.randomUUID(),
              role: msg.role,
              content: msg.content,
              timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
              intent: msg.intent,
            });
            break;
            
          case 'error':
            logger.error('Server error', msg);
            setError(msg.message);
            onError?.(new Error(msg.message));
            break;
        }
      } catch (err) {
        logger.warn('Failed to parse WS message', event.data);
      }
    }
  }, [updateStatus, addTranscript, playNextAudio, onError]);

  // Start call
  const startCall = useCallback(async (callerName: string) => {
    if (status !== 'idle' && status !== 'ended' && status !== 'error') {
      logger.warn('Cannot start call: already in progress');
      return;
    }
    
    // Validate name
    const sanitized = sanitizeInput(callerName);
    if (sanitized.length < 2) {
      setError('Name must be at least 2 characters');
      return;
    }
    
    logger.info(`Starting call for: ${sanitized}`);
    setError(null);
    setTranscripts([]);
    audioQueueRef.current = [];
    updateStatus('connecting');
    
    try {
      // Request microphone permission
      logger.debug('Requesting microphone access');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: CAPTURE_SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        } 
      });
      mediaStreamRef.current = stream;
      logger.info('Microphone access granted');
      
      // Create audio context
      audioContextRef.current = new AudioContext({ sampleRate: CAPTURE_SAMPLE_RATE });
      const ctx = audioContextRef.current;
      
      // Setup audio processing
      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      
      // Connect WebSocket
      const wsUrl = api.getWebSocketUrl(sanitized);
      logger.debug(`Connecting to: ${wsUrl}`);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.binaryType = 'arraybuffer';
      
      ws.onopen = () => {
        logger.info('WebSocket connected');
        // Don't set status here - wait for server to send 'connected' status
        
        // Start duration timer
        startTimeRef.current = Date.now();
        durationIntervalRef.current = setInterval(() => {
          if (startTimeRef.current) {
            setCallDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
          }
        }, 1000);
        
        // Start sending audio
        processor.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          
          const inputData = e.inputBuffer.getChannelData(0);
          const int16Array = new Int16Array(inputData.length);
          
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          
          try {
            ws.send(int16Array.buffer);
          } catch (err) {
            logger.error('Failed to send audio', err);
          }
        };
        
        source.connect(processor);
        processor.connect(ctx.destination);
      };
      
      ws.onmessage = handleMessage;
      
      ws.onerror = (event) => {
        logger.error('WebSocket error', event);
        setError('Connection error');
        updateStatus('error');
      };
      
      ws.onclose = (event) => {
        logger.info(`WebSocket closed: ${event.code} ${event.reason}`);
        if (status !== 'ended' && status !== 'error') {
          updateStatus('ended');
        }
      };
      
    } catch (err) {
      logger.error('Failed to start call', err);
      const message = err instanceof Error ? err.message : 'Failed to start call';
      setError(message);
      updateStatus('error');
      onError?.(err instanceof Error ? err : new Error(message));
    }
  }, [status, updateStatus, handleMessage, onError]);

  // End call
  const endCall = useCallback(() => {
    logger.info('Ending call');
    
    // Stop duration timer
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }
    
    // Send end message
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end', reason: 'user_ended' }));
      wsRef.current.close(1000, 'user_ended');
    }
    wsRef.current = null;
    
    // Stop audio processing
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    
    // Stop media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    // Close audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    // Clear audio queue
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setIsSpeaking(false);
    
    updateStatus('ended');
  }, [updateStatus]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
    };
  }, []);

  return {
    status,
    sessionId,
    isConnected: status === 'connected' || status === 'active' || status === 'processing' || status === 'speaking',
    isSpeaking,
    transcripts,
    startCall,
    endCall,
    error,
    callDuration,
  };
}

export default useVoiceCall;
