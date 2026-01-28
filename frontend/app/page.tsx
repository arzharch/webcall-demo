"use client";

import { useCallback, useState, useEffect } from "react";
import { useVoiceCall } from "@/hooks/useVoiceCall";
import {
  CallButton,
  NameInput,
  StatusIndicator,
  TranscriptDisplay,
  Waveform,
} from "@/components";
import { validateCallerName, sanitizeInput } from "@/lib/api";
import { createLogger } from "@/lib/logger";

const logger = createLogger("HomePage");

export default function HomePage() {
  const [callerName, setCallerName] = useState("");
  const [nameError, setNameError] = useState<string | undefined>();
  const [isReady, setIsReady] = useState(false);

  const {
    status,
    sessionId,
    isConnected,
    isSpeaking,
    transcripts,
    startCall,
    endCall,
    error,
    callDuration,
    getAnalyserNode,
  } = useVoiceCall();

  // Log mount
  useEffect(() => {
    logger.info("Page mounted");
    return () => logger.info("Page unmounted");
  }, []);

  // Log status changes
  useEffect(() => {
    logger.info("Call status changed", { status, sessionId, isConnected });
  }, [status, sessionId, isConnected]);

  // Log errors
  useEffect(() => {
    if (error) {
      logger.error("Call error", { error });
    }
  }, [error]);

  // Handle name input change with validation
  const handleNameChange = useCallback((value: string) => {
    const sanitized = sanitizeInput(value);
    setCallerName(sanitized);
    
    if (sanitized.length === 0) {
      setNameError(undefined);
      setIsReady(false);
      return;
    }

    const validation = validateCallerName(sanitized);
    if (!validation.valid) {
      setNameError(validation.error);
      setIsReady(false);
    } else {
      setNameError(undefined);
      setIsReady(true);
    }
  }, []);

  // Handle call button click
  const handleCallClick = useCallback(async () => {
    if (status === "idle" || status === "error") {
      // Starting call
      if (!isReady) {
        logger.warn("Attempted to start call without valid name");
        setNameError("Please enter a valid name");
        return;
      }
      logger.info("Starting call", { callerName });
      await startCall(callerName.trim());
    } else if (status === "active" || status === "connected" || status === "connecting" || status === "processing" || status === "speaking") {
      // Ending call
      logger.info("Ending call");
      endCall();
    }
  }, [status, isReady, callerName, startCall, endCall]);

  // Determine if we're in an active call state
  const isInCall = status === "connecting" || status === "connected" || status === "active" || status === "processing" || status === "speaking";
  const canStartCall = isReady && (status === "idle" || status === "error");

  return (
    <main className="min-h-screen bg-stone-50 text-stone-900 flex flex-col items-center justify-center p-4 selection:bg-red-100 pattern-grid">
      
      {/* Main Container */}
      <div className="w-full max-w-md animate-fade-in flex flex-col gap-6 relative z-10">
        
        {/* Header - Restaurant Branding */}
        <header className="text-center font-serif">
          <div className="inline-block p-1 border-b-2 border-red-600 mb-3">
             <h1 className="text-5xl font-bold text-stone-800 tracking-tighter font-serif italic drop-shadow-sm">
              Bella Cucina
            </h1>
          </div>
          <p className="text-stone-500 text-sm font-medium tracking-widest uppercase mt-1">
            Authentic Italian Dining
          </p>
        </header>

        {/* Main Card */}
        <div className="bg-white rounded-[2rem] shadow-2xl shadow-stone-300/60 p-6 md:p-8 w-full border border-stone-100 animate-fade-in-up relative overflow-hidden">
          {/* Decorative Corner accent - subtle vine/organic shape */}
          <div className="absolute -top-12 -right-12 w-40 h-40 bg-green-50 rounded-full blur-3xl opacity-60 pointer-events-none" />
          <div className="absolute -bottom-12 -left-12 w-40 h-40 bg-red-50 rounded-full blur-3xl opacity-60 pointer-events-none" />
          
          {/* Status Bar */}
          <div className="flex items-center justify-between mb-8 relative z-10">
            <StatusIndicator
              status={status}
              duration={callDuration}
              sessionId={sessionId}
            />
          </div>

          {/* Visualization Area */}
          <div className="h-32 mb-8 flex flex-col items-center justify-center bg-stone-50/50 rounded-xl border border-stone-100 overflow-hidden relative">
             
             {/* Hint when idle */}
             {!isConnected && status !== 'error' && (
               <div className="absolute top-4 w-full text-center z-20">
                 <span className="text-stone-400 text-[10px] font-bold tracking-widest uppercase">
                   Ready to connect
                 </span>
               </div>
             )}

             {/* Waveform Visualization */}
             <div className="w-full h-full flex items-end justify-center pb-2">
                <Waveform
                  isActive={isConnected}
                  getAnalyserNode={getAnalyserNode}
                  barCount={20} // Fewer bars, thicker = cleaner look
                />
             </div>
          </div>

          {/* Name Input - only show when not in call */}
          {!isInCall && (
            <div className="mb-6 animate-fade-in relative z-10">
               <label className="block text-[10px] font-bold text-stone-400 mb-2 ml-1 uppercase tracking-widest">
                 Your Name
               </label>
              <NameInput
                value={callerName}
                onChange={handleNameChange}
                error={nameError}
                disabled={isInCall}
                placeholder="Name for reservation..."
              />
            </div>
          )}

          {/* Caller display */}
          {isInCall && callerName && (
            <div className="text-center mb-8 animate-fade-in">
              <p className="text-stone-400 text-xs uppercase tracking-wider font-bold mb-1">Reservation For</p>
              <p className="text-stone-800 font-serif font-bold text-xl italic">{callerName}</p>
            </div>
          )}

          {/* Action Button */}
          <div className="flex justify-center flex-col items-center gap-2">
            <CallButton
              status={status}
              onClick={handleCallClick}
              disabled={!canStartCall && status === "idle"}
            />
            {status === "idle" && (
                <span className="text-xs text-stone-400 font-medium mt-2">
                    {canStartCall ? "Tap to Call Reservation Desk" : "Please enter name first"}
                </span>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-6 text-center animate-fade-in">
              <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-red-50 text-red-700 text-sm font-medium border border-red-100">
                <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            </div>
          )}
        </div>

        {/* Live Transcript */}
        {transcripts.length > 0 && (
          <div className="w-full animate-fade-in-up animation-delay-200">
            <TranscriptDisplay
              transcripts={transcripts}
              maxHeight="200px"
            />
          </div>
        )}

        <footer className="text-center mt-4">
            <p className="text-stone-400 text-xs font-semibold uppercase tracking-widest mb-1">
                Voice Technology by
            </p>
            <p className="text-stone-600 font-bold text-sm">
                Synthion AI
            </p>
        </footer>
      </div>
    </main>
  );
}
