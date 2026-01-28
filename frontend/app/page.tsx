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
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white flex flex-col">
      {/* Decorative gradient orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-amber-500/20 to-orange-500/10 rounded-full blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-gradient-to-tr from-blue-500/10 to-purple-500/10 rounded-full blur-3xl" />
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center justify-center flex-1 px-4 py-8 animate-fade-in">
        {/* Header */}
        <header className="text-center mb-8 animate-fade-in-up">
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-amber-400 via-orange-400 to-amber-500 bg-clip-text text-transparent mb-2">
            Bella
          </h1>
          <p className="text-slate-400 text-sm md:text-base">
            Voice Assistant for Restaurant Reservations
          </p>
        </header>

        {/* Main Card */}
        <div className="glass-effect rounded-3xl p-6 md:p-8 w-full max-w-md animate-fade-in-up animation-delay-100">
          {/* Status Indicator */}
          <div className="flex justify-center mb-6">
            <StatusIndicator
              status={status}
              duration={callDuration}
              sessionId={sessionId}
            />
          </div>

          {/* Waveform Animation */}
          <div className="flex justify-center mb-8">
            <Waveform
              isActive={isConnected}
              isSpeaking={isSpeaking}
              barCount={24}
            />
          </div>

          {/* Name Input - only show when not in call */}
          {!isInCall && (
            <div className="mb-6 animate-fade-in">
              <NameInput
                value={callerName}
                onChange={handleNameChange}
                error={nameError}
                disabled={isInCall}
                placeholder="Enter your name"
              />
            </div>
          )}

          {/* Caller name display during call */}
          {isInCall && callerName && (
            <div className="text-center mb-6 animate-fade-in">
              <span className="text-slate-400 text-sm">Calling as </span>
              <span className="text-amber-400 font-medium">{callerName}</span>
            </div>
          )}

          {/* Call Button */}
          <div className="flex justify-center mb-6">
            <CallButton
              status={status}
              onClick={handleCallClick}
              disabled={!canStartCall && status === "idle"}
            />
          </div>

          {/* Error Display */}
          {error && (
            <div className="text-center mb-4 animate-fade-in">
              <p className="text-red-400 text-sm bg-red-500/10 rounded-lg px-4 py-2">
                {error}
              </p>
            </div>
          )}

          {/* Hint Text */}
          {status === "idle" && (
            <p className="text-center text-slate-500 text-xs animate-fade-in">
              {canStartCall
                ? "Tap the button to start your call"
                : "Enter your name to begin"}
            </p>
          )}
        </div>

        {/* Transcript Section */}
        {transcripts.length > 0 && (
          <div className="w-full max-w-md mt-6 animate-fade-in-up animation-delay-200">
            <TranscriptDisplay
              transcripts={transcripts}
              maxHeight="240px"
            />
          </div>
        )}

        {/* Footer */}
        <footer className="mt-8 text-center text-slate-600 text-xs animate-fade-in animation-delay-300">
          <p>Powered by AI Voice Technology</p>
        </footer>
      </div>
    </main>
  );
}
