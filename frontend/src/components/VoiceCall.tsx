import React, { useState, useRef, useEffect } from 'react';
import './VoiceCall.css';

interface VoiceCallProps {
    callId: string;
    onEnd: () => void;
}

const VoiceCall: React.FC<VoiceCallProps> = ({ callId, onEnd }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState<string[]>([]);
    const [isListening, setIsListening] = useState(true);
    const [status, setStatus] = useState('Connecting...');

    const socketRef = useRef<WebSocket | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const processorRef = useRef<ScriptProcessorAudioNode | null>(null);

    useEffect(() => {
        connectWebSocket();
        initializeAudio();

        return () => {
            if (socketRef.current) {
                socketRef.current.close();
            }
            if (mediaStreamRef.current) {
                mediaStreamRef.current.getTracks().forEach(track => track.stop());
            }
        };
    }, [callId]);

    const connectWebSocket = () => {
        const wsUrl = `ws://localhost:8000/ws/audio/${callId}`;
        socketRef.current = new WebSocket(wsUrl);

        socketRef.current.onopen = () => {
            setStatus('Connected');
            setIsListening(true);
        };

        socketRef.current.onmessage = (event) => {
            // Receive audio response
            playAudio(event.data);
        };

        socketRef.current.onerror = (error) => {
            console.error('WebSocket error:', error);
            setStatus('Connection error');
        };

        socketRef.current.onclose = () => {
            setStatus('Disconnected');
            setIsListening(false);
        };
    };

    const initializeAudio = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaStreamRef.current = stream;

            // Proper AudioContext initialization
            const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
            const audioContext = new AudioContextClass();
            audioContextRef.current = audioContext;
            
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            source.connect(processor);
            processor.connect(audioContext.destination);

            processor.onaudioprocess = (event) => {
                if (isListening && socketRef.current?.readyState === WebSocket.OPEN) {
                    const audioData = event.inputBuffer.getChannelData(0);
                    const pcmData = new Int16Array(audioData.length);
                    
                    for (let i = 0; i < audioData.length; i++) {
                        pcmData[i] = Math.max(-1, Math.min(1, audioData[i])) * 0x7FFF;
                    }

                    socketRef.current.send(pcmData.buffer);
                }
            };

            setStatus('Listening...');
        } catch (error) {
            console.error('Audio initialization error:', error);
            setStatus('Microphone access denied');
        }
    };

    const playAudio = (audioBuffer: ArrayBuffer) => {
        if (!audioContextRef.current) return;

        const audioContext = audioContextRef.current;
        audioContext.decodeAudioData(audioBuffer, (audioBuffer) => {
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start(0);
        });
    };

    return (
        <div className="voice-call">
            <div className="call-status">
                <div className={`status-indicator ${isListening ? 'active' : ''}`}></div>
                <span className="status-text">{status}</span>
            </div>

            <div className="transcript-box">
                <h3>Conversation</h3>
                <div className="transcript-content">
                    {transcript.length === 0 ? (
                        <p className="empty-text">Listening for your voice...</p>
                    ) : (
                        transcript.map((line, idx) => (
                            <p key={idx} className="transcript-line">
                                {line}
                            </p>
                        ))
                    )}
                </div>
            </div>

            <button className="end-call-button" onClick={onEnd}>
                End Call
            </button>
        </div>
    );
};

export default VoiceCall;