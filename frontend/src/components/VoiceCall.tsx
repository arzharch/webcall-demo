import React, { useState, useRef, useEffect } from 'react';
import './VoiceCall.css';

interface VoiceCallProps {
    callId: string;
    onEnd: () => void;
}

const VoiceCall: React.FC<VoiceCallProps> = ({ callId, onEnd }) => {
    const [transcript, setTranscript] = useState<string[]>([]);
    const [status, setStatus] = useState('Connecting...');

    const webrtcSocketRef = useRef<WebSocket | null>(null);
    const transcriptSocketRef = useRef<WebSocket | null>(null);
    const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
    const localStreamRef = useRef<MediaStream | null>(null);
    const remoteAudioRef = useRef<HTMLAudioElement | null>(null);

    useEffect(() => {
        initializeCall();

        return () => {
            cleanup();
        };
    }, [callId]);

    const initializeCall = async () => {
        await initializeAudio();
        connectWebRTC();
        connectTranscript();
    };

    const cleanup = () => {
        if (webrtcSocketRef.current) {
            webrtcSocketRef.current.close();
        }
        if (transcriptSocketRef.current) {
            transcriptSocketRef.current.close();
        }
        if (peerConnectionRef.current) {
            peerConnectionRef.current.close();
        }
        if (localStreamRef.current) {
            localStreamRef.current.getTracks().forEach(track => track.stop());
        }
    };

    const initializeAudio = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            localStreamRef.current = stream;
        } catch (error) {
            console.error('Failed to initialize audio:', error);
            setStatus('Microphone access denied');
        }
    };

    const connectWebRTC = () => {
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        peerConnectionRef.current = pc;

        localStreamRef.current?.getTracks().forEach(track => {
            pc.addTrack(track, localStreamRef.current!);
        });

        pc.ontrack = (event) => {
            if (remoteAudioRef.current) {
                remoteAudioRef.current.srcObject = event.streams[0];
            }
        };

        const wsUrl = `ws://localhost:8000/ws/webrtc/${callId}`;
        const ws = new WebSocket(wsUrl);
        webrtcSocketRef.current = ws;

        ws.onopen = async () => {
            console.log('WebRTC signaling connected');
            setStatus('Establishing connection...');
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            ws.send(JSON.stringify({ type: 'offer', sdp: offer.sdp }));
        };

        ws.onmessage = async (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'answer') {
                const answer = new RTCSessionDescription(data);
                await pc.setRemoteDescription(answer);
            } else if (data.type === 'candidate') {
                await pc.addIceCandidate(data.candidate);
            }
        };

        pc.onicecandidate = (event) => {
            if (event.candidate) {
                ws.send(JSON.stringify({ type: 'candidate', candidate: event.candidate.toJSON() }));
            }
        };
    };

    const connectTranscript = () => {
        const wsUrl = `ws://localhost:8000/ws/transcript/${callId}`;
        const ws = new WebSocket(wsUrl);
        transcriptSocketRef.current = ws;

        ws.onopen = () => {
            console.log('Transcript WebSocket connected');
            setStatus('Connected');
            // This is where you would send the user's transcribed speech
            // For now, we'll just log a message
            console.log("Ready to send/receive transcripts.");
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'transcript') {
                const speaker = message.role === 'user' ? 'You' : 'Maria';
                setTranscript(prev => [...prev, `${speaker}: ${message.content}`]);
            }
        };

        ws.onerror = (error) => {
            console.error('Transcript WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('Transcript WebSocket disconnected');
        };
    };

    return (
        <div className="voice-call">
            <div className="call-status">
                <div className={`status-indicator ${status.includes('Connected') ? 'active' : ''}`}></div>
            </div>

            <div className="transcript-box">
                <h3>Conversation</h3>
                <div className="transcript-content">
                    {transcript.length === 0 ? (
                        <p className="empty-text">Conversation will appear here...</p>
                    ) : (
                        transcript.map((line, index) => (
                            <div key={index} className="transcript-line">
                                {line}
                            </div>
                        ))
                    )}
                </div>
            </div>

            <audio ref={remoteAudioRef} autoPlay />

            <button className="end-call-button" onClick={onEnd}>
                End Call
            </button>
        </div>
    );
};

export default VoiceCall;