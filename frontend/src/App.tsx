import React, { useState, useRef, useEffect } from 'react';
import VoiceCall from './components/VoiceCall';
import './App.css';

const App: React.FC = () => {
    const [callActive, setCallActive] = useState(false);
    const [callId, setCallId] = useState<string>('');

    const handleStartCall = async () => {
        try {
            const response = await fetch('http://localhost:8000/session/start', {
                method: 'POST'
            });
            const data = await response.json();
            setCallId(data.call_id);
            setCallActive(true);
        } catch (error) {
            console.error('Error starting call:', error);
            alert('Failed to start call. Is the backend running?');
        }
    };

    const handleEndCall = () => {
        setCallActive(false);
        setCallId('');
    };

    return (
        <div className="app">
            <header className="app-header">
                <h1>🎤 Bella Cucina</h1>
                <p>AI Restaurant Reservation Assistant</p>
            </header>

            <main className="app-main">
                {!callActive ? (
                    <button className="start-button" onClick={handleStartCall}>
                        Start Voice Call
                    </button>
                ) : (
                    <VoiceCall callId={callId} onEnd={handleEndCall} />
                )}
            </main>
        </div>
    );
};

export default App;

// fetch() uses http:// but WebSocket uses ws://
// Both need to match backend CORS and protocol
// Should handle ws:// upgrade properly
