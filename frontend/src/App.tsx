import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// --- Basic CSS for styling ---
const styles: { [key: string]: React.CSSProperties } = {
  container: {
    fontFamily: 'Arial, sans-serif',
    maxWidth: '600px',
    margin: '50px auto',
    padding: '20px',
    border: '1px solid #ccc',
    borderRadius: '10px',
    boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
    backgroundColor: '#f9f9f9',
  },
  header: {
    textAlign: 'center',
    marginBottom: '20px',
    color: '#333',
  },
  buttonContainer: {
    display: 'flex',
    justifyContent: 'center',
    gap: '20px',
    marginBottom: '20px',
  },
  button: {
    padding: '10px 20px',
    fontSize: '16px',
    cursor: 'pointer',
    border: 'none',
    borderRadius: '5px',
    color: 'white',
  },
  status: {
    textAlign: 'center',
    marginBottom: '20px',
    padding: '10px',
    borderRadius: '5px',
  },
  transcriptContainer: {
    height: '300px',
    overflowY: 'auto',
    border: '1px solid #eee',
    borderRadius: '5px',
    padding: '10px',
    backgroundColor: 'white',
  },
  transcriptMessage: {
    marginBottom: '10px',
    padding: '8px',
    borderRadius: '5px',
  }
};

type TranscriptMessage = {
  sender: 'user' | 'bot';
  text: string;
};

const App: React.FC = () => {
  const [callActive, setCallActive] = useState(false);
  const [callId, setCallId] = useState<string>('');
  const [isConnected, setIsConnected] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const audioQueue = useRef<ArrayBuffer[]>([]);
  const isPlaying = useRef(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  // Function to play audio from the queue
  const playAudio = async () => {
    if (isPlaying.current || audioQueue.current.length === 0) return;
    
    isPlaying.current = true;
    const audioData = audioQueue.current.shift();
    if (!audioData || !audioContext.current) {
      isPlaying.current = false;
      return;
    }

    try {
      const audioBuffer = await audioContext.current.decodeAudioData(audioData);
      const source = audioContext.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.current.destination);
      source.onended = () => {
        isPlaying.current = false;
        playAudio(); // Play next in queue
      };
      source.start();
    } catch (e) {
      console.error("Error decoding or playing audio:", e);
      isPlaying.current = false;
    }
  };

  const startCall = async () => {
    // Generate a unique call ID
    const newCallId = `call_${Date.now()}`;
    setCallId(newCallId);
    setCallActive(true);
    setTranscript([]);

    ws.current = new WebSocket(`ws://localhost:8000/ws/audio/${newCallId}`);

    ws.current.onopen = async () => {
      console.log("WebSocket connected");
      setIsConnected(true);
      if (!audioContext.current) {
        audioContext.current = new AudioContext();
      }
      
      // Automatically start recording when connection is established
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder.current = new MediaRecorder(stream);
        
        mediaRecorder.current.ondataavailable = (event) => {
          if (event.data.size > 0 && ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(event.data);
          }
        };

        mediaRecorder.current.start(250); // Send data every 250ms
        console.log("Recording started automatically");
      } catch (error) {
        console.error("Error accessing microphone:", error);
        alert("Could not access microphone. Please check permissions.");
      }
    };

    ws.current.onmessage = async (event) => {
      // Handle JSON messages (transcripts)
      if (typeof event.data === 'string') {
        try {
          const message = JSON.parse(event.data);
          if (message.type === 'transcript') {
            setTranscript(prev => [...prev, {
              sender: message.role === 'user' ? 'user' : 'bot',
              text: message.content
            }]);
          }
        } catch (e) {
          console.log("Non-JSON text message:", event.data);
        }
      }
      // Handle binary messages (audio)
      else if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
        const arrayBuffer = (event.data instanceof Blob) ? await event.data.arrayBuffer() : event.data;
        audioQueue.current.push(arrayBuffer);
        playAudio();
      }
    };

    ws.current.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
      setCallActive(false);
      mediaRecorder.current?.stop();
    };

    ws.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
      setCallActive(false);
      mediaRecorder.current?.stop();
    };
  };

  const endCall = () => {
    mediaRecorder.current?.stop();
    ws.current?.close();
    setCallActive(false);
    setCallId('');
    setIsConnected(false);
  };

  // Effect to clean up on component unmount
  useEffect(() => {
    return () => {
      ws.current?.close();
      mediaRecorder.current?.stream.getTracks().forEach(track => track.stop());
    };
  }, []);
  
  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Bella Cucina Voice Assistant</h1>

      <div style={{ ...styles.status, backgroundColor: isConnected ? '#e8f5e9' : '#ffebee', color: isConnected ? '#2e7d32' : '#c62828' }}>
        {isConnected ? '🎙️ Connected - Listening...' : 'Disconnected'}
      </div>

      <div style={styles.buttonContainer}>
        <button onClick={startCall} disabled={isConnected} style={{ ...styles.button, backgroundColor: isConnected ? '#aaa' : '#4CAF50' }}>
          Start Call
        </button>
        <button onClick={endCall} disabled={!isConnected} style={{ ...styles.button, backgroundColor: !isConnected ? '#aaa' : '#f44336' }}>
          End Call
        </button>
      </div>
      
      <h3 style={{ textAlign: 'center', color: '#555', marginTop: '20px' }}>Live Transcript</h3>
      <div style={styles.transcriptContainer}>
        {transcript.length === 0 ? (
          <p style={{color: '#888', textAlign: 'center', fontStyle: 'italic'}}>
            Start a call to see the conversation transcript...
          </p>
        ) : (
          transcript.map((msg, idx) => (
            <div 
              key={idx} 
              style={{
                ...styles.transcriptMessage,
                backgroundColor: msg.sender === 'user' ? '#e3f2fd' : '#f1f8e9',
                borderLeft: `4px solid ${msg.sender === 'user' ? '#2196F3' : '#8BC34A'}`,
                textAlign: msg.sender === 'user' ? 'right' : 'left'
              }}
            >
              <strong>{msg.sender === 'user' ? 'You' : 'Maria'}:</strong> {msg.text}
            </div>
          ))
        )}
        <div ref={transcriptEndRef} />
      </div>
    </div>
  );
};

export default App;