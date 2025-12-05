import React, { useState, useRef, useEffect } from 'react';

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
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptMessage[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const audioQueue = useRef<ArrayBuffer[]>([]);
  const isPlaying = useRef(false);

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

  const startCall = () => {
    const callId = `call_${Date.now()}`;
    ws.current = new WebSocket(`ws://localhost:8000/ws/audio/${callId}`);

    ws.current.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
      if (!audioContext.current) {
        audioContext.current = new AudioContext();
      }
    };

    ws.current.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        const arrayBuffer = await event.data.arrayBuffer();
        audioQueue.current.push(arrayBuffer);
        playAudio();
      } else {
        // For text-based status or transcript updates if ever needed
        console.log("Received text message:", event.data);
      }
    };

    ws.current.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
      setIsRecording(false);
    };

    ws.current.onerror = (error) => {
      console.error("WebSocket error:", error);
      setIsConnected(false);
      setIsRecording(false);
    };
  };

  const endCall = () => {
    ws.current?.close();
  };

  const toggleRecording = async () => {
    if (!isRecording) {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
        
        mediaRecorder.current.ondataavailable = (event) => {
          if (event.data.size > 0 && ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(event.data);
          }
        };

        mediaRecorder.current.start(1000); // Send data every 1000ms (1 second)
        setIsRecording(true);
      } catch (error) {
        console.error("Error accessing microphone:", error);
        alert("Could not access microphone. Please check permissions.");
      }
    } else {
      // Stop recording
      mediaRecorder.current?.stop();
      setIsRecording(false);
    }
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
        {isConnected ? 'Connected' : 'Disconnected'}
      </div>

      <div style={styles.buttonContainer}>
        <button onClick={startCall} disabled={isConnected} style={{ ...styles.button, backgroundColor: isConnected ? '#aaa' : '#4CAF50' }}>
          Start Call
        </button>
        <button onClick={endCall} disabled={!isConnected} style={{ ...styles.button, backgroundColor: !isConnected ? '#aaa' : '#f44336' }}>
          End Call
        </button>
      </div>

      <div style={styles.buttonContainer}>
        <button onClick={toggleRecording} disabled={!isConnected} style={{ ...styles.button, backgroundColor: isRecording ? '#f44336' : '#2196F3' }}>
          {isRecording ? 'Stop Listening' : 'Start Listening'}
        </button>
      </div>
      
      {/* A real transcript would require the backend to send text messages, this is a placeholder */}
      <h3 style={{ textAlign: 'center', color: '#555' }}>Live Transcript</h3>
      <div style={styles.transcriptContainer}>
          <p style={{color: '#888', textAlign: 'center'}}>Transcript will appear here...</p>
      </div>
    </div>
  );
};

export default App;
