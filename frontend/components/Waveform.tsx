
import React, { useEffect, useRef } from 'react';

interface WaveformProps {
  isActive: boolean;
  getAnalyserNode?: () => AnalyserNode | null;
  className?: string;
  barCount?: number;
}

export function Waveform({ isActive, getAnalyserNode, className = '', barCount = 40 }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    let animationId: number;
    let analyser: AnalyserNode | null = null;
    
    // Config
    const bufferLength = 128; // Reduced buffer for chunkier "retro" look
    const dataArray = new Uint8Array(bufferLength);
    
    // Style config
    const barWidthRatio = 0.5; // Thinner bars, more gap
    const borderRadius = 100; // Fully rounded pills

    const draw = () => {
      // Lazy fetch analyzer
      if (isActive && getAnalyserNode && !analyser) {
        analyser = getAnalyserNode();
      }

      // Check signal
      let hasSignal = false;
      if (isActive && analyser) {
        analyser.getByteFrequencyData(dataArray);
        for(let i=0; i<dataArray.length; i++) {
           if (dataArray[i] > 10) { hasSignal = true; break; }
        }
      } else {
        dataArray.fill(0);
      }

      const w = rect.width;
      const h = rect.height;
      const totalBarSpace = w / barCount;
      const barWidth = totalBarSpace * barWidthRatio;
      
      ctx.clearRect(0, 0, w, h);

      for (let i = 0; i < barCount; i++) {
        // Focus on vocal range bins (index 2 to ~25 out of 128)
        const dataIndex = Math.floor(2 + (i / barCount) * 20); 
        
        let value = dataArray[dataIndex] || 0;
        
        // Idle animation
        if (isActive && !hasSignal) {
           // Gentle wave
           const idleHeight = 6 + Math.sin(Date.now() / 400 + i * 0.5) * 4;
           value = idleHeight * 3; 
        } else if (!isActive) {
           value = 4; // Flat line dot
        }

        // Height Calc
        const percent = Math.min(1, value / 200); 
        // Non-linear scaling for better visuals
        const scaledPercent = Math.pow(percent, 0.8);
        const barHeight = Math.max(4, h * 0.8 * scaledPercent);
        
        const x = i * totalBarSpace + (totalBarSpace - barWidth) / 2;
        const y = (h - barHeight) / 2;

        // Draw Pill
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(x, y, barWidth, barHeight, borderRadius);
        } else {
            ctx.rect(x, y, barWidth, barHeight);
        }
        
        // Elegant Color
        if (isActive) {
             // Gradient from bottom to top for each bar? 
             // Or solid color based on height?
             // Let's do a solid "Tomato" compatible color
             ctx.fillStyle = `rgba(220, 38, 38, ${0.4 + percent * 0.6})`; // Red-600 with opacity fade
        } else {
             ctx.fillStyle = '#e7e5e4'; // Stone-200 (Inactive)
        }
        
        ctx.fill();
      }

      animationId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [isActive, getAnalyserNode, barCount]);


  return (
    <canvas 
      ref={canvasRef}
      className={`w-full h-16 ${className}`}
      style={{ width: '100%', height: '64px' }}
    />
  );
}

export default Waveform;

