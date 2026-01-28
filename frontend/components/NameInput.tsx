/**
 * Name Input Component
 * Styled input with validation
 */

import React, { useState, useCallback } from 'react';

interface NameInputProps {
  value: string;
  onChange: (value: string) => void;
  error?: string;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

export function NameInput({ 
  value, 
  onChange, 
  error, 
  disabled, 
  placeholder = "Enter your name", 
  className = '' 
}: NameInputProps) {
  const [isFocused, setIsFocused] = useState(false);
  
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(e.target.value);
  }, [onChange]);
  
  const isValid = value.length >= 2 && !error;
  
  return (
    <div className={`relative ${className}`}>
      <div className="relative">
        {/* Icon */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={1.5} 
              d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" 
            />
          </svg>
        </div>
        
        {/* Input */}
        <input
          type="text"
          value={value}
          onChange={handleChange}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          disabled={disabled}
          placeholder={placeholder}
          maxLength={50}
          autoComplete="name"
          className={`
            input-field pl-12 pr-12
            ${error ? 'border-red-500/50 focus:ring-red-500/30' : ''}
            ${isValid ? 'border-green-500/30' : ''}
          `}
          aria-label="Your name"
          aria-invalid={!!error}
        />
        
        {/* Validation indicator */}
        {value.length > 0 && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2">
            {isValid ? (
              <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : error ? (
              <svg className="w-5 h-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : null}
          </div>
        )}
      </div>
      
      {/* Error message */}
      {error && (
        <p className="mt-2 text-sm text-red-400 animate-fade-in">
          {error}
        </p>
      )}
      
      {/* Helper text */}
      {!error && value.length === 0 && (
        <p className="mt-2 text-xs text-slate-500">
          2-50 characters, letters and spaces only
        </p>
      )}
    </div>
  );
}

export default NameInput;
