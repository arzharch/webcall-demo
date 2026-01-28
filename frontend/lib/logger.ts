/**
 * Frontend Logger
 * Provides structured logging with levels and context
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  timestamp: string;
  level: LogLevel;
  component: string;
  message: string;
  data?: unknown;
}

// Log level priority
const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

// Get minimum log level from environment
const getMinLevel = (): LogLevel => {
  if (typeof window === 'undefined') return 'info';
  const env = process.env.NEXT_PUBLIC_LOG_LEVEL?.toLowerCase();
  if (env && env in LOG_LEVELS) return env as LogLevel;
  return process.env.NODE_ENV === 'development' ? 'debug' : 'info';
};

const MIN_LEVEL = getMinLevel();

// Color codes for console
const COLORS: Record<LogLevel, string> = {
  debug: '#9ca3af',
  info: '#22d3ee',
  warn: '#f59e0b',
  error: '#ef4444',
};

// Emoji indicators
const ICONS: Record<LogLevel, string> = {
  debug: '🔍',
  info: 'ℹ️',
  warn: '⚠️',
  error: '❌',
};

class Logger {
  private component: string;
  private buffer: LogEntry[] = [];
  private maxBuffer = 100;

  constructor(component: string) {
    this.component = component;
  }

  private shouldLog(level: LogLevel): boolean {
    return LOG_LEVELS[level] >= LOG_LEVELS[MIN_LEVEL];
  }

  private formatTime(): string {
    return new Date().toISOString().slice(11, 23);
  }

  private log(level: LogLevel, message: string, data?: unknown): void {
    if (!this.shouldLog(level)) return;

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      component: this.component,
      message,
      data,
    };

    // Buffer for potential export
    this.buffer.push(entry);
    if (this.buffer.length > this.maxBuffer) {
      this.buffer.shift();
    }

    // Console output with styling
    const time = this.formatTime();
    const prefix = `${ICONS[level]} [${time}] [${this.component}]`;

    if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
      const style = `color: ${COLORS[level]}; font-weight: bold;`;
      
      if (data !== undefined) {
        console.groupCollapsed(`%c${prefix} ${message}`, style);
        console.log('Data:', data);
        console.groupEnd();
      } else {
        console.log(`%c${prefix} ${message}`, style);
      }
    } else {
      // Simple output for SSR/production
      const logFn = level === 'error' ? console.error : 
                    level === 'warn' ? console.warn : console.log;
      logFn(`${prefix} ${message}`, data ?? '');
    }
  }

  debug(message: string, data?: unknown): void {
    this.log('debug', message, data);
  }

  info(message: string, data?: unknown): void {
    this.log('info', message, data);
  }

  warn(message: string, data?: unknown): void {
    this.log('warn', message, data);
  }

  error(message: string, data?: unknown): void {
    this.log('error', message, data);
  }

  // Get buffered logs for debugging/export
  getBuffer(): LogEntry[] {
    return [...this.buffer];
  }

  // Clear buffer
  clearBuffer(): void {
    this.buffer = [];
  }
}

// Factory function to create loggers
export function createLogger(component: string): Logger {
  return new Logger(component);
}

// Default logger
export const logger = createLogger('App');

export type { LogLevel, LogEntry, Logger };
