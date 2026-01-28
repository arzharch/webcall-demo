/**
 * API Client for Bella Voice AI Backend
 * Handles all REST API calls with validation and error handling
 */

import { createLogger } from './logger';
import type { HealthCheck, DailyStats, BookingInfo, CallSession } from './types';

const logger = createLogger('API');

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT = 10000; // 10 seconds

// Request ID for tracking
let requestId = 0;

/**
 * Validates and sanitizes user input
 */
export function sanitizeInput(input: string): string {
  return input
    .trim()
    .slice(0, 100) // Max 100 chars
    .replace(/[<>'"]/g, ''); // Remove potential XSS chars
}

/**
 * Validates caller name
 */
export function validateCallerName(name: string): { valid: boolean; error?: string } {
  const sanitized = sanitizeInput(name);
  
  if (sanitized.length < 2) {
    return { valid: false, error: 'Name must be at least 2 characters' };
  }
  
  if (sanitized.length > 50) {
    return { valid: false, error: 'Name must be less than 50 characters' };
  }
  
  if (!/^[a-zA-Z\s'-]+$/.test(sanitized)) {
    return { valid: false, error: 'Name can only contain letters, spaces, hyphens, and apostrophes' };
  }
  
  return { valid: true };
}

/**
 * Custom error class for API errors
 */
export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Fetch wrapper with timeout and error handling
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout = REQUEST_TIMEOUT
): Promise<Response> {
  const id = ++requestId;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  logger.debug(`[${id}] Request: ${options.method || 'GET'} ${url}`);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    
    clearTimeout(timeoutId);
    
    logger.debug(`[${id}] Response: ${response.status}`);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.detail || `Request failed with status ${response.status}`,
        response.status,
        errorData.code
      );
    }
    
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error instanceof APIError) throw error;
    
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        logger.error(`[${id}] Request timeout`);
        throw new APIError('Request timed out', 408, 'TIMEOUT');
      }
      logger.error(`[${id}] Request error: ${error.message}`);
      throw new APIError(error.message, 0, 'NETWORK_ERROR');
    }
    
    throw new APIError('Unknown error', 0, 'UNKNOWN');
  }
}

/**
 * API Client
 */
export const api = {
  /**
   * Health check
   */
  async health(): Promise<HealthCheck> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/health`);
    return response.json();
  },
  
  /**
   * Get daily statistics
   */
  async getStats(date?: string): Promise<DailyStats> {
    const params = date ? `?date=${date}` : '';
    const response = await fetchWithTimeout(`${API_BASE_URL}/stats${params}`);
    return response.json();
  },
  
  /**
   * List active sessions
   */
  async listSessions(): Promise<{ active_count: number; sessions: CallSession[] }> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/sessions`);
    return response.json();
  },
  
  /**
   * Get call details
   */
  async getCall(callId: string): Promise<CallSession> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/calls/${callId}`);
    return response.json();
  },
  
  /**
   * List bookings
   */
  async listBookings(filters?: { name?: string; date?: string }): Promise<BookingInfo[]> {
    const params = new URLSearchParams();
    if (filters?.name) params.set('name', filters.name);
    if (filters?.date) params.set('date', filters.date);
    const query = params.toString() ? `?${params}` : '';
    const response = await fetchWithTimeout(`${API_BASE_URL}/bookings${query}`);
    return response.json();
  },
  
  /**
   * Get booking by ID
   */
  async getBooking(bookingId: number): Promise<BookingInfo> {
    const response = await fetchWithTimeout(`${API_BASE_URL}/bookings/${bookingId}`);
    return response.json();
  },
  
  /**
   * Get WebSocket URL for voice call
   */
  getWebSocketUrl(callerName: string, phone?: string): string {
    const sanitizedName = encodeURIComponent(sanitizeInput(callerName));
    const baseUrl = API_BASE_URL.replace(/^http/, 'ws');
    let url = `${baseUrl}/ws/call?name=${sanitizedName}`;
    if (phone) {
      url += `&phone=${encodeURIComponent(phone)}`;
    }
    return url;
  },
};

export default api;
