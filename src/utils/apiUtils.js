import { API_ENDPOINT } from '../config';

/**
 * Tests API connectivity with a simple OPTIONS request
 * @returns {Promise<boolean>} true if API is reachable, false otherwise
 */
export const checkApiConnectivity = async () => {
  try {
    const response = await fetch(`${API_ENDPOINT}/summarize`, {
      method: 'OPTIONS',
      headers: {
        'Accept': 'application/json'
      }
    });
    
    return response.status >= 200 && response.status < 500;
  } catch (error) {
    console.error('API connectivity check failed:', error);
    return false;
  }
};

/**
 * Formats API endpoint for display/debugging
 * @param {string} endpoint The full API endpoint URL
 * @returns {string} A masked/formatted version of the URL
 */
export const formatApiEndpoint = (endpoint) => {
  try {
    const url = new URL(endpoint);
    return `${url.protocol}//${url.host}${url.pathname.length > 1 ? '/..' : ''}`;
  } catch (error) {
    return endpoint;
  }
};
