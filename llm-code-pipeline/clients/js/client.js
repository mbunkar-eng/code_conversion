/**
 * JavaScript/TypeScript Client SDK for LLM Code Pipeline.
 *
 * OpenAI-compatible client that can be used as a drop-in replacement.
 *
 * @example
 * const client = new LLMClient({
 *   baseUrl: 'http://localhost:8000',
 *   apiKey: 'your-api-key'
 * });
 *
 * // Chat completion
 * const response = await client.chat.completions.create({
 *   model: 'qwen2.5-coder-7b',
 *   messages: [
 *     { role: 'user', content: 'Convert Python to Java: print("hello")' }
 *   ]
 * });
 * console.log(response.choices[0].message.content);
 *
 * // Streaming
 * const stream = await client.chat.completions.create({
 *   model: 'qwen2.5-coder-7b',
 *   messages: [{ role: 'user', content: 'Write a function' }],
 *   stream: true
 * });
 *
 * for await (const chunk of stream) {
 *   process.stdout.write(chunk.choices[0]?.delta?.content || '');
 * }
 */

class LLMClientError extends Error {
    constructor(message, statusCode = null, body = null) {
        super(message);
        this.name = 'LLMClientError';
        this.statusCode = statusCode;
        this.body = body;
    }
}

class AuthenticationError extends LLMClientError {
    constructor(message, statusCode, body) {
        super(message, statusCode, body);
        this.name = 'AuthenticationError';
    }
}

class RateLimitError extends LLMClientError {
    constructor(message, statusCode, body) {
        super(message, statusCode, body);
        this.name = 'RateLimitError';
    }
}

class APIError extends LLMClientError {
    constructor(message, statusCode, body) {
        super(message, statusCode, body);
        this.name = 'APIError';
    }
}

/**
 * Chat Completions namespace
 */
class ChatCompletions {
    constructor(client) {
        this._client = client;
    }

    /**
     * Create a chat completion.
     *
     * @param {Object} params - Request parameters
     * @param {string} params.model - Model to use
     * @param {Array} params.messages - Conversation messages
     * @param {number} [params.temperature=0.7] - Sampling temperature
     * @param {number} [params.top_p=1.0] - Nucleus sampling parameter
     * @param {number} [params.n=1] - Number of completions
     * @param {boolean} [params.stream=false] - Stream responses
     * @param {string|Array} [params.stop] - Stop sequences
     * @param {number} [params.max_tokens] - Maximum tokens
     * @param {Object} [params.response_format] - Response format
     * @returns {Promise<Object|AsyncGenerator>}
     */
    async create(params) {
        const {
            model,
            messages,
            temperature = 0.7,
            top_p = 1.0,
            n = 1,
            stream = false,
            stop = null,
            max_tokens = null,
            presence_penalty = 0,
            frequency_penalty = 0,
            response_format = null,
            ...rest
        } = params;

        const payload = {
            model,
            messages,
            temperature,
            top_p,
            n,
            stream,
            presence_penalty,
            frequency_penalty,
            ...rest
        };

        if (stop) payload.stop = stop;
        if (max_tokens) payload.max_tokens = max_tokens;
        if (response_format) payload.response_format = response_format;

        if (stream) {
            return this._streamCompletion(payload);
        }

        return this._client._request('POST', '/v1/chat/completions', payload);
    }

    async *_streamCompletion(payload) {
        const response = await this._client._streamRequest('POST', '/v1/chat/completions', payload);

        for await (const chunk of response) {
            yield chunk;
        }
    }
}

/**
 * Chat namespace
 */
class Chat {
    constructor(client) {
        this.completions = new ChatCompletions(client);
    }
}

/**
 * Completions namespace (legacy API)
 */
class Completions {
    constructor(client) {
        this._client = client;
    }

    /**
     * Create a completion.
     *
     * @param {Object} params - Request parameters
     * @param {string} params.model - Model to use
     * @param {string|Array} params.prompt - Prompt(s) to complete
     * @param {number} [params.max_tokens=256] - Maximum tokens
     * @param {number} [params.temperature=0.7] - Sampling temperature
     * @param {boolean} [params.stream=false] - Stream responses
     * @returns {Promise<Object|AsyncGenerator>}
     */
    async create(params) {
        const {
            model,
            prompt,
            max_tokens = 256,
            temperature = 0.7,
            top_p = 1.0,
            n = 1,
            stream = false,
            stop = null,
            echo = false,
            ...rest
        } = params;

        const payload = {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            n,
            stream,
            echo,
            ...rest
        };

        if (stop) payload.stop = stop;

        if (stream) {
            return this._client._streamRequest('POST', '/v1/completions', payload);
        }

        return this._client._request('POST', '/v1/completions', payload);
    }
}

/**
 * Models namespace
 */
class Models {
    constructor(client) {
        this._client = client;
    }

    /**
     * List available models.
     * @returns {Promise<Array>}
     */
    async list() {
        const response = await this._client._request('GET', '/v1/models');
        return response.data || [];
    }

    /**
     * Retrieve model information.
     * @param {string} modelId - Model identifier
     * @returns {Promise<Object>}
     */
    async retrieve(modelId) {
        return this._client._request('GET', `/v1/models/${modelId}`);
    }
}

/**
 * LLM Client for interacting with the LLM Code Pipeline API.
 */
class LLMClient {
    /**
     * Create a new LLM client.
     *
     * @param {Object} options - Client options
     * @param {string} [options.baseUrl] - API base URL
     * @param {string} [options.apiKey] - API key
     * @param {number} [options.timeout=120000] - Request timeout in ms
     */
    constructor(options = {}) {
        this.baseUrl = (options.baseUrl || process.env.LLM_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
        this.apiKey = options.apiKey || process.env.LLM_API_KEY;
        this.timeout = options.timeout || 120000;

        // Namespaced interfaces
        this.chat = new Chat(this);
        this.completions = new Completions(this);
        this.models = new Models(this);
    }

    _getHeaders() {
        const headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };

        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }

        return headers;
    }

    async _request(method, endpoint, body = null) {
        const url = `${this.baseUrl}${endpoint}`;

        const options = {
            method,
            headers: this._getHeaders()
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        // Add timeout using AbortController
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        options.signal = controller.signal;

        try {
            const response = await fetch(url, options);
            clearTimeout(timeoutId);

            if (!response.ok) {
                await this._handleError(response);
            }

            return response.json();
        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError') {
                throw new APIError('Request timeout', 408);
            }

            if (error instanceof LLMClientError) {
                throw error;
            }

            throw new APIError(`Request failed: ${error.message}`);
        }
    }

    async *_streamRequest(method, endpoint, body = null) {
        const url = `${this.baseUrl}${endpoint}`;

        const options = {
            method,
            headers: this._getHeaders()
        };

        if (body) {
            options.body = JSON.stringify(body);
        }

        const response = await fetch(url, options);

        if (!response.ok) {
            await this._handleError(response);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();

                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);

                        if (data === '[DONE]') {
                            return;
                        }

                        try {
                            yield JSON.parse(data);
                        } catch {
                            // Skip invalid JSON
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    async _handleError(response) {
        let body = null;
        let message = response.statusText;

        try {
            body = await response.json();
            message = body.error?.message || message;
        } catch {
            // Use status text if JSON parsing fails
        }

        if (response.status === 401) {
            throw new AuthenticationError(message, response.status, body);
        } else if (response.status === 429) {
            throw new RateLimitError(message, response.status, body);
        } else {
            throw new APIError(message, response.status, body);
        }
    }

    /**
     * Check API health status.
     * @returns {Promise<Object>}
     */
    async health() {
        return this._request('GET', '/health');
    }
}

// Export for different module systems
if (typeof module !== 'undefined' && module.exports) {
    // CommonJS
    module.exports = {
        LLMClient,
        LLMClientError,
        AuthenticationError,
        RateLimitError,
        APIError
    };
} else if (typeof window !== 'undefined') {
    // Browser
    window.LLMClient = LLMClient;
    window.LLMClientError = LLMClientError;
    window.AuthenticationError = AuthenticationError;
    window.RateLimitError = RateLimitError;
    window.APIError = APIError;
}

// ES Module export (for bundlers)
export {
    LLMClient,
    LLMClientError,
    AuthenticationError,
    RateLimitError,
    APIError
};

export default LLMClient;
