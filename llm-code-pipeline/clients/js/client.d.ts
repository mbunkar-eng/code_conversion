/**
 * TypeScript declarations for LLM Code Pipeline Client SDK.
 */

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string;
    name?: string;
}

export interface Usage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
}

export interface ChatChoice {
    index: number;
    message: ChatMessage;
    finish_reason: 'stop' | 'length' | 'content_filter';
}

export interface ChatCompletionResponse {
    id: string;
    object: 'chat.completion';
    created: number;
    model: string;
    choices: ChatChoice[];
    usage: Usage;
    system_fingerprint?: string;
}

export interface ChatCompletionChunk {
    id: string;
    object: 'chat.completion.chunk';
    created: number;
    model: string;
    choices: {
        index: number;
        delta: {
            role?: string;
            content?: string;
        };
        finish_reason?: string;
    }[];
}

export interface CompletionChoice {
    text: string;
    index: number;
    logprobs?: any;
    finish_reason: 'stop' | 'length' | 'content_filter';
}

export interface CompletionResponse {
    id: string;
    object: 'text_completion';
    created: number;
    model: string;
    choices: CompletionChoice[];
    usage: Usage;
    system_fingerprint?: string;
}

export interface ResponseFormat {
    type: 'text' | 'json_object' | 'json_schema';
    json_schema?: {
        name: string;
        schema: Record<string, any>;
        strict?: boolean;
    };
}

export interface ChatCompletionParams {
    model: string;
    messages: ChatMessage[];
    temperature?: number;
    top_p?: number;
    n?: number;
    stream?: boolean;
    stop?: string | string[];
    max_tokens?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    response_format?: ResponseFormat;
    seed?: number;
}

export interface CompletionParams {
    model: string;
    prompt: string | string[];
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    n?: number;
    stream?: boolean;
    stop?: string | string[];
    echo?: boolean;
}

export interface ModelInfo {
    id: string;
    object: 'model';
    created: number;
    owned_by: string;
}

export interface HealthResponse {
    status: string;
    model_loaded: boolean;
    model_name?: string;
    gpu_available: boolean;
    version: string;
}

export interface ClientOptions {
    baseUrl?: string;
    apiKey?: string;
    timeout?: number;
}

export class LLMClientError extends Error {
    statusCode: number | null;
    body: any;
    constructor(message: string, statusCode?: number | null, body?: any);
}

export class AuthenticationError extends LLMClientError {}
export class RateLimitError extends LLMClientError {}
export class APIError extends LLMClientError {}

export class ChatCompletions {
    create(params: ChatCompletionParams & { stream: true }): Promise<AsyncGenerator<ChatCompletionChunk>>;
    create(params: ChatCompletionParams & { stream?: false }): Promise<ChatCompletionResponse>;
    create(params: ChatCompletionParams): Promise<ChatCompletionResponse | AsyncGenerator<ChatCompletionChunk>>;
}

export class Chat {
    completions: ChatCompletions;
}

export class Completions {
    create(params: CompletionParams & { stream: true }): Promise<AsyncGenerator<any>>;
    create(params: CompletionParams & { stream?: false }): Promise<CompletionResponse>;
    create(params: CompletionParams): Promise<CompletionResponse | AsyncGenerator<any>>;
}

export class Models {
    list(): Promise<ModelInfo[]>;
    retrieve(modelId: string): Promise<ModelInfo>;
}

export class LLMClient {
    baseUrl: string;
    apiKey: string | undefined;
    timeout: number;

    chat: Chat;
    completions: Completions;
    models: Models;

    constructor(options?: ClientOptions);
    health(): Promise<HealthResponse>;
}

export default LLMClient;
