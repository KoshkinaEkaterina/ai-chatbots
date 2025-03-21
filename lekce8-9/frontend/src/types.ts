export interface Product {
    id: string;
    name: string;
    price: number;
    category: string[];
    description: string;
    score: number;
    dimensions: Record<string, number>;
    weight?: number;
    confidence: number;
}

export interface ChatResponse {
    message: string;
    conversation_id: string;
    products: Product[];
    criteria?: Record<string, any>;
}

export interface Message {
    role: 'user' | 'assistant';
    content: string;
    products?: Product[];
    criteria?: Record<string, any>;
} 