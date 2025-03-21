import React, { useState, useRef, useEffect } from 'react';
import { Message, Product } from '../types';
import ProductList from './ProductList';

export default function Chat() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [conversationId, setConversationId] = useState<string | null>(null);
    const messagesEndRef = useRef<null | HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim()) return;

        // Add user message
        setMessages(prev => [...prev, { role: 'user', content: input }]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: input,
                    conversation_id: conversationId
                }),
            });

            const data = await response.json();
            
            // Store conversation ID
            if (!conversationId) {
                setConversationId(data.conversation_id);
            }

            // Add assistant message with products
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.message,
                products: data.products,
                criteria: data.criteria
            }]);

        } catch (error) {
            console.error('Error:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: 'Sorry, there was an error processing your request.'
            }]);
        }

        setIsLoading(false);
    };

    return (
        <div className="flex flex-col h-screen max-w-4xl mx-auto p-4">
            <div className="flex-1 overflow-y-auto mb-4 space-y-4">
                {messages.map((message, index) => (
                    <div key={index} className="space-y-2">
                        <div className={`p-4 rounded-lg ${
                            message.role === 'user' 
                                ? 'bg-blue-100 ml-auto max-w-[80%]' 
                                : 'bg-gray-100 mr-auto max-w-[80%]'
                        }`}>
                            <p className="whitespace-pre-wrap">{message.content}</p>
                        </div>
                        
                        {/* Show products if available */}
                        {message.products && message.products.length > 0 && (
                            <div className="ml-4">
                                <ProductList 
                                    products={message.products}
                                    criteria={message.criteria}
                                />
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && (
                    <div className="bg-gray-100 p-4 rounded-lg mr-auto">
                        Thinking...
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="flex gap-2">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    className="flex-1 p-2 border rounded"
                    placeholder="Type your message..."
                />
                <button 
                    type="submit"
                    disabled={isLoading}
                    className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-blue-300"
                >
                    Send
                </button>
            </form>
        </div>
    );
} 