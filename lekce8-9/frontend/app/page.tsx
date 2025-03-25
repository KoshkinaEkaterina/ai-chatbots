'use client';

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ApiResponse {
  message: string;  // This contains the markdown formatted response
  conversation_id: string;
  products: any[];  // Add proper typing if needed
  criteria: any;    // Add proper typing if needed
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    
    if (!trimmedInput || isLoading) return;

    try {
      // Add user message
      setMessages(prev => [...prev, { role: 'user', content: trimmedInput }]);
      setIsLoading(true);
      setInput('');

      // Send to API
      const response = await fetch('http://localhost:8001/chat', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: trimmedInput,
          conversation_id: localStorage.getItem('conversation_id') || undefined,
          products: [],  // Only send if we have products to compare
          criteria: {}   // Only send if we have criteria
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      
      // Store conversation ID
      if (data.conversation_id) {
        localStorage.setItem('conversation_id', data.conversation_id);
      }
      
      // Add bot response
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.message
      }]);

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error in communication. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = (content: string, role: 'user' | 'assistant') => {
    if (role === 'user') {
      return <div className="whitespace-pre-wrap">{content}</div>;
    }

    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({node, ...props}) => <h1 className="text-2xl font-bold mb-4 mt-2" {...props}/>,
          h2: ({node, ...props}) => <h2 className="text-xl font-bold mb-3 mt-2" {...props}/>,
          h3: ({node, ...props}) => <h3 className="text-lg font-bold mb-2 mt-2" {...props}/>,
          p: ({node, ...props}) => <p className="mb-4" {...props}/>,
          ul: ({node, ...props}) => <ul className="list-disc ml-6 mb-4" {...props}/>,
          ol: ({node, ...props}) => <ol className="list-decimal ml-6 mb-4" {...props}/>,
          li: ({node, ...props}) => <li className="mb-2" {...props}/>,
          blockquote: ({node, ...props}) => (
            <blockquote className="border-l-4 border-blue-500 pl-4 italic my-4" {...props}/>
          ),
          code: ({node, inline, ...props}) => (
            inline ? 
              <code className="bg-gray-100 rounded px-1 py-0.5" {...props}/> :
              <code className="block bg-gray-100 rounded p-4 my-4 overflow-x-auto" {...props}/>
          ),
          table: ({node, ...props}) => (
            <div className="overflow-x-auto my-4">
              <table className="min-w-full border-collapse border border-gray-300" {...props}/>
            </div>
          ),
          th: ({node, ...props}) => (
            <th className="border border-gray-300 bg-gray-100 px-4 py-2" {...props}/>
          ),
          td: ({node, ...props}) => (
            <td className="border border-gray-300 px-4 py-2" {...props}/>
          ),
          hr: ({node, ...props}) => <hr className="my-4 border-t border-gray-300" {...props}/>,
          strong: ({node, ...props}) => <strong className="font-bold" {...props}/>,
          em: ({node, ...props}) => <em className="italic" {...props}/>,
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4">
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-xl overflow-hidden">
        {/* Chat header */}
        <div className="bg-blue-600 text-white p-4">
          <h1 className="text-xl font-bold">Chat Bot</h1>
        </div>

        {/* Messages container */}
        <div className="h-[600px] overflow-y-auto p-4 space-y-4">
          {messages.map((message, i) => (
            <div
              key={i}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] p-4 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                {renderMessage(message.content, message.role)}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 text-gray-600 p-4 rounded-lg">
                Thinking...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input form */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
          <div className="flex gap-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </main>
  );
} 