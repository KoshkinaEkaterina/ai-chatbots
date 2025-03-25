'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant' | 'system' | 'analysis';
  content: string;
  metrics?: {
    emotional?: any;
    cognitive?: any;
    engagement?: any;
    formality?: any;
  };
  topic_stats?: {
    topic_id: string;
    topic_question: string;
    questions_asked: number;
    factor_coverage: {
      [key: string]: {
        coverage: number;
        questions: number;
      };
    };
  };
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const makeRequest = async (url: string, message: string | null) => {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ message }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    try {
      setMessages(prev => [...prev, { role: 'user', content: input }]);
      setInput('');
      const apiUrl = `${process.env.NEXT_PUBLIC_API_URL}/chat`;

      // Make initial request if this is the first message
      if (messages.length === 0) {
        const initData = await makeRequest(apiUrl, null);
        if (initData.response) {
          // Add the initial bot message
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: initData.response
          }]);
        }
      }

      // Send actual user message
      const data = await makeRequest(apiUrl, input);
      
      if (data.response) {
        // Add humanity analysis if present
        if (data.humanity_analysis) {
          setMessages(prev => [...prev, {
            role: 'analysis',
            content: 'Humanity Analysis',
            metrics: {
              emotional: data.humanity_analysis.emotional,
              cognitive: data.humanity_analysis.cognitive,
              engagement: data.humanity_analysis.engagement,
              formality: data.humanity_analysis.formality,
            }
          }]);
        }

        // Add bot response
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.response
        }]);

        if (data.topic_stats) {
          setMessages(prev => [...prev, {
            role: 'system',
            content: 'Topic Coverage Analysis',
            topic_stats: data.topic_stats
          }]);
        }
      }

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Omlouvám se, došlo k chybě při komunikaci.'
      }]);
    }
  };

  const renderAnalysisMetrics = (metrics: any) => {
    return (
      <div className="space-y-2 font-mono text-sm">
        {metrics.emotional && (
          <div>
            <div className="font-bold">Emotional State:</div>
            <div>Weight: {metrics.emotional.emotional_weight?.toFixed(2)}</div>
            <div>Key Emotions: {metrics.emotional.key_emotions?.join(', ')}</div>
            <div>Complexity: {metrics.emotional.emotional_complexity?.toFixed(2)}</div>
          </div>
        )}
        {metrics.cognitive && (
          <div>
            <div className="font-bold">Cognitive State:</div>
            <div>Load: {metrics.cognitive.current_load?.toFixed(2)}</div>
            <div>Mental Effort: {metrics.cognitive.mental_effort_level?.toFixed(2)}</div>
          </div>
        )}
        {metrics.engagement && (
          <div>
            <div className="font-bold">Engagement:</div>
            <div>Level: {metrics.engagement.engagement_level?.toFixed(2)}</div>
          </div>
        )}
      </div>
    );
  };

  const renderTopicStats = (stats: any) => {
    return (
      <div className="font-mono text-sm">
        <div className="font-bold mb-2">Topic: {stats.topic_question}</div>
        <div className="mb-2">Questions Asked: {stats.questions_asked}</div>
        <div className="font-bold mb-1">Factor Coverage:</div>
        <table className="w-full text-left">
          <thead>
            <tr>
              <th className="pr-4">Factor</th>
              <th className="px-2">Coverage</th>
              <th className="px-2">Questions</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(stats.factor_coverage).map(([factor, data]: [string, any]) => (
              <tr key={factor}>
                <td className="pr-4">{factor}</td>
                <td className="px-2">{(data.coverage * 100).toFixed(1)}%</td>
                <td className="px-2">{data.questions}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4">
      <div className="w-full max-w-4xl bg-white rounded-lg shadow-xl overflow-hidden">
        {/* Chat header */}
        <div className="bg-blue-600 text-white p-4">
          <h1 className="text-xl font-bold">Interview Bot</h1>
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
                    : message.role === 'system'
                    ? 'bg-yellow-100 text-gray-900 font-mono'
                    : message.role === 'analysis'
                    ? 'bg-purple-100 text-gray-900 w-full'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                {message.role === 'analysis' && message.metrics 
                  ? renderAnalysisMetrics(message.metrics)
                  : message.role === 'system' && message.topic_stats
                  ? renderTopicStats(message.topic_stats)
                  : message.content}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-center">
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