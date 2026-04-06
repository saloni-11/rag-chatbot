import { useState, useRef, useEffect } from "react";
import ChatMessage from "./components/ChatMessage";
import SourcePanel from "./components/SourcePanel";

async function askQuestion(question) {
  const response = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!response.ok) {
    throw new Error("API error: " + response.status);
  }

  return response.json();
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeSources, setActiveSources] = useState(null);
  const [showSources, setShowSources] = useState(false);

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function sendQuestion(question) {
    if (!question || isLoading) return;

    const userMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const data = await askQuestion(question);
      const botMessage = {
        role: "bot",
        content: data.answer,
        sources: data.sources,
        guardrailAction: data.guardrail_action,
      };
      setMessages((prev) => [...prev, botMessage]);

      if (data.sources && data.sources.length > 0) {
        setActiveSources(data.sources);
      } else {
        setActiveSources(null);
      }
    } catch (error) {
      const errorMessage = {
        role: "bot",
        content:
          "Sorry, I couldn't reach the server. Make sure the FastAPI backend is running: uvicorn src.api.main:app --reload",
        guardrailAction: "error",
      };
      setMessages((prev) => [...prev, errorMessage]);
      setActiveSources(null);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSend() {
    sendQuestion(input.trim());
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex h-screen bg-slate-50">
      <div className="flex flex-1 flex-col">
        <header className="border-b border-slate-200 bg-white px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-slate-800">
                AI/ML Interview Prep
              </h1>
              <p className="text-sm text-slate-500">
                Ask questions about ML, deep learning, NLP and data analytics
              </p>
            </div>
            {activeSources && (
              <button
                onClick={() => setShowSources(!showSources)}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-600 transition-colors hover:bg-slate-100"
              >
                {showSources ? "Hide" : "Show"} sources ({activeSources.length})
              </button>
            )}
          </div>
        </header>

        <div className="chat-scrollbar flex-1 overflow-y-auto px-4 py-6">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <div className="mb-6 text-5xl">&#x1F916;</div>
              <h2 className="mb-2 text-xl font-medium text-slate-700">
                Ready to prep for your interview
              </h2>
              <p className="mb-8 max-w-md text-slate-500">
                Ask me anything about machine learning, transformers, BERT,
                attention mechanisms, and more.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "What is self-attention?",
                  "How does BERT handle bidirectional context?",
                  "Explain positional encoding",
                ].map((q) => (
                  <button
                    key={q}
                    onClick={() => sendQuestion(q)}
                    className="rounded-full border border-slate-200 bg-white px-4 py-2 text-sm text-slate-600 transition-all hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, index) => (
            <ChatMessage key={index} message={msg} />
          ))}

          {isLoading && (
            <div className="mb-4 flex justify-start">
              <div className="rounded-2xl rounded-bl-md bg-white px-5 py-3 shadow-sm border border-slate-100">
                <div className="dot-animation flex gap-1">
                  <span className="inline-block h-2 w-2 rounded-full bg-slate-400" />
                  <span className="inline-block h-2 w-2 rounded-full bg-slate-400" />
                  <span className="inline-block h-2 w-2 rounded-full bg-slate-400" />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <div className="border-t border-slate-200 bg-white px-4 py-4">
          <div className="mx-auto flex max-w-3xl items-center gap-3">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask an AI/ML interview question..."
              disabled={isLoading}
              className="flex-1 rounded-xl border border-slate-300 px-4 py-3 text-sm outline-none transition-colors placeholder:text-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:bg-slate-50 disabled:text-slate-400"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              Send
            </button>
          </div>
        </div>
      </div>

      {showSources && activeSources && (
        <SourcePanel
          sources={activeSources}
          onClose={() => setShowSources(false)}
        />
      )}
    </div>
  );
}
