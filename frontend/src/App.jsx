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

const SUGGESTED_QUESTIONS = [
  {
    category: "Transformers",
    icon: "\u2699\uFE0F",
    questions: [
      "What is self-attention and why is it important?",
      "Explain positional encoding in transformers",
    ],
  },
  {
    category: "BERT",
    icon: "\uD83E\uDDE0",
    questions: [
      "How does BERT handle bidirectional context?",
      "What is masked language modeling?",
    ],
  },
  {
    category: "Fundamentals",
    icon: "\uD83D\uDCCA",
    questions: [
      "What is the difference between encoder and decoder?",
      "How does multi-head attention work?",
    ],
  },
];

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
    <div className="flex h-screen bg-gradient-to-br from-slate-50 to-blue-50/30">
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <header className="border-b border-slate-200/80 bg-white/80 backdrop-blur-sm px-6 py-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600">
                <svg className="h-5 w-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.62 48.62 0 0 1 12 20.904a48.62 48.62 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.636 50.636 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.903 59.903 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0 1 12 13.489a50.702 50.702 0 0 1 7.74-3.342M6.75 15a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm0 0v-3.675A55.378 55.378 0 0 1 12 8.443m-7.007 11.55A5.981 5.981 0 0 0 6.75 15.75v-1.5" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
                  AI/ML Study Companion
                  <span className="rounded-md bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
                    RAG
                  </span>
                </h1>
                <p className="text-xs text-slate-500">
                  Answers grounded in research papers — not hallucinated
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {activeSources && (
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-1.5 text-sm text-slate-600 transition-colors hover:bg-slate-50"
                >
                  <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                  </svg>
                  {showSources ? "Hide" : "Show"} sources ({activeSources.length})
                </button>
              )}
            </div>
          </div>
        </header>

        {/* Chat area */}
        <div className="chat-scrollbar flex-1 overflow-y-auto px-4 py-6">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center px-4">
              <div className="mb-2 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 text-3xl shadow-lg shadow-blue-600/20">
                <svg className="h-8 w-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.438 60.438 0 0 0-.491 6.347A48.62 48.62 0 0 1 12 20.904a48.62 48.62 0 0 1 8.232-4.41 60.46 60.46 0 0 0-.491-6.347m-15.482 0a50.636 50.636 0 0 0-2.658-.813A59.906 59.906 0 0 1 12 3.493a59.903 59.903 0 0 1 10.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0 1 12 13.489a50.702 50.702 0 0 1 7.74-3.342M6.75 15a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Zm0 0v-3.675A55.378 55.378 0 0 1 12 8.443m-7.007 11.55A5.981 5.981 0 0 0 6.75 15.75v-1.5" />
                </svg>
              </div>
              <h2 className="mb-1.5 text-2xl font-semibold text-slate-800">
                Your AI/ML study companion
              </h2>
              <p className="mb-8 max-w-lg text-slate-500 text-sm leading-relaxed">
                Learn machine learning concepts with answers retrieved from
                real research papers. Source chunks are shown for deeper study.
              </p>

              <div className="w-full max-w-2xl">
                <div className="grid grid-cols-3 gap-3">
                  {SUGGESTED_QUESTIONS.map((cat) => (
                    <div
                      key={cat.category}
                      className="rounded-xl border border-slate-200/80 bg-white p-4 text-left"
                    >
                      <div className="mb-3 flex items-center gap-2">
                        <span className="text-base">{cat.icon}</span>
                        <span className="text-xs font-semibold text-slate-700">
                          {cat.category}
                        </span>
                      </div>
                      <div className="flex flex-col gap-2">
                        {cat.questions.map((q) => (
                          <button
                            key={q}
                            onClick={() => sendQuestion(q)}
                            className="rounded-lg bg-slate-50 px-3 py-2 text-left text-xs text-slate-600 transition-all hover:bg-blue-50 hover:text-blue-700"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-6 flex items-center justify-center gap-6 text-xs text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <div className="h-1.5 w-1.5 rounded-full bg-blue-400"></div>
                    Your question is embedded
                  </div>
                  <span>&#x2192;</span>
                  <div className="flex items-center gap-1.5">
                    <div className="h-1.5 w-1.5 rounded-full bg-teal-400"></div>
                    Similar chunks retrieved
                  </div>
                  <span>&#x2192;</span>
                  <div className="flex items-center gap-1.5">
                    <div className="h-1.5 w-1.5 rounded-full bg-indigo-400"></div>
                    LLM generates grounded answer
                  </div>
                </div>
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
                  <span className="inline-block h-2 w-2 rounded-full bg-blue-400" />
                  <span className="inline-block h-2 w-2 rounded-full bg-blue-400" />
                  <span className="inline-block h-2 w-2 rounded-full bg-blue-400" />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input area */}
        <div className="border-t border-slate-200/80 bg-white/80 backdrop-blur-sm px-4 py-4">
          <div className="mx-auto flex max-w-3xl items-center gap-3">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask an AI/ML question..."
              disabled={isLoading}
              className="flex-1 rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm outline-none transition-all placeholder:text-slate-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 disabled:bg-slate-50 disabled:text-slate-400"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 px-5 py-3 text-sm font-medium text-white transition-all hover:shadow-md hover:shadow-blue-600/25 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-300 disabled:shadow-none"
            >
              Send
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
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
