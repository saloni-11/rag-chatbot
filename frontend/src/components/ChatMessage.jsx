const GUARDRAIL_BADGES = {
  passed: null,
  scope_rejected: {
    label: "Out of scope",
    color: "bg-amber-50 text-amber-700 border border-amber-200",
  },
  low_confidence: {
    label: "Low confidence",
    color: "bg-orange-50 text-orange-700 border border-orange-200",
  },
  error: {
    label: "Error",
    color: "bg-red-50 text-red-700 border border-red-200",
  },
};

export default function ChatMessage({ message }) {
  const isUser = message.role === "user";

  const badge =
    !isUser && message.guardrailAction
      ? GUARDRAIL_BADGES[message.guardrailAction]
      : null;

  return (
    <div className={`mb-4 flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className="flex items-start gap-2.5 max-w-2xl">
        {/* Bot avatar */}
        {!isUser && (
          <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600">
            <svg className="h-3.5 w-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </div>
        )}

        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "rounded-br-md bg-gradient-to-r from-blue-600 to-indigo-600 text-white"
              : "rounded-bl-md border border-slate-100 bg-white text-slate-700 shadow-sm"
          }`}
        >
          {badge && (
            <span
              className={`mb-2 inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${badge.color}`}
            >
              {badge.label}
            </span>
          )}

          <div className="whitespace-pre-wrap">{message.content}</div>

          {!isUser && message.sources && message.sources.length > 0 && (
            <div className="mt-2.5 flex items-center gap-1.5 text-xs text-slate-400">
              <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
              Based on {message.sources.length} source
              {message.sources.length > 1 ? "s" : ""}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
