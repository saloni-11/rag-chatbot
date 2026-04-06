const GUARDRAIL_BADGES = {
  passed: null,
  scope_rejected: {
    label: "Out of scope",
    color: "bg-amber-100 text-amber-700",
  },
  low_confidence: {
    label: "Low confidence",
    color: "bg-orange-100 text-orange-700",
  },
  error: {
    label: "Error",
    color: "bg-red-100 text-red-700",
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
      <div
        className={`max-w-2xl rounded-2xl px-5 py-3 text-sm leading-relaxed ${
          isUser
            ? "rounded-br-md bg-blue-600 text-white"
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
          <div className="mt-2 text-xs text-slate-400">
            Based on {message.sources.length} source
            {message.sources.length > 1 ? "s" : ""}
          </div>
        )}
      </div>
    </div>
  );
}
