export default function SourcePanel({ sources, onClose }) {
  return (
    <div className="w-96 shrink-0 border-l border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-4">
        <h2 className="text-sm font-semibold text-slate-700">
          Retrieved Sources
        </h2>
        <button
          onClick={onClose}
          className="rounded-lg p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
        >
          <svg
            className="h-5 w-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      <div
        className="chat-scrollbar overflow-y-auto p-4"
        style={{ height: "calc(100vh - 65px)" }}
      >
        {sources.map((source, index) => (
          <div
            key={index}
            className="mb-3 rounded-xl border border-slate-100 bg-slate-50 p-4"
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="text-xs font-medium text-slate-600">
                {source.file_name}
              </span>
              {source.score && (
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                    source.score >= 0.6
                      ? "bg-green-100 text-green-700"
                      : source.score >= 0.45
                        ? "bg-amber-100 text-amber-700"
                        : "bg-red-100 text-red-700"
                  }`}
                >
                  {(source.score * 100).toFixed(0)}% match
                </span>
              )}
            </div>

            <p className="text-xs leading-relaxed text-slate-500">
              {source.text.length > 300
                ? source.text.slice(0, 300) + "..."
                : source.text}
            </p>
          </div>
        ))}

        <p className="mt-4 text-center text-xs text-slate-400">
          These are the document chunks retrieved from the source papers. The
          answer is generated based on these chunks.
        </p>
      </div>
    </div>
  );
}
