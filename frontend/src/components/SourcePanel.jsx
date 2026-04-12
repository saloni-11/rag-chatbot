export default function SourcePanel({ sources, onClose }) {
  return (
    <div className="w-96 shrink-0 border-l border-slate-200/80 bg-white/80 backdrop-blur-sm">
      <div className="flex items-center justify-between border-b border-slate-200/80 px-4 py-3.5">
        <div className="flex items-center gap-2">
          <svg className="h-4 w-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
          <h2 className="text-sm font-semibold text-slate-700">
            Retrieved sources
          </h2>
        </div>
        <button
          onClick={onClose}
          className="rounded-lg p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div
        className="chat-scrollbar overflow-y-auto p-4"
        style={{ height: "calc(100vh - 60px)" }}
      >
        {sources.map((source, index) => (
          <div
            key={index}
            className="mb-3 rounded-xl border border-slate-100 bg-white p-4 shadow-sm"
          >
            <div className="mb-2 flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <svg className="h-3.5 w-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                </svg>
                <span className="text-xs font-medium text-slate-600">
                  {source.file_name}
                </span>
              </div>
              {source.score && (
                <span
                  className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                    source.score >= 0.6
                      ? "bg-emerald-50 text-emerald-700"
                      : source.score >= 0.45
                        ? "bg-amber-50 text-amber-700"
                        : "bg-red-50 text-red-700"
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

        <div className="mt-4 rounded-lg bg-slate-50 p-3 text-center">
          <p className="text-xs text-slate-400">
            Answers are generated exclusively from these retrieved chunks.
            No external knowledge is used.
          </p>
        </div>
      </div>
    </div>
  );
}
