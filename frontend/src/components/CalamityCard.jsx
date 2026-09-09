const THEMES = {
  tsunami:    { bg: "bg-[var(--color-tsunami-bg)]", border: "border-[var(--color-tsunami)]", text: "text-[var(--color-tsunami)]", icon: "🌊" },
  earthquake: { bg: "bg-[var(--color-earthquake-bg)]", border: "border-[var(--color-earthquake)]", text: "text-[var(--color-earthquake)]", icon: "🔶" },
  typhoon:    { bg: "bg-[var(--color-typhoon-bg)]", border: "border-[var(--color-typhoon)]", text: "text-[var(--color-typhoon)]", icon: "🌀" },
  none:       { bg: "bg-[var(--color-surface)]", border: "border-[var(--color-border)]", text: "text-[var(--color-text-muted)]", icon: "✅" },
};

const SEV_BADGES = {
  low:      "bg-green-100 text-green-700",
  moderate: "bg-yellow-100 text-yellow-700",
  high:     "bg-orange-100 text-orange-700",
  critical: "bg-red-100 text-red-700",
};

const SEV_BAR_COLORS = {
  low:      "bg-green-500",
  moderate: "bg-yellow-500",
  high:     "bg-orange-500",
  critical: "bg-red-500",
};

export default function CalamityCard({ data }) {
  if (!data) return null;

  const t = THEMES[data.calamity] || THEMES.none;
  const badge = SEV_BADGES[data.severity_label] || SEV_BADGES.low;
  const barColor = SEV_BAR_COLORS[data.severity_label] || "bg-gray-400";

  return (
    <div className={`rounded-xl p-4 ${t.bg} border ${t.border}/30 animate-slide-up`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xl">{t.icon}</span>
          <div>
            <h3 className={`text-base font-bold uppercase tracking-wide ${t.text}`}>
              {data.calamity}
            </h3>
            <p className="text-xs text-[var(--color-text-muted)]">
              Source: {data.source}
            </p>
          </div>
        </div>
        <span className={`px-2.5 py-1 text-xs font-bold rounded-full uppercase ${badge}`}>
          {data.severity_label}
        </span>
      </div>

      {/* Description */}
      <p className="text-sm text-[var(--color-text-secondary)] mb-3 leading-relaxed">
        {data.description}
      </p>

      {/* Severity bar */}
      <div className="mb-3">
        <div className="flex justify-between text-xs text-[var(--color-text-muted)] mb-1">
          <span>Severity</span>
          <span className="font-mono font-semibold">{(data.severity * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${barColor}`}
            style={{ width: `${Math.max(data.severity * 100, 2)}%` }}
          />
        </div>
      </div>

      {/* Probabilities */}
      <div className="grid grid-cols-3 gap-2">
        {Object.entries(data.probabilities || {}).map(([cls, prob]) => (
          <div key={cls} className="text-center bg-white/70 rounded-lg py-1.5 px-1 border border-gray-100">
            <p className="text-xs text-[var(--color-text-muted)] capitalize">{cls}</p>
            <p className="text-base font-bold text-[var(--color-text)] font-mono">
              {(prob * 100).toFixed(1)}%
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
