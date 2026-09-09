const THEMES = {
  tsunami:    { bg: "bg-sky-50", border: "border-sky-300", text: "text-sky-700", icon: "🌊" },
  earthquake: { bg: "bg-orange-50", border: "border-orange-300", text: "text-orange-700", icon: "🔶" },
  typhoon:    { bg: "bg-violet-50", border: "border-violet-300", text: "text-violet-700", icon: "🌀" },
  none:       { bg: "bg-gray-50", border: "border-gray-200", text: "text-gray-500", icon: "✅" },
};

export default function CalamityBanner({ calamity }) {
  if (!calamity) return null;

  const t = THEMES[calamity.calamity] || THEMES.none;
  const isAlert = calamity.calamity !== "none";

  return (
    <div className={`absolute top-3 left-1/2 -translate-x-1/2 z-[500] px-4 py-2 rounded-full
                     ${t.bg} border ${t.border} shadow-md
                     animate-slide-up flex items-center gap-2.5`}>
      {/* Pulsing dot */}
      <span className="relative flex h-2.5 w-2.5">
        {isAlert && (
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-50
            ${calamity.calamity === "tsunami" ? "bg-sky-400" :
              calamity.calamity === "earthquake" ? "bg-orange-400" :
              calamity.calamity === "typhoon" ? "bg-violet-400" : "bg-gray-400"}`} />
        )}
        <span className={`relative inline-flex rounded-full h-2.5 w-2.5
          ${isAlert ? (calamity.calamity === "tsunami" ? "bg-sky-500" :
            calamity.calamity === "earthquake" ? "bg-orange-500" : "bg-violet-500") : "bg-green-500"}`} />
      </span>

      <span className={`text-sm font-semibold ${t.text}`}>
        {t.icon}{" "}
        {isAlert
          ? `${calamity.calamity.toUpperCase()} — ${calamity.severity_label.toUpperCase()} (${(calamity.severity * 100).toFixed(0)}%)`
          : "All Clear — No Active Calamity"
        }
      </span>
    </div>
  );
}
