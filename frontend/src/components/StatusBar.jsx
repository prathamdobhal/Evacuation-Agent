export default function StatusBar({ backendOnline, calamity, route }) {
  return (
    <div className="absolute bottom-4 left-4 z-[500] flex items-center gap-2">
      {/* Backend status */}
      <div className="bg-white/90 backdrop-blur-sm rounded-full px-3 py-1.5 flex items-center gap-1.5 shadow-md border border-white/60">
        <span className={`w-1.5 h-1.5 rounded-full ${
          backendOnline === true ? "bg-emerald-500" :
          backendOnline === false ? "bg-red-400" :
          "bg-amber-400 animate-pulse"
        }`} />
        <span className="text-[11px] text-[var(--color-text-muted)] font-semibold">
          {backendOnline === true ? "API Online" :
           backendOnline === false ? "API Offline" : "Checking…"}
        </span>
      </div>

      {/* Route quick stats */}
      {route && (
        <div className="bg-white/90 backdrop-blur-sm rounded-full px-3 py-1.5 flex items-center gap-3 shadow-md border border-white/60">
          <span className="text-[11px] text-[var(--color-text-secondary)] font-semibold flex items-center gap-1">
            <span className="text-blue-500">📏</span> {route.distance_km} km
          </span>
          <span className="w-px h-3 bg-[var(--color-border)]" />
          <span className="text-[11px] text-[var(--color-text-secondary)] font-semibold flex items-center gap-1">
            <span>🚶</span> {route.walk_time_minutes} min
          </span>
          <span className="w-px h-3 bg-[var(--color-border)]" />
          <span className="text-[11px] text-[var(--color-text-secondary)] font-semibold flex items-center gap-1">
            <span>📍</span> {route.path_coords?.length || 0} nodes
          </span>
        </div>
      )}
    </div>
  );
}