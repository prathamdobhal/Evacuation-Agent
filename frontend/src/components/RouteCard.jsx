export default function RouteCard({ data }) {
  if (!data) return null;

  const stats = [
    { label: "Distance",  value: `${data.distance_km} km`, icon: "📏" },
    { label: "Walk Time", value: `${data.walk_time_minutes} min`, icon: "🚶" },
    { label: "Capacity",  value: `${data.camp_capacity.toLocaleString()}`, icon: "👥" },
  ];

  return (
    <div className="rounded-xl p-4 bg-[var(--color-success-light)] border border-green-200">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl">🏁</span>
        <div>
          <h3 className="text-base font-bold text-green-700">Evacuation Route Found</h3>
          <p className="text-xs text-[var(--color-text-muted)]">
            Destination: {data.camp_name}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-3">
        {stats.map((s) => (
          <div key={s.label} className="text-center bg-white rounded-lg py-2 px-1 border border-green-100">
            <p className="text-base mb-0.5">{s.icon}</p>
            <p className="text-sm font-bold text-[var(--color-text)]">{s.value}</p>
            <p className="text-[11px] text-[var(--color-text-muted)]">{s.label}</p>
          </div>
        ))}
      </div>

      {data.blocked_roads?.length > 0 && (
        <div>
          <p className="text-xs text-red-600 font-semibold mb-1">⚠️ Blocked Roads</p>
          <div className="flex flex-wrap gap-1">
            {data.blocked_roads.map((road, i) => (
              <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-red-50 text-red-600 border border-red-200">
                {road}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
