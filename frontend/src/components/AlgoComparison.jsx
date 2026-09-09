export default function AlgoComparison({ data }) {
  if (!data || data.length === 0) return null;

  const valid = data.filter((a) => a.cost_meters > 0 && a.path_length > 0);
  const bestCost = valid.length > 0 ? Math.min(...valid.map((a) => a.cost_meters)) : -1;
  const bestTime = valid.length > 0 ? Math.min(...valid.map((a) => a.time_ms)) : -1;

  return (
    <div className="rounded-xl p-3 bg-white border border-[var(--color-border)]">
      <div className="flex items-center gap-2 mb-2.5">
        <span className="text-sm">📊</span>
        <h3 className="text-xs font-bold text-[var(--color-text)] uppercase tracking-wider">Algorithm Comparison</h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-[var(--color-text-muted)] border-b border-[var(--color-border)]">
              <th className="text-left pb-1.5 font-bold">Algo</th>
              <th className="text-right pb-1.5 font-bold">Cost (m)</th>
              <th className="text-right pb-1.5 font-bold">Nodes</th>
              <th className="text-right pb-1.5 font-bold">Time</th>
              <th className="text-right pb-1.5 font-bold">Path</th>
            </tr>
          </thead>
          <tbody>
            {data.map((algo) => {
              const isBest = algo.cost_meters === bestCost && algo.cost_meters > 0;
              const isFastest = algo.time_ms === bestTime && algo.time_ms >= 0;
              const noPath = algo.cost_meters <= 0 || algo.path_length === 0;

              return (
                <tr key={algo.algorithm}
                    className={`border-b border-[var(--color-border)]/40 last:border-0 ${noPath ? "opacity-30" : ""}`}>
                  <td className="py-1.5 font-bold text-[var(--color-text)]">
                    {algo.algorithm === "ASTAR" ? "A*" : algo.algorithm}
                    {isBest && <span className="ml-1 text-emerald-500 text-[9px]">★</span>}
                  </td>
                  <td className={`py-1.5 text-right font-mono ${isBest ? "text-emerald-600 font-bold" : "text-[var(--color-text-secondary)]"}`}>
                    {noPath ? "—" : algo.cost_meters.toLocaleString()}
                  </td>
                  <td className="py-1.5 text-right font-mono text-[var(--color-text-secondary)]">
                    {algo.nodes_expanded.toLocaleString()}
                  </td>
                  <td className={`py-1.5 text-right font-mono ${isFastest ? "text-blue-500 font-bold" : "text-[var(--color-text-secondary)]"}`}>
                    {algo.time_ms.toFixed(1)}ms
                  </td>
                  <td className="py-1.5 text-right font-mono text-[var(--color-text-secondary)]">
                    {noPath ? "—" : algo.path_length}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-[var(--color-text-muted)] mt-2">
        ★ lowest cost · <span className="text-blue-500 font-bold">Blue</span> = fastest
      </p>
    </div>
  );
}