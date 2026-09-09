import { useEffect, useState, useCallback } from "react";
import { getHealth, getCalamity, predictCalamity, getRoute, getCamps } from "../api/client.js";
import CalamityCard from "./CalamityCard.jsx";
import RouteCard from "./RouteCard.jsx";
import AlgoComparison from "./AlgoComparison.jsx";
import ManualSelector from "./ManualSelector.jsx";

function StepBadge({ num, label, active, done }) {
  return (
    <div className="flex items-center gap-1.5">
      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-all duration-300
        ${done ? "bg-emerald-500 text-white shadow-sm shadow-emerald-200" :
          active ? "bg-[var(--color-accent)] text-white shadow-sm shadow-blue-200" :
          "bg-[var(--color-surface)] text-[var(--color-text-muted)] border border-[var(--color-border)]"}`}>
        {done ? "✓" : num}
      </div>
      <span className={`text-xs font-semibold tracking-wide ${active || done ? "text-[var(--color-text)]" : "text-[var(--color-text-muted)]"}`}>
        {label}
      </span>
    </div>
  );
}

const ALGO_LABELS = { astar: "A*", ucs: "UCS", bfs: "BFS", dfs: "DFS" };

export default function Sidebar({
  calamity, setCalamity, route, setRoute,
  camps, setCamps, userPos,
  loading, setLoading, setError,
  backendOnline, setBackendOnline,
}) {
  const [tab, setTab] = useState("live");
  const [algorithm, setAlgorithm] = useState("astar");

  const step = route ? 3 : calamity && calamity.calamity !== "none" ? 2 : 1;

  useEffect(() => {
    getHealth().then(() => setBackendOnline(true)).catch(() => setBackendOnline(false));
  }, []);

  const fetchLive = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getCalamity();
      setCalamity(data);
      const campData = await getCamps(data.calamity);
      setCamps(campData.camps || []);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  const handleManualPredict = useCallback(async (params) => {
    setLoading(true);
    try {
      const data = await predictCalamity(params);
      setCalamity(data);
      const campData = await getCamps(data.calamity);
      setCamps(campData.camps || []);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, []);

  const handleFindRoute = useCallback(async () => {
    if (!userPos) { setError("Click on the map to set your location first."); return; }
    if (!calamity) { setError("Detect a calamity first."); return; }
    setLoading(true);
    try {
      const data = await getRoute({
        lat: userPos.lat, lng: userPos.lng,
        calamity: calamity.calamity, severity: calamity.severity, algorithm,
      });
      setRoute(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, [userPos, calamity, algorithm]);

  return (
    <aside className="w-[360px] min-w-[360px] h-full bg-white border-r border-[var(--color-border)]
                       flex flex-col z-10 overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-[var(--color-border)] bg-white">
        <div className="flex items-center gap-2.5 mb-3">
          <div className="w-8 h-8 rounded-lg bg-[var(--color-accent)] flex items-center justify-center shadow-md shadow-blue-200">
            <span className="text-white text-base">🛡️</span>
          </div>
          <div>
            <h1 className="text-sm font-bold text-[var(--color-text)] tracking-tight leading-tight">
              Evacuation Agent
            </h1>
            <p className="text-[11px] text-[var(--color-text-muted)] leading-none mt-0.5">
              Sendai, Miyagi · AI-Powered Routing
            </p>
          </div>

          {/* Online indicator in header */}
          <div className="ml-auto flex items-center gap-1.5 px-2 py-1 rounded-full bg-[var(--color-surface)] border border-[var(--color-border)]">
            <span className={`w-1.5 h-1.5 rounded-full ${
              backendOnline === true ? "bg-emerald-500" :
              backendOnline === false ? "bg-red-400" : "bg-amber-400 animate-pulse"
            }`} />
            <span className="text-[10px] font-medium text-[var(--color-text-muted)]">
              {backendOnline === true ? "Live" : backendOnline === false ? "Offline" : "…"}
            </span>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex gap-1 p-1 rounded-lg bg-[var(--color-surface)]">
          {[
            { key: "live", label: "Live Feed", icon: "🌐" },
            { key: "manual", label: "Manual", icon: "🎛️" }
          ].map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all duration-200
                ${tab === t.key
                  ? "bg-white text-[var(--color-text)] shadow-sm"
                  : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]"
                }`}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Progress steps */}
      <div className="px-4 py-2.5 border-b border-[var(--color-border)] bg-[var(--color-surface)]/40 flex items-center justify-between">
        <StepBadge num={1} label="Detect" active={step === 1} done={step > 1} />
        <div className="flex-1 h-px bg-[var(--color-border)] mx-2" />
        <StepBadge num={2} label="Locate" active={step === 2} done={step > 2} />
        <div className="flex-1 h-px bg-[var(--color-border)] mx-2" />
        <StepBadge num={3} label="Route" active={step === 3} done={false} />
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">

        {/* Step 1: Detect calamity */}
        {tab === "live" ? (
          <div className="space-y-3">
            <button
              onClick={fetchLive}
              disabled={loading || backendOnline === false}
              className="w-full py-2.5 rounded-xl text-sm font-semibold transition-all duration-200
                bg-[var(--color-accent)] text-white
                hover:bg-[var(--color-accent-hover)]
                disabled:opacity-40 disabled:cursor-not-allowed
                shadow-sm active:scale-[0.98]"
            >
              {loading && !calamity ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Scanning...
                </span>
              ) : "🔍  Detect Current Calamity"}
            </button>

            {calamity && <CalamityCard data={calamity} />}
          </div>
        ) : (
          <ManualSelector onPredict={handleManualPredict} loading={loading} calamity={calamity} />
        )}

        {/* Step 2: Location + algorithm */}
        {calamity && calamity.calamity !== "none" && (
          <div className="animate-slide-up space-y-2.5 pt-2.5 border-t border-[var(--color-border)]">

            {/* Location display */}
            <div className="rounded-xl bg-[var(--color-surface)] p-3 border border-[var(--color-border)]">
              <p className="text-[10px] text-[var(--color-text-muted)] font-bold uppercase tracking-widest mb-1">
                📍 Your Location
              </p>
              {userPos ? (
                <p className="text-sm text-[var(--color-text)] font-mono font-medium">
                  {userPos.lat.toFixed(5)}, {userPos.lng.toFixed(5)}
                </p>
              ) : (
                <p className="text-xs text-[var(--color-text-muted)] italic">
                  Tap anywhere on the map to drop your pin
                </p>
              )}
            </div>

            {/* Algorithm selector */}
            <div className="rounded-xl bg-[var(--color-surface)] p-3 border border-[var(--color-border)]">
              <p className="text-[10px] text-[var(--color-text-muted)] font-bold uppercase tracking-widest mb-2">
                ⚡ Search Algorithm
              </p>
              <div className="grid grid-cols-4 gap-1">
                {["astar", "ucs", "bfs", "dfs"].map((a) => (
                  <button
                    key={a}
                    onClick={() => setAlgorithm(a)}
                    className={`py-1.5 text-xs font-bold rounded-lg transition-all
                      ${algorithm === a
                        ? "bg-[var(--color-accent)] text-white shadow-sm"
                        : "bg-white text-[var(--color-text-secondary)] hover:bg-white/80 border border-[var(--color-border)]"
                      }`}
                  >
                    {ALGO_LABELS[a]}
                  </button>
                ))}
              </div>
            </div>

            {/* Find route button */}
            <button
              onClick={handleFindRoute}
              disabled={loading || !userPos}
              className="w-full py-2.5 rounded-xl text-sm font-semibold transition-all duration-200
                bg-emerald-500 text-white
                hover:bg-emerald-600
                disabled:opacity-40 disabled:cursor-not-allowed
                shadow-sm active:scale-[0.98]"
            >
              {loading && calamity ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Computing route...
                </span>
              ) : "🚀  Find Evacuation Route"}
            </button>
          </div>
        )}

        {/* Step 3: Results */}
        {route && (
          <div className="animate-slide-up space-y-3 pt-2.5 border-t border-[var(--color-border)]">
            <RouteCard data={route} />
            <AlgoComparison data={route.algo_comparison} />
          </div>
        )}
      </div>
    </aside>
  );
}