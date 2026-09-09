import { useState, useCallback } from "react";
import MapView from "./components/MapView.jsx";
import Sidebar from "./components/Sidebar.jsx";
import CalamityBanner from "./components/CalamityBanner.jsx";
import StatusBar from "./components/StatusBar.jsx";

export default function App() {
  const [calamity, setCalamity] = useState(null);
  const [route, setRoute] = useState(null);
  const [camps, setCamps] = useState([]);
  const [userPos, setUserPos] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null);

  const clearError = useCallback(() => setError(null), []);

  return (
    <div className="relative h-full w-full flex overflow-hidden bg-[var(--color-bg)]">
      {/* Sidebar */}
      <Sidebar
        calamity={calamity}
        setCalamity={setCalamity}
        route={route}
        setRoute={setRoute}
        camps={camps}
        setCamps={setCamps}
        userPos={userPos}
        loading={loading}
        setLoading={setLoading}
        setError={setError}
        backendOnline={backendOnline}
        setBackendOnline={setBackendOnline}
      />

      {/* Map area */}
      <main className="flex-1 relative">
        <CalamityBanner calamity={calamity} />

        <MapView
          route={route}
          camps={camps}
          userPos={userPos}
          setUserPos={setUserPos}
          calamity={calamity}
        />

        <StatusBar
          backendOnline={backendOnline}
          calamity={calamity}
          route={route}
        />

        {/* Error toast */}
        {error && (
          <div
            className="absolute top-16 right-4 bg-white rounded-xl px-4 py-3 text-sm
                        border border-red-200 shadow-lg animate-slide-up cursor-pointer
                        max-w-xs z-[1000] flex items-start gap-2"
            onClick={clearError}
          >
            <span className="text-red-500 text-base mt-0.5">⚠️</span>
            <div>
              <p className="font-semibold text-red-600 text-xs">Error</p>
              <p className="text-[var(--color-text-secondary)] text-xs mt-0.5">{error}</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
