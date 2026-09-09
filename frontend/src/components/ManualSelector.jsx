import { useState } from "react";
import CalamityCard from "./CalamityCard.jsx";

const PRESETS = {
  earthquake_moderate: {
    label: "🔶 Moderate Quake",
    magnitude: 6.5, depth_km: 25, lat: 38.3, lng: 140.9,
    central_pressure_hpa: 1013, max_wind_knots: 0, wave_intensity: 0,
  },
  earthquake_major: {
    label: "🔶 Major Quake",
    magnitude: 8.5, depth_km: 10, lat: 38.27, lng: 141.02,
    central_pressure_hpa: 1013, max_wind_knots: 0, wave_intensity: 0,
  },
  tsunami: {
    label: "🌊 Tsunami",
    magnitude: 7.0, depth_km: 15, lat: 38.28, lng: 141.05,
    central_pressure_hpa: 1013, max_wind_knots: 0, wave_intensity: 8.5,
  },
  typhoon: {
    label: "🌀 Typhoon",
    magnitude: 0, depth_km: 0, lat: 38.25, lng: 140.85,
    central_pressure_hpa: 935, max_wind_knots: 95, wave_intensity: 0,
  },
  super_typhoon: {
    label: "🌀 Super Typhoon",
    magnitude: 0, depth_km: 0, lat: 38.25, lng: 140.85,
    central_pressure_hpa: 890, max_wind_knots: 140, wave_intensity: 0,
  },
};

const FIELDS = [
  { key: "magnitude",            label: "Magnitude",       min: 0, max: 10, step: 0.1 },
  { key: "depth_km",             label: "Depth (km)",      min: 0, max: 700, step: 1 },
  { key: "lat",                  label: "Latitude",        min: 30, max: 46, step: 0.01 },
  { key: "lng",                  label: "Longitude",       min: 130, max: 146, step: 0.01 },
  { key: "central_pressure_hpa", label: "Pressure (hPa)",  min: 870, max: 1030, step: 1 },
  { key: "max_wind_knots",       label: "Wind (knots)",    min: 0, max: 200, step: 1 },
  { key: "wave_intensity",       label: "Wave Intensity",  min: 0, max: 15, step: 0.1 },
];

const defaults = {
  magnitude: 0, depth_km: 10, lat: 38.27, lng: 140.9,
  central_pressure_hpa: 1013, max_wind_knots: 0, wave_intensity: 0,
};

export default function ManualSelector({ onPredict, loading, calamity }) {
  const [params, setParams] = useState({ ...defaults });

  const set = (key, val) => setParams((p) => ({ ...p, [key]: parseFloat(val) || 0 }));

  const applyPreset = (key) => {
    const { label, ...vals } = PRESETS[key];
    setParams(vals);
  };

  return (
    <div className="space-y-3 animate-fade-in">
      {/* Presets */}
      <div>
        <p className="text-xs text-[var(--color-text-muted)] mb-2 font-semibold uppercase tracking-wider">
          Quick Presets
        </p>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(PRESETS).map(([key, preset]) => (
            <button
              key={key}
              onClick={() => applyPreset(key)}
              className="text-xs px-3 py-2 rounded-lg bg-[var(--color-surface)]
                         text-[var(--color-text-secondary)] border border-[var(--color-border)]
                         hover:bg-white hover:border-[var(--color-accent)] hover:text-[var(--color-accent)]
                         transition-all"
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-2">
        {FIELDS.map((f) => (
          <div key={f.key}>
            <div className="flex justify-between text-xs mb-0.5">
              <span className="text-[var(--color-text-muted)] font-medium">{f.label}</span>
              <span className="text-[var(--color-text)] font-mono font-bold">{params[f.key]}</span>
            </div>
            <input
              type="range"
              min={f.min} max={f.max} step={f.step}
              value={params[f.key]}
              onChange={(e) => set(f.key, e.target.value)}
              className="w-full h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer
                         [&::-webkit-slider-thumb]:appearance-none
                         [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
                         [&::-webkit-slider-thumb]:rounded-full
                         [&::-webkit-slider-thumb]:bg-[var(--color-accent)]
                         [&::-webkit-slider-thumb]:shadow-md
                         [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>
        ))}
      </div>

      {/* Predict button */}
      <button
        onClick={() => onPredict(params)}
        disabled={loading}
        className="w-full py-3 rounded-xl text-base font-semibold transition-all duration-200
          bg-[var(--color-accent)] text-white
          hover:bg-[var(--color-accent-hover)]
          disabled:opacity-40 disabled:cursor-not-allowed
          shadow-sm active:scale-[0.98]"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Predicting...
          </span>
        ) : "🧠  Run ML Prediction"}
      </button>

      {calamity && <CalamityCard data={calamity} />}
    </div>
  );
}
