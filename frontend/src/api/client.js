/**
 * API client for the Evacuation Agent backend.
 * Uses native fetch + proxy (configured in vite.config.js).
 */

const BASE = import.meta.env.VITE_API_URL || "";

async function request(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const getCalamity = () => request("/calamity/current");

export const predictCalamity = (params) =>
  request("/calamity/predict", {
    method: "POST",
    body: JSON.stringify(params),
  });

export const getRoute = ({ lat, lng, calamity, severity, algorithm = "astar" }) => {
  const qs = new URLSearchParams({ lat, lng, calamity, severity, algorithm });
  return request(`/route?${qs}`);
};

export const getCamps = (calamity = "none") =>
  request(`/camps?calamity=${calamity}`);

export const getHealth = () => request("/health");