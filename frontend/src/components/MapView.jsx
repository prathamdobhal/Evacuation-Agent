import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const CENTER = [38.268, 140.87];
const ZOOM = 13;

function makeIcon(color, size = 24, border = "#fff") {
  return L.divIcon({
    html: `<div style="
      width:${size}px;height:${size}px;border-radius:50%;
      background:${color};border:3px solid ${border};
      box-shadow:0 2px 12px ${color}66, 0 1px 3px rgba(0,0,0,0.2);
    "></div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    className: "",
  });
}

const CAMP_COLORS = {
  tsunami_tower: "#0ea5e9",
  school: "#f59e0b",
  community_center: "#8b5cf6",
  gymnasium: "#3b82f6",
  park: "#22c55e",
  hospital: "#ef4444",
};

// User: vivid indigo pulse ring effect
const userIcon = L.divIcon({
  html: `<div style="position:relative;width:32px;height:32px;">
    <div style="position:absolute;inset:0;border-radius:50%;background:#4f6ef7;opacity:0.25;animation:ping 1.5s cubic-bezier(0,0,0.2,1) infinite;"></div>
    <div style="position:absolute;inset:4px;border-radius:50%;background:#4f6ef7;border:3px solid #fff;box-shadow:0 2px 10px #4f6ef766;"></div>
  </div>
  <style>@keyframes ping{75%,100%{transform:scale(2);opacity:0}}</style>`,
  iconSize: [32, 32],
  iconAnchor: [16, 16],
  className: "",
});

const destIcon = makeIcon("#22c55e", 28, "#fff");

export default function MapView({ route, camps, userPos, setUserPos, calamity }) {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const routeLayer = useRef(null);
  const campMarkers = useRef([]);
  const userMarker = useRef(null);
  const destMarker = useRef(null);

  useEffect(() => {
    if (mapInstance.current) return;

    const map = L.map(mapRef.current, {
      center: CENTER,
      zoom: ZOOM,
      zoomControl: true,
      attributionControl: true,
    });

    L.tileLayer("https://tile.openstreetmap.de/{z}/{x}/{y}.png", {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19,
    }).addTo(map);

    map.on("click", (e) => {
      setUserPos({ lat: e.latlng.lat, lng: e.latlng.lng });
    });

    mapInstance.current = map;
    return () => { map.remove(); mapInstance.current = null; };
  }, []);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    if (userMarker.current) { map.removeLayer(userMarker.current); userMarker.current = null; }
    if (userPos) {
      userMarker.current = L.marker([userPos.lat, userPos.lng], { icon: userIcon })
        .addTo(map)
        .bindPopup(`
          <div style="font-family:'Plus Jakarta Sans',system-ui,sans-serif;font-size:13px;line-height:1.5;">
            <strong style="color:#4f6ef7;">📍 Your Location</strong><br/>
            <span style="color:#5f6577;">${userPos.lat.toFixed(5)}, ${userPos.lng.toFixed(5)}</span>
          </div>
        `);
    }
  }, [userPos]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    campMarkers.current.forEach((m) => map.removeLayer(m));
    campMarkers.current = [];

    camps.forEach((camp) => {
      const color = CAMP_COLORS[camp.type] || "#6b7280";
      const marker = L.marker([camp.lat, camp.lng], { icon: makeIcon(color, 18) })
        .addTo(map)
        .bindPopup(`
          <div style="font-family:'Plus Jakarta Sans',system-ui,sans-serif;font-size:13px;line-height:1.6;">
            <strong style="color:#1a1d26;">${camp.name}</strong><br/>
            <span style="color:#5f6577;">
              Type: ${camp.type.replace(/_/g, " ")}<br/>
              Capacity: ${camp.capacity.toLocaleString()}<br/>
              ${camp.elevation_m ? `Elevation: ${camp.elevation_m}m` : ""}
            </span>
          </div>
        `);
      campMarkers.current.push(marker);
    });
  }, [camps]);

  useEffect(() => {
    const map = mapInstance.current;
    if (!map) return;
    if (routeLayer.current) { map.removeLayer(routeLayer.current); routeLayer.current = null; }
    if (destMarker.current) { map.removeLayer(destMarker.current); destMarker.current = null; }

    if (route && route.path_coords?.length > 1) {
      // Always blue route line — clear, bold, high-contrast
      routeLayer.current = L.polyline(route.path_coords, {
        color: "#3b82f6",
        weight: 5,
        opacity: 0.9,
        lineCap: "round",
        lineJoin: "round",
      }).addTo(map);

      // Subtle shadow/glow layer underneath
      L.polyline(route.path_coords, {
        color: "#93c5fd",
        weight: 9,
        opacity: 0.35,
        lineCap: "round",
        lineJoin: "round",
      }).addTo(map);

      const last = route.path_coords[route.path_coords.length - 1];
      destMarker.current = L.marker(last, { icon: destIcon })
        .addTo(map)
        .bindPopup(`
          <div style="font-family:'Plus Jakarta Sans',system-ui,sans-serif;font-size:13px;line-height:1.6;">
            <strong style="color:#22c55e;">🏁 ${route.camp_name}</strong><br/>
            <span style="color:#5f6577;">
              Distance: ${route.distance_km} km<br/>
              Walk time: ~${route.walk_time_minutes} min
            </span>
          </div>
        `)
        .openPopup();

      map.fitBounds(routeLayer.current.getBounds().pad(0.15));
    }
  }, [route]);

  return <div ref={mapRef} className="absolute inset-0 z-0" />;
}