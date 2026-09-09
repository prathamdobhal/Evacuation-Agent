"""
backend/routers/calamity.py

FIXES vs uploaded version:
  1. MLService is now imported from services.ml_service (absolute import).
  2. ml_svc is instantiated ONCE at module level (not inside the handler).
  3. CalamityResponse now receives all required fields: probabilities, description.
  4. _fetch_usgs() returns None gracefully on any error — no 500s from USGS outages.
  5. dist_to_coast_km estimated from USGS lat/lng vs Sendai coast reference point.
"""
from __future__ import annotations  # add this at the very top of the file
import httpx
import math
from fastapi import APIRouter

from models.schemas import CalamityResponse, MLInferenceRequest
from services.ml_service import MLService

router  = APIRouter()
ml_svc  = MLService()

# Sendai coast reference (Arahama beach)
SENDAI_COAST_LAT = 38.2688
SENDAI_COAST_LNG = 141.0251

USGS_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=geojson&starttime=now-1day&minmagnitude=5.0"
    "&minlatitude=30&maxlatitude=46&minlongitude=130&maxlongitude=146"
    "&orderby=time&limit=1"
)


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def _fetch_usgs() -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp  = await client.get(USGS_URL)
            data  = resp.json()
            feats = data.get("features", [])
            if not feats:
                return None
            feat   = feats[0]
            props  = feat["properties"]
            coords = feat["geometry"]["coordinates"]   # [lng, lat, depth]
            lat    = float(coords[1])
            lng    = float(coords[0])
            return {
                "magnitude":        props.get("mag", 0),
                "depth_km":         float(coords[2]) if len(coords) > 2 else 10.0,
                "lat":              lat,
                "lng":              lng,
                "tsunami_flag":     int(props.get("tsunami", 0)),
                "dist_to_coast_km": _haversine_km(lat, lng, SENDAI_COAST_LAT, SENDAI_COAST_LNG),
                "source":           "usgs",
            }
    except Exception:
        return None


@router.get("/current", response_model=CalamityResponse)
async def get_current_calamity():
    """
    Polls USGS (2 s timeout) → ML inference.
    Falls back to 'none' if no signal.
    """
    usgs = await _fetch_usgs()

    if usgs:
        result = ml_svc.predict(
            magnitude=usgs["magnitude"],
            depth_km=usgs["depth_km"],
            lat=usgs["lat"],
            lng=usgs["lng"],
            wave_intensity=5.0 if usgs["tsunami_flag"] else None,
            dist_to_coast_km=usgs["dist_to_coast_km"],
        )
        result["source"] = "usgs+ml"
        return CalamityResponse(**result)

    return CalamityResponse(
        calamity="none",
        severity=0.0,
        severity_label="low",
        source="offline_fallback",
        probabilities={"earthquake": 0.0, "tsunami": 0.0, "typhoon": 0.0},
        description="No active calamity detected. All clear.",
    )


@router.post("/predict", response_model=CalamityResponse)
def predict_calamity(req: MLInferenceRequest):
    """Direct ML inference from raw sensor inputs."""
    result = ml_svc.predict(
        magnitude=req.magnitude,
        depth_km=req.depth_km,
        lat=req.lat,
        lng=req.lng,
        central_pressure_hpa=req.central_pressure_hpa,
        max_wind_knots=req.max_wind_knots,
        wave_intensity=req.wave_intensity,
    )
    return CalamityResponse(**result)