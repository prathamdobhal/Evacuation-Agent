"""
backend/models/schemas.py

FIX: severity_label is lowercase ("low"|"moderate"|"high") to match
     what ml_service.py returns. Original had mixed case in docs but
     validators would reject "Low" if the field had a pattern validator.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class CalamityResponse(BaseModel):
    calamity:       str                  # "tsunami" | "earthquake" | "typhoon" | "none"
    severity:       float                # 0.0 – 1.0
    severity_label: str                  # "low" | "moderate" | "high"
    source:         str                  # "ml_model" | "rule_based" | "usgs+ml" | "offline_fallback"
    probabilities:  Dict[str, float]     # {"tsunami": 0.91, "earthquake": 0.06, ...}
    description:    str                  # human-readable summary


class AlgoStats(BaseModel):
    algorithm:      str
    nodes_expanded: int
    cost_meters:    float
    time_ms:        float
    path_length:    int


class RouteResponse(BaseModel):
    calamity:          str
    severity:          float
    camp_name:         str
    camp_lat:          float
    camp_lng:          float
    camp_capacity:     int
    distance_km:       float
    walk_time_minutes: float
    path_coords:       List[List[float]]   # [[lat, lng], ...]
    blocked_roads:     List[str]
    algo_comparison:   List[AlgoStats]


class Camp(BaseModel):
    id:          str
    name:        str
    lat:         float
    lng:         float
    capacity:    int
    type:        str
    elevation_m: Optional[float] = None
    address:     Optional[str]   = None
    valid_for:   List[str]


class CampsResponse(BaseModel):
    calamity: str
    camps:    List[Camp]
    total:    int


class MLInferenceRequest(BaseModel):
    magnitude:            Optional[float] = Field(None, ge=0, le=10)
    depth_km:             float           = Field(10.0, ge=0)
    lat:                  float           = Field(38.2, ge=-90,  le=90)
    lng:                  float           = Field(140.9, ge=-180, le=180)
    central_pressure_hpa: Optional[float] = Field(None)
    max_wind_knots:       Optional[float] = Field(None, ge=0)
    wave_intensity:       Optional[float] = Field(None)