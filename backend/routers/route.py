"""
backend/routers/route.py

FIX: Uses get_graph_service() singleton instead of GraphService() directly.
     This prevents the 85k-node graph from being loaded twice at startup
     (route.py and camps.py both imported GraphService before).
"""

from fastapi import APIRouter, Query, HTTPException
from models.schemas import RouteResponse
from services.graph_service import get_graph_service

router = APIRouter()


@router.get("/", response_model=RouteResponse)
def get_route(
    lat:       float = Query(...,      description="User latitude",  ge=-90,  le=90),
    lng:       float = Query(...,      description="User longitude", ge=-180, le=180),
    calamity:  str   = Query("none",   description="Calamity type: earthquake|tsunami|typhoon|none"),
    severity:  float = Query(0.5,      description="Severity 0–1",  ge=0,    le=1),
    algorithm: str   = Query("astar",  description="Algorithm: astar|ucs|bfs|dfs"),
):
    graph_svc = get_graph_service()
    result = graph_svc.find_route(lat, lng, calamity, severity, algorithm)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return RouteResponse(**result)