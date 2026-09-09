"""
backend/routers/camps.py
GET /camps?calamity=
"""

from fastapi import APIRouter, Query
from models.schemas import CampsResponse
from services.graph_service import GraphService

router = APIRouter()
graph_svc = GraphService()


@router.get("/", response_model=CampsResponse)
def get_camps(
    calamity: str = Query("none", description="Filter camps valid for this calamity")
):
    """Returns all valid relief camps for the given calamity type."""
    valid_camps = graph_svc.get_valid_camps(calamity)
    return CampsResponse(
        calamity=calamity,
        camps=valid_camps,
        total=len(valid_camps)
    )