"""
backend/main.py

FIX: Uses lifespan context to warm up GraphService once at startup
     instead of letting each router module load it independently.
     Also uses absolute imports (required when running: uvicorn main:app from backend/).
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import calamity, route, camps
from services.graph_service import get_graph_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: load graph + pkl files once before first request
    print("[startup] Loading GraphService...")
    get_graph_service()
    print("[startup] Ready.")
    yield


app = FastAPI(
    title="Sendai Evacuation Agent API",
    description="ML-powered disaster routing for Sendai, Miyagi, Japan",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calamity.router, prefix="/calamity", tags=["Calamity Detection"])
app.include_router(route.router,    prefix="/route",    tags=["Routing"])
app.include_router(camps.router,    prefix="/camps",    tags=["Shelter Camps"])


@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "ok", "city": "Sendai, Miyagi, Japan"}


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Sendai Evacuation Agent API",
        "endpoints": ["/calamity/current", "/calamity/predict",
                      "/route", "/camps", "/health", "/docs"],
    }