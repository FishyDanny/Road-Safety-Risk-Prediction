"""FastAPI application for DeepRisk Road Safety prediction API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings
from src.model_loader import ModelRegistry
from src.logger import get_logger
from src.api.routes.predict import router as predict_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup and cleanup at shutdown.

    Uses ModelRegistry to load all three models (Random Forest, FFNN, Residual)
    into memory once, storing the registry in app.state for access by routes.
    """
    settings = Settings()
    registry = ModelRegistry()

    models_config = {
        "random_forest": "final_baseline_rf_model.pkl",
        "ffnn": "final_first_dl_model.keras",
        "residual": "final_second_dl_model.keras",
    }

    for model_name, model_file in models_config.items():
        try:
            model_path = settings.get_model_path(model_file)
            registry.get(model_name, str(model_path))
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")

    app.state.registry = registry
    yield

    registry.clear()
    logger.info("All models cleared from memory")


app = FastAPI(
    title="DeepRisk Road Safety API",
    description="API for predicting road crash types (single vs multiple vehicle) using machine learning",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1/predict", tags=["Prediction"])


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Return API information."""
    return {
        "name": "DeepRisk Road Safety API",
        "version": "1.0.0",
        "description": "Predict road crash types using machine learning",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "docs": "/docs",
        },
    }


@app.get("/health", tags=["Health"], response_model=dict)
async def health_check() -> dict:
    """Check API health and model loading status.

    Returns status of the API and which models are loaded.
    """
    registry: ModelRegistry = app.state.registry
    model_names: List[str] = list(registry._models.keys())
    models_loaded = len(model_names) > 0

    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "model_names": model_names,
    }
