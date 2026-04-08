"""Prediction endpoints for DeepRisk API."""

from __future__ import annotations

from typing import List
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    RawInputFeatures,
    PreprocessedInput,
    PredictionResponse,
    ErrorResponse,
)
from src.preprocessing import load_preprocessor, transform_raw_input
from src.model_loader import ModelRegistry
from src.config import Settings
from src.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/raw",
    response_model=List[PredictionResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_raw(
    request: Request, features: RawInputFeatures
) -> List[PredictionResponse]:
    """Predict crash type from raw human-readable input features.

    Accepts 13 raw features, transforms them using the preprocessor pipeline,
    and returns predictions from all three models (Random Forest, FFNN, Residual NN).

    Args:
        request: FastAPI request object (access to app.state)
        features: Raw input features from the request body

    Returns:
        List of predictions from all three models

    Raises:
        HTTPException: If preprocessor not found or model error
    """
    if (
        not hasattr(request.app.state, "preprocessor")
        or request.app.state.preprocessor is None
    ):
        settings = Settings()
        preprocessor_path = settings.get_model_path("preprocessor.pkl")
        try:
            request.app.state.preprocessor = load_preprocessor(str(preprocessor_path))
            logger.info(f"Loaded preprocessor from {preprocessor_path}")
        except FileNotFoundError:
            logger.error(f"Preprocessor file not found at {preprocessor_path}")
            raise HTTPException(status_code=500, detail="Preprocessor file not found")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load preprocessor: {str(e)}"
            )

    preprocessor = request.app.state.preprocessor

    input_dict = features.model_dump(by_alias=True, mode="python")
    try:
        transformed = transform_raw_input(input_dict, preprocessor)
    except ValueError as e:
        logger.error(f"Input transformation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    registry: ModelRegistry = request.app.state.registry
    predictions: List[PredictionResponse] = []

    for model_name in ["random_forest", "ffnn", "residual"]:
        model = registry._models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not loaded")
            raise HTTPException(
                status_code=500, detail=f"Model {model_name} not loaded"
            )

        if model_name == "random_forest":
            probs = model.predict_proba(transformed)[0]
            probability = float(probs[1])
            prediction = int(probs[1] > 0.5)
        else:
            prob_array = model.predict(transformed, verbose=0)
            probability = float(prob_array[0][0])
            prediction = int(probability > 0.5)

        predictions.append(
            PredictionResponse(
                prediction=prediction,
                probability=probability,
                model_name=model_name,
            )
        )

    logger.info(
        f"Raw prediction complete: predictions={[p.prediction for p in predictions]}"
    )
    return predictions


@router.post(
    "/preprocessed",
    response_model=List[PredictionResponse],
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_preprocessed(
    request: Request, features: PreprocessedInput
) -> List[PredictionResponse]:
    """Predict crash type from preprocessed feature vector.

    Accepts a preprocessed feature vector (output of ColumnTransformer),
    skipping the preprocessing step. Returns predictions from all three models.

    Args:
        request: FastAPI request object (access to app.state)
        features: Preprocessed feature vector

    Returns:
        List of predictions from all three models

    Raises:
        HTTPException: If model error
    """
    import numpy as np

    try:
        feature_array = np.array([features.features])
        if feature_array.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got {feature_array.ndim}D")
    except Exception as e:
        logger.error(f"Feature conversion failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feature format: {str(e)}")

    registry: ModelRegistry = request.app.state.registry
    predictions: List[PredictionResponse] = []

    for model_name in ["random_forest", "ffnn", "residual"]:
        model = registry._models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not loaded")
            raise HTTPException(
                status_code=500, detail=f"Model {model_name} not loaded"
            )

        if model_name == "random_forest":
            probs = model.predict_proba(feature_array)[0]
            probability = float(probs[1])
            prediction = int(probs[1] > 0.5)
        else:
            prob_array = model.predict(feature_array, verbose=0)
            probability = float(prob_array[0][0])
            prediction = int(probability > 0.5)

        predictions.append(
            PredictionResponse(
                prediction=prediction,
                probability=probability,
                model_name=model_name,
            )
        )

    logger.info(
        f"Preprocessed prediction complete: predictions={[p.prediction for p in predictions]}"
    )
    return predictions
