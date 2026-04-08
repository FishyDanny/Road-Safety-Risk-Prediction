"""
Pytest configuration and fixtures for DeepRisk API tests.
"""

import os
from pathlib import Path

import joblib
import pytest
import tensorflow as tf
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.api.main import app
from src.config import Settings
from src.model_loader import ModelRegistry
from src.preprocessing import load_preprocessor


@pytest.fixture(scope="session")
def settings():
    """Create settings instance with correct model directory."""
    return Settings()


@pytest.fixture(scope="session")
def model_registry(settings):
    """Load all three models once per test session.

    This fixture loads the Random Forest, FFNN, and Residual models
    into a ModelRegistry and caches them for the entire test session,
    avoiding expensive model reloads for each test.

    Yields:
        ModelRegistry: Registry with all three models loaded.
    """
    registry = ModelRegistry()

    models_config = {
        "random_forest": "final_baseline_rf_model.pkl",
        "ffnn": "final_first_dl_model.keras",
        "residual": "final_second_dl_model.keras",
    }

    for model_name, model_file in models_config.items():
        model_path = settings.get_model_path(model_file)
        if model_path.exists():
            registry.get(model_name, str(model_path))

    yield registry

    # Cleanup after session
    registry.clear()


@pytest.fixture(scope="session")
def preprocessor(settings):
    """Load the preprocessor pipeline once per test session.

    Yields:
        ColumnTransformer: Fitted preprocessor for transforming raw inputs.
    """
    preprocessor_path = settings.get_model_path("preprocessor.pkl")
    return load_preprocessor(str(preprocessor_path))


@pytest.fixture(scope="session")
def test_client():
    """Create a TestClient for the FastAPI app.

    Uses httpx.AsyncClient internally for async endpoint testing.

    Yields:
        TestClient: FastAPI test client with app mounted.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="function")
def app_with_models(model_registry, preprocessor):
    """Configure app state with loaded models and preprocessor.

    This fixture sets up the FastAPI app's state to include the
    model registry and preprocessor, simulating a real startup.

    Args:
        model_registry: Registry with all models loaded.
        preprocessor: Fitted preprocessor pipeline.

    Yields:
        FastAPI: App instance with configured state.
    """
    app.state.registry = model_registry
    app.state.preprocessor = preprocessor
    yield app


@pytest.fixture(scope="function")
def client_with_models(app_with_models):
    """Create test client with models and preprocessor loaded.

    This is the primary fixture for integration tests that need
    the full API with loaded models.

    Args:
        app_with_models: App instance with models configured.

    Yields:
        TestClient: Test client ready for integration tests.
    """
    with TestClient(app_with_models) as client:
        yield client


@pytest.fixture
def sample_raw_input():
    """Provide sample raw input features for Australian road crash data.

    This represents a realistic crash scenario in Australia.

    Returns:
        dict: Dictionary with all 13 raw features matching Australian dataset format.
    """
    return {
        "State": "NSW",
        "Speed Limit": 60,
        "National Road Type": "Arterial Road",
        "Road User": "Driver",
        "Age": 30,
        "Gender": "Male",
        "Bus Involvement": "No",
        "Articulated Truck Involvement": "No",
        "Heavy Rigid Truck Involvement": "No",
        "Dayweek": "Friday",
        "Time": 18,
        "Christmas Period": "No",
        "Easter Period": "No",
    }


@pytest.fixture
def sample_preprocessed_input():
    """Provide sample preprocessed input features.

    This represents features after ColumnTransformer processing.
    The actual values come from real test data to ensure compatibility
    with the models.

    Returns:
        list: List of preprocessed feature values.
    """
    # Load actual test data to get realistic preprocessed features
    settings = Settings()
    X_test_path = settings.get_model_path("X_test.pkl")

    if X_test_path.exists():
        X_test = joblib.load(X_test_path)
        # Get the first sample as test input
        sample = X_test[0:1]  # Keep 2D shape
        return sample.flatten().tolist()
    else:
        # Fallback to dummy data if test data not available
        # This is approximate feature count after preprocessing
        return [0.0] * 60
