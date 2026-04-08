"""Integration tests for DeepRisk API prediction endpoints."""

import numpy as np
import pytest
from fastapi.testclient import TestClient


class TestPredictRawEndpoint:
    """Integration tests for POST /api/v1/predict/raw endpoint."""

    def test_predict_raw_with_valid_input_returns_predictions(
        self, client_with_models, sample_raw_input
    ):
        """Test that valid raw input returns predictions from all three models."""
        response = client_with_models.post("/api/v1/predict/raw", json=sample_raw_input)

        assert response.status_code == 200
        predictions = response.json()

        assert isinstance(predictions, list)
        assert len(predictions) == 3

        model_names = [p["model_name"] for p in predictions]
        assert "random_forest" in model_names
        assert "ffnn" in model_names
        assert "residual" in model_names

    def test_predict_raw_returns_correct_response_format(
        self, client_with_models, sample_raw_input
    ):
        """Test that predictions contain prediction, probability, and model_name."""
        response = client_with_models.post("/api/v1/predict/raw", json=sample_raw_input)

        assert response.status_code == 200
        predictions = response.json()

        for prediction in predictions:
            assert "prediction" in prediction
            assert "probability" in prediction
            assert "model_name" in prediction

            assert isinstance(prediction["prediction"], int)
            assert prediction["prediction"] in [0, 1]

            assert isinstance(prediction["probability"], float)
            assert 0.0 <= prediction["probability"] <= 1.0

    def test_predict_raw_with_australian_state_values(self, client_with_models):
        """Test prediction with various valid Australian states."""
        aus_states = ["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]

        for state in aus_states:
            input_data = {
                "State": state,
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

            response = client_with_models.post("/api/v1/predict/raw", json=input_data)

            assert response.status_code == 200, f"Failed for state {state}"
            predictions = response.json()
            assert len(predictions) == 3

    def test_predict_raw_with_night_time_crash(self, client_with_models):
        """Test prediction for night-time crash (single vehicle indicator)."""
        night_time_input = {
            "State": "NSW",
            "Speed Limit": 100,
            "National Road Type": "National Highway",
            "Road User": "Driver",
            "Age": 25,
            "Gender": "Male",
            "Bus Involvement": "No",
            "Articulated Truck Involvement": "No",
            "Heavy Rigid Truck Involvement": "No",
            "Dayweek": "Saturday",
            "Time": 2,
            "Christmas Period": "No",
            "Easter Period": "No",
        }

        response = client_with_models.post("/api/v1/predict/raw", json=night_time_input)

        assert response.status_code == 200
        predictions = response.json()
        assert len(predictions) == 3

        for pred in predictions:
            assert pred["model_name"] in ["random_forest", "ffnn", "residual"]

    def test_predict_raw_with_different_age_groups(self, client_with_models):
        """Test predictions across different age demographics."""
        ages = [18, 30, 50, 70, 90]

        for age in ages:
            input_data = {
                "State": "VIC",
                "Speed Limit": 80,
                "National Road Type": "Arterial Road",
                "Road User": "Driver",
                "Age": age,
                "Gender": "Male",
                "Bus Involvement": "No",
                "Articulated Truck Involvement": "No",
                "Heavy Rigid Truck Involvement": "No",
                "Dayweek": "Wednesday",
                "Time": 14,
                "Christmas Period": "No",
                "Easter Period": "No",
            }

            response = client_with_models.post("/api/v1/predict/raw", json=input_data)

            assert response.status_code == 200, f"Failed for age {age}"

    def test_predict_raw_with_truck_involvement(self, client_with_models):
        """Test prediction when trucks are involved (multiple vehicle indicator)."""
        truck_input = {
            "State": "QLD",
            "Speed Limit": 110,
            "National Road Type": "National Highway",
            "Road User": "Driver",
            "Age": 45,
            "Gender": "Male",
            "Bus Involvement": "No",
            "Articulated Truck Involvement": "Yes",
            "Heavy Rigid Truck Involvement": "No",
            "Dayweek": "Tuesday",
            "Time": 10,
            "Christmas Period": "No",
            "Easter Period": "No",
        }

        response = client_with_models.post("/api/v1/predict/raw", json=truck_input)

        assert response.status_code == 200
        predictions = response.json()
        assert len(predictions) == 3


class TestPredictPreprocessedEndpoint:
    """Integration tests for POST /api/v1/predict/preprocessed endpoint."""

    def test_predict_preprocessed_with_valid_input_returns_predictions(
        self, client_with_models, sample_preprocessed_input
    ):
        """Test that valid preprocessed input returns predictions from all three models."""
        response = client_with_models.post(
            "/api/v1/predict/preprocessed",
            json={"features": sample_preprocessed_input},
        )

        assert response.status_code == 200
        predictions = response.json()

        assert isinstance(predictions, list)
        assert len(predictions) == 3

        model_names = [p["model_name"] for p in predictions]
        assert "random_forest" in model_names
        assert "ffnn" in model_names
        assert "residual" in model_names

    def test_predict_preprocessed_returns_correct_response_format(
        self, client_with_models, sample_preprocessed_input
    ):
        """Test that predictions contain prediction, probability, and model_name."""
        response = client_with_models.post(
            "/api/v1/predict/preprocessed",
            json={"features": sample_preprocessed_input},
        )

        assert response.status_code == 200
        predictions = response.json()

        for prediction in predictions:
            assert "prediction" in prediction
            assert "probability" in prediction
            assert "model_name" in prediction

            assert isinstance(prediction["prediction"], int)
            assert prediction["prediction"] in [0, 1]

            assert isinstance(prediction["probability"], float)
            assert 0.0 <= prediction["probability"] <= 1.0

    def test_predict_preprocessed_all_models_consistent_format(
        self, client_with_models, sample_preprocessed_input
    ):
        """Test that all three models return consistent response format."""
        response = client_with_models.post(
            "/api/v1/predict/preprocessed",
            json={"features": sample_preprocessed_input},
        )

        assert response.status_code == 200
        predictions = response.json()

        assert len(predictions) == 3

        model_set = {pred["model_name"] for pred in predictions}
        assert model_set == {"random_forest", "ffnn", "residual"}

        for pred in predictions:
            assert set(pred.keys()) == {"prediction", "probability", "model_name"}


class TestModelPredictions:
    """Tests verifying real model predictions work correctly."""

    def test_random_forest_prediction_is_valid(
        self, client_with_models, sample_raw_input
    ):
        """Test Random Forest model returns valid prediction."""
        response = client_with_models.post("/api/v1/predict/raw", json=sample_raw_input)

        assert response.status_code == 200
        predictions = response.json()

        rf_pred = next(
            (p for p in predictions if p["model_name"] == "random_forest"), None
        )
        assert rf_pred is not None
        assert rf_pred["prediction"] in [0, 1]
        assert 0.0 <= rf_pred["probability"] <= 1.0

    def test_ffnn_prediction_is_valid(self, client_with_models, sample_raw_input):
        """Test FFNN model returns valid prediction."""
        response = client_with_models.post("/api/v1/predict/raw", json=sample_raw_input)

        assert response.status_code == 200
        predictions = response.json()

        ffnn_pred = next((p for p in predictions if p["model_name"] == "ffnn"), None)
        assert ffnn_pred is not None
        assert ffnn_pred["prediction"] in [0, 1]
        assert 0.0 <= ffnn_pred["probability"] <= 1.0

    def test_residual_prediction_is_valid(self, client_with_models, sample_raw_input):
        """Test Residual NN model returns valid prediction."""
        response = client_with_models.post("/api/v1/predict/raw", json=sample_raw_input)

        assert response.status_code == 200
        predictions = response.json()

        residual_pred = next(
            (p for p in predictions if p["model_name"] == "residual"), None
        )
        assert residual_pred is not None
        assert residual_pred["prediction"] in [0, 1]
        assert 0.0 <= residual_pred["probability"] <= 1.0

    def test_predictions_from_preprocessed_match_model_formats(
        self, client_with_models, sample_preprocessed_input
    ):
        """Test preprocessed endpoint returns valid predictions from all models."""
        response = client_with_models.post(
            "/api/v1/predict/preprocessed",
            json={"features": sample_preprocessed_input},
        )

        assert response.status_code == 200
        predictions = response.json()

        assert len(predictions) == 3

        for pred in predictions:
            assert pred["prediction"] in [0, 1]
            assert 0.0 <= pred["probability"] <= 1.0

    def test_raw_and_preprocessed_endpoints_consistent_predictions(
        self, client_with_models, sample_raw_input, sample_preprocessed_input
    ):
        """Test that both endpoints return predictions from same models."""
        raw_response = client_with_models.post(
            "/api/v1/predict/raw", json=sample_raw_input
        )
        preprocessed_response = client_with_models.post(
            "/api/v1/predict/preprocessed",
            json={"features": sample_preprocessed_input},
        )

        assert raw_response.status_code == 200
        assert preprocessed_response.status_code == 200

        raw_preds = raw_response.json()
        preprocessed_preds = preprocessed_response.json()

        raw_models = {p["model_name"] for p in raw_preds}
        preprocessed_models = {p["model_name"] for p in preprocessed_preds}

        assert raw_models == preprocessed_models
