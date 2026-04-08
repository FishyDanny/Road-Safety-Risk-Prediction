import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_healthy_status_with_loaded_models(self, client_with_models):
        response = client_with_models.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True
        assert "model_names" in data
        assert set(data["model_names"]) == {"random_forest", "ffnn", "residual"}

    def test_health_returns_model_names_list(self, client_with_models):
        response = client_with_models.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["model_names"], list)
        assert len(data["model_names"]) == 3


class TestRootEndpoint:
    def test_root_returns_api_info(self, client_with_models):
        response = client_with_models.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "DeepRisk Road Safety API"
        assert data["version"] == "1.0.0"
        assert "description" in data
        assert "endpoints" in data

    def test_root_endpoints_field_contains_expected_paths(self, client_with_models):
        response = client_with_models.get("/")

        data = response.json()
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "predict" in endpoints
        assert "docs" in endpoints
        assert endpoints["health"] == "/health"
        assert endpoints["predict"] == "/api/v1/predict"


class TestRawInputValidationError:
    def test_missing_required_field_returns_422(self, client_with_models):
        invalid_payload = {
            "State": "NSW",
            "Speed Limit": 60,
        }

        response = client_with_models.post("/api/v1/predict/raw", json=invalid_payload)

        assert response.status_code == 422

    def test_invalid_state_returns_422(self, client):
        invalid_payload = {
            "State": "INVALID_STATE",
            "Speed Limit": 60,
            "National Road Type": "Arterial",
            "Road User": "Driver",
            "Age": 35,
            "Gender": "Male",
            "Bus Involvement": "No",
            "Articulated Truck Involvement": "No",
            "Heavy Rigid Truck Involvement": "No",
            "Dayweek": "Monday",
            "Time": 14,
            "Christmas Period": "No",
            "Easter Period": "No",
        }

        response = client.post("/api/v1/predict/raw", json=invalid_payload)

        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data

    def test_invalid_gender_returns_422(self, client):
        invalid_payload = {
            "State": "NSW",
            "Speed Limit": 60,
            "National Road Type": "Arterial",
            "Road User": "Driver",
            "Age": 35,
            "Gender": "InvalidGender",
            "Bus Involvement": "No",
            "Articulated Truck Involvement": "No",
            "Heavy Rigid Truck Involvement": "No",
            "Dayweek": "Monday",
            "Time": 14,
            "Christmas Period": "No",
            "Easter Period": "No",
        }

        response = client.post("/api/v1/predict/raw", json=invalid_payload)

        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data

    def test_invalid_yes_no_field_returns_422(self, client):
        invalid_payload = {
            "State": "NSW",
            "Speed Limit": 60,
            "National Road Type": "Arterial",
            "Road User": "Driver",
            "Age": 35,
            "Gender": "Male",
            "Bus Involvement": "Maybe",
            "Articulated Truck Involvement": "No",
            "Heavy Rigid Truck Involvement": "No",
            "Dayweek": "Monday",
            "Time": 14,
            "Christmas Period": "No",
            "Easter Period": "No",
        }

        response = client.post("/api/v1/predict/raw", json=invalid_payload)

        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data

    def test_invalid_dayweek_returns_422(self, client):
        invalid_payload = {
            "State": "NSW",
            "Speed Limit": 60,
            "National Road Type": "Arterial",
            "Road User": "Driver",
            "Age": 35,
            "Gender": "Male",
            "Bus Involvement": "No",
            "Articulated Truck Involvement": "No",
            "Heavy Rigid Truck Involvement": "No",
            "Dayweek": "InvalidDay",
            "Time": 14,
            "Christmas Period": "No",
            "Easter Period": "No",
        }

        response = client.post("/api/v1/predict/raw", json=invalid_payload)

        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data


class TestPreprocessedInputValidationError:
    def test_invalid_feature_count_returns_400(self, client):
        invalid_payload = {"features": [1.0, 2.0, 3.0]}

        response = client.post("/api/v1/predict/preprocessed", json=invalid_payload)

        assert response.status_code == 400
        error_data = response.json()
        assert "detail" in error_data
