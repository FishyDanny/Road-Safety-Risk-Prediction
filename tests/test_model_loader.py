"""Unit tests for model loading module."""

import pytest
from unittest.mock import Mock, patch


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rf_model_path():
    """Path to the trained Random Forest model."""
    return "models/final_baseline_rf_model.pkl"


@pytest.fixture
def keras_model_path():
    """Path to the trained Keras feedforward model."""
    return "models/final_first_dl_model.keras"


@pytest.fixture
def keras_residual_model_path():
    """Path to the trained Keras residual model."""
    return "models/final_second_dl_model.keras"


@pytest.fixture
def mock_rf_model():
    """Mock RandomForestClassifier."""
    mock = Mock()
    mock.n_estimators = 100
    mock.classes_ = [0, 1]
    mock.n_features_in_ = 10
    return mock


@pytest.fixture
def mock_keras_model():
    """Mock Keras Model."""
    mock = Mock()
    mock.layers = [Mock(), Mock()]
    return mock


# ============================================================================
# ModelRegistry Tests
# ============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_get_loads_random_forest(self, rf_model_path, mock_rf_model):
        """Test that get() loads Random Forest model correctly."""
        from src.model_loader import ModelRegistry

        with patch("src.model_loader.load_random_forest") as mock_load:
            mock_load.return_value = mock_rf_model
            registry = ModelRegistry()
            model = registry.get("random_forest", rf_model_path)

            assert model is not None
            assert model == mock_rf_model
            mock_load.assert_called_once_with(rf_model_path)

    def test_get_loads_keras_model(self, keras_model_path, mock_keras_model):
        """Test that get() loads Keras model correctly."""
        from src.model_loader import ModelRegistry

        with patch("src.model_loader.load_keras_model") as mock_load:
            mock_load.return_value = mock_keras_model
            registry = ModelRegistry()
            model = registry.get("keras_ff", keras_model_path)

            assert model is not None
            assert model == mock_keras_model
            mock_load.assert_called_once_with(keras_model_path)

    def test_get_caches_model(self, rf_model_path, mock_rf_model):
        """Test that get() caches models and returns same instance."""
        from src.model_loader import ModelRegistry

        with patch("src.model_loader.load_random_forest") as mock_load:
            mock_load.return_value = mock_rf_model
            registry = ModelRegistry()

            model1 = registry.get("random_forest", rf_model_path)
            model2 = registry.get("random_forest", rf_model_path)

            assert model1 is model2
            mock_load.assert_called_once()

    def test_get_different_models_not_cached_together(
        self, rf_model_path, keras_model_path, mock_rf_model, mock_keras_model
    ):
        """Test that different model types are cached separately."""
        from src.model_loader import ModelRegistry

        with (
            patch("src.model_loader.load_random_forest") as mock_rf_load,
            patch("src.model_loader.load_keras_model") as mock_keras_load,
        ):
            mock_rf_load.return_value = mock_rf_model
            mock_keras_load.return_value = mock_keras_model

            registry = ModelRegistry()
            rf_model = registry.get("random_forest", rf_model_path)
            keras_model = registry.get("keras_ff", keras_model_path)

            assert rf_model is not keras_model

    def test_clear_removes_all_models(
        self, rf_model_path, keras_model_path, mock_rf_model, mock_keras_model
    ):
        """Test that clear() removes all cached models."""
        from src.model_loader import ModelRegistry

        with (
            patch("src.model_loader.load_random_forest") as mock_rf_load,
            patch("src.model_loader.load_keras_model") as mock_keras_load,
        ):
            mock_rf_load.return_value = mock_rf_model
            mock_keras_load.return_value = mock_keras_model

            registry = ModelRegistry()
            registry.get("random_forest", rf_model_path)
            registry.get("keras_ff", keras_model_path)

            assert registry.is_loaded("random_forest")
            assert registry.is_loaded("keras_ff")

            registry.clear()

            assert not registry.is_loaded("random_forest")
            assert not registry.is_loaded("keras_ff")

    def test_clear_on_empty_registry(self):
        """Test that clear() works on empty registry without error."""
        from src.model_loader import ModelRegistry

        registry = ModelRegistry()
        registry.clear()

        assert not registry.is_loaded("random_forest")

    def test_is_loaded_returns_false_initially(self):
        """Test that is_loaded() returns False when model not cached."""
        from src.model_loader import ModelRegistry

        registry = ModelRegistry()

        assert not registry.is_loaded("random_forest")
        assert not registry.is_loaded("keras_ff")
        assert not registry.is_loaded("nonexistent_model")

    def test_is_loaded_returns_true_after_get(self, rf_model_path, mock_rf_model):
        """Test that is_loaded() returns True after model is loaded."""
        from src.model_loader import ModelRegistry

        with patch("src.model_loader.load_random_forest") as mock_load:
            mock_load.return_value = mock_rf_model
            registry = ModelRegistry()

            assert not registry.is_loaded("random_forest")

            registry.get("random_forest", rf_model_path)

            assert registry.is_loaded("random_forest")

    def test_multiple_keras_models_cached_separately(
        self, keras_model_path, keras_residual_model_path
    ):
        """Test that different Keras models are cached separately."""
        from src.model_loader import ModelRegistry

        mock_model1 = Mock()
        mock_model2 = Mock()

        with patch("src.model_loader.load_keras_model") as mock_load:
            mock_load.side_effect = [mock_model1, mock_model2]

            registry = ModelRegistry()
            model1 = registry.get("keras_ff", keras_model_path)
            model2 = registry.get("keras_residual", keras_residual_model_path)

            assert model1 is not model2
            assert registry.is_loaded("keras_ff")
            assert registry.is_loaded("keras_residual")


# ============================================================================
# load_random_forest Tests
# ============================================================================


class TestLoadRandomForest:
    """Tests for load_random_forest function."""

    def test_loads_random_forest_classifier(self, rf_model_path, mock_rf_model):
        """Test that load_random_forest returns RandomForestClassifier."""
        with patch("src.model_loader.joblib.load") as mock_load:
            mock_load.return_value = mock_rf_model

            from src.model_loader import load_random_forest

            model = load_random_forest(rf_model_path)

            assert model is not None
            assert model == mock_rf_model
            mock_load.assert_called_once_with(rf_model_path)

    def test_returns_different_instances(self, rf_model_path, mock_rf_model):
        """Test that each call returns a new model instance."""
        with patch("src.model_loader.joblib.load") as mock_load:
            model1_mock = Mock()
            model2_mock = Mock()
            mock_load.side_effect = [model1_mock, model2_mock]

            from src.model_loader import load_random_forest

            model1 = load_random_forest(rf_model_path)
            model2 = load_random_forest(rf_model_path)

            assert model1 is not model2


# ============================================================================
# load_keras_model Tests
# ============================================================================


class TestLoadKerasModel:
    """Tests for load_keras_model function."""

    def test_loads_keras_model(self, keras_model_path, mock_keras_model):
        """Test that load_keras_model returns tf.keras.Model."""
        with patch("src.model_loader.tf.keras.models.load_model") as mock_load:
            mock_load.return_value = mock_keras_model

            from src.model_loader import load_keras_model

            model = load_keras_model(keras_model_path)

            assert model is not None
            assert model == mock_keras_model
            mock_load.assert_called_once_with(keras_model_path)

    def test_returns_different_instances(self, keras_model_path):
        """Test that each call returns a new model instance."""
        with patch("src.model_loader.tf.keras.models.load_model") as mock_load:
            mock_model1 = Mock()
            mock_model2 = Mock()
            mock_load.side_effect = [mock_model1, mock_model2]

            from src.model_loader import load_keras_model

            model1 = load_keras_model(keras_model_path)
            model2 = load_keras_model(keras_model_path)

            assert model1 is not model2


# ============================================================================
# get_model Tests
# ============================================================================


class TestGetModel:
    """Tests for get_model convenience function."""

    def test_get_model_loads_random_forest(self, rf_model_path, mock_rf_model):
        """Test that get_model correctly loads Random Forest."""
        from src.model_loader import ModelRegistry, get_model

        with patch("src.model_loader.load_random_forest") as mock_load:
            mock_load.return_value = mock_rf_model

            registry = ModelRegistry()
            model = get_model("random_forest", registry, rf_model_path)

            assert model is not None
            assert model == mock_rf_model

    def test_get_model_loads_keras(self, keras_model_path, mock_keras_model):
        """Test that get_model correctly loads Keras model."""
        from src.model_loader import ModelRegistry, get_model

        with patch("src.model_loader.load_keras_model") as mock_load:
            mock_load.return_value = mock_keras_model

            registry = ModelRegistry()
            model = get_model("keras_ff", registry, keras_model_path)

            assert model is not None
            assert model == mock_keras_model

    def test_get_model_uses_registry_caching(self, rf_model_path, mock_rf_model):
        """Test that get_model uses registry caching properly."""
        from src.model_loader import ModelRegistry, get_model

        with patch("src.model_loader.load_random_forest") as mock_load:
            mock_load.return_value = mock_rf_model

            registry = ModelRegistry()

            model1 = get_model("random_forest", registry, rf_model_path)
            model2 = get_model("random_forest", registry, rf_model_path)

            assert model1 is model2
            mock_load.assert_called_once()

    def test_get_model_with_custom_model_name(self, rf_model_path, mock_rf_model):
        """Test that get_model works with custom model names."""
        from src.model_loader import ModelRegistry, get_model

        with patch("src.model_loader.load_random_forest") as mock_load:
            mock_load.return_value = mock_rf_model

            registry = ModelRegistry()
            model = get_model("my_custom_rf", registry, rf_model_path)

            assert model is not None
