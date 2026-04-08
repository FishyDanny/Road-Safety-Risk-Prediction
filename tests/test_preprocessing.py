"""Unit tests for preprocessing module.

Tests cover:
- load_preprocessor: Loading ColumnTransformer from disk
- validate_raw_input: Validating input dictionaries
- transform_raw_input: Transforming raw data to preprocessed arrays
"""

import pytest
import numpy as np
from sklearn.compose import ColumnTransformer
from unittest.mock import Mock, patch
import tempfile
import os

from src.preprocessing import (
    load_preprocessor,
    validate_raw_input,
    transform_raw_input,
    PreprocessingConfig,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_input_dict():
    """Complete valid input with all 13 required features."""
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
def incomplete_input_dict():
    """Input missing several required features."""
    return {"State": "NSW", "Speed Limit": 60, "Age": 30}


@pytest.fixture
def single_missing_feature_input():
    """Input missing just one feature."""
    valid_input = {
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
        # 'Easter Period': 'No'  # Missing this one
    }
    return valid_input


@pytest.fixture
def mock_preprocessor():
    """Mock ColumnTransformer for testing without loading real model."""
    preprocessor = Mock(spec=ColumnTransformer)
    preprocessor.transform.return_value = np.array([[0.5] * 60])  # Shape (1, 60)
    return preprocessor


@pytest.fixture
def preprocessor_path():
    """Path to real preprocessor file for integration-like tests."""
    return "models/preprocessor.pkl"


# ============================================================================
# TESTS: load_preprocessor
# ============================================================================


class TestLoadPreprocessor:
    """Tests for load_preprocessor function."""

    def test_load_preprocessor_returns_column_transformer(self, preprocessor_path):
        """Test that load_preprocessor returns a ColumnTransformer instance."""
        preprocessor = load_preprocessor(preprocessor_path)

        assert isinstance(preprocessor, ColumnTransformer), (
            f"Expected ColumnTransformer, got {type(preprocessor)}"
        )

    def test_load_preprocessor_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_preprocessor("nonexistent/path/to/preprocessor.pkl")

    def test_load_preprocessor_invalid_object(self):
        """Test that ValueError is raised when file doesn't contain ColumnTransformer."""
        # Create a temp file with a non-ColumnTransformer object
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            import joblib

            joblib.dump({"not": "a preprocessor"}, temp_path)

            with pytest.raises(ValueError) as exc_info:
                load_preprocessor(temp_path)

            assert "not a ColumnTransformer" in str(exc_info.value)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_preprocessor_is_fitted(self, preprocessor_path):
        """Test that loaded preprocessor is fitted (has transformers_)."""
        preprocessor = load_preprocessor(preprocessor_path)

        # Fitted ColumnTransformer should have transformers_ attribute
        assert hasattr(preprocessor, "transformers_"), (
            "Preprocessor should have 'transformers_' attribute after fitting"
        )


# ============================================================================
# TESTS: validate_raw_input
# ============================================================================


class TestValidateRawInput:
    """Tests for validate_raw_input function."""

    def test_validate_complete_input(self, valid_input_dict):
        """Test that complete input returns True."""
        result = validate_raw_input(valid_input_dict)

        assert result is True, "Complete input should validate to True"

    def test_validate_incomplete_input(self, incomplete_input_dict):
        """Test that incomplete input returns False."""
        result = validate_raw_input(incomplete_input_dict)

        assert result is False, "Incomplete input should validate to False"

    def test_validate_missing_single_feature(self, single_missing_feature_input):
        """Test that input missing one feature returns False."""
        result = validate_raw_input(single_missing_feature_input)

        assert result is False, (
            "Input missing even one feature should validate to False"
        )

    def test_validate_empty_input(self):
        """Test that empty dictionary returns False."""
        result = validate_raw_input({})

        assert result is False, "Empty input should validate to False"

    def test_validate_extra_features_still_valid(self, valid_input_dict):
        """Test that extra features don't prevent validation."""
        valid_input_dict["Extra Feature"] = "Some Value"
        valid_input_dict["Another Extra"] = 123

        result = validate_raw_input(valid_input_dict)

        assert result is True, "Extra features should not affect validation"

    def test_validate_checks_all_features(self, valid_input_dict):
        """Test that ALL 13 features are checked."""
        config = PreprocessingConfig()

        # Test that all features are required
        for feature in config.FEATURES:
            test_input = valid_input_dict.copy()
            del test_input[feature]

            result = validate_raw_input(test_input)
            assert result is False, f"Missing {feature} should cause validation to fail"


# ============================================================================
# TESTS: transform_raw_input
# ============================================================================


class TestTransformRawInput:
    """Tests for transform_raw_input function."""

    def test_transform_returns_numpy_array(self, valid_input_dict, mock_preprocessor):
        """Test that transform returns a numpy array."""
        result = transform_raw_input(valid_input_dict, mock_preprocessor)

        assert isinstance(result, np.ndarray), (
            f"Expected numpy array, got {type(result)}"
        )

    def test_transform_returns_correct_shape(self, valid_input_dict, mock_preprocessor):
        """Test that transformed array has shape (1, N)."""
        result = transform_raw_input(valid_input_dict, mock_preprocessor)

        assert result.shape[0] == 1, (
            f"Expected batch dimension of 1, got {result.shape[0]}"
        )
        assert result.ndim == 2, f"Expected 2D array, got {result.ndim}D"

    def test_transform_calls_preprocessor_transform(
        self, valid_input_dict, mock_preprocessor
    ):
        """Test that preprocessor.transform is called."""
        import pandas as pd

        transform_raw_input(valid_input_dict, mock_preprocessor)

        # Should have called transform with a DataFrame
        mock_preprocessor.transform.assert_called_once()
        call_args = mock_preprocessor.transform.call_args
        assert isinstance(call_args[0][0], pd.DataFrame), (
            "transform should be called with a DataFrame"
        )

    def test_transform_incomplete_input_raises_error(
        self, incomplete_input_dict, mock_preprocessor
    ):
        """Test that incomplete input raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            transform_raw_input(incomplete_input_dict, mock_preprocessor)

        assert "Missing required features" in str(exc_info.value)

    def test_transform_shows_missing_features_in_error(
        self, single_missing_feature_input, mock_preprocessor
    ):
        """Test that error message lists missing features."""
        with pytest.raises(ValueError) as exc_info:
            transform_raw_input(single_missing_feature_input, mock_preprocessor)

        error_msg = str(exc_info.value)
        assert "Missing required features" in error_msg
        assert "Easter Period" in error_msg, (
            "Error should specify which features are missing"
        )

    def test_transform_with_real_preprocessor(
        self, valid_input_dict, preprocessor_path
    ):
        """Integration test: transform with real preprocessor file."""
        # Load the actual preprocessor
        preprocessor = load_preprocessor(preprocessor_path)

        # Transform the input
        result = transform_raw_input(valid_input_dict, preprocessor)

        # Check output
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        # The real preprocessor should produce a specific number of features
        # This will depend on your one-hot encoding
        assert result.shape[1] > 0


# ============================================================================
# TESTS: PreprocessingConfig
# ============================================================================


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_config_has_all_features(self):
        """Test that config contains all 13 features."""
        config = PreprocessingConfig()

        assert len(config.FEATURES) == 13, (
            f"Expected 13 features, got {len(config.FEATURES)}"
        )

    def test_config_features_are_strings(self):
        """Test that all feature names are strings."""
        config = PreprocessingConfig()

        for feature in config.FEATURES:
            assert isinstance(feature, str), (
                f"Feature name {feature} should be a string"
            )

    def test_config_features_match_expected(self):
        """Test that features match expected names."""
        config = PreprocessingConfig()

        expected_features = [
            "State",
            "Speed Limit",
            "National Road Type",
            "Road User",
            "Age",
            "Gender",
            "Bus Involvement",
            "Articulated Truck Involvement",
            "Heavy Rigid Truck Involvement",
            "Dayweek",
            "Time",
            "Christmas Period",
            "Easter Period",
        ]

        assert config.FEATURES == tuple(expected_features), (
            "Feature names should match expected list"
        )
