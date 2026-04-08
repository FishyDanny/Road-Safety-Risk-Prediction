"""Preprocessing module for DeepRisk road safety prediction.

This module provides utilities for loading the fitted ColumnTransformer
and transforming raw input data for model prediction.
"""

from dataclasses import dataclass
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
import joblib


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing features.

    Attributes:
        FEATURES: Tuple of 13 feature names required for prediction.
            - State: Australian state where crash occurred
            - Speed Limit: Speed limit at crash location
            - National Road Type: Classification of road type
            - Road User: Type of road user involved
            - Age: Age of the person involved
            - Gender: Gender of the person involved
            - Bus Involvement: Whether a bus was involved
            - Articulated Truck Involvement: Whether an articulated truck was involved
            - Heavy Rigid Truck Involvement: Whether a heavy rigid truck was involved
            - Dayweek: Day of the week
            - Time: Hour of the day (0-23)
            - Christmas Period: Whether during Christmas period
            - Easter Period: Whether during Easter period
    """

    FEATURES: tuple = (
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
    )


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a fitted ColumnTransformer preprocessor from disk.

    Args:
        path: Path to the pickled preprocessor file (e.g., 'models/preprocessor.pkl')

    Returns:
        The fitted ColumnTransformer ready for transforming new data.

    Raises:
        FileNotFoundError: If the preprocessor file doesn't exist.
        ValueError: If the loaded object is not a ColumnTransformer.

    Example:
        >>> preprocessor = load_preprocessor('models/preprocessor.pkl')
        >>> type(preprocessor)
        <class 'sklearn.compose._column_transformer.ColumnTransformer'>
    """
    preprocessor = joblib.load(path)
    if not isinstance(preprocessor, ColumnTransformer):
        raise ValueError(
            f"Loaded object is not a ColumnTransformer, got {type(preprocessor)}"
        )
    return preprocessor


def validate_raw_input(input_dict: dict) -> bool:
    """Validate that all required features are present in input dict.

    Checks that the input dictionary contains all 13 features required
    for prediction. Does not validate the values themselves - that's
    handled by the Pydantic schemas.

    Args:
        input_dict: Dictionary containing raw input features.

    Returns:
        True if all features are present, False otherwise.

    Example:
        >>> valid_input = {'State': 'NSW', 'Speed Limit': 60, ...}
        >>> validate_raw_input(valid_input)
        True
        >>> incomplete_input = {'State': 'NSW'}
        >>> validate_raw_input(incomplete_input)
        False
    """
    config = PreprocessingConfig()
    return all(feature in input_dict for feature in config.FEATURES)


def transform_raw_input(
    input_dict: dict, preprocessor: ColumnTransformer
) -> np.ndarray:
    """Transform raw input dict to preprocessed numpy array.

    Converts a dictionary of raw human-readable features into a
    preprocessed numpy array suitable for model prediction, using
    the fitted ColumnTransformer pipeline.

    The preprocessor handles:
    - StandardScaler for numeric features (Time, Speed Limit, Age)
    - OneHotEncoder for categorical features (State, Road User, etc.)
    - OrdinalEncoder for ordinal features (Dayweek, National Road Type)

    Args:
        input_dict: Dictionary containing all 13 required features.
        preprocessor: Fitted ColumnTransformer from load_preprocessor().

    Returns:
        Preprocessed numpy array with shape (1, N) where N is the
        number of transformed features.

    Raises:
        ValueError: If input_dict doesn't contain all required features.

    Example:
        >>> preprocessor = load_preprocessor('models/preprocessor.pkl')
        >>> raw_input = {
        ...     'State': 'NSW', 'Speed Limit': 60, 'National Road Type': 'Arterial Road',
        ...     'Road User': 'Driver', 'Age': 30, 'Gender': 'Male',
        ...     'Bus Involvement': 'No', 'Articulated Truck Involvement': 'No',
        ...     'Heavy Rigid Truck Involvement': 'No', 'Dayweek': 'Friday',
        ...     'Time': 18, 'Christmas Period': 'No', 'Easter Period': 'No'
        ... }
        >>> transformed = transform_raw_input(raw_input, preprocessor)
        >>> transformed.shape
        (1, 60)
    """
    if not validate_raw_input(input_dict):
        config = PreprocessingConfig()
        missing = [f for f in config.FEATURES if f not in input_dict]
        raise ValueError(f"Missing required features: {missing}")

    df = pd.DataFrame([input_dict])
    return preprocessor.transform(df)
