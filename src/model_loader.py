"""Model loading utilities for DeepRisk."""

from typing import Union
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier


class ModelRegistry:
    """Registry to cache loaded models in memory."""

    def __init__(self):
        self._models: dict = {}

    def get(
        self, model_name: str, path: str
    ) -> Union[RandomForestClassifier, tf.keras.Model]:
        """Get a cached model or load it if not in cache.

        Args:
            model_name: Identifier for the model type ('random_forest' or keras model name).
            path: Path to the model file on disk.

        Returns:
            The loaded model, either from cache or freshly loaded.
        """
        if model_name not in self._models:
            if model_name == "random_forest":
                self._models[model_name] = load_random_forest(path)
            else:
                self._models[model_name] = load_keras_model(path)
        return self._models[model_name]

    def clear(self) -> None:
        """Clear all cached models from memory."""
        self._models.clear()

    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is already cached.

        Args:
            model_name: Identifier for the model.

        Returns:
            True if model is in cache, False otherwise.
        """
        return model_name in self._models


def load_random_forest(path: str) -> RandomForestClassifier:
    """Load a trained Random Forest model from disk.

    Args:
        path: Path to the saved model file (.pkl or .joblib).

    Returns:
        Loaded RandomForestClassifier instance.
    """
    model = joblib.load(path)
    return model


def load_keras_model(path: str) -> tf.keras.Model:
    """Load a trained Keras model from disk.

    Args:
        path: Path to the saved model file (.h5 or SavedModel directory).

    Returns:
        Loaded tf.keras.Model instance.
    """
    model = tf.keras.models.load_model(path)
    return model


def get_model(
    model_name: str, registry: ModelRegistry, path: str
) -> Union[RandomForestClassifier, tf.keras.Model]:
    """Convenience function to get a model through a registry.

    Args:
        model_name: Identifier for the model type ('random_forest' or keras model name).
        registry: ModelRegistry instance to use for caching.
        path: Path to the model file on disk.

    Returns:
        The loaded model, either from cache or freshly loaded.
    """
    return registry.get(model_name, path)
