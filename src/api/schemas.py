"""Pydantic schemas for DeepRisk API request/response validation."""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, field_validator


class RawInputFeatures(BaseModel):
    """Raw input features for crash prediction.

    Represents the 13 features required for prediction before preprocessing.
    All features match the original dataset format.

    Attributes:
        State: Australian state where crash occurred (e.g., 'NSW', 'VIC').
        Speed Limit: Speed limit at crash location (40-130 km/h).
        National Road Type: Classification of road type.
        Road User: Type of road user involved.
        Age: Age of the person involved (0-100).
        Gender: Gender of the person involved.
        Bus Involvement: Whether a bus was involved ('Yes'/'No').
        Articulated Truck Involvement: Whether an articulated truck was involved.
        Heavy Rigid Truck Involvement: Whether a heavy rigid truck was involved.
        Dayweek: Day of the week.
        Time: Hour of the day (0-23).
        Christmas Period: Whether during Christmas period.
        Easter Period: Whether during Easter period.
    """

    State: str = Field(..., description="Australian state where crash occurred")
    Speed_Limit: int = Field(
        ..., alias="Speed Limit", description="Speed limit at crash location (km/h)"
    )
    National_Road_Type: str = Field(
        ..., alias="National Road Type", description="Classification of road type"
    )
    Road_User: str = Field(
        ..., alias="Road User", description="Type of road user involved"
    )
    Age: int = Field(..., ge=0, le=120, description="Age of the person involved")
    Gender: str = Field(..., description="Gender of the person involved")
    Bus_Involvement: str = Field(
        ..., alias="Bus Involvement", description="Whether a bus was involved"
    )
    Articulated_Truck_Involvement: str = Field(
        ...,
        alias="Articulated Truck Involvement",
        description="Whether an articulated truck was involved",
    )
    Heavy_Rigid_Truck_Involvement: str = Field(
        ...,
        alias="Heavy Rigid Truck Involvement",
        description="Whether a heavy rigid truck was involved",
    )
    Dayweek: str = Field(..., description="Day of the week")
    Time: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    Christmas_Period: str = Field(
        ..., alias="Christmas Period", description="Whether during Christmas period"
    )
    Easter_Period: str = Field(
        ..., alias="Easter Period", description="Whether during Easter period"
    )

    model_config = {"populate_by_name": True}

    @field_validator("State")
    @classmethod
    def validate_state(cls, v: str) -> str:
        """Validate Australian state/territory code."""
        valid_states = {"NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"}
        if v.upper() not in valid_states:
            raise ValueError(
                f"Invalid state '{v}'. Must be one of: {', '.join(sorted(valid_states))}"
            )
        return v.upper()

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        """Validate gender value."""
        valid_genders = {"Male", "Female", "Unknown"}
        if v.title() not in valid_genders:
            raise ValueError(
                f"Invalid gender '{v}'. Must be one of: {', '.join(valid_genders)}"
            )
        return v.title()

    @field_validator(
        "Bus_Involvement",
        "Articulated_Truck_Involvement",
        "Heavy_Rigid_Truck_Involvement",
    )
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        """Validate Yes/No fields."""
        if v.title() not in {"Yes", "No"}:
            raise ValueError(f"Invalid value '{v}'. Must be 'Yes' or 'No'")
        return v.title()

    @field_validator("Christmas_Period", "Easter_Period")
    @classmethod
    def validate_holiday_period(cls, v: str) -> str:
        """Validate holiday period fields."""
        if v.title() not in {"Yes", "No"}:
            raise ValueError(f"Invalid value '{v}'. Must be 'Yes' or 'No'")
        return v.title()

    @field_validator("Dayweek")
    @classmethod
    def validate_dayweek(cls, v: str) -> str:
        """Validate day of week."""
        valid_days = {
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        }
        if v.title() not in valid_days:
            raise ValueError(
                f"Invalid day '{v}'. Must be one of: {', '.join(sorted(valid_days))}"
            )
        return v.title()


class PreprocessedInput(BaseModel):
    """Preprocessed input features ready for model prediction.

    Represents the transformed features after preprocessing pipeline.
    This is the output of the ColumnTransformer pipeline.

    Attributes:
        features: List of preprocessed feature values (numpy array flattened).
    """

    features: List[float] = Field(
        ..., description="Preprocessed feature vector (output of ColumnTransformer)"
    )


class PredictionResponse(BaseModel):
    """Response from prediction endpoint.

    Attributes:
        prediction: Binary prediction (0=single vehicle, 1=multiple vehicle).
        probability: Probability score for the predicted class.
        model_name: Name of the model used for prediction.
    """

    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Predicted crash type (0=single, 1=multiple vehicle)",
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of predicted class"
    )
    model_name: str = Field(..., description="Model used for prediction")


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Health status ('healthy' or 'unhealthy').
        models_loaded: Whether models are loaded in memory.
        model_names: List of loaded model names.
    """

    status: str = Field(..., description="Health status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    model_names: List[str] = Field(
        default_factory=list, description="Names of loaded models"
    )


class ErrorResponse(BaseModel):
    """Error response.

    Attributes:
        error: Error type or code.
        detail: Detailed error message.
    """

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
