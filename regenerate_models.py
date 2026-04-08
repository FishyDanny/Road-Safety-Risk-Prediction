#!/usr/bin/env python3
"""
Regenerate model artifacts locally from notebooks.
Downloads data and processes it outside of Google Colab.
"""

import ssl
import urllib.request

# Fix SSL certificate verification issue
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras

# Set paths
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("Step 1: Downloading data...")
url = "https://aueprod01ckanstg.blob.core.windows.net/public-catalogue/1633969d-46d3-437f-82e5-4d468db04a9f/bitre_fatalities_may2025.xlsx"
data = pd.read_excel(url, header=4)
print(f"  Downloaded {len(data)} records")

print("Step 2: Preprocessing data...")
# Translate unknown indicators to NaN
unknown_indicators = ["Unknown", "unknown", "-9"]
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].replace(unknown_indicators, np.nan)

# Fill missing values
data["Bus Involvement"] = data["Bus Involvement"].fillna("Missing")
data["Articulated Truck Involvement"] = data["Articulated Truck Involvement"].fillna("Missing")
data["Heavy Rigid Truck Involvement"] = data["Heavy Rigid Truck Involvement"].fillna("Missing")
data["National Road Type"] = data["National Road Type"].fillna("Missing")
data["Road User"] = data["Road User"].fillna("Missing")
data["Gender"] = data["Gender"].fillna("Missing")
data["Time"] = pd.to_datetime(data["Time"], errors="coerce").dt.hour
data = data.dropna(subset=["Crash Type", "Speed Limit", "Time"])

# Split data
train, temp_data = train_test_split(
    data,
    test_size=0.4,
    random_state=42,
    stratify=data["Crash Type"]
)
val, test = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_data["Crash Type"]
)

# Define features
features = [
    "State", "Speed Limit", "National Road Type",
    "Road User", "Age", "Gender",
    "Bus Involvement", "Articulated Truck Involvement", "Heavy Rigid Truck Involvement",
    "Dayweek", "Time", "Christmas Period", "Easter Period"
]
target = "Crash Type"

numeric_features = ["Time", "Speed Limit", "Age"]
categorical_features = [
    "State", "Road User", "Gender", "Bus Involvement",
    "Heavy Rigid Truck Involvement", "Articulated Truck Involvement",
    "Christmas Period", "Easter Period"
]
ordinal_features = ["Dayweek", "National Road Type"]

# Create preprocessor
preprocessor = ColumnTransformer([
    ("numerical", StandardScaler(), numeric_features),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("ordinal", OrdinalEncoder(), ordinal_features)
])

# Transform features
X_train = preprocessor.fit_transform(train[features])
X_val = preprocessor.transform(val[features])
X_test = preprocessor.transform(test[features])

# Convert targets
y_train = train[target].map({"Single": 0, "Multiple": 1}).values
y_val = val[target].map({"Single": 0, "Multiple": 1}).values
y_test = test[target].map({"Single": 0, "Multiple": 1}).values

print("Step 3: Saving preprocessed data...")
joblib.dump(X_train, os.path.join(MODELS_DIR, "X_train.pkl"))
joblib.dump(X_val, os.path.join(MODELS_DIR, "X_val.pkl"))
joblib.dump(X_test, os.path.join(MODELS_DIR, "X_test.pkl"))
joblib.dump(y_train, os.path.join(MODELS_DIR, "y_train.pkl"))
joblib.dump(y_val, os.path.join(MODELS_DIR, "y_val.pkl"))
joblib.dump(y_test, os.path.join(MODELS_DIR, "y_test.pkl"))
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
print(f"  Saved to {MODELS_DIR}/")

print("Step 4: Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, os.path.join(MODELS_DIR, "final_baseline_rf_model.pkl"))
print("  Random Forest trained and saved")

print("Step 5: Training Feedforward Neural Network...")
input_dim = X_train.shape[1]

ffnn_model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid")
])

ffnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

ffnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=1
)

ffnn_model.save(os.path.join(MODELS_DIR, "final_first_dl_model.keras"))
print("  Feedforward NN trained and saved")

print("Step 6: Training Residual Neural Network...")
# Simplified residual model
inputs = keras.layers.Input(shape=(input_dim,))
x = keras.layers.Dense(64, activation="relu")(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.3)(x)

# Residual block
residual = keras.layers.Dense(64, activation="relu")(x)
residual = keras.layers.BatchNormalization()(residual)
residual = keras.layers.Dropout(0.3)(residual)
x = keras.layers.Add()([x, residual])

x = keras.layers.Dense(32, activation="relu")(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

residual_model = keras.Model(inputs, outputs)
residual_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

residual_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
    verbose=1
)

residual_model.save(os.path.join(MODELS_DIR, "final_second_dl_model.keras"))
print("  Residual NN trained and saved")

print("\nAll models regenerated successfully!")
print(f"Models saved to: {MODELS_DIR}/")
print("\nGenerated files:")
for f in os.listdir(MODELS_DIR):
    if f.endswith((".pkl", ".keras")):
        print(f"  - {f}")
