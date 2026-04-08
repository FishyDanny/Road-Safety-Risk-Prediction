# DeepRisk Implementation Notepad

## Task 6: Preprocessing Module - 2025-04-01

### Findings

1. **Feature Count Discrepancy**: The notebook `01_preprocess.ipynb` shows **13 features** (not 12 as mentioned in the task):
   - Road features: State, Speed Limit, National Road Type
   - Victim features: Road User, Age, Gender
   - Vehicle involvement: Bus Involvement, Articulated Truck Involvement, Heavy Rigid Truck Involvement
   - Temporal features: Dayweek, Time, Christmas Period, Easter Period

2. **Module Created**: `src/preprocessing.py` with:
   - `PreprocessingConfig` dataclass with correct 13 features
   - `load_preprocessor(path)` - loads fitted ColumnTransformer
   - `validate_raw_input(input_dict)` - validates all 13 features present
   - `transform_raw_input(input_dict, preprocessor)` - transforms raw dict to numpy array

3. **Preprocessor Pipeline** (from notebook):
   - StandardScaler: Time, Speed Limit, Age
   - OneHotEncoder: State, Road User, Gender, Bus Involvement, Heavy Rigid Truck Involvement, Articulated Truck Involvement, Christmas Period, Easter Period
   - OrdinalEncoder: Dayweek, National Road Type

### Next Steps
- Task 7: Model loading utilities (depends on models from Task 5)
- Task 11: Raw prediction endpoint (depends on preprocessing module)
## Task 15: SHAP Analysis - 2025-04-01

### Findings

1. **SHAP Analysis Added to 05_analysis.ipynb**:
   - TreeExplainer for Random Forest (1000 row sample for efficiency)
   - KernelExplainer for Neural Networks (100 background, 500 explain samples)
   - Summary plots for all three models
   - Waterfall plots for individual predictions
   - Interactive HTML visualizations

2. **Output Files Created**:
   - `reports/shap_rf_summary.png` - Random Forest feature importance
   - `reports/shap_ffnn_summary.png` - Feedforward NN feature importance
   - `reports/shap_residualnn_summary.png` - Residual NN feature importance
   - `reports/shap_waterfall_sample*.png` - Individual prediction explanations
   - `reports/shap_interactive/*.html` - Interactive beeswarm plots

3. **Key Implementation Details**:
   - Environment-agnostic path handling (Colab vs Local)
   - Feature names extracted from preprocessor pipeline
   - Positive SHAP = increased probability of multiple vehicle crash
   - Negative SHAP = increased probability of single vehicle crash

4. **Actuarial Interpretation Added**:
   - Base value explanation (average prediction)
   - SHAP value interpretation for insurance pricing
   - Regulatory compliance notes ("right to explanation")
   - Cross-model feature importance comparison

## Task 9: FastAPI Application & Schemas - 2025-04-01

### Findings

1. **Files Created**:
   - `src/api/main.py` - FastAPI application with lifespan context manager
   - `src/api/schemas.py` - Pydantic schemas for request/response validation

2. **main.py Implementation**:
   - FastAPI app with title "DeepRisk Road Safety API"
   - CORS middleware for localhost:3000, localhost:5173, 127.0.0.1:3000
   - `lifespan` context manager loads all 3 models at startup via ModelRegistry
   - Models stored in `app.state.registry` for route access
   - Root endpoint `/` returns API info JSON
   - Health check `/health` returns status, models_loaded, model_names
   - Models loaded: random_forest, ffnn, residual

3. **schemas.py Implementation**:
   - `RawInputFeatures` - 13 features with Pydantic validation (uses Field aliases for names with spaces)
   - `PreprocessedInput` - features as `List[float]`
   - `PredictionResponse` - prediction (int), probability (float), model_name (str)
   - `HealthResponse` - status, models_loaded, model_names
   - `ErrorResponse` - error, detail
   - Validators for: State (Australian states), Gender, Yes/No fields, Dayweek, Christmas/Easter Period

4. **Routes Structure**:
   - `src/api/routes/__init__.py` exists (placeholder for Tasks 11-12)
   - Prediction routers will be added in Tasks 11-12

5. **Verification**:
   - Python compilation successful
   - App object imports correctly
   - All models load successfully at startup

## Task 11: Prediction Endpoints - 2025-04-02

### Findings

1. **Files Created/Modified**:
   - Created: `src/api/routes/predict.py` with both endpoints
   - Modified: `src/api/main.py` to register router at `/api/v1/predict`

2. **Endpoints Implemented**:
   - POST `/api/v1/predict/raw`: Accepts 13 human-readable features, returns predictions from all 3 models
   - POST `/api/v1/predict/preprocessed`: Accepts preprocessed feature vector, returns predictions from all 3 models

3. **Implementation Details**:
   - Preprocessor lazy-loaded on first request and cached in `app.state.preprocessor`
   - Model predictions via ModelRegistry from `app.state.registry`
   - Random Forest: `predict_proba()` returns `[prob_0, prob_1]`, class 1 = multiple vehicle
   - Neural Networks: `predict()` returns probability directly
   - Input dictionary uses `by_alias=True` for Pydantic field conversion (Speed_Limit -> Speed Limit)

4. **Error Handling**:
   - 400: Invalid input or transformation failure
   - 500: Preprocessor/model not found or model loading error

5. **Verification**:
   - Python compilation successful for both files
   - Router registered with prefix `/api/v1/predict`, tag "Prediction"

### Follow-Up Tasks
- Task 12: Evaluation endpoint with probability thresholds
