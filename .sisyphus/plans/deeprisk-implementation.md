# DeepRisk Implementation Plan

## TL;DR

> **Quick Summary**: Implement SHAP explainability, FastAPI prediction API with dual endpoints, and Docker deployment for a road safety risk prediction system using TensorFlow models.
>
> **Deliverables**:
> - SHAP analysis added to `05_analysis.ipynb` (static images + interactive HTML)
> - FastAPI app with `/predict/raw` and `/predict/preprocessed` endpoints
> - Model artifacts regenerated locally in `models/` directory
> - Complete test suite (unit + integration + API tests)
> - Production-ready Dockerfile for AWS EC2 deployment
> - requirements.txt with pinned dependencies
>
> **Estimated Effort**: Medium (foundation + API + Docker + tests)
> **Parallel Execution**: YES - 5 waves with up to 4 parallel tasks each
> **Critical Path**: Setup → Model Regeneration → Core Modules → API Endpoints → Docker

---

## Context

### Original Request
Implement 100% of DEVELOPMENT_PLAN.md tasks:
1. Implement SHAP values script within `05_analysis.ipynb` for actuarial transparency
2. Build a `Dockerfile` to export the heavy PyTorch environment to AWS EC2 easily
   - **CORRECTION**: Actual framework is TensorFlow, not PyTorch
3. Create a `predict()` REST endpoint using FastAPI

### Interview Summary

**Key Discussions**:
- **Framework**: Use existing TensorFlow/Keras models (not PyTorch) — DEVELOPMENT_PLAN.md was aspirational, actual code uses TensorFlow
- **SHAP Coverage**: All three models (Random Forest, Feedforward NN, Residual NN) for complete actuarial transparency
- **API Input**: Dual endpoints — `/predict/raw` accepts human-readable values, `/predict/preprocessed` accepts encoded arrays
- **Docker Strategy**: Embed models in image for portability, use `tensorflow/tensorflow` base image
- **Model Artifacts**: Regenerate locally from notebooks (user has data, ~30 min runtime)
- **Testing**: TDD approach with pytest; unit tests (preprocessing, prediction), integration tests (full API workflow), API tests (health, errors)
- **SHAP Output**: Both static images embedded in notebook AND interactive HTML files for exploration

**Research Findings**:
- **No test infrastructure exists** — No pytest, no test files, no CI/CD
- **No local model artifacts** — Everything stored on Google Drive, needs regeneration
- **Preprocessing pipeline**: sklearn ColumnTransformer with StandardScaler + OneHotEncoder + OrdinalEncoder
- **12 raw features** → preprocessing → model input
- **Target**: Crash Type (Single=0, Multiple=1)
- **Best model performance**: Random Forest (F1=0.71, AUC-ROC=0.81)

### Technical Stack
- **Framework**: TensorFlow 2.x + Keras (existing `.keras` models)
- **ML Libraries**: scikit-learn (Random Forest), pandas, numpy
- **Artifacts**: joblib/pickle for sklearn, keras.models for TensorFlow
- **API**: FastAPI + Uvicorn (new)
- **Explainability**: SHAP (new)
- **Testing**: pytest + httpx for API testing (new)
- **Deployment**: Docker with TensorFlow base image (new)

### Metis Review

**Identified Gaps** (addressed):
- **Model artifacts not local**: Resolved by regenerating from notebooks in Wave 2
- **No requirements.txt**: Created in Wave 1 with pinned versions
- **No test infrastructure**: Created in Wave 1 with pytest.ini
- **Input schema ambiguity**: Resolved by dual endpoints (raw + preprocessed)
- **Docker base image**: Using `tensorflow/tensorflow:latest` with Python 3.10
- **SHAP computation scope**: User confirmed both formats (static + interactive)

**Guardrails Applied**:
- MUST NOT rewrite or regenerate existing notebooks (00-04) — only regenerate artifacts
- MUST NOT add database integration, authentication, or frontend dashboard
- MUST NOT train or retrain models — use frozen weights from notebooks
- MUST use Pydantic models for FastAPI input validation
- MUST validate model loading in FastAPI startup event (not per-request)

---

## Work Objectives

### Core Objective
Create a production-ready FastAPI prediction API with SHAP explainability and Docker deployment for the DeepRisk road safety risk prediction system, implementing all three tasks from DEVELOPMENT_PLAN.md with 100% test coverage using TDD methodology.

### Concrete Deliverables
- `requirements.txt` — Pinned dependencies for reproducibility
- `pytest.ini` — Test configuration
- `models/` directory — Local artifacts (preprocessor.pkl, three model files, processed data)
- `src/__init__.py`, `src/config.py`, `src/logger.py` — Project infrastructure
- `src/preprocessing.py` — ColumnTransformer pipeline module
- `src/model_loader.py` — Model and preprocessor loading utilities
- `src/api/__init__.py`, `src/api/main.py` — FastAPI application
- `src/api/schemas.py` — Pydantic input/output schemas
- `src/api/routes/predict.py` — Prediction endpoints (raw + preprocessed)
- `src/api/routes/health.py` — Health check endpoint
- `05_analysis.ipynb` — Updated with SHAP analysis cells
- `reports/shap_*.png` — Static SHAP visualizations
- `reports/shap_interactive/` — Interactive HTML explanations
- `tests/conftest.py` — Test fixtures and configuration
- `tests/test_preprocessing.py` — Unit tests for preprocessing module
- `tests/test_model_loader.py` — Unit tests for model loading
- `tests/test_api_integration.py` — Integration tests for full API workflow
- `tests/test_api_health.py` — API health and error case tests
- `Dockerfile` — Multi-stage build for AWS EC2 deployment
- `.dockerignore` — Exclude unnecessary files from Docker context

### Definition of Done
- [ ] `pytest tests/ -v` passes with 100% coverage on `src/`
- [ ] `uvicorn src.api:app --host 0.0.0.0 --port 8000` starts successfully
- [ ] `curl http://localhost:8000/health` returns `{"status": "healthy", "models_loaded": true}`
- [ ] `curl -X POST http://localhost:8000/predict/raw` with valid input returns prediction JSON
- [ ] `curl -X POST http://localhost:8000/predict/preprocessed` with valid input returns prediction JSON
- [ ] `docker build -t road-safety-api:latest .` completes successfully with image size < 3GB
- [ ] `docker run -p 8000:8000 road-safety-api:latest` serves working API
- [ ] `05_analysis.ipynb` contains SHAP summary plots and waterfall plots
- [ ] `reports/shap_*.png` files exist (at least 3 model plots)
- [ ] All code follows TDD: tests written before implementation

### Must Have
1. **Model Artifacts**: All models and preprocessors regenerated locally in `models/` directory
2. **Dual API Endpoints**: Both `/predict/raw` (human-readable) and `/predict/preprocessed` (encoded) working
3. **SHAP Analysis**: Actuarial transparency for all three models with both static and interactive outputs
4. **Full Test Coverage**: Unit tests (preprocessing, prediction), integration tests (API workflow), API tests (health, errors)
5. **Working Docker**: Production-ready Dockerfile that builds and runs successfully
6. **Requirements File**: Complete `requirements.txt` with pinned versions for reproducibility

### Must NOT Have (Guardrails)
- **No Database Integration**: No PostgreSQL, SQLite, or any data persistence layer
- **No Authentication**: No JWT, API keys, OAuth, or security layer
- **No Frontend Dashboard**: No React, Mapbox, or visualization UI (out of scope)
- **No CI/CD Pipeline**: No GitHub Actions, Jenkins, or automated deployment
- **No Model Retraining**: Use frozen weights from existing notebooks only
- **No Notebook Rewriting**: Do not modify notebooks 00-04, only add to `05_analysis.ipynb`
- **No Per-Request Model Loading**: Models must load once at startup, not on every prediction

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: NO (needs setup)
- **Automated tests**: TDD (tests written before implementation)
- **Framework**: pytest
- **TDD Workflow**: Each TODO follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Backend/API**: Use Bash (curl) — Send requests, assert status + response fields
- **Library/Module**: Use Bash (python -c) — Import, call functions, compare output
- **Notebooks**: Use Bash (jupyter execute) — Run notebook, check cell outputs
- **Docker**: Use Bash (docker commands) — Build, run, test, cleanup

---

## Execution Strategy

### Parallel Execution Waves

> Maximize throughput by grouping independent tasks into parallel waves.
> Each wave completes before the next begins.
> Target: 3-4 tasks per wave. Fewer than 3 per wave (except final) = under-splitting.

```
Wave 1 (Start Immediately — foundation + scaffolding):
├── Task 1: Project configuration (requirements.txt, pytest.ini) [quick]
├── Task 2: Test infrastructure setup (tests/, conftest.py) [quick]
├── Task 3: Source directory structure (src/) [quick]
└── Task 4: .dockerignore configuration [quick]
    → 4 tasks, ALL can start immediately
    → Blocks: everything else (foundation layer)

Wave 2 (After Wave 1 — core modules + model regeneration):
├── Task 5: Regenerate model artifacts locally [deep] — Regenerate from notebooks
├── Task 6: Preprocessing module (src/preprocessing.py) [unspecified-high]
├── Task 7: Model loading utilities (src/model_loader.py) [unspecified-high]
└── Task 8: Config and logging (src/config.py, src/logger.py) [quick]
    → 4 tasks, Task 5-6-7-8 can run in parallel
    → Task 5 blocks: Tasks 9-10 (need models)
    → Task 6 blocks: Task 11 (preprocessing needed)
    → Task 7 blocks: Task 10 (model loading needed)

Wave 3 (After Wave 2 — SHAP + API core):
├── Task 9: SHAP analysis in 05_analysis.ipynb [ai-engineer] — Depends: Task 5 (models)
├── Task 10: FastAPI application (src/api/main.py, src/api/schemas.py) [quick] — Depends: Task 7
├── Task 11: Raw prediction endpoint (src/api/routes/predict.py) [unspecified-high] — Depends: Task 6, Task 7
└── Task 12: Preprocessed prediction endpoint (add to predict.py) [unspecified-high] — Depends: Task 7
    → 4 tasks, Task 9-10-11-12 can run in parallel (different concerns)
    → Task 11 and Task 12 are codependent (same file)

Wave 4 (After Wave 3 — TDD unit + integration tests):
├── Task 13: Unit tests - preprocessing (tests/test_preprocessing.py) [quick] — Depends: Task 6
├── Task 14: Unit tests - model prediction (tests/test_model_loader.py) [quick] — Depends: Task 7
├── Task 15: Integration tests - API workflow (tests/test_api_integration.py) [unspecified-high] — Depends: Task 11, Task 12
└── Task 16: API tests - health and errors (tests/test_api_health.py) [quick] — Depends: Task 10
    → 4 tasks, Task 13-14-15-16 can run in parallel
    → All are test tasks following TDD

Wave 5 (After Wave 4 — Docker deployment):
├── Task 17: Dockerfile creation [quick] — Depends: ALL previous tasks
└── Task 18: Docker build and test [quick] — Depends: Task 17
    → 2 tasks, must be sequential (Dockerfile first, then test)
    → Final verification before completion

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
    → Present results → Get explicit user okay

Critical Path: Task 1 → Task 5 → Task 9 → Task 11 → Task 15 → Task 17 → Task 18 → F1-F4 → user okay
Parallel Speedup: ~70% faster than sequential (4 tasks in Waves 1-4 can run concurrently)
Max Concurrent: 4 (Waves 1-4)
```

### Dependency Matrix

- **1-4**: — — 5-8, —
- **5**: 1 — 9, 2
- **6-8**: 1 — 11, 2
- **9**: 5 — 14, 2
- **10**: 7 — 15, 2
- **11**: 6, 7 — 15, 2
- **12**: 7 — 15, 2
- **13-16**: respective dependencies — 17, 2
- **17**: ALL — 18, —
- **18**: 17 — F1-F4, —

> Abbreviated for reference. YOUR generated plan must include FULL matrix.

### Agent Dispatch Summary

- **Wave 1**: **4 quick** — T1-T4 → `quick`
- **Wave 2**: **4 mixed** — T5 → `deep`, T6 → `unspecified-high`, T7 → `unspecified-high`, T8 → `quick`
- **Wave 3**: **4 mixed** — T9 → `ai-engineer`, T10 → `quick`, T11 → `unspecified-high`, T12 → `unspecified-high`
- **Wave 4**: **4 mixed** — T13 → `quick`, T14 → `quick`, T15 → `unspecified-high`, T16 → `quick`
- **Wave 5**: **2 quick** — T17 → `quick`, T18 → `quick`
- **FINAL**: **4 reviews** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

- [x] 1. Create requirements.txt with pinned dependencies

  **What to do**:
  - Create `requirements.txt` in project root with pinned versions for reproducibility
  - Include: tensorflow==2.15.0, keras==2.15.0, scikit-learn==1.3.2, pandas==2.1.3, numpy==1.26.2, joblib==1.3.2, fastapi==0.108.0, uvicorn[standard]==0.25.0, pydantic==2.5.3, shap==0.44.1, matplotlib==3.8.2, seaborn==0.13.0, jupyter==1.0.0, pytest==7.4.3, pytest-cov==4.1.0, httpx==0.26.0
  - Pin all versions with `==` to ensure consistency
  - Add comment grouping: "# ML Libraries", "# API Framework", "# Explainability", "# Development and Testing"

  **Must NOT do**:
  - Do NOT use unpinned versions (e.g., `tensorflow` without version)
  - Do NOT include packages not needed for this project
  - Do NOT add development tools like black, mypy, pylint (out of scope)

  **Recommended Agent Profile**:
  > Select category + skills based on task domain. Justify each choice.
  - **Category**: `quick`
    - Reason: Simple file creation with well-defined content, no complex logic
  - **Skills**: []
    - No specialized skills needed for requirements.txt creation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5 (needs dependencies installed)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `05_analysis.ipynb:import cells` - Current imports to ensure consistency
  - `README.md:Prerequisites` - Existing dependency documentation

  **Acceptance Criteria**:
  - [ ] requirements.txt file created in project root
  - [ ] All packages have pinned versions (e.g., `tensorflow==2.15.0`)
  - [ ] Comments separate package groups (ML, API, Explainability, Testing)
  - [ ] `pip install -r requirements.txt --dry-run` succeeds without conflicts

  **QA Scenarios**:

  ```
  Scenario: Requirements installation dry-run
    Tool: Bash
    Preconditions: requirements.txt created
    Steps:
      1. pip install -r requirements.txt --dry-run
      2. Verify all packages resolve without errors
    Expected Result: "Would install tensorflow-2.15.0, keras-2.15.0, ..."
    Failure Indicators: "ERROR: No matching distribution", "Conflicting requirements"
    Evidence: .sisyphus/evidence/task-01-requirements-dry-run.txt

  Scenario: TensorFlow version check
    Tool: Bash
    Preconditions: requirements.txt created
    Steps:
      1. python -c "import tensorflow as tf; print(tf.__version__)"
    Expected Result: "2.15.0" (or specified version)
    Failure Indicators: "ModuleNotFoundError", different version number
    Evidence: .sisyphus/evidence/task-01-tf-version.txt
  ```

  **Evidence to Capture**:
  - [ ] Dry-run output showing all packages resolve
  - [ ] TensorFlow version confirmation

  **Commit**: YES
  - Message: `chore: add requirements.txt with pinned dependencies`
  - Files: `requirements.txt`
  - Pre-commit: `pip install -r requirements.txt --dry-run`

- [x] 2. Set up test infrastructure

  **What to do**:
  - Create `tests/` directory in project root
  - Create `tests/conftest.py` with pytest fixtures for API client, model paths
  - Create `pytest.ini` in project root with configuration: testpaths=tests, python_files=*.py, python_functions=test_*, addopts=--verbose --tb=short
  - Add pytest fixtures for: mock prediction data, temporary model files, API test client
  - Create `.gitignore` entry for `__pycache__/`, `.pytest_cache/`, `.coverage`

  **Must NOT do**:
  - Do NOT create actual test files yet (done in later tasks)
  - Do NOT add CI/CD configuration (out of scope)
  - Do NOT include test fixtures for features not in scope (database, auth)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Directory creation and configuration files, standard pytest setup
  - **Skills**: []
    - No specialized skills needed for test infrastructure setup

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Tasks 13-16 (need test infrastructure)
  - **Blocked By**: None (can start immediately)

  **References**:
  - `pytest.ini` configuration patterns from FastAPI documentation

  **Acceptance Criteria**:
  - [ ] `tests/` directory exists
  - [ ] `tests/conftest.py` has base fixtures defined
  - [ ] `pytest.ini` exists with testpaths=tests configuration
  - [ ] `.gitignore` includes pytest cache paths

  **QA Scenarios**:

  ```
  Scenario: Pytest configuration check
    Tool: Bash
    Preconditions: pytest.ini created
    Steps:
      1. pytest --collect-only
    Expected Result: "no tests collected" (tests not written yet)
    Failure Indicators: "ERROR: pytest.ini not found", configuration errors
    Evidence: .sisyphus/evidence/task-02-pytest-config.txt

  Scenario: Conftest imports work
    Tool: Bash
    Preconditions: tests/conftest.py created
    Steps:
      1. python -c "from tests.conftest import *; print('Conftest loaded')"
    Expected Result: "Conftest loaded"
    Failure Indicators: "ModuleNotFoundError", "ImportError"
    Evidence: .sisyphus/evidence/task-02-conftest-import.txt
  ```

  **Evidence to Capture**:
  - [ ] Pytest collection output (empty but valid)
  - [ ] Conftest import verification

  **Commit**: YES
  - Message: `test: add pytest infrastructure and configuration`
  - Files: `tests/conftest.py`, `pytest.ini`, `.gitignore`
  - Pre-commit: None

- [x] 3. Create source directory structure

  **What to do**:
  - Create `src/` directory in project root
  - Create `src/__init__.py` (empty for package initialization)
  - Create `src/api/__init__.py` (empty for API subpackage)
  - Create `src/api/routes/__init__.py` (empty for routes subpackage)
  - Create directory structure matching plan: `src/`, `src/api/`, `src/api/routes/`, `models/`, `reports/`

  **Must NOT do**:
  - Do NOT create Python files yet (done in later tasks)
  - Do NOT create `data/` directory (out of scope - no database)
  - Do NOT add frontend directories (out of scope)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Directory creation with __init__.py files, no logic
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Tasks 6-12 (need src/ structure)
  - **Blocked By**: None (can start immediately)

  **References**:
  - Python package structure best practices

  **Acceptance Criteria**:
  - [ ] `src/` directory exists with `__init__.py`
  - [ ] `src/api/` directory exists with `__init__.py`
  - [ ] `src/api/routes/` directory exists with `__init__.py`
  - [ ] `models/` directory exists (for artifacts)
  - [ ] `reports/` directory exists (for SHAP outputs)
  - [ ] All `__init__.py` files are empty or have package docstrings

  **QA Scenarios**:

  ```
  Scenario: Directory structure exists
    Tool: Bash
    Preconditions: Directories created
    Steps:
      1. ls -la src/ src/api/ src/api/routes/ models/ reports/
    Expected Result: All directories listed with __init__.py files
    Failure Indicators: "No such file or directory"
    Evidence: .sisyphus/evidence/task-03-directory-structure.txt

  Scenario: Package import works
    Tool: Bash
    Preconditions: __init__.py files created
    Steps:
      1. python -c "import sys; sys.path.insert(0, '.'); import src; print('Source package imported')"
    Expected Result: "Source package imported"
    Failure Indicators: "ModuleNotFoundError"
    Evidence: .sisyphus/evidence/task-03-package-import.txt
  ```

  **Evidence to Capture**:
  - [ ] Directory listing showing all structure
  - [ ] Package import verification

  **Commit**: YES
  - Message: `chore: create source directory structure`
  - Files: `src/__init__.py`, `src/api/__init__.py`, `src/api/routes/__init__.py`
  - Pre-commit: None

- [x] 4. Configure .dockerignore

  **What to do**:
  - Create `.dockerignore` in project root
  - Exclude from Docker build context: `__pycache__/`, `*.pyc`, `*.pyo`, `.pytest_cache/`, `.coverage`, `htmlcov/`, `.git/`, `.sisyphus/`, `*.ipynb`, `.ipynb_checkpoints/`, `tests/`, `*.md`, `ACTL3143 Project Part 2.pdf`
  - Include in build: `src/`, `models/`, `requirements.txt`, `Dockerfile`

  **Must NOT do**:
  - Do NOT exclude `requirements.txt` (needed in image)
  - Do NOT exclude `models/` (needed in image)
  - Do NOT use overly broad patterns that might exclude needed files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Configuration file for Docker build optimization
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 17 (Dockerfile build needs .dockerignore)
  - **Blocked By**: None (can start immediately)

  **References**:
  - Docker best practices for .dockerignore

  **Acceptance Criteria**:
  - [ ] `.dockerignore` file created in project root
  - [ ] Excludes notebooks, tests, git, cache files
  - [ ] Includes src/, models/, requirements.txt
  - [ ] Docker build context size reduced (verification in Task 17)

  **QA Scenarios**:

  ```
  Scenario: .dockerignore exists and has content
    Tool: Bash
    Preconditions: .dockerignore created
    Steps:
      1. cat .dockerignore
      2. grep -c "\.pyc\|__pycache__\|\.git" .dockerignore
    Expected Result: Count >= 3 (at least .pyc, __pycache__, .git excluded)
    Failure Indicators: File not found, count < 3
    Evidence: .sisyphus/evidence/task-04-dockerignore.txt

  Scenario: Docker build context size check
    Tool: Bash
    Preconditions: Docker installed, .dockerignore created
    Steps:
      1. docker build -f Dockerfile --no-cache --progress=plain . 2>&1 | grep "Sending build context"
      Note: This will be verified in Task 17, placeholder check here
    Expected Result: Build context < 100MB (notebooks excluded)
    Failure Indicators: Build context > 500MB (notebooks included)
    Evidence: Verified in Task 17
  ```

  **Evidence to Capture**:
  - [ ] .dockerignore content verification
  - [ ] Build context size (deferred to Task 17)

  **Commit**: YES
  - Message: `chore: add .dockerignore to exclude notebooks and tests`
  - Files: `.dockerignore`
  - Pre-commit: None

- [ ] 5. Regenerate model artifacts locally

  **What to do**:
  - Run preprocessing notebook `01_preprocess.ipynb` locally to generate processed data splits (X_train.pkl, X_val.pkl, X_test.pkl, y_train.pkl, y_val.pkl, y_test.pkl)
  - Save fitted preprocessor (ColumnTransformer) to `models/preprocessor.pkl` using joblib
  - Run baseline model notebook `02_baseline_model.ipynb` to train Random Forest and save to `models/final_baseline_rf_model.pkl`
  - Run deep learning notebook `04_train_deep_learning_models.ipynb` to train Feedforward NN and Residual NN, save to `models/final_first_dl_model.keras` and `models/final_second_dl_model.keras`
  - Verify all artifacts load correctly with shapes matching

  **Must NOT do**:
  - Do NOT modify existing notebooks 01-04 — only execute them
  - Do NOT retrain with different hyperparameters — use frozen weights
  - Do NOT skip validation steps — verify each artifact loads

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Time-consuming model training (approximately 30 minutes), requires monitoring long-running processes
  - **Skills**: []
    - No specialized skills needed — standard notebook execution
  - **Skills Evaluated but Omitted**:
    - `ai-engineer`: Not needed for execution, only for architecture changes

  **Parallelization**:
  - **Can Run In Parallel**: NO (sequential notebook execution required)
  - **Parallel Group**: Sequential (must complete before Wave 3)
  - **Blocks**: Tasks 9 (SHAP needs models), Tasks 10-12 (API needs models)
  - **Blocked By**: Task 1 (needs dependencies installed)

  **References**:
  - `01_preprocess.ipynb:cell 1-end` - Data preprocessing pipeline
  - `02_baseline_model.ipynb:cell 1-end` - Random Forest training
  - `04_train_deep_learning_models.ipynb:cell 1-end` - Deep learning training
  - `README.md:How to Run` - Execution order

  **Acceptance Criteria**:
  - [ ] `models/preprocessor.pkl` exists and loads successfully
  - [ ] `models/final_baseline_rf_model.pkl` exists and has correct shape
  - [ ] `models/final_first_dl_model.keras` exists and loads with keras.models.load_model
  - [ ] `models/final_second_dl_model.keras` exists and loads with keras.models.load_model
  - [ ] `models/X_test.pkl`, `models/y_test.pkl` exist for SHAP analysis
  - [ ] All model prediction shapes match (single output: probability of Multiple vehicle crash)

  **QA Scenarios**:

  ```
  Scenario: Preprocessor loads and transforms sample data
    Tool: Bash (python)
    Preconditions: Models regenerated, preprocessor.pkl created
    Steps:
      1. python -c "import joblib; p = joblib.load('models/preprocessor.pkl'); print(type(p).__name__)"
      2. python -c "import pandas as pd; import joblib; p = joblib.load('models/preprocessor.pkl'); sample = pd.DataFrame([{'State': 'NSW', 'Speed Limit': 60, ...}]); transformed = p.transform(sample); print(transformed.shape)"
    Expected Result: "ColumnTransformer" and transformed shape matches expected feature count
    Failure Indicators: "FileNotFoundError", "EOFError", shape mismatch
    Evidence: .sisyphus/evidence/task-05-preprocessor-load.txt

  Scenario: Random Forest model loads and predicts
    Tool: Bash (python)
    Preconditions: final_baseline_rf_model.pkl created
    Steps:
      1. python -c "import joblib; rf = joblib.load('models/final_baseline_rf_model.pkl'); print(type(rf).__name__)"
      2. python -c "import joblib; import numpy as np; rf = joblib.load('models/final_baseline_rf_model.pkl'); X = np.random.randn(1, 60); pred = rf.predict(X); print(pred.shape)"
    Expected Result: "RandomForestClassifier" and output shape (1,)
    Failure Indicators: "FileNotFoundError", shape mismatch
    Evidence: .sisyphus/evidence/task-05-rf-load.txt

  Scenario: TensorFlow models load and predict
    Tool: Bash (python)
    Preconditions: .keras models created
    Steps:
      1. python -c "import tensorflow as tf; model = tf.keras.models.load_model('models/final_first_dl_model.keras'); print(type(model).__name__)"
      2. python -c "import tensorflow as tf; import numpy as np; model = tf.keras.models.load_model('models/final_first_dl_model.keras'); X = np.random.randn(1, 60); pred = model.predict(X); print(pred.shape)"
    Expected Result: "Functional" (or model type) and output shape (1, 1)
    Failure Indicators: "OSError", "ValueError", shape mismatch
    Evidence: .sisyphus/evidence/task-05-dl-load.txt

  Scenario: All artifacts present in models/
    Tool: Bash
    Preconditions: Regeneration complete
    Steps:
      1. ls -la models/ | grep -c "pkl\|keras"
    Expected Result: Count >= 7 (preprocessor, X_test, y_test, rf, two keras models, plus train/val data)
    Failure Indicators: Count < 7
    Evidence: .sisyphus/evidence/task-05-models-directory.txt
  ```

  **Evidence to Capture**:
  - [ ] Preprocessor load output
  - [ ] Random Forest load and prediction output
  - [ ] TensorFlow models load and prediction output
  - [ ] models/ directory listing with all artifacts

  **Commit**: YES
  - Message: `feat: regenerate and save all model artifacts locally`
  - Files: `models/*.pkl`, `models/*.keras`
  - Pre-commit: Verify all files load successfully

- [x] 6. Create preprocessing module

  **What to do**:
  - Create `src/preprocessing.py` module
  - Define `PreprocessingConfig` class or dataclass with feature names: State, Speed Limit, National Road Type, Road User, Age, Gender, Bus Involvement, Articulated Truck Involvement, Heavy Rigid Truck Involvement, Dayweek, Time, Christmas Period, Easter Period
  - Implement `load_preprocessor(path: str) -> ColumnTransformer` function
  - Implement `transform_raw_input(input_dict: dict, preprocessor: ColumnTransformer) -> np.ndarray` function
  - Implement `validate_raw_input(input_dict: dict) -> bool` function to check all 12 features present
  - Add docstrings explaining the 12 raw features and transformation pipeline

  **Must NOT do**:
  - Do NOT create new preprocessing logic — use the saved fitted preprocessor from models/
  - Do NOT hardcode feature transformations — the ColumnTransformer handles it
  - Do NOT modify the preprocessor — it's frozen from notebook training

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Moderate complexity module with validation logic, needs careful implementation
  - **Skills**: []
    - Standard Python module, no specialized ML skills needed (just use loaded preprocessor)
  - **Skills Evaluated but Omitted**:
    - `ai-engineer`: Not needed — using existing preprocessor, not designing new

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 7, 8)
  - **Blocks**: Task 11 (prediction endpoint needs preprocessing)
  - **Blocked By**: Task 1 (needs dependencies), Task 3 (needs src/ directory)

  **References**:
  - `01_preprocess.ipynb:cell with ColumnTransformer` - Preprocessor definition
  - `05_analysis.ipynb:input features` - Feature list validation
  - Metis review: "Use joblib for sklearn ColumnTransformer"

  **Acceptance Criteria**:
  - [ ] `src/preprocessing.py` exists with PreprocessingConfig, load_preprocessor, transform_raw_input, validate_raw_input
  - [ ] All functions have comprehensive docstrings
  - [ ] `validate_raw_input` correctly validates presence of all 12 features
  - [ ] `transform_raw_input` uses loaded preprocessor to transform dict to array

  **QA Scenarios**:

  ```
  Scenario: Load preprocessor module
    Tool: Bash (python)
    Preconditions: src/preprocessing.py created
    Steps:
      1. python -c "from src.preprocessing import load_preprocessor; p = load_preprocessor('models/preprocessor.pkl'); print(type(p).__name__)"
    Expected Result: "ColumnTransformer"
    Failure Indicators: "ImportError", "FileNotFoundError"
    Evidence: .sisyphus/evidence/task-06-load-preprocessor.txt

  Scenario: Validate raw input
    Tool: Bash (python)
    Preconditions: validate_raw_input implemented
    Steps:
      1. python -c "from src.preprocessing import validate_raw_input; valid = {'State': 'NSW', 'Speed Limit': 60, 'National Road Type': 'Arterial Road', 'Road User': 'Driver', 'Age': 30, 'Gender': 'Male', 'Bus Involvement': 'No', 'Articulated Truck Involvement': 'No', 'Heavy Rigid Truck Involvement': 'No', 'Dayweek': 'Friday', 'Time': 18, 'Christmas Period': 'No', 'Easter Period': 'No'}; print(validate_raw_input(valid))"
      2. python -c "from src.preprocessing import validate_raw_input; invalid = {'State': 'NSW'}; print(validate_raw_input(invalid))"
    Expected Result: True for valid input, False for invalid input
    Failure Indicators: Exception raised, incorrect boolean
    Evidence: .sisyphus/evidence/task-06-validate-input.txt

  Scenario: Transform raw input to array
    Tool: Bash (python)
    Preconditions: transform_raw_input implemented, preprocessor exists
    Steps:
      1. python -c "from src.preprocessing import load_preprocessor, transform_raw_input; import joblib; p = joblib.load('models/preprocessor.pkl'); valid = {'State': 'NSW', 'Speed Limit': 60, 'National Road Type': 'Arterial Road', 'Road User': 'Driver', 'Age': 30, 'Gender': 'Male', 'Bus Involvement': 'No', 'Articulated Truck Involvement': 'No', 'Heavy Rigid Truck Involvement': 'No', 'Dayweek': 'Friday', 'Time': 18, 'Christmas Period': 'No', 'Easter Period': 'No'}; arr = transform_raw_input(valid, p); print(arr.shape)"
    Expected Result: Shape (1, N) where N is transformed feature count
    Failure Indicators: "ValueError", shape mismatch, transformation error
    Evidence: .sisyphus/evidence/task-06-transform-input.txt
  ```

  **Evidence to Capture**:
  - [ ] Preprocessor load output
  - [ ] Validation function correct responses
  - [ ] Transformation output shape

  **Commit**: YES
  - Message: `feat: add preprocessing module with validation and transformation`
  - Files: `src/preprocessing.py`
  - Pre-commit: `python -m py_compile src/preprocessing.py`

- [x] 7. Create model loading utilities

  **What to do**:
  - Create `src/model_loader.py` module
  - Implement `load_random_forest(path: str) -> RandomForestClassifier` using joblib
  - Implement `load_keras_model(path: str) -> tf.keras.Model` using tf.keras.models.load_model
  - Implement `ModelRegistry` class to cache loaded models in memory (load once at startup, reuse for predictions)
  - Implement `get_model(model_name: str, registry: ModelRegistry) -> Union[RandomForestClassifier, tf.keras.Model]` to retrieve cached model
  - Add model validation: check input/output shapes match expectations

  **Must NOT do**:
  - Do NOT load models on every prediction request — must load once and cache
  - Do NOT retrain models — use frozen weights
  - Do NOT add model versioning or rollback (out of scope)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Moderate complexity with caching logic, model loading strategies
  - **Skills**: []
    - Standard Python module with ML model loading patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 8)
  - **Blocks**: Task 10 (FastAPI startup), Tasks 11-12 (prediction endpoints)
  - **Blocked By**: Task 1 (needs dependencies), Task 3 (needs src/ directory)

  **References**:
  - `02_baseline_model.ipynb:model saving` - Random Forest loading pattern
  - `04_train_deep_learning_models.ipynb:model saving` - Keras loading pattern
  - Metis review: "Validate model loading in FastAPI startup event, not per-request"

  **Acceptance Criteria**:
  - [ ] `src/model_loader.py` exists with load_random_forest, load_keras_model, ModelRegistry, get_model
  - [ ] ModelRegistry implements caching (dictionary-based for simplicity)
  - [ ] All loading functions validate model input/output shapes
  - [ ] All functions have comprehensive docstrings

  **QA Scenarios**:

  ```
  Scenario: Load Random Forest model
    Tool: Bash (python)
    Preconditions: model_loader.py created, models exist
    Steps:
      1. python -c "from src.model_loader import load_random_forest; rf = load_random_forest('models/final_baseline_rf_model.pkl'); print(type(rf).__name__)"
    Expected Result: "RandomForestClassifier"
    Failure Indicators: "ImportError", "FileNotFoundError", wrong type
    Evidence: .sisyphus/evidence/task-07-load-rf.txt

  Scenario: Load Keras model
    Tool: Bash (python)
    Preconditions: model_loader.py created, models exist
    Steps:
      1. python -c "import tensorflow as tf; from src.model_loader import load_keras_model; model = load_keras_model('models/final_first_dl_model.keras'); print(type(model).__name__); print(model.input_shape); print(model.output_shape)"
    Expected Result: Model type and shapes printed, output_shape is (None, 1)
    Failure Indicators: "OSError", "InvalidGraphDef", wrong shapes
    Evidence: .sisyphus/evidence/task-07-load-keras.txt

  Scenario: ModelRegistry caching
    Tool: Bash (python)
    Preconditions: ModelRegistry implemented
    Steps:
      1. python -c "from src.model_loader import ModelRegistry, load_random_forest; registry = ModelRegistry(); rf1 = registry.get('random_forest', 'models/final_baseline_rf_model.pkl'); rf2 = registry.get('random_forest', 'models/final_baseline_rf_model.pkl'); print(rf1 is rf2)"
    Expected Result: "True" (same object reference, cached)
    Failure Indicators: "False" (models loaded multiple times)
    Evidence: .sisyphus/evidence/task-07-caching.txt
  ```

  **Evidence to Capture**:
  - [ ] Random Forest load output
  - [ ] Keras model load output with shapes
  - [ ] Caching verification (object identity)

  **Commit**: YES
  - Message: `feat: add model loading utilities with caching`
  - Files: `src/model_loader.py`
  - Pre-commit: `python -m py_compile src/model_loader.py`

- [x] 8. Create configuration and logging modules

  **What to do**:
  - Create `src/config.py` module with configuration management:
    - Define `Settings` class with model paths, API settings, logging settings
    - Use environment variables with defaults: MODEL_DIR (default: "./models"), API_HOST (default: "0.0.0.0"), API_PORT (default: 8000), LOG_LEVEL (default: "INFO")
    - Provide method `get_model_path(model_name: str) -> Path` for consistent path resolution
  - Create `src/logger.py` module with logging setup:
    - Configure structured logging with timestamp, level, module, message
    - Provide `get_logger(name: str) -> logging.Logger` function
    - Log to console with configurable level from Settings

  **Must NOT do**:
  - Do NOT add database configuration (out of scope)
  - Do NOT add authentication configuration (out of scope)
  - Do NOT use external config files (YAML, JSON) — keep it simple with Python + env vars

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard configuration and logging setup, straightforward
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: Task 10 (FastAPI needs config)
  - **Blocked By**: Task 3 (needs src/ directory)

  **References**:
  - FastAPI configuration best practices
  - Python logging best practices

  **Acceptance Criteria**:
  - [ ] `src/config.py` exists with Settings class and environment variable handling
  - [ ] `src/logger.py` exists with structured logging setup
  - [ ] Settings provides default values and model path resolution
  - [ ] Logger outputs to console with configurable level

  **QA Scenarios**:

  ```
  Scenario: Configuration loads with defaults
    Tool: Bash (python)
    Preconditions: config.py created
    Steps:
      1. python -c "from src.config import Settings; s = Settings(); print(f'MODEL_DIR={s.MODEL_DIR}'); print(f'API_HOST={s.API_HOST}'); print(f'API_PORT={s.API_PORT}')"
    Expected Result: "MODEL_DIR=models", "API_HOST=0.0.0.0", "API_PORT=8000"
    Failure Indicators: "AttributeError", different defaults
    Evidence: .sisyphus/evidence/task-08-config-defaults.txt

  Scenario: Logger creates structured logs
    Tool: Bash (python)
    Preconditions: logger.py created
    Steps:
      1. python -c "from src.logger import get_logger; logger = get_logger('test'); logger.info('Test message')"
    Expected Result: Log line with timestamp, level, module, message format
    Failure Indicators: Unstructured log output, missing fields
    Evidence: .sisyphus/evidence/task-08-logger-output.txt

  Scenario: get_model_path resolves correctly
    Tool: Bash (python)
    Preconditions: config.py created
    Steps:
      1. python -c "from src.config import Settings; s = Settings(); print(s.get_model_path('final_baseline_rf_model.pkl'))"
    Expected Result: Path ending with "models/final_baseline_rf_model.pkl"
    Failure Indicators: Wrong path, exception
    Evidence: .sisyphus/evidence/task-08-model-path.txt
  ```

  **Evidence to Capture**:
  - [ ] Configuration defaults output
  - [ ] Logger structured output
  - [ ] Model path resolution output

  **Commit**: YES
  - Message: `feat: add configuration and logging modules`
  - Files: `src/config.py`, `src/logger.py`
  - Pre-commit: `python -m py_compile src/config.py src/logger.py`

- [ ] 9. Implement SHAP analysis in 05_analysis.ipynb

  **What to do**:
  - Add new cells to `05_analysis.ipynb` after existing analysis cells
  - Import shap library: `import shap`
  - For Random Forest: Use `shap.TreeExplainer` for fast computation on X_test sample (1000 rows for performance)
  - For Feedforward NN and Residual NN: Use `shap.KernelExplainer` on background data subset (100 rows)
  - Generate summary plots for all three models: `shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names)`
  - Generate waterfall plots for individual predictions: `shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test.iloc[0], feature_names=feature_names))`
  - Save static images to `reports/shap_rf_summary.png`, `reports/shap_ffnn_summary.png`, `reports/shap_residualnn_summary.png`
  - Save interactive HTML explanations to `reports/shap_interactive/` directory using `shap.save_html()`
  - Add markdown cells explaining SHAP values for actuarial interpretation

  **Must NOT do**:
  - Do NOT delete or modify existing cells in 05_analysis.ipynb — only add new cells
  - Do NOT run SHAP on entire test set (11,500+ rows) — use sampling for performance
  - Do NOT retrain models — use frozen weights from Task 5

  **Recommended Agent Profile**:
  - **Category**: `ai-engineer`
    - Reason: Deep learning explainability requires understanding of neural networks and SHAP library
  - **Skills**: [`ai-engineer`]
    - `ai-engineer`: SHAP implementation, explainability patterns, neural network interpretation
  - **Skills Evaluated but Omitted**:
    - `visual-engineering`: Notebook visualizations are adequate, no frontend needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10, 11, 12)
  - **Blocks**: None (SHAP is independent of API)
  - **Blocked By**: Task 5 (needs models and test data)

  **References**:
  - `05_analysis.ipynb:existing cells` - Model loading and prediction patterns
  - SHAP documentation: https://shap.readthedocs.io/en/latest/
  - Metis review: "TreeExplainer for RF, KernelExplainer for NNs, subsample for performance"

  **Acceptance Criteria**:
  - [ ] New SHAP cells added to `05_analysis.ipynb` after existing analysis
  - [ ] Random Forest SHAP analysis uses TreeExplainer
  - [ ] Feedforward NN and Residual NN use KernelExplainer
  - [ ] Static summary plots saved to `reports/shap_*.png` (3 files)
  - [ ] Interactive HTML explanations saved to `reports/shap_interactive/`
  - [ ] Markdown cells explain SHAP for actuarial transparency
  - [ ] Notebook executes without errors from top to bottom

  **QA Scenarios**:

  ```
  Scenario: SHAP library imports and runs
    Tool: Bash (jupyter)
    Preconditions: shap installed, models exist
    Steps:
      1. python -c "import shap; print(shap.__version__)"
      2. python -c "import shap; print(shap.TreeExplainer.__doc__[:50])"
    Expected Result: SHAP version printed, TreeExplainer documentation accessible
    Failure Indicators: "ModuleNotFoundError", "ImportError"
    Evidence: .sisyphus/evidence/task-09-shap-import.txt

  Scenario: Notebook executes successfully with SHAP cells
    Tool: Bash (jupyter)
    Preconditions: 05_analysis.ipynb updated with SHAP cells
    Steps:
      1. jupyter execute 05_analysis.ipynb --stdout | tail -20
      2. echo "Notebook execution exit code: $?"
    Expected Result: Exit code 0, no error traces in output
    Failure Indicators: "Exception", "Error", exit code != 0
    Evidence: .sisyphus/evidence/task-09-notebook-execute.txt

  Scenario: Static SHAP plots exist
    Tool: Bash
    Preconditions: Notebook executed
    Steps:
      1. ls -la reports/shap_*.png
      2. file reports/shap_*.png
    Expected Result: 3 PNG files (rf, ffnn, residualnn), all identified as images
    Failure Indicators: Files missing, wrong format
    Evidence: .sisyphus/evidence/task-09-shap-plots.txt

  Scenario: Interactive HTML exists
    Tool: Bash
    Preconditions: Notebook executed
    Steps:
      1. ls -la reports/shap_interactive/
      2. grep -c "shap" reports/shap_interactive/*.html || find reports/shap_interactive -name "*.html" | wc -l
    Expected Result: At least 1 HTML file in shap_interactive directory
    Failure Indicators: Directory missing, no HTML files
    Evidence: .sisyphus/evidence/task-09-shap-html.txt
  ```

  **Evidence to Capture**:
  - [ ] SHAP import verification
  - [ ] Notebook execution output (last 20 lines)
  - [ ] Static plot file listing
  - [ ] Interactive HTML file listing

  **Commit**: YES
  - Message: `feat: add SHAP explainability analysis to notebook`
  - Files: `05_analysis.ipynb`, `reports/shap_*.png`, `reports/shap_interactive/*.html`
  - Pre-commit: `jupyter execute 05_analysis.ipynb --stdout`

- [x] 10. Create FastAPI application and schemas

  **What to do**:
  - Create `src/api/main.py` with FastAPI app initialization:
    - Initialize FastAPI app with title, description, version
    - Add CORS middleware for localhost access
    - Define `lifespan` context manager to load models at startup (use ModelRegistry)
    - Include routers from routes (will be added in Tasks 11-12)
    - Add root endpoint `/` with API information
  - Create `src/api/schemas.py` with Pydantic models:
    - `RawInputFeatures` class: 12 features with validation (State, Speed Limit, Age, Gender, etc.)
    - `PreprocessedInput` class: features as List[float]
    - `PredictionResponse` class: prediction (int), probability (float), model_name (str)
    - `HealthResponse` class: status (str), models_loaded (bool), model_names (List[str])
    - `ErrorResponse` class: error (str), detail (str)
  - Add proper type hints and validators to Pydantic models (use field validators for categorical fields)

  **Must NOT do**:
  - Do NOT add database connections (out of scope)
  - Do NOT add authentication middleware (out of scope)
  - Do NOT load models on every request — use startup event
  - Do NOT create prediction endpoints yet (done in Tasks 11-12)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: FastAPI boilerplate setup, straightforward configuration
  - **Skills**: []
    - Standard FastAPI patterns, no specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 11, 12)
  - **Blocks**: Task 16 (health endpoint tests)
  - **Blocked By**: Task 3 (needs src/api structure), Task 7 (needs ModelRegistry), Task 8 (needs config)

  **References**:
  - FastAPI documentation: https://fastapi.tiangolo.com/
  - Pydantic validation: https://docs.pydantic.dev/latest/
  - `06_analysis.ipynb:feature list` - Feature names for schema

  **Acceptance Criteria**:
  - [ ] `src/api/main.py` exists with FastAPI app, lifespan context manager, model loading
  - [ ] `src/api/schemas.py` exists with all Pydantic models
  - [ ] Models loaded at startup, not on every request
  - [ ] Health check endpoint `/health` responds with model status
  - [ ] Root endpoint `/` provides API information

  **QA Scenarios**:

  ```
  Scenario: FastAPI app imports and initializes
    Tool: Bash (python)
    Preconditions: main.py created
    Steps:
      1. python -c "from src.api.main import app; print(app.title)"
      2. python -c "from src.api.main import app; print(type(app).__name__)"
    Expected Result: App title printed, "FastAPI" type
    Failure Indicators: "ImportError", "AttributeError"
    Evidence: .sisyphus/evidence/task-10-app-import.txt

  Scenario: Pydantic schemas validate input
    Tool: Bash (python)
    Preconditions: schemas.py created
    Steps:
      1. python -c "from src.api.schemas import RawInputFeatures; data = {'State': 'NSW', 'Speed Limit': 60, 'National Road Type': 'Arterial Road', 'Road User': 'Driver', 'Age': 30, 'Gender': 'Male', 'Bus Involvement': 'No', 'Articulated Truck Involvement': 'No', 'Heavy Rigid Truck Involvement': 'No', 'Dayweek': 'Friday', 'Time': 18, 'Christmas Period': 'No', 'Easter Period': 'No'}; features = RawInputFeatures(**data); print(features.State)"
    Expected Result: "NSW" (validated input)
    Failure Indicators: "ValidationError", different output
    Evidence: .sisyphus/evidence/task-10-schema-validation.txt

  Scenario: Health endpoint responds
    Tool: Bash (uvicorn)
    Preconditions: App running (uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &)
    Steps:
      1. sleep 5 && curl http://localhost:8000/health
      2. curl http://localhost:8000/health | python -c "import sys, json; data = json.load(sys.stdin); print(data['status'])"
    Expected Result: JSON with status "healthy" and models_loaded true
    Failure Indicators: Connection refused, missing fields, models_loaded false
    Evidence: .sisyphus/evidence/task-10-health-endpoint.txt

  Scenario: Root endpoint provides API info
    Tool: Bash (curl)
    Preconditions: App running
    Steps:
      1. curl http://localhost:8000/
      2. curl http://localhost:8000/ | python -c "import sys, json; data = json.load(sys.stdin); print(data.get('title', 'No title'))"
    Expected Result: JSON with title, description, version
    Failure Indicators: Empty response, missing fields
    Evidence: .sisyphus/evidence/task-10-root-endpoint.txt
  ```

  **Evidence to Capture**:
  - [ ] App import output
  - [ ] Schema validation output
  - [ ] Health endpoint response
  - [ ] Root endpoint response

  **Commit**: YES
  - Message: `feat: create FastAPI app with Pydantic schemas`
  - Files: `src/api/main.py`, `src/api/schemas.py`
  - Pre-commit: `python -m py_compile src/api/main.py src/api/schemas.py`

- [x] 11. Create raw prediction endpoint

  **What to do**:
  - Create `src/api/routes/predict.py` module
  - Import necessary modules: `from fastapi import APIRouter, HTTPException`, `from src.api.schemas import RawInputFeatures, PredictionResponse`, `from src.preprocessing import load_preprocessor, transform_raw_input`, `from src.model_loader import ModelRegistry`
  - Create `APIRouter()` instance with prefix="/predict"
  - Implement `@router.post("/raw", response_model=PredictionResponse)` endpoint:
    - Accept `RawInputFeatures` (12 human-readable features)
    - Get preprocessor from app.state (loaded at startup)
    - Get models from ModelRegistry
    - Call `transform_raw_input()` to preprocess the raw input
    - Get prediction from all three models (Random Forest, Feedforward NN, Residual NN)
    - Return prediction, probability, and model name for each
  - Add error handling: 400 for invalid input, 500 for model/prediction errors
  - Include endpoint in main app in `src/api/main.py`: `app.include_router(predict_router, prefix="/api/v1")`

  **Must NOT do**:
  - Do NOT load preprocessor/model on every request — use cached instances
  - Do NOT create `/predict/preprocessed` endpoint yet (done in Task 12)
  - Do NOT return model internals (weights, architecture) — only prediction and probability
  - Do NOT add authentication/authorization (out of scope)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Moderate complexity endpoint with preprocessing pipeline integration
  - **Skills**: []
    - Standard FastAPI routing, preprocessing integration

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 12) — Same file as Task 12, coordinate carefully
  - **Blocks**: None (independent)
  - **Blocked By**: Task 6 (needs preprocessing), Task 7 (needs model_loader), Task 10 (needs schemas)

  **References**:
  - `src/preprocessing.py` - transform_raw_input function
  - `src/model_loader.py` - ModelRegistry and get_model
  - `src/api/schemas.py` - RawInputFeatures, PredictionResponse
  - `05_analysis.ipynb:model prediction patterns` - How models are called

  **Acceptance Criteria**:
  - [ ] `src/api/routes/predict.py` exists with `/raw` POST endpoint
  - [ ] Endpoint accepts human-readable input (State, Speed Limit, Age, etc.)
  - [ ] Preprocessor loaded at startup, retrieved from app.state
  - [ ] Models loaded from ModelRegistry, not on every request
  - [ ] Returns prediction, probability, and model name for all three models
  - [ ] Error handling returns 400 for invalid input, 500 for prediction errors
  - [ ] Endpoint registered in main app at `/api/v1/predict/raw`

  **QA Scenarios**:

  ```
  Scenario: Raw prediction endpoint responds to valid input
    Tool: Bash (curl)
    Preconditions: App running, models loaded
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/raw -H "Content-Type: application/json" -d '{"State": "NSW", "Speed Limit": 60, "National Road Type": "Arterial Road", "Road User": "Driver", "Age": 30, "Gender": "Male", "Bus Involvement": "No", "Articulated Truck Involvement": "No", "Heavy Rigid Truck Involvement": "No", "Dayweek": "Friday", "Time": 18, "Christmas Period": "No", "Easter Period": "No"}'
    Expected Result: JSON with prediction (0 or 1), probability (float between 0 and 1), model_name (string)
    Failure Indicators: 404, 500, missing fields, invalid JSON
    Evidence: .sisyphus/evidence/task-11-raw-valid.txt

  Scenario: Raw endpoint rejects invalid categorical input
    Tool: Bash (curl)
    Preconditions: App running
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/raw -H "Content-Type: application/json" -d '{"State": "INVALID_STATE", "Speed Limit": 60, ...}'
    Expected Result: HTTP 400 Bad Request with validation error
    Failure Indicators: 200 OK with invalid prediction, 500 internal error
    Evidence: .sisyphus/evidence/task-11-raw-invalid.txt

  Scenario: Raw endpoint handles missing fields
    Tool: Bash (curl)
    Preconditions: App running
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/raw -H "Content-Type: application/json" -d '{"State": "NSW"}'
    Expected Result: HTTP 400 Bad Request with field validation error
    Failure Indicators: 200 OK with defaults, 500 internal error
    Evidence: .sisyphus/evidence/task-11-raw-missing.txt

  Scenario: All three models return predictions
    Tool: Bash (python + curl)
    Preconditions: App running
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/raw ... (valid input)
      2. python -c "import json, sys; data = json.load(sys.stdin); models = ['Random Forest', 'Feedforward NN', 'Residual NN']; print('All models present:', all(m in data['model_name'] for m in models))"
    Expected Result: True (all three models present in response)
    Failure Indicators: False, missing models
    Evidence: .sisyphus/evidence/task-11-all-models.txt
  ```

  **Evidence to Capture**:
  - [ ] Valid prediction response
  - [ ] Invalid categorical input response (400)
  - [ ] Missing fields response (400)
  - [ ] All models present verification

  **Commit**: YES
  - Message: `feat: add raw prediction endpoint with human-readable input`
  - Files: `src/api/routes/predict.py`, `src/api/main.py` (updated)
  - Pre-commit: `python -m py_compile src/api/routes/predict.py`

- [x] 12. Add preprocessed prediction endpoint

  **What to do**:
  - Add to existing `src/api/routes/predict.py` file (same file as Task 11)
  - Implement `@router.post("/preprocessed", response_model=PredictionResponse)` endpoint:
    - Accept `PreprocessedInput` (features as List[float])
    - Get models from ModelRegistry
    - Skip preprocessing (input already preprocessed)
    - Get prediction from all three models
    - Return prediction, probability, and model name
  - Add input validation: check feature array length matches expected shape
  - Add error handling: 400 for invalid feature count, 500 for model errors

  **Must NOT do**:
  - Do NOT preprocess the input — assume it's already preprocessed
  - Do NOT create a new router file — add to existing predict.py
  - Do NOT return different response format from `/raw` endpoint — keep consistent
  - Do NOT add batch prediction (out of scope)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Moderate complexity endpoint, continuation of Task 11
  - **Skills**: []
    - Standard FastAPI routing, same patterns as Task 11

  **Parallelization**:
  - **Can Run In Parallel**: YES (but coordinate with Task 11 — same file)
  - **Parallel Group**: Wave 3 (with Tasks 9, 10, 11)
  - **Blocks**: Task 15 (integration tests need both endpoints)
  - **Blocked By**: Task 7 (needs model_loader), Task 10 (needs schemas)

  **References**:
  - `src/api/routes/predict.py` - Existing `/raw` endpoint (add `/preprocessed` to same file)
  - `src/api/schemas.py` - PreprocessedInput, PredictionResponse
  - `src/model_loader.py` - ModelRegistry and get_model

  **Acceptance Criteria**:
  - [ ] `/preprocessed` POST endpoint added to existing predict.py
  - [ ] Endpoint accepts List[float] features (preprocessed array)
  - [ ] Validates feature array length matches expected shape
  - [ ] Returns prediction, probability, and model name for all three models
  - [ ] Error handling returns 400 for invalid feature count, 500 for model errors
  - [ ] Endpoint registered in main app at `/api/v1/predict/preprocessed`
  - [ ] Response format identical to `/raw` endpoint

  **QA Scenarios**:

  ```
  Scenario: Preprocessed endpoint responds to valid input
    Tool: Bash (curl)
    Preconditions: App running, models loaded
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/preprocessed -H "Content-Type: application/json" -d '{"features": [0.5, -1.2, ...]}'
      Note: Need actual preprocessed feature array from test data
    Expected Result: JSON with prediction (0 or 1), probability (float), model_name
    Failure Indicators: 404, 500, missing fields
    Evidence: .sisyphus/evidence/task-12-preprocessed-valid.txt

  Scenario: Preprocessed endpoint rejects wrong feature count
    Tool: Bash (curl)
    Preconditions: App running
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/preprocessed -H "Content-Type: application/json" -d '{"features": [0.5, -1.2]}'
    Expected Result: HTTP 400 Bad Request with feature count error
    Failure Indicators: 200 OK, 500 internal error
    Evidence: .sisyphus/evidence/task-12-preprocessed-invalid.txt

  Scenario: Both endpoints return identical format
    Tool: Bash (python + curl)
    Preconditions: App running, /raw and /preprocessed working
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/raw ... > raw_response.json
      2. curl -X POST http://localhost:8000/api/v1/predict/preprocessed ... > preprocessed_response.json
      3. python -c "import json; r1 = json.load(open('raw_response.json')); r2 = json.load(open('preprocessed_response.json')); print('Keys match:', set(r1.keys()) == set(r2.keys()))"
    Expected Result: True (same response keys)
    Failure Indicators: False, different keys
    Evidence: .sisyphus/evidence/task-12-format-match.txt

  Scenario: All three models return predictions from preprocessed input
    Tool: Bash (python + curl)
    Preconditions: App running
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/preprocessed ... | python -c "import json, sys; data = json.load(sys.stdin); print('Models:', data.get('predictions', []))"
    Expected Result: Three predictions (one per model) with probabilities
    Failure Indicators: Missing models, wrong count
    Evidence: .sisyphus/evidence/task-12-all-models-preprocessed.txt
  ```

  **Evidence to Capture**:
  - [ ] Valid preprocessed prediction response
  - [ ] Invalid feature count response (400)
  - [ ] Format match verification
  - [ ] All models present verification

  **Commit**: YES (coordinate with Task 11 if working in parallel)
  - Message: `feat: add preprocessed prediction endpoint`
  - Files: `src/api/routes/predict.py` (additional endpoint)
  - Pre-commit: `python -m py_compile src/api/routes/predict.py`

- [ ] 13. Write unit tests for preprocessing

  **What to do** (TDD: RED → GREEN → REFACTOR):
  - **RED**: Create `tests/test_preprocessing.py` and write failing tests first:
    - Test `load_preprocessor()` returns ColumnTransformer
    - Test `validate_raw_input()` returns True for valid input with all 12 features
    - Test `validate_raw_input()` returns False for invalid input missing fields
    - Test `transform_raw_input()` returns correct shape numpy array
    - Test `transform_raw_input()` raises error for unknown categorical values
  - **GREEN**: Run `pytest tests/test_preprocessing.py -v` to see failures, then verify preprocessing module passes all tests
  - **REFACTOR**: Optimize imports, add edge case tests (None values, empty dict)

  **Must NOT do**:
  - Do NOT write integration tests here — focus on unit tests only
  - Do NOT test model prediction (separate task)
  - Do NOT test FastAPI endpoints (separate task)
  - Do NOT skip the RED phase — write failing tests first

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard unit test patterns, straightforward pytest usage
  - **Skills**: []
    - No specialized skills needed for unit testing

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 14, 15, 16)
  - **Blocks**: None
  - **Blocked By**: Task 6 (needs preprocessing module)

  **References**:
  - `src/preprocessing.py` - Functions to test
  - `tests/conftest.py` - Test fixtures
  - `01_preprocess.ipynb` - Preprocessing pipeline reference

  **Acceptance Criteria**:
  - [ ] `tests/test_preprocessing.py` exists with at least 5 test functions
  - [ ] All tests follow TDD: written first, failed, then passed
  - [ ] `pytest tests/test_preprocessing.py -v` passes with 100% success
  - [ ] Tests cover: valid input, invalid input, missing fields, unknown values, shape validation

  **QA Scenarios**:

  ```
  Scenario: All preprocessing tests pass
    Tool: Bash (pytest)
    Preconditions: tests/test_preprocessing.py created, preprocessing module exists
    Steps:
      1. pytest tests/test_preprocessing.py -v --tb=short
      2. echo "Exit code: $?"
    Expected Result: All tests pass, exit code 0
    Failure Indicators: FAILED tests, exit code != 0
    Evidence: .sisyphus/evidence/task-13-preprocessing-tests.txt

  Scenario: Test coverage for preprocessing module
    Tool: Bash (pytest-cov)
    Preconditions: pytest-cov installed
    Steps:
      1. pytest tests/test_preprocessing.py --cov=src.preprocessing --cov-report=term-missing
    Expected Result: Coverage > 80% for preprocessing module
    Failure Indicators: Coverage < 80%, missing lines reported
    Evidence: .sisyphus/evidence/task-13-coverage.txt

  Scenario: TDD verification - tests written first
    Tool: Bash (git, manual)
    Preconditions: Git history available
    Steps:
      1. git log --oneline --all -- tests/test_preprocessing.py src/preprocessing.py | head -5
      2. Verify test file commit exists
    Expected Result: Test file created, preprocessing module tested
    Failure Indicators: No test file commits, only implementation changes
    Evidence: .sisyphus/evidence/task-13-tdd-verification.txt
  ```

  **Evidence to Capture**:
  - [ ] Pytest output showing all tests pass
  - [ ] Coverage report showing > 80% coverage
  - [ ] Git log showing TDD workflow

  **Commit**: YES
  - Message: `test: add unit tests for preprocessing module`
  - Files: `tests/test_preprocessing.py`
  - Pre-commit: `pytest tests/test_preprocessing.py -v`

- [ ] 14. Write unit tests for model loading

  **What to do** (TDD: RED → GREEN → REFACTOR):
  - **RED**: Create `tests/test_model_loader.py` and write failing tests first:
    - Test `load_random_forest()` returns RandomForestClassifier
    - Test `load_keras_model()` returns tf.keras.Model with correct input/output shapes
    - Test `ModelRegistry.get()` caches model (same object reference)
    - Test `ModelRegistry.get()` returns correct model type
    - Test model prediction shape validation
  - **GREEN**: Run `pytest tests/test_model_loader.py -v` to see failures, then verify model_loader passes all tests
  - **REFACTOR**: Add error case tests (invalid paths, corrupted models)

  **Must NOT do**:
  - Do NOT test actual prediction accuracy (focus on loading and caching)
  - Do NOT test integration with FastAPI (separate task)
  - Do NOT mock models — use actual model files from Task 5

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard unit tests for model loading utilities
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 13, 15, 16)
  - **Blocks**: None
  - **Blocked By**: Task 7 (needs model_loader module)

  **References**:
  - `src/model_loader.py` - Functions to test
  - `tests/conftest.py` - Test fixtures for model paths
  - `02_baseline_model.ipynb` - Random Forest reference
  - `04_train_deep_learning_models.ipynb` - Keras model reference

  **Acceptance Criteria**:
  - [ ] `tests/test_model_loader.py` exists with at least 5 test functions
  - [ ] All tests follow TDD: written first, failed, then passed
  - [ ] `pytest tests/test_model_loader.py -v` passes with 100% success
  - [ ] Tests cover: RF loading, Keras loading, caching, shape validation, error cases

  **QA Scenarios**:

  ```
  Scenario: All model loader tests pass
    Tool: Bash (pytest)
    Preconditions: tests/test_model_loader.py created, model_loader module exists
    Steps:
      1. pytest tests/test_model_loader.py -v --tb=short
      2. echo "Exit code: $?"
    Expected Result: All tests pass, exit code 0
    Failure Indicators: FAILED tests, exit code != 0
    Evidence: .sisyphus/evidence/task-14-model-loader-tests.txt

  Scenario: Caching verification
    Tool: Bash (pytest)
    Preconditions: Caching test in test_model_loader.py
    Steps:
      1. pytest tests/test_model_loader.py::test_model_caching -v
    Expected Result: Test passes, same object reference verified
    Failure Indicators: Test fails, different object references
    Evidence: .sisyphus/evidence/task-14-caching-test.txt

  Scenario: Model shape validation
    Tool: Bash (pytest)
    Preconditions: Shape validation test exists
    Steps:
      1. pytest tests/test_model_loader.py -k "shape" -v
    Expected Result: Tests pass, input/output shapes validated
    Failure Indicators: Shape mismatch errors
    Evidence: .sisyphus/evidence/task-14-shape-tests.txt
  ```

  **Evidence to Capture**:
  - [ ] Pytest output showing all tests pass
  - [ ] Caching test output
  - [ ] Shape validation test output

  **Commit**: YES
  - Message: `test: add unit tests for model loading utilities`
  - Files: `tests/test_model_loader.py`
  - Pre-commit: `pytest tests/test_model_loader.py -v`

- [ ] 15. Write integration tests for API workflow

  **What to do** (TDD: RED → GREEN → REFACTOR):
  - **RED**: Create `tests/test_api_integration.py` and write failing tests first:
    - Test POST `/api/v1/predict/raw` with valid input returns 200 and prediction
    - Test POST `/api/v1/predict/raw` with invalid input returns 400
    - Test POST `/api/v1/predict/preprocessed` with valid input returns 200
    - Test POST `/api/v1/predict/preprocessed` with wrong feature count returns 400
    - Test full workflow: raw input → preprocessing → prediction → response
    - Test all three models return predictions (Random Forest, Feedforward NN, Residual NN)
  - **GREEN**: Run `pytest tests/test_api_integration.py -v` to see failures, then verify API passes all tests
  - **REFACTOR**: Add edge case tests (large payloads, concurrent requests)

  **Must NOT do**:
  - Do NOT use mock requests — use httpx AsyncClient for real integration tests
  - Do NOT test unit functionality (preprocessing, model loading) — focus on integration
  - Do NOT skip startup/shutdown lifecycle — use test fixtures to simulate

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration tests require understanding of full API workflow
  - **Skills**: []
    - Standard pytest with httpx AsyncClient

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 13, 14, 16)
  - **Blocks**: None
  - **Blocked By**: Task 11 (needs /raw endpoint), Task 12 (needs /preprocessed endpoint)

  **References**:
  - `src/api/main.py` - FastAPI app
  - `src/api/routes/predict.py` - Prediction endpoints
  - `tests/conftest.py` - Test client fixture
  - FastAPI testing docs: https://fastapi.tiangolo.com/tutorial/testing/

  **Acceptance Criteria**:
  - [ ] `tests/test_api_integration.py` exists with at least 6 test functions
  - [ ] All tests follow TDD: written first, failed, then passed
  - [ ] `pytest tests/test_api_integration.py -v` passes with 100% success
  - [ ] Tests use httpx AsyncClient for real HTTP requests
  - [ ] Tests cover: both endpoints, valid/invalid inputs, full workflow, all models

  **QA Scenarios**:

  ```
  Scenario: All integration tests pass
    Tool: Bash (pytest)
    Preconditions: tests/test_api_integration.py created, API running
    Steps:
      1. pytest tests/test_api_integration.py -v --tb=short
      2. echo "Exit code: $?"
    Expected Result: All tests pass, exit code 0
    Failure Indicators: FAILED tests, exit code != 0
    Evidence: .sisyphus/evidence/task-15-integration-tests.txt

  Scenario: Raw endpoint integration test
    Tool: Bash (pytest)
    Preconditions: API test client configured
    Steps:
      1. pytest tests/test_api_integration.py::test_raw_prediction_valid -v
    Expected Result: Test passes, 200 response with prediction
    Failure Indicators: Test fails, 400/500 response
    Evidence: .sisyphus/evidence/task-15-raw-test.txt

  Scenario: Preprocessed endpoint integration test
    Tool: Bash (pytest)
    Preconditions: API test client configured
    Steps:
      1. pytest tests/test_api_integration.py::test_preprocessed_prediction_valid -v
    Expected Result: Test passes, 200 response with prediction
    Failure Indicators: Test fails, connection error
    Evidence: .sisyphus/evidence/task-15-preprocessed-test.txt

  Scenario: All three models test
    Tool: Bash (pytest)
    Preconditions: API test client configured
    Steps:
      1. pytest tests/test_api_integration.py -k "all_models" -v
    Expected Result: Test passes, three predictions in response
    Failure Indicators: Test fails, missing models
    Evidence: .sisyphus/evidence/task-15-all-models-test.txt
  ```

  **Evidence to Capture**:
  - [ ] Integration tests output
  - [ ] Raw endpoint test output
  - [ ] Preprocessed endpoint test output
  - [ ] All models test output

  **Commit**: YES
  - Message: `test: add integration tests for full API workflow`
  - Files: `tests/test_api_integration.py`
  - Pre-commit: `pytest tests/test_api_integration.py -v`

- [ ] 16. Write API health and error tests

  **What to do** (TDD: RED → GREEN → REFACTOR):
  - **RED**: Create `tests/test_api_health.py` and write failing tests first:
    - Test GET `/health` returns 200 with models_loaded true
    - Test GET `/health` returns model names list
    - Test GET `/` returns API information with title, version
    - Test POST `/api/v1/predict/raw` with malformed JSON returns 422
    - Test POST `/api/v1/predict/raw` with missing Content-Type returns 422
    - Test POST `/api/v1/predict/preprocessed` with non-array features returns 422
  - **GREEN**: Run `pytest tests/test_api_health.py -v` to see failures, then verify API passes all tests
  - **REFACTOR**: Add performance tests (response time < 500ms)

  **Must NOT do**:
  - Do NOT duplicate integration tests from Task 15 — focus on health and error cases
  - Do NOT test business logic — focus on API contract
  - Do NOT mock responses — use real test client

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple API health and error case tests
  - **Skills**: []
    - Standard pytest patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 13, 14, 15)
  - **Blocks**: None
  - **Blocked By**: Task 10 (needs FastAPI app)

  **References**:
  - `src/api/main.py` - FastAPI app
  - `tests/conftest.py` - Test client fixture
  - FastAPI testing docs: https://fastapi.tiangolo.com/tutorial/testing/

  **Acceptance Criteria**:
  - [ ] `tests/test_api_health.py` exists with at least 6 test functions
  - [ ] All tests follow TDD: written first, failed, then passed
  - [ ] `pytest tests/test_api_health.py -v` passes with 100% success
  - [ ] Tests cover: health endpoint, root endpoint, error cases (malformed JSON, missing Content-Type, type errors)

  **QA Scenarios**:

  ```
  Scenario: All health and error tests pass
    Tool: Bash (pytest)
    Preconditions: tests/test_api_health.py created, API running
    Steps:
      1. pytest tests/test_api_health.py -v --tb=short
      2. echo "Exit code: $?"
    Expected Result: All tests pass, exit code 0
    Failure Indicators: FAILED tests, exit code != 0
    Evidence: .sisyphus/evidence/task-16-health-tests.txt

  Scenario: Health endpoint test
    Tool: Bash (pytest)
    Preconditions: API test client configured
    Steps:
      1. pytest tests/test_api_health.py::test_health_endpoint -v
    Expected Result: Test passes, models_loaded true
    Failure Indicators: Test fails, models_loaded false
    Evidence: .sisyphus/evidence/task-16-health-test.txt

  Scenario: Error case tests
    Tool: Bash (pytest)
    Preconditions: API test client configured
    Steps:
      1. pytest tests/test_api_health.py -k "error" -v
    Expected Result: All error tests pass, 422 responses
    Failure Indicators: Tests fail, wrong status codes
    Evidence: .sisyphus/evidence/task-16-error-tests.txt
  ```

  **Evidence to Capture**:
  - [ ] Health tests output
  - [ ] Health endpoint test output
  - [ ] Error tests output

  **Commit**: YES
  - Message: `test: add API health and error case tests`
  - Files: `tests/test_api_health.py`
  - Pre-commit: `pytest tests/test_api_health.py -v`

- [ ] 17. Create Dockerfile

  **What to do**:
  - Create `Dockerfile` in project root
  - Use `tensorflow/tensorflow:latest` as base image (includes Python 3.10 + TensorFlow)
  - Set working directory to `/app`
  - Copy `requirements.txt` and install dependencies: `RUN pip install --no-cache-dir -r requirements.txt`
  - Copy source code: `COPY src/ ./src/`
  - Copy models: `COPY models/ ./models/`
  - Create reports directory: `RUN mkdir -p reports/shap_interactive`
  - Expose port: `EXPOSE 8000`
  - Set environment variables: `ENV PYTHONUNBUFFERED=1`
  - Run with uvicorn: `CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]`
  - Verify Dockerfile syntax: `docker build --no-cache --progress=plain .`

  **Must NOT do**:
  - Do NOT use `python:latest` as base — must include TensorFlow
  - Do NOT copy Jupyter notebooks — use .dockerignore to exclude
  - Do NOT copy tests/ directory — not needed in production
  - Do NOT use `tensorflow/tensorflow:latest-gpu` (CPU-only for simplicity, AWS EC2 standard)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard Dockerfile creation, straightforward Docker patterns
  - **Skills**: []
    - No specialized skills needed for Dockerfile

  **Parallelization**:
  - **Can Run In Parallel**: NO (all previous tasks must complete)
  - **Parallel Group**: Sequential (depends on ALL previous tasks)
  - **Blocks**: Task 18 (Docker build)
  - **Blocked By**: ALL previous tasks (need src/, models/, requirements.txt)

  **References**:
  - `.dockerignore` - Files to exclude
  - `requirements.txt` - Dependencies to install
  - TensorFlow Docker Hub: https://hub.docker.com/r/tensorflow/tensorflow

  **Acceptance Criteria**:
  - [ ] `Dockerfile` exists in project root
  - [ ] Base image is `tensorflow/tensorflow:latest`
  - [ ] Installs requirements.txt dependencies
  - [ ] Copies src/ and models/ directories
  - [ ] Exposes port 8000
  - [ ] Entrypoint uses uvicorn with correct app path
  - [ ] `docker build -t road-safety-api:latest .` succeeds

  **QA Scenarios**:

  ```
  Scenario: Dockerfile exists and is valid
    Tool: Bash
    Preconditions: Dockerfile created
    Steps:
      1. cat Dockerfile
      2. grep -c "FROM\|RUN\|COPY\|EXPOSE\|CMD" Dockerfile
    Expected Result: Count >= 6 (FROM, RUN, COPY, EXPOSE, CMD present)
    Failure Indicators: File missing, syntax errors
    Evidence: .sisyphus/evidence/task-17-dockerfile.txt

  Scenario: Docker build succeeds
    Tool: Bash (docker)
    Preconditions: Docker installed, all dependencies ready
    Steps:
      1. docker build -t road-safety-api:latest .
      2. echo "Build exit code: $?"
    Expected Result: Build succeeds, exit code 0
    Failure Indicators: Build fails, exit code != 0
    Evidence: .sisyphus/evidence/task-17-docker-build.txt

  Scenario: Docker image size < 3GB
    Tool: Bash (docker)
    Preconditions: Docker build complete
    Steps:
      1. docker images road-safety-api:latest | awk '{print $7}'
    Expected Result: Size < 3GB (TensorFlow base ~2GB + models ~500MB)
    Failure Indicators: Size >= 3GB (notebooks included, .dockerignore not working)
    Evidence: .sisyphus/evidence/task-17-docker-size.txt

  Scenario: Base image is TensorFlow
    Tool: Bash (docker)
    Preconditions: Docker image built
    Steps:
      1. docker run --rm road-safety-api:latest python -c "import tensorflow as tf; print(tf.__version__)"
    Expected Result: TensorFlow version printed (e.g., "2.15.0")
    Failure Indicators: "ModuleNotFoundError", base image wrong
    Evidence: .sisyphus/evidence/task-17-tf-version.txt
  ```

  **Evidence to Capture**:
  - [ ] Dockerfile content
  - [ ] Docker build output
  - [ ] Docker image size
  - [ ] TensorFlow version in container

  **Commit**: YES
  - Message: `feat: add Dockerfile for AWS EC2 deployment`
  - Files: `Dockerfile`
  - Pre-commit: `docker build -t test-build:latest . && docker rmi test-build:latest`

- [ ] 18. Build and test Docker container

  **What to do**:
  - Build Docker image: `docker build -t road-safety-api:latest .`
  - Run container: `docker run -d -p 8000:8000 --name test-api road-safety-api:latest`
  - Wait for startup: `sleep 10`
  - Test health endpoint: `curl http://localhost:8000/health`
  - Test raw prediction: `curl -X POST http://localhost:8000/api/v1/predict/raw -H "Content-Type: application/json" -d '{"State": "NSW", "Speed Limit": 60, ...}'`
  - Test preprocessed prediction: `curl -X POST http://localhost:8000/api/v1/predict/preprocessed -H "Content-Type: application/json" -d '{"features": [...]}')`
  - Verify all three models respond
  - Stop and cleanup: `docker stop test-api && docker rm test-api`
  - Document deployment: Add `DEPLOYMENT.md` with AWS EC2 instructions

  **Must NOT do**:
  - Do NOT push to Docker Hub (out of scope)
  - Do NOT create AWS infrastructure (out of scope)
  - Do NOT set up CI/CD pipeline (out of scope)
  - Do NOT use GPU variants (CPU-only for simplicity)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Docker testing and documentation, verification step
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO (must wait for Dockerfile build)
  - **Parallel Group**: Sequential (final task)
  - **Blocks**: Final verification wave
  - **Blocked By**: Task 17 (Dockerfile created)

  **References**:
  - `Dockerfile` - Image definition
  - `src/api/main.py` - Application entrypoint
  - `README.md` - Update with deployment instructions

  **Acceptance Criteria**:
  - [ ] Docker image builds successfully
  - [ ] Container starts and responds on port 8000
  - [ ] Health endpoint returns `{"status": "healthy", "models_loaded": true}`
  - [ ] Raw prediction endpoint works correctly
  - [ ] Preprocessed prediction endpoint works correctly
  - [ ] All three models return predictions
  - [ ] Container stops and cleans up successfully
  - [ ] `DEPLOYMENT.md` created with AWS EC2 deployment steps

  **QA Scenarios**:

  ```
  Scenario: Container starts and health check passes
    Tool: Bash (docker)
    Preconditions: Docker image built
    Steps:
      1. docker run -d -p 8000:8000 --name test-api road-safety-api:latest
      2. sleep 10
      3. curl http://localhost:8000/health
    Expected Result: {"status": "healthy", "models_loaded": true, "model_names": [...]}
    Failure Indicators: Connection refused, models_loaded false
    Evidence: .sisyphus/evidence/task-18-container-health.txt

  Scenario: Raw prediction endpoint works in container
    Tool: Bash (docker, curl)
    Preconditions: Container running
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/raw -H "Content-Type: application/json" -d '{"State": "NSW", "Speed Limit": 60, ...}'
    Expected Result: JSON with prediction, probability, model_name
    Failure Indicators: 404, 500, connection error
    Evidence: .sisyphus/evidence/task-18-raw-prediction.txt

  Scenario: Preprocessed prediction endpoint works in container
    Tool: Bash (docker, curl)
    Preconditions: Container running
    Steps:
      1. curl -X POST http://localhost:8000/api/v1/predict/preprocessed -H "Content-Type: application/json" -d '{"features": [...]}')
    Expected Result: JSON with prediction, probability, model_name
    Failure Indicators: 404, 500, connection error
    Evidence: .sisyphus/evidence/task-18-preprocessed-prediction.txt

  Scenario: Container cleanup succeeds
    Tool: Bash (docker)
    Preconditions: Container running
    Steps:
      1. docker stop test-api
      2. docker rm test-api
      3. docker ps -a | grep test-api
    Expected Result: No containers named "test-api" found
    Failure Indicators: Container still exists
    Evidence: .sisyphus/evidence/task-18-cleanup.txt

  Scenario: Deployment documentation exists
    Tool: Bash
    Preconditions: DEPLOYMENT.md created
    Steps:
      1. cat DEPLOYMENT.md
      2. grep -c "AWS EC2\|docker run" DEPLOYMENT.md
    Expected Result: Count >= 2 (AWS EC2 and docker run mentioned)
    Failure Indicators: File missing, insufficient content
    Evidence: .sisyphus/evidence/task-18-deployment-doc.txt
  ```

  **Evidence to Capture**:
  - [ ] Container health check response
  - [ ] Raw prediction response
  - [ ] Preprocessed prediction response
  - [ ] Container cleanup verification
  - [ ] Deployment documentation content

  **Commit**: YES
  - Message: `test: build and verify Docker deployment, add deployment docs`
  - Files: `DEPLOYMENT.md`
  - Pre-commit: `docker build -t test-api:latest . && docker run --rm test-api:latest python -c "import tensorflow; print('OK')" && docker rmi test-api:latest`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
>
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `tsc --noEmit` (if TypeScript) or `python -m py_compile` for all Python files + `flake8` or `pylint` + `pytest tests/`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, `print()` in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp).
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill if UI)
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (features working together, not isolation). Test edge cases: empty state, invalid input, rapid actions. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

All commits follow conventional commit format with atomic, focused changes:

1. `chore: add requirements.txt with pinned dependencies` — Task 1
2. `test: add pytest infrastructure and configuration` — Task 2
3. `chore: create source directory structure` — Task 3
4. `chore: add .dockerignore to exclude notebooks and tests` — Task 4
5. `feat: regenerate and save all model artifacts locally` — Task 5
6. `feat: add preprocessing module with validation and transformation` — Task 6
7. `feat: add model loading utilities with caching` — Task 7
8. `feat: add configuration and logging modules` — Task 8
9. `feat: add SHAP explainability analysis to notebook` — Task 9
10. `feat: create FastAPI app with Pydantic schemas` — Task 10
11. `feat: add raw prediction endpoint with human-readable input` — Task 11
12. `feat: add preprocessed prediction endpoint` — Task 12
13. `test: add unit tests for preprocessing module` — Task 13
14. `test: add unit tests for model loading utilities` — Task 14
15. `test: add integration tests for full API workflow` — Task 15
16. `test: add API health and error case tests` — Task 16
17. `feat: add Dockerfile for AWS EC2 deployment` — Task 17
18. `test: build and verify Docker deployment, add deployment docs` — Task 18

Pre-commit hooks enforced:
- Python syntax check: `python -m py_compile <file>`
- Test run: `pytest tests/<file> -v`

---

## Success Criteria

### Verification Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v --cov=src --cov-report=term-missing

# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health
# Expected: {"status": "healthy", "models_loaded": true, "model_names": [...]}

# Test raw prediction endpoint
curl -X POST http://localhost:8000/api/v1/predict/raw \
  -H "Content-Type: application/json" \
  -d '{"State": "NSW", "Speed Limit": 60, "National Road Type": "Arterial Road", "Road User": "Driver", "Age": 30, "Gender": "Male", "Bus Involvement": "No", "Articulated Truck Involvement": "No", "Heavy Rigid Truck Involvement": "No", "Dayweek": "Friday", "Time": 18, "Christmas Period": "No", "Easter Period": "No"}'
# Expected: {"predictions": [{"prediction": 0, "probability": 0.XX, "model_name": "..."}, ...]}

# Build Docker image
docker build -t road-safety-api:latest .
docker run -d -p 8000:8000 --name test-api road-safety-api:latest
curl http://localhost:8000/health
docker stop test-api && docker rm test-api

# Run Jupyter notebook with SHAP
jupyter execute 05_analysis.ipynb --stdout

# Verify SHAP outputs
ls -la reports/shap_*.png reports/shap_interactive/
```

### Final Checklist

- [ ] All "Must Have" present:
  - [ ] Model artifacts regenerated locally in `models/`
  - [ ] Dual API endpoints (`/predict/raw` and `/predict/preprocessed`)
  - [ ] SHAP analysis in `05_analysis.ipynb` with static + interactive outputs
  - [ ] Complete test suite (unit + integration + API tests)
  - [ ] Working Dockerfile that builds successfully
  - [ ] requirements.txt with pinned versions

- [ ] All "Must NOT Have" absent:
  - [ ] No database integration (PostgreSQL, SQLite, etc.)
  - [ ] No authentication (JWT, API keys, OAuth)
  - [ ] No frontend dashboard (React, Mapbox, etc.)
  - [ ] No CI/CD pipeline (GitHub Actions, Jenkins)
  - [ ] No model retraining (frozen weights only)
  - [ ] No notebook rewriting (00-04 untouched)
  - [ ] No per-request model loading (cached at startup)

- [ ] All tests pass:
  - [ ] `pytest tests/test_preprocessing.py -v` — 100% pass
  - [ ] `pytest tests/test_model_loader.py -v` — 100% pass
  - [ ] `pytest tests/test_api_integration.py -v` — 100% pass
  - [ ] `pytest tests/test_api_health.py -v` — 100% pass

- [ ] Docker builds and runs:
  - [ ] `docker build -t road-safety-api:latest .` — succeeds
  - [ ] `docker run -d -p 8000:8000 road-safety-api:latest` — starts
  - [ ] `curl http://localhost:8000/health` — returns healthy
  - [ ] Image size < 3GB

- [ ] SHAP analysis complete:
  - [ ] `reports/shap_rf_summary.png` exists
  - [ ] `reports/shap_ffnn_summary.png` exists
  - [ ] `reports/shap_residualnn_summary.png` exists
  - [ ] `reports/shap_interactive/` directory has HTML files
  - [ ] `05_analysis.ipynb` executes without errors
