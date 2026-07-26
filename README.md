# Road Safety Risk Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688.svg?logo=fastapi&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-purple.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A machine learning project that predicts whether road crashes in Australia involve single or multiple vehicles. Built with TensorFlow and scikit-learn, comparing Random Forest against deep learning models (Feedforward and Residual Neural Networks) to help inform insurance pricing and road safety interventions.

Beyond the modelling, the trained models are served behind a validated **FastAPI** endpoint with SHAP
explanations and a pytest suite — see [Serving the model](#serving-the-model).

## Technologies Used

- **Python 3.8+** - Core programming language
- **TensorFlow & Keras** - Deep learning frameworks
- **scikit-learn** - Machine learning library for Random Forest baseline
- **pandas & NumPy** - Data manipulation and analysis
- **matplotlib & seaborn** - Data visualisation
- **Jupyter Notebook** - Interactive development environment
- **Bayesian Optimisation** - Hyperparameter tuning
- **FastAPI & Pydantic** - Model serving with request validation
- **SHAP** - Per-prediction feature attribution
- **pytest** - API, integration, and preprocessing tests

## Features

- **Binary classification** of crash types (single vs. multiple vehicle)
- **Three model architectures**: Random Forest (baseline), Feedforward NN, and Residual NN
- **Comprehensive data preprocessing** with stratified train/validation/test splits
- **Automated hyperparameter tuning** using Bayesian optimisation
- **Performance metrics**: F1-score, AUC-ROC, precision, recall, and accuracy
- **Exploratory data analysis** with visualisations of temporal patterns, speed correlations, and demographics

## What Users Can Do

1. **Explore the dataset** - Understand patterns in Australian road fatalities from 1989-2025
2. **Train models** - Run and compare three different classification approaches
3. **Evaluate performance** - Analyse model predictions with detailed metrics and confusion matrices
4. **Tune hyperparameters** - Optimise model architectures using Bayesian optimisation
5. **Apply predictions** - Use trained models for insurance risk assessment or road safety planning

## The Process

### 1. Data Exploration (00_eda.ipynb)

Started by exploring the Australian Road Deaths Database (57,430 observations). Discovered key patterns:

- 55% single vehicle vs. 45% multiple vehicle crashes (class imbalance)
- Single vehicle crashes peak at night (12 AM - 6 AM) - linked to fatigue/impaired driving
- Multiple vehicle crashes occur during afternoon rush hour (12 PM - 6 PM)
- 72% of fatalities are male, with peaks among young adults (20-30 years)

### 2. Data Preprocessing (01_preprocess.ipynb)

Cleaned and prepared the data:

- Removed 153 observations (0.3%) with missing critical features
- Created 60/20/20 train/validation/test split (stratified to maintain class balance)
- Standardised numerical features (age, time, speed limit)
- One-hot encoded categorical features (state, road user type, vehicle involvement)

### 3. Baseline Model (02_baseline_model.ipynb)

Built a Random Forest classifier as the benchmark:

- Chose Random Forest for its ability to handle mixed data types and robustness to outliers
- Achieved 72% accuracy and 0.81 AUC-ROC
- This became the bar for deep learning models to beat

### 4. Hyperparameter Tuning (03_hyperparameter_tuning.ipynb)

Optimised all three models using Bayesian optimisation:

- More efficient than grid search (fewer iterations needed)
- Tuned number of layers, units per layer, dropout rates, and learning rates
- Used validation AUC-ROC as the optimisation metric

### 5. Deep Learning Models (04_train_deep_learning_models.ipynb)

Implemented two neural network architectures:

- **Feedforward NN**: 2 dense layers with batch normalisation and dropout
- **Residual NN**: 4 residual blocks with skip connections to address vanishing gradients

### 6. Analysis (05_analysis.ipynb)

Compared all models and visualised results:

- Random Forest performed best overall (F1: 0.71, AUC-ROC: 0.81)
- Feedforward NN had highest recall for single vehicle crashes (76%)
- Residual NN showed marginal improvement due to skip connections

## What I Learned

### Technical Skills

- **Bayesian optimisation** is far more efficient than grid search for hyperparameter tuning
- **Residual connections** help with gradient flow in deeper networks, though the improvement was marginal for this dataset
- **Class imbalance** matters - using balanced class weights significantly improved model performance
- **Feature engineering** temporal data (converting time to categorical bins) revealed important patterns

### Domain Knowledge

- Single vehicle crashes have very different characteristics than multiple vehicle crashes
- Temporal patterns are strong predictors (night vs. day)
- Speed limits and road types are highly correlated with crash severity

### Challenges Overcome

- Handling missing data across 36 years of records
- Balancing model complexity vs. interpretability for insurance applications
- Preventing overfitting with appropriate dropout and early stopping

## How It Could Be Improved

1. **Add more features** - Weather conditions, traffic density, driver blood alcohol content
2. **Try other architectures** - LSTM for temporal sequences, attention mechanisms
3. **Address class imbalance further** - SMOTE oversampling, cost-sensitive learning
4. **Ensemble methods** - Combine Random Forest with neural networks
5. **Geospatial analysis** - Add location-based features and clustering
6. **Multi-class classification** - Predict crash severity levels, not just single/multiple

> Items previously on this list — SHAP explainability and API deployment — have since been built. See
> [Serving the model](#serving-the-model) below.

## How to Run the Project

### Prerequisites

```bash
python >= 3.8
jupyter notebook
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/FishyDanny/Road-Safety-Risk-Prediction.git
cd Road-Safety-Risk-Prediction
```

2. Install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn openpyxl
```

### Running the Notebooks

Execute notebooks in this order:

```bash
# 1. Explore the data
jupyter notebook 00_eda.ipynb

# 2. Preprocess and split data
jupyter notebook 01_preprocess.ipynb

# 3. Train baseline Random Forest
jupyter notebook 02_baseline_model.ipynb

# 4. Tune hyperparameters
jupyter notebook 03_hyperparameter_tuning.ipynb

# 5. Train deep learning models
jupyter notebook 04_train_deep_learning_models.ipynb

# 6. Analyse and compare results
jupyter notebook 05_analysis.ipynb
```

## Serving the model

The trained models are exposed behind a versioned **FastAPI** service. Models are loaded once at
application startup via a lifespan handler rather than per request, so inference does not pay
deserialisation cost on every call.

```bash
uvicorn src.api.main:app --reload
# Interactive docs at http://localhost:8000/docs
```

| Endpoint                            | Purpose                                                                                                                                                                                     |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /api/v1/predict/raw`          | Takes 13 human-readable features (state, speed limit, age, day, hour, …), runs them through the same preprocessing pipeline used in training, and returns predictions from all three models |
| `POST /api/v1/predict/preprocessed` | Takes an already-encoded feature vector, for batch or downstream use                                                                                                                        |
| `GET /health`                       | Liveness check reporting model-load status                                                                                                                                                  |

**Input validation** is handled by Pydantic schemas with domain constraints — speed limits bounded to
40–130 km/h, age 0–120, hour 0–23, and Australian state codes validated against an allow-list — so
malformed requests fail with a 400 before reaching the model.

**Explainability** uses SHAP to attribute each prediction to its contributing features, which matters
for the insurance use case: a risk score that cannot be explained cannot be justified to a regulator
or a customer.

### Tests

```bash
pytest                  # API health, integration, model loading, preprocessing
pytest --cov=src        # with coverage
```

## Results

| Model             | F1-Score | AUC-ROC  | Precision (Single) | Recall (Single) | Precision (Multiple) | Recall (Multiple) | Accuracy |
| ----------------- | -------- | -------- | ------------------ | --------------- | -------------------- | ----------------- | -------- |
| **Random Forest** | **0.71** | **0.81** | **0.78**           | 0.70            | 0.67                 | **0.75**          | **0.72** |
| Feedforward NN    | 0.67     | 0.80     | 0.73               | **0.77**        | 0.69                 | 0.66              | 0.72     |
| Residual NN       | 0.68     | 0.80     | 0.74               | 0.76            | 0.69                 | 0.66              | 0.72     |

**Key Takeaway**: Random Forest achieved the best overall performance, making it the most suitable for insurance risk assessments. However, Feedforward NN's higher recall for single vehicle crashes makes it valuable for road safety interventions.

## Dataset

**Source**: [Australian Road Deaths Database](https://www.bitre.gov.au/statistics/safety/fatal_road_crash_database) - Bureau of Infrastructure and Transport Research Economics

**Size**: 57,430 fatality records (January 1989 - May 2025)

**Features**: State, speed limit, road type, road user type, age, gender, time, day of week, vehicle involvement (bus, trucks), holiday periods

**Target**: Crash Type (Single or Multiple vehicle)

## Project Structure

```
Road-Safety-Risk-Prediction/
│
├── 00_eda.ipynb                        # Exploratory data analysis and visualisations
├── 01_preprocess.ipynb                 # Data cleaning and feature engineering
├── 02_baseline_model.ipynb             # Random Forest benchmark model
├── 03_hyperparameter_tuning.ipynb      # Bayesian optimisation for all models
├── 04_train_deep_learning_models.ipynb # Feedforward and Residual neural networks
├── 05_analysis.ipynb                   # Model comparison and evaluation
│
├── src/
│   ├── api/
│   │   ├── main.py                     # FastAPI app, model loading via lifespan handler
│   │   ├── schemas.py                  # Pydantic request/response models with validation
│   │   └── routes/predict.py           # /predict/raw and /predict/preprocessed
│   ├── model_loader.py                 # Loads RF + both Keras models and the preprocessor
│   ├── preprocessing.py                # Shared transform pipeline (training and serving)
│   ├── config.py                       # Settings
│   └── logger.py                       # Structured logging
│
├── models/                             # Serialised models and train/val/test splits
├── tests/                              # pytest: API health, integration, loader, preprocessing
├── regenerate_models.py                # Rebuild all model artefacts from scratch
├── requirements.txt
└── README.md                           # This file
```

## Use Cases

- **Insurance pricing**: Predict crash risk for motor insurance premiums
- **Road safety planning**: Identify high-risk roads and times for interventions
- **Policy decisions**: Target education campaigns based on demographic patterns
- **Infrastructure planning**: Design safer roads based on speed limit and crash type correlations

## References

1. [Australian Road Deaths Database](https://www.bitre.gov.au/statistics/safety/fatal_road_crash_database) - Bureau of Infrastructure and Transport Research Economics
2. [Skip Connections in Deep Learning](https://theaisummer.com/skip-connections/) - Understanding residual networks
3. [Machine Bias in Risk Assessment](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) - Ethical considerations

## Licence

Released under the [MIT Licence](LICENSE) © Danny Li.

Originally developed as a deep-learning coursework project at UNSW (2025), since extended with a
FastAPI serving layer, a test suite, and reproducible model regeneration.

## Contact

For questions or collaborations, please open an issue in this repository.

## AI Usage

The notebooks were written by me, with AI assistance used for code review and editing passes. This
README was drafted with AI assistance.
