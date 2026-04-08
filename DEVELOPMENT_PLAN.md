# DEVELOPMENT_PLAN.md
**Project Name:** DeepRisk (Road Safety Risk Prediction)
**Focus:** Actuarial Deep Learning & Predictive Neural Networks

---

## 1. Purpose & Vision
The purpose of DeepRisk is to utilize massive sets of geographic and historical urban traffic data to dynamically predict the probability and severity of vehicular accidents across a city grid. The vision is to demonstrate advanced Actuarial Machine Learning application—shifting from legacy regression models to deep neural network infrastructures that ingest hundreds of non-linear features (weather, geometry, time, volume) to accurately price systemic road risks.

## 2. UNSW Course Foundation
*   **ACTL3143 (AI and Deep Learning for Actuarial Applications):** Demonstrates full mastery over PyTorch, loss function tuning, dropout regularization, and hyperparameter optimization for actuarial outcome prediction.
*   **COMP9417 (Machine Learning and Data Mining):** Comprehensive understanding of the statistical backbone, pre-processing edge cases, gradient descent logic, and ROC/AUC performance evaluation metrics.
*   **ACTL3142 (Actuarial Data & Analysis):** Dealing with severely imbalanced historical damage datasets requiring SMOTE up-sampling and strict feature engineering.

## 3. Core Topics & Technical Concerns
*   **Topic 1: Neural Architecture Design:** Deciding between Multi-Layer Perceptrons (MLPs) versus Graph Neural Networks (GNNs) given that roadways represent a mathematical graph of interconnected nodes.
*   **Topic 2: Overfitting & Bias:** The core concern of neural networks in finance/insurance. Using validation hold-out sets and strict dropout layers to ensure our model doesn't blindly memorize the training data.
*   **Concern: Explainability (XAI):** Deep learning is notoriously "black box". To be adopted by regulators or actuaries legally, we must implement SHAP values to explain *why* an intersection is deemed "High Risk" (e.g., lack of streetlights vs high traffic volume).

## 4. Final Product Blueprint
The end state is a beautiful Python ecosystem containing heavily documented Jupyter Notebooks (`01_preprocess.ipynb` to `05_analysis.ipynb`) wrapped in a final PDF scientific paper. Additionally, a frontend web dashboard (React/Mapbox GL JS) visualizing the city map with bright red/orange "Risk Heatmaps" overlaying the exact roads the AI has pinpointed as critical accident hotspots over the next 24 hours.

## 5. Monetization Strategy
*   **Aviation/Auto Insurance Licensing:** Selling the prediction engine's API endpoints to major automotive insurers (e.g., Suncorp, IAG). If a client is driving down a "High Risk" path during a storm, the insurer can dynamically adjust micro-premiums or send mobile alerts to the driver.
*   **B2G (Business-to-Government) Consulting:** Selling the risk-analysis reports directly to City Councils or Transport bodies (TpNSW) so they can optimally allocate tax budgets to fix the most dangerous infrastructural geometries.

## 6. Implementation Plan & Next Steps
1.  **Refine Jupyter Notebooks:** Clean the base exploratory data analysis (EDA) charts so they are professional and presentation-ready.
2.  **API Deployment:** Wrap the final trained `.pt` (PyTorch model weights) file inside a lightweight FastAPI Python container hosted on AWS.
3.  **Frontend Heatmap Integration:** Pass geographic coordinates to the specific Mapbox UI visualization.

## 7. Tasks to be Done
- [ ] Implement SHAP values script within `05_analysis.ipynb` for actuarial transparency.
- [ ] Build a `Dockerfile` to export the heavy PyTorch environment to AWS EC2 easily.
- [ ] Create a `predict()` REST endpoint using FastAPI.
