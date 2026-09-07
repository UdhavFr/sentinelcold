# SentinelCold: Predictive Risk Modeling for Pharmaceutical Cold-Chain

**SentinelCold** is an end-to-end machine learning system designed to reframe pharmaceutical cold-chain monitoring from reactive, threshold-based alerting to **proactive, explainable risk prediction**. 

Developed as part of the research *Predictive Cold Chain Disruption Modeling for Pharmaceutical Logistics* (Udhav Dhandia, Christ University), this project aims to detect **silent failures** in cold-chain logistics using multi-sensor IoT telemetry (temperature, humidity, vibration, door-state, routing metadata).

## 🚀 Key Features

* **Proactive Prediction**: Detects cumulative degradation patterns before they breach hard-coded thresholds, preventing irrecoverable product damage.
* **Explainable AI (XAI)**: Generates per-shipment causal explanations using SHAP and LIME. Every high-risk prediction is accompanied by a regulator-legible attribution log (compliant with FDA 21 CFR Part 11 / EU GxP Annex 11).
* **Robust Data Pipeline**: A 6-tier architecture that handles class imbalance (CGAN/DeepSMOTE), non-linear feature interactions, and outliers without diluting the predictive signal.
* **Cost-Sensitive Optimization**: Models are tuned using an asymmetric cost matrix, heavily penalizing False Negatives (missed failures) over False Positives (unnecessary QA checks).

## 🏗 System Architecture (The 6-Tier Pipeline)

1. **Ingestion & Data Profiling**: Strict Pydantic schema validation and statistical profiling.
2. **DCAI Preprocessing & Cleaning**: Class-conditional median imputation and outlier-preserving winsorization.
3. **Multi-Sensor Feature Engineering**: Synthesizes physically-motivated interaction features (e.g., `excursion_intensity`).
4. **Predictive Modeling Core**: Gradient-boosted ensembles (XGBoost, LightGBM, CatBoost) and temporal anomaly modeling via Conv1D Autoencoder.
5. **Explainable AI Engine**: Global and local model attribution.
6. **Decision Support**: Risk tiering and recommended action payload.

For complete architectural details, consult the Product Requirement Document: [`prd.md`](prd.md).

## 📂 Repository Structure

- `tier1_ingestion/` to `tier6_decision_support/`: The core 6-tier ML pipeline modules.
- `config.yaml`: Centralized configuration for hyperparameters, thresholds, and paths.
- `data/raw/`: Contains the reference benchmark dataset (`cold_chain_shipment_data.csv`, n=8,000).
- `outputs/`: Auto-generated directory for models, experiments, figures, tables, and reports.
- `relevant submissions/`: Contains academic submissions by Udhav Dhandia (2343065), including:
  - Exploratory Data Analysis (`2343065_eda.pdf`, `eda_code_2343065.docx`)
  - Literature Review (`2343065_lit_review.pdf`, `Lab_02_..._Literature_Review.docx/xlsx`)
  - Architecture Diagram (`2343065 architecture diagram.png`)
- `utils/`: Shared utilities and helpers.

## 📊 Dataset & Research Basis

- **Benchmark Dataset**: Kaggle Cold Chain Shipment Silent Failure Dataset
- **Base Academic Reference**: Xie et al. (2025), *An Anomaly Detection Scheme for Data Stream in Cold Chain Logistics*, PLOS ONE

## 🛠 Setup & Usage

Ensure you have Python installed and create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

*(Further execution scripts for the pipeline steps will be orchestrated via `main.py`)*

---
*Confidential / Internal - Draft v1.0*
