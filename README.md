# SentinelCold: Predictive Risk Modeling for Pharmaceutical Cold-Chain

**SentinelCold** is an end-to-end machine learning system designed to reframe pharmaceutical cold-chain monitoring from reactive, threshold-based alerting to **proactive, explainable risk prediction**. 

Developed as part of the research *Predictive Cold Chain Disruption Modeling for Pharmaceutical Logistics* (Udhav Dhandia, Index: 2343065, Christ University), this project aims to detect **silent failures** in cold-chain logistics using multi-sensor IoT telemetry including temperature, humidity, vibration, door-state, and routing metadata.

## 1. Key Features

* **Proactive Prediction**: Detects cumulative degradation patterns before they breach hard-coded thresholds, mitigating the risk of irrecoverable product damage.
* **Explainable AI (XAI)**: Generates per-shipment causal explanations using SHAP and LIME methodologies. High-risk predictions are accompanied by a regulator-legible attribution log, ensuring compliance with FDA 21 CFR Part 11 and EU GxP Annex 11.
* **Robust Data Pipeline**: Implements a 6-tier architecture that addresses class imbalance (via CGAN/DeepSMOTE), non-linear feature interactions, and outliers without diluting the underlying predictive signals.
* **Cost-Sensitive Optimization**: Models are tuned using an asymmetric cost matrix that heavily penalizes False Negatives (missed failures) over False Positives (unnecessary quality assurance checks).

## 2. System Architecture

The system is logically divided into a 6-tier pipeline:

1. **Ingestion & Data Profiling**: Implements strict Pydantic schema validation and statistical profiling.
2. **DCAI Preprocessing & Cleaning**: Utilizes class-conditional median imputation and outlier-preserving winsorization.
3. **Multi-Sensor Feature Engineering**: Synthesizes physically-motivated interaction features (e.g., `excursion_intensity`).
4. **Predictive Modeling Core**: Employs gradient-boosted ensembles (XGBoost, LightGBM, CatBoost) alongside temporal anomaly modeling via a Conv1D Autoencoder.
5. **Explainable AI Engine**: Conducts global and local model attribution.
6. **Decision Support**: Outputs a risk tiering evaluation and recommended action payload.

For comprehensive architectural details, please consult the Product Requirement Document: [`prd.md`](prd.md).

## 3. Repository Structure

- `tier1_ingestion/` to `tier6_decision_support/`: The core modules comprising the 6-tier machine learning pipeline.
- `config.yaml`: Centralized configuration governing hyperparameters, thresholds, and relative file paths.
- `data/raw/`: Contains the reference benchmark dataset (`cold_chain_shipment_data.csv`, n=8,000).
- `outputs/`: Auto-generated directory designated for trained models, experiments, figures, tables, and reports.
- `relevant submissions/`: Contains the core academic submissions by Udhav Dhandia (2343065), including:
  - Exploratory Data Analysis (`2343065_eda.pdf`, `eda_code_2343065.docx`)
  - Literature Review (`2343065_lit_review.pdf`, `Lab_02_..._Literature_Review.docx`, `Lab_02_..._Literature_Review_Spreadsheet.xlsx`)
  - Architecture Diagram (`2343065 architecture diagram.png`)
- `utils/`: Shared functional utilities and helpers.

## 4. Dataset and Research Basis

- **Benchmark Dataset**: Kaggle Cold Chain Shipment Silent Failure Dataset.
- **Base Academic Reference**: Xie et al. (2025), *An Anomaly Detection Scheme for Data Stream in Cold Chain Logistics*, PLOS ONE.

## 5. Setup and Usage Instructions

Ensure Python is installed on the host machine, and create a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows environments use: venv\Scripts\activate
pip install -r requirements.txt
```

*(Further execution scripts for the pipeline steps will be orchestrated via the `main.py` entry point)*

---
*Classification: Confidential / Internal - Draft v1.0*
