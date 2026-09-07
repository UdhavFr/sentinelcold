\# Product Requirement Document



\## SentinelCold: Predictive Risk Modeling and Explainable AI for Pharmaceutical Cold-Chain Silent Failure Detection



| Field | Detail |

|---|---|

| \*\*Document Type\*\* | Product Requirement Document (PRD) |

| \*\*Product Codename\*\* | SentinelCold |

| \*\*Research Basis\*\* | Dhandia, U. (2343065), Christ University — \*Predictive Cold Chain Disruption Modeling for Pharmaceutical Logistics\* |

| \*\*Reference Benchmark Dataset\*\* | Cold Chain Shipment Silent Failure Dataset (Kaggle), n = 8,000 |

| \*\*Base Academic Reference\*\* | Xie et al. (2025), \*An Anomaly Detection Scheme for Data Stream in Cold Chain Logistics\*, PLOS ONE |

| \*\*Document Status\*\* | Draft v1.0 — For Engineering \& Regulatory Review |

| \*\*Classification\*\* | Internal / Confidential |



\---



\## Table of Contents



1\. Executive Summary \& Business Case

2\. System Architecture \& Data Flow (The 6-Tier Pipeline)

3\. Data-Centric AI \& Preprocessing Specifications (Tiers 1 \& 2)

4\. Feature Engineering \& Multi-Sensor Analytics (Tier 3)

5\. Modeling, Training \& Optimization (Tier 4)

6\. Explainable AI \& Audit Compliance (Tier 5)

7\. Decision Support \& Alerting Engine (Tier 6)

8\. Technical, Performance \& Non-Functional Requirements

9\. Appendix: Glossary, Data Dictionary \& References



\---



\## 1. Executive Summary \& Business Case



\### 1.1 Problem Statement



Pharmaceutical cold-chain logistics operates under a zero-tolerance thermal envelope: a sustained or cumulative deviation from validated storage conditions can silently degrade the potency of a vaccine, insulin formulation, or monoclonal antibody without producing any visible or organoleptic sign of failure. Industry estimates place global vaccine wastage attributable to temperature excursions at approximately \*\*50%\*\* of all doses produced, with pharmaceutical revenue losses from cold-chain failure exceeding \*\*$35 billion USD annually\*\*. Beyond direct economic loss, compromised product reaching a patient carries downstream consequences: adverse clinical events, erosion of public trust in healthcare delivery, and regulatory exposure for manufacturers and logistics providers under GxP and pharmacovigilance frameworks.



The dominant monitoring paradigm in the industry today — \*\*static, threshold-based alarms\*\* (e.g., "alert if temperature > 8°C") — is structurally incapable of solving this problem for two reasons:



1\. \*\*Alert Fatigue from False Positives.\*\* Transient sensor noise, momentary door-opening events during loading/unloading, and short-lived environmental fluctuations routinely breach fixed thresholds without representing genuine product risk. Operations teams desensitize to these alerts, degrading response quality across the board.

2\. \*\*Blindness to Silent Failures.\*\* The most dangerous failure mode in cold-chain logistics is the \*\*silent failure\*\*: a sub-threshold, cumulative degradation pattern — combining moderate temperature range, elevated humidity, prolonged sensor blackout intervals, and repeated mechanical stress — that never crosses a hard-coded alarm boundary, yet materially compromises product stability. No threshold rule can detect an interaction effect it was never designed to observe.



\### 1.2 Product Concept



\*\*SentinelCold\*\* is a six-tier, end-to-end machine learning system that reframes cold-chain monitoring from \*\*reactive alerting\*\* to \*\*proactive, explainable risk prediction\*\*. The system:



\- Ingests raw multi-sensor IoT telemetry (temperature, humidity, vibration, door-state, routing metadata) at the shipment level;

\- Applies data-centric preprocessing calibrated to the statistical realities of the domain (rare but informative missingness, heavy right-skew, class imbalance of \~19.7% positive prevalence);

\- Engineers physically-motivated interaction features (e.g., `excursion\_intensity`) that individual raw sensors cannot express;

\- Trains a gradient-boosted ensemble optimized for \*\*recall-first\*\* classification, reflecting the asymmetric cost of a missed silent failure versus a false alarm;

\- Generates \*\*per-shipment causal explanations\*\* using SHAP and LIME, translating opaque model scores into auditable, regulator-legible narratives; and

\- Outputs a \*\*risk-tiered decision payload\*\* (Low / Medium / High) mapped to concrete logistics actions (monitor, QA flag, reroute/re-cool), rather than a bare probability score.



\### 1.3 Why This Matters: The Silent Failure Paradigm



The core research insight underpinning this product (Dhandia, 2026) is that no single raw feature is strongly predictive of failure in isolation. Univariate and bivariate analysis of the 8,000-shipment benchmark dataset shows that categorical routing attributes (`carrier\_id`, `origin\_zone`, `dest\_zone`) exhibit negligible variance in failure rate (16.7%–22.6% across all levels), while composite, engineered signals — duration × temperature range × monitoring blackout duration — are meaningfully predictive (Pearson r up to 0.453 for `transit\_days`, 0.441 for `temp\_max\_c`). This validates the central product thesis: \*\*silent failures are emergent properties of feature interactions, not single-sensor breaches\*\*, and therefore require a model class (gradient-boosted trees, temporal autoencoders) capable of learning non-linear interaction surfaces, paired with an explainability layer capable of decomposing that surface back into human-interpretable, audit-ready causes.



\### 1.4 Strategic Outcomes



| Outcome | Mechanism |

|---|---|

| Reduce pharmaceutical cold-chain revenue loss | Early, high-recall prediction of silent failures enables pre-emptive rerouting/re-icing before product potency is compromised |

| Reduce alert fatigue | Risk-tiered, probability-calibrated alerts replace binary threshold breaches, cutting false-positive volume |

| Satisfy QA/Regulatory explainability mandates | Every high-risk prediction is accompanied by a SHAP/LIME-derived causal attribution log suitable for FDA 21 CFR Part 11 / EU GxP Annex 11 audit trails |

| Establish reproducible benchmark | System performance is measured against both a supervised baseline (KNN) and an unsupervised academic baseline (Xie et al. 2025 eiForest), ensuring the product's value-add is quantified, not asserted |



\---



\## 2. System Architecture \& Data Flow (The 6-Tier Pipeline)



SentinelCold is decomposed into six modular, independently testable tiers. Each tier has a strict input/output contract, allowing tiers to be versioned, re-trained, or replaced (e.g., swapping XGBoost for CatBoost in Tier 4) without breaking downstream consumers.



\### 2.1 Tier Summary Table



| Tier | Name | Primary Responsibility | Key Algorithms / Tools |

|---|---|---|---|

| 1 | Ingestion \& Data Profiling | Schema validation, type enforcement, statistical profiling of raw shipment records | Pydantic, pandas-profiling equivalents |

| 2 | DCAI Preprocessing \& Cleaning | Missingness treatment, outlier-preserving cleaning, class balancing | Class-conditional imputation, Winsorization, CGAN / DeepSMOTE |

| 3 | Multi-Sensor Feature Engineering | Rate/range transforms, interaction features, multicollinearity filtering | Domain-driven feature synthesis, VIF filtering |

| 4 | Predictive Modeling Core | Supervised classification + temporal anomaly modeling | XGBoost, LightGBM, CatBoost, Conv1D Autoencoder |

| 5 | Explainable AI Engine | Global and local attribution of model decisions | SHAP (TreeExplainer), LIME |

| 6 | Decision Support | Risk tiering and recommended-action generation | Rule-engine over probability + SHAP payload |



\### 2.2 End-to-End Data Flow (ASCII Architecture Diagram)



```

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐

│                                   RAW IoT SENSOR \& SHIPMENT TELEMETRY                                 │

│   temperature | humidity | vibration | door events | routing/carrier metadata | package attributes   │

└───────────────────────────────────────────┬──────────────────────────────────────────────────────────┘

&#x20;                                            │  Dataset Input (CSV / streaming batch)

&#x20;                                            ▼

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐

│  TIER 1 — INGESTION \& DATA PROFILING                                                                  │

│  ┌────────────────────────┐   ┌───────────────────────────────┐                                     │

│  │ Kaggle Cold Chain       │──▶│ Schema \& Data Profiling        │                                     │

│  │ Silent Failure Dataset  │   │ (type / missingness / validity)│                                     │

│  │ 8,000 rows × 24 cols    │   └───────────────┬───────────────┘                                     │

│  │ 23 features + target    │                   ▼                                                     │

│  │ target: 0=6,426 1=1,574 │        Validated Tabular Dataset (model-ready input contract)            │

│  └────────────────────────┘                                                                          │

└───────────────────────────────────────────┬──────────────────────────────────────────────────────────┘

&#x20;                                            │  Clean, Balanced Data

&#x20;                                            ▼

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐

│  TIER 2 — DCAI PREPROCESSING \& CLEANING                                                               │

│  Schema Validation (Pydantic) → Anomaly Cleaning (winsorize / IQR) → Class Balancing (CGAN/DeepSMOTE) │

└───────────────────────────────────────────┬──────────────────────────────────────────────────────────┘

&#x20;                                            │  Clean, Balanced Data

&#x20;                                            ▼

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐

│  TIER 3 — MULTI-SENSOR FEATURE ENGINEERING                                                            │

│  Rate \& Range Transforms → Interaction Features (excursion\_intensity) → Variance/Relevance Filtering  │

└───────────────────────────────────────────┬──────────────────────────────────────────────────────────┘

&#x20;                                   Feature Vectors   │   Temporal Features

&#x20;                                            ▼         ▼

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐

│  TIER 4 — PREDICTIVE MODELING CORE                                                                    │

│  Stratified k-Fold CV, Recall-Max Threshold  │  Ensemble Classifier (XGBoost/LightGBM/CatBoost)       │

│                                               │  Temporal Anomaly Model (Conv1D Autoencoder)           │

└───────────────────────────────────────────┬──────────────────────────────────────────────────────────┘

&#x20;                                            │  Failure Probability + Anomaly / Reconstruction Signal

&#x20;                                            ▼

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐

│  TIER 5 — EXPLAINABLE AI ENGINE                                                                       │

│  Global SHAP (dataset-wide feature importance)  →  Local SHAP / LIME (per-shipment explanation)       │

└───────────────────────────────────────────┬──────────────────────────────────────────────────────────┘

&#x20;                                            │  Attribution Payload

&#x20;                                            ▼

┌────────────────────────────────────────────────────────────────────────────────────────────────────┐

│  TIER 6 — DECISION SUPPORT                                                                            │

│  Risk-Tiering Engine (Low / Medium / High)  →  Recommended Response (Reroute / Cool / QA Flag / OK)   │

└────────────────────────────────────────────────────────────────────────────────────────────────────┘

```



\### 2.3 Design Principles Governing the Architecture



1\. \*\*Explainability is a first-class output, not a post-hoc bolt-on.\*\* Tier 5 is architecturally mandatory before Tier 6 can fire an alert — no risk-tier decision is ever issued without an accompanying attribution payload.

2\. \*\*The pipeline is imbalance-aware at every tier\*\*, not only at the modeling stage. Schema validation flags target-conditional missingness (Tier 1), balancing is deferred to Tier 2 (not applied blindly at ingestion), and evaluation in Tier 4 never reports raw accuracy as a headline metric.

3\. \*\*Outliers are signal, not noise.\*\* Consistent with the empirical finding that failure rates exceed 50% among duration/temperature outliers, no tier silently drops outlier records; Tier 2 winsorizes only for linear-model compatibility while preserving the outlier's directional signal for tree-based learners.

4\. \*\*Tier boundaries are versioned contracts.\*\* Each tier consumes and emits a strictly typed schema (see Section 3.2), enabling independent CI/CD promotion of, e.g., a new Tier 4 model without touching Tier 1–3 code.



\---



\## 3. Data-Centric AI \& Preprocessing Specifications (Tiers 1 \& 2)



\### 3.1 Tier 1 — Ingestion \& Data Profiling



\*\*Objective:\*\* Guarantee that no downstream tier ever receives malformed, mistyped, or silently corrupted shipment data.



\*\*Functional Requirements:\*\*



\- \*\*FR-1.1 Type Validation.\*\* All 24 columns (`shipment\_id` + 22 features + `silent\_failure`) are validated against a strict Pydantic schema at ingestion. No implicit type coercion is permitted; a record failing type validation is quarantined, not auto-corrected, to prevent silent information loss.

\- \*\*FR-1.2 Identifier Handling.\*\* `shipment\_id` is validated for uniqueness (100% uniqueness expected across 8,000 records) and is excluded from all statistical/modeling feature sets, retained only as a join key for audit-trail reconstruction.

\- \*\*FR-1.3 Missingness Profiling.\*\* The profiler computes per-column missingness percentage and flags any column exceeding a 5% missingness threshold for special handling in Tier 2, rather than blanket row deletion. Empirically, only two columns require this treatment:



| Column | Missing Count | Missing % |

|---|---|---|

| `rh\_max` | 334 | 4.18% |

| `temp\_recovery\_rate` | 64 | 0.80% |



\- \*\*FR-1.4 Duplicate \& Consistency Checks.\*\* Full-row duplicate detection and physically-impossible-value range checks (e.g., `fill\_ratio` outside \[0,1]) are executed; the reference dataset shows zero duplicates and zero out-of-range values, but this check is mandatory in production regardless of the reference dataset's cleanliness.

\- \*\*FR-1.5 Target Distribution Audit.\*\* Every ingestion run logs the class balance of `silent\_failure`. A drift alert fires if positive-class prevalence deviates by more than ±5 percentage points from the training-time baseline of 19.68%, as this signals either a labeling pipeline fault or a genuine (and operationally significant) shift in failure incidence.



\*\*Tier 1 Output Contract:\*\* A \*Validated Tabular Dataset\* object — schema-locked, deduplicated, missingness-annotated — passed to Tier 2 with a machine-readable profiling report attached as metadata.



\### 3.2 Pydantic Data Schema (Illustrative)



```python

from pydantic import BaseModel, confloat, conint, constr

from typing import Optional



class ShipmentRecord(BaseModel):

&#x20;   shipment\_id: constr(strip\_whitespace=True, min\_length=1)

&#x20;   transit\_days: confloat(ge=0)

&#x20;   door\_opens: conint(ge=0)

&#x20;   leg\_count: conint(ge=1, le=3)

&#x20;   temp\_mean\_c: float

&#x20;   temp\_max\_c: float

&#x20;   temp\_min\_c: float

&#x20;   temp\_std\_c: confloat(ge=0)

&#x20;   temp\_recovery\_rate: Optional\[confloat(ge=0)]        # 0.80% missing — allowed

&#x20;   rh\_mean: confloat(ge=0, le=1)

&#x20;   rh\_std: confloat(ge=0)

&#x20;   rh\_max: Optional\[confloat(ge=0, le=1)]              # 4.18% missing — allowed

&#x20;   package\_type: conint(ge=0, le=2)

&#x20;   product\_volume\_l: confloat(gt=0)

&#x20;   fill\_ratio: confloat(ge=0, le=1)

&#x20;   carrier\_id: constr(strip\_whitespace=True)

&#x20;   origin\_zone: constr(strip\_whitespace=True)

&#x20;   dest\_zone: constr(strip\_whitespace=True)

&#x20;   sensor\_gap\_hours: confloat(ge=0)

&#x20;   vibration\_index: confloat(ge=0)

&#x20;   feature\_x1: float

&#x20;   feature\_x2: float

&#x20;   feature\_x3: float

&#x20;   silent\_failure: conint(ge=0, le=1)

```



\### 3.3 Tier 2 — DCAI Preprocessing \& Cleaning



\*\*Objective:\*\* Convert a validated-but-raw dataset into a model-ready, class-balanced dataset without diluting the predictive signal carried by missingness patterns or outliers.



\#### 3.3.1 Class-Conditional Median Imputation



Naïve global-median imputation is explicitly \*\*rejected\*\* for `rh\_max` and `temp\_recovery\_rate` because both variables are empirically correlated with the target class (`rh\_max` shows a Pearson r of 0.324 with `silent\_failure`). Replacing missing values with a single unconditional median would erase this signal by regressing all imputed values toward the majority-class center.



\*\*Specification — Class-Conditional Median Imputation (CCMI):\*\*



```

for column c in {rh\_max, temp\_recovery\_rate}:

&#x20;   median\_0 = median(c | silent\_failure == 0)

&#x20;   median\_1 = median(c | silent\_failure == 1)

&#x20;   for row r where c\[r] is missing:

&#x20;       c\[r] = median\_1 if silent\_failure\[r] == 1 else median\_0

```



At inference time (where the true label is unknown by definition), the imputation model instead uses a \*\*regression-based interpolation\*\* fallback: a lightweight regressor (e.g., ridge regression or k-NN regression, k=5) trained on the remaining 21 features predicts the missing value, conditioned on the correlated feature cluster (e.g., `rh\_max` is imputed using `rh\_mean`, `rh\_std`, and `temp\_mean\_c` as predictors, given the observed weak cross-family correlation between temperature and humidity groups).



\#### 3.3.2 Outlier Handling — Winsorization, Not Deletion



\*\*Empirical justification:\*\* IQR-based outlier detection identifies outlier rates ranging from 0.69% (`door\_opens`) to 9.36% (`transit\_days`). Because failure incidence is disproportionately concentrated among these outlier records, deletion would remove the exact population the model is designed to detect.



\*\*Specification:\*\*



\- Outlier bounds: `\[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` computed per numeric feature, per training fold (to prevent leakage).

\- \*\*Tree-based models (Tier 4 primary path):\*\* outliers are retained unmodified — gradient-boosted trees are split-based and are not distorted by extreme values in the same way distance- or gradient-based linear models are.

\- \*\*Linear/distance-based auxiliary models (e.g., logistic regression baselines, KNN benchmark):\*\* outliers are winsorized (capped at the IQR fence value) rather than dropped, preserving directional signal while bounding leverage.

\- No record is ever deleted solely on the basis of univariate outlier status.



\#### 3.3.3 Class Balancing — CGAN / DeepSMOTE



\*\*Problem:\*\* 19.68% positive prevalence (1,574 / 8,000) is significant enough to bias naive classifiers toward the majority class, but not so extreme (unlike, e.g., 1:1000 fraud detection) that pure class-weighting is guaranteed sufficient on its own, particularly for the minority sub-populations that fall near the decision boundary (moderate — not extreme — temperature range combined with elevated humidity).



\*\*Specification:\*\*



\- \*\*Primary technique — Conditional GAN (CGAN) oversampling:\*\* A CGAN is trained to generate synthetic minority-class (`silent\_failure = 1`) feature vectors, conditioned on the joint distribution of `temp\_range\_c`, `transit\_days`, and `door\_opens` (the three highest-correlation engineered/raw features). Synthetic samples are validated via a two-sample Kolmogorov–Smirnov test against the real minority distribution per-feature (target: p > 0.05, indicating no statistically significant distributional divergence) before being admitted to the training set.

\- \*\*Secondary/fallback technique — DeepSMOTE:\*\* For lower-latency retraining cycles where a full CGAN training run is not cost-justified, DeepSMOTE (SMOTE performed in a learned autoencoder latent space rather than raw feature space) is used to synthesize minority examples that respect non-linear feature manifolds — critical given the known non-normality of temperature/humidity distributions.

\- \*\*Balancing ratio:\*\* Target training-set balance is capped at 1:2 (minority:majority) rather than 1:1, based on the empirical finding that over-balancing synthetic minority data measurably degrades precision without a compensating recall gain in validation experiments; this ratio is a tunable hyperparameter subject to re-validation each retraining cycle.

\- \*\*Guardrail:\*\* All balancing is applied \*\*only\*\* to the training fold within each cross-validation split (Tier 4). Validation and test folds always retain the true, unaltered 19.68% prevalence to ensure reported metrics reflect real-world deployment conditions.



\*\*Tier 2 Output Contract:\*\* A \*Clean, Balanced Dataset\* — imputed, winsorized (where applicable), and class-augmented — passed to Tier 3.



\---



\## 4. Feature Engineering \& Multi-Sensor Analytics (Tier 3)



\### 4.1 Objective



Raw per-sensor summary statistics (mean, max, min temperature; mean humidity, etc.) are individually weak predictors of silent failure. Tier 3's function is to synthesize \*\*physically-motivated interaction and rate features\*\* that better encode the actual failure mechanism: cumulative, unmonitored, multi-modal environmental stress.



\### 4.2 Engineered Feature Specifications



\#### 4.2.1 `temp\_range\_c` — Temperature Range



\*\*Formula:\*\*

```

temp\_range\_c = temp\_max\_c − temp\_min\_c

```



\*\*Physical rationale:\*\* A shipment can maintain a compliant \*mean\* temperature while still experiencing wide excursions above and below setpoint. `temp\_range\_c` isolates excursion magnitude independent of the central tendency, and is empirically superior to `temp\_mean\_c` alone (r = 0.387 vs. failure) with only marginally lower correlation than `temp\_std\_c` (r = 0.392), while remaining more interpretable to non-technical QA reviewers than a standard deviation.



\#### 4.2.2 `excursion\_intensity` — Composite Excursion Severity



\*\*Formula:\*\*

```

excursion\_intensity = 0.4 × z(temp\_range\_c) + 0.6 × z(sensor\_gap\_hours)

```

where `z(·)` denotes the training-fold z-score standardization of the respective feature.



\*\*Physical rationale:\*\* This is the single most conceptually important engineered feature in the pipeline. It directly operationalizes the "silent failure" hypothesis: a temperature excursion is dangerous in proportion both to (a) its magnitude (`temp\_range\_c`) and (b) how long the shipment went \*\*unmonitored\*\* while it was occurring (`sensor\_gap\_hours`) — since an excursion caught immediately can be corrected, while one discovered after a long telemetry blackout cannot. The 0.6 weighting on monitoring gap versus 0.4 on magnitude reflects the empirical and domain-expert judgment that \*detection latency\*, not excursion size alone, is the dominant driver of irrecoverable product damage. `excursion\_intensity` shows a positive correlation with the target (r = 0.197) that exceeds several raw humidity features, despite being derived from only two inputs.



\#### 4.2.3 `vibration\_rate` — Duration-Normalized Mechanical Stress



\*\*Formula:\*\*

```

vibration\_rate = vibration\_index / transit\_days

```



\*\*Physical rationale:\*\* Raw `vibration\_index` conflates trip length with mechanical stress intensity — a long, smooth trip can accumulate a similar raw vibration count to a short, rough one. Normalizing by duration isolates stress \*intensity\*. Empirically, `vibration\_rate` is \*\*negatively\*\* correlated with failure (r = −0.263), consistent with the domain interpretation that higher-intensity, well-monitored express routes may in fact use more robust packaging/handling protocols than slower multi-leg routes — this counter-intuitive relationship is exactly the kind of insight Tier 5's explainability layer must surface transparently rather than suppress.



\#### 4.2.4 `rh\_range` — Humidity Range



\*\*Formula:\*\*

```

rh\_range = rh\_max − rh\_baseline

```

where `rh\_baseline` is the shipment-level minimum or setpoint-reference humidity reading captured in the raw telemetry stream. In the reference benchmark dataset this is pre-computed; in the production streaming pipeline it is derived per-shipment from the raw humidity time series minimum.



\*\*Physical rationale:\*\* Mirrors `temp\_range\_c` for the humidity channel. Elevated `rh\_range` correlates positively with failure (r = 0.176), consistent with the domain finding that humidity and temperature are only weakly cross-correlated (i.e., a humidity excursion is a partially independent failure pathway, not merely a proxy for temperature failure), and must therefore be engineered and modeled as a distinct signal rather than dropped in favor of temperature-only features.



\#### 4.2.5 `door\_open\_rate` — Rejected Transform (Documented Negative Result)



\*\*Formula considered:\*\*

```

door\_open\_rate = door\_opens / transit\_days

```



\*\*Finding:\*\* Contrary to the pattern observed for `vibration\_rate`, normalizing `door\_opens` by duration \*\*reduces\*\* predictive power relative to the raw count. The raw count of door-opening events is retained as a first-class feature rather than replaced by its rate-normalized counterpart, because a fixed number of door-openings on a short shipment carries genuinely higher risk in absolute terms — normalizing by duration disguises this by making short, high-touch shipments appear "average." This negative result is retained in the codebase and PRD explicitly to document that \*\*feature engineering is validated empirically, not assumed from prior domain intuition alone.\*\*



\### 4.3 Multicollinearity Management



\*\*Diagnosis:\*\* Pearson correlation analysis of the full feature set (Tier 1 profiling + Tier 3 engineered set) reveals two tightly-clustered feature families:



| Family | Members | Collinearity Pattern |

|---|---|---|

| Temperature | `temp\_mean\_c`, `temp\_max\_c`, `temp\_min\_c`, `temp\_std\_c`, `temp\_recovery\_rate`, `temp\_range\_c` | High pairwise correlation; `temp\_range\_c` ≈ `temp\_std\_c` with slightly lower variance |

| Humidity | `rh\_mean`, `rh\_std`, `rh\_max`, `rh\_range` | High pairwise correlation within family; weak cross-correlation with temperature family |



\*\*Filtering Rule (applied only to linear/distance-based auxiliary models, not to the production tree ensemble):\*\*



\- Compute Variance Inflation Factor (VIF) for all numeric features within each correlated family.

\- Retain at most \*\*one\*\* representative temperature feature (default: `temp\_range\_c`, selected over `temp\_std\_c` for interpretability) plus `temp\_mean\_c` as a location statistic, when the downstream model is linear.

\- Retain at most \*\*three of four\*\* humidity features (drop `rh\_std` by default, retaining `rh\_mean`, `rh\_max`, `rh\_range`) under the same condition.

\- Classifying attributes (`carrier\_id`, `origin\_zone`, `dest\_zone`) are explicitly exempted from collinearity filtering, as they show negligible correlation with all other feature families and are retained as low-signal-but-low-cost categorical context features.



\*\*Production note:\*\* Because Tier 4's primary models are gradient-boosted tree ensembles, which are largely robust to multicollinearity (correlated features simply share split importance rather than destabilizing coefficients), this filtering rule is applied \*\*only\*\* to auxiliary linear baselines and SHAP interpretation sanity-checks, not to the primary XGBoost/LightGBM/CatBoost training pipeline, to avoid discarding information the ensemble can safely exploit.



\*\*Tier 3 Output Contract:\*\* \*Feature Vectors\* (tabular, engineered) and \*Temporal Features\* (raw time-indexed sensor sequences, preserved for the Tier 4 Conv1D Autoencoder) are emitted as two parallel artifacts.



\---



\## 5. Modeling, Training \& Optimization (Tier 4)



\### 5.1 Training Regimen



\*\*FR-4.1 Stratified k-Fold Cross-Validation.\*\* All model training uses \*\*Stratified 5-Fold Cross-Validation\*\* (k=5, configurable), preserving the 19.68% positive-class prevalence in every fold. Class balancing (Tier 2, Section 3.3.3) is re-applied independently within each training fold only — never leaked across the validation boundary.



\*\*FR-4.2 Ensemble Composition.\*\* The production classification core is an ensemble of three gradient-boosted tree architectures, selected over the KNN and K-Means baselines for the reasons documented in Section 5.2:



| Model | Role | Key Configuration |

|---|---|---|

| XGBoost | Primary classifier | `scale\_pos\_weight` tuned to inverse class ratio; `max\_depth` 4–8 (grid-searched) |

| LightGBM | Secondary classifier / speed benchmark | Leaf-wise growth; native categorical handling for `carrier\_id`, `origin\_zone`, `dest\_zone` |

| CatBoost | Tertiary classifier / categorical-robustness benchmark | Ordered boosting to reduce prediction shift on categorical routing features |



Final production inference uses either the best single model (selected per validation cycle) or a soft-voting ensemble average, configurable via deployment flag.



\### 5.2 Baseline Justification — Why Not KNN or K-Means



| Model | Result | Verdict |

|---|---|---|

| K-Means (K=2), unsupervised | Silhouette Score = 0.274; Adjusted Rand Index = 0.2142 | \*\*Rejected.\*\* Confirms the failure/non-failure boundary is not recoverable from global geometric clustering alone — the classes are not globally separable, only locally/interactively separable. |

| K-Nearest Neighbors (KNN), supervised | Best F1 = 0.4922 (k=7); Best ROC-AUC = 0.8729 (k=31) | \*\*Rejected for production\*\*, retained as documented baseline. KNN establishes that supervised structure exists and is learnable, but is disqualified by: (1) O(N) query-time cost incompatible with real-time/edge inference latency targets (Section 8.2); (2) inherent sensitivity to class imbalance in the local neighborhood vote; (3) absence of a native, feature-level explainability mechanism required by Tier 5. |

| Gradient-Boosted Trees (XGBoost/LightGBM/CatBoost) | Target: F1 > 0.90, ROC-AUC > 0.92 (Section 8.1) | \*\*Selected for production.\*\* Native handling of class imbalance (`scale\_pos\_weight`/`class\_weights`), native non-linear feature-interaction learning without manual polynomial expansion, O(log N) inference via pre-built tree structures, and first-class, mathematically exact compatibility with TreeSHAP for Tier 5. |



\### 5.3 Threshold Tuning — Asymmetric Cost Optimization



\*\*Problem framing:\*\* In cold-chain pharmaceutical logistics, the cost of a \*\*False Negative\*\* (a genuine silent failure classified as safe, leading to a compromised product reaching a patient) is categorically higher than the cost of a \*\*False Positive\*\* (a safe shipment flagged for QA review, incurring inspection labor and minor delay). The default 0.5 probability threshold implicitly assumes symmetric costs and is therefore \*\*not acceptable\*\* for production deployment.



\*\*Specification:\*\*



\- An asymmetric cost matrix is defined at deployment configuration time, e.g.:



| | Predicted Safe | Predicted Failure |

|---|---|---|

| \*\*Actual Safe\*\* | Cost = 0 | Cost = 1 (inspection overhead) |

| \*\*Actual Failure\*\* | Cost = 20 (patient safety / recall / brand risk) | Cost = 2 (mitigation action cost) |



\- The classification threshold is selected, per retraining cycle, as:

```

θ\* = argmin\_θ  E\[cost(θ)]  subject to Recall(θ) ≥ 0.95

```

This is implemented as a constrained grid search over `θ ∈ \[0.05, 0.50]` on the held-out validation fold, selecting the highest-precision threshold that still satisfies the hard recall floor, rather than optimizing F1 or accuracy directly.

\- The resulting \*\*recall-biased threshold\*\* is expected to sit meaningfully below the naive 0.5 cutoff (empirically anticipated in the 0.15–0.30 range based on the class prior and cost ratio above), and this exact value is logged and versioned alongside each model artifact for audit reproducibility.



\### 5.4 Temporal Anomaly Modeling — Conv1D Autoencoder



\*\*Objective:\*\* The tabular ensemble (Section 5.1) operates on shipment-level summary statistics. A complementary \*\*Conv1D Autoencoder\*\* operates directly on the raw, time-indexed multi-sensor stream (temperature, humidity, vibration sampled at native telemetry frequency) to detect anomalous sub-sequences that summary statistics may average away — e.g., a brief but severe spike that is diluted in a shipment-long mean.



\*\*Specification:\*\*



\- \*\*Architecture:\*\* 1D convolutional encoder (3 conv blocks, stride-2 downsampling) → latent bottleneck → symmetric 1D transposed-convolutional decoder, trained to minimize reconstruction MSE on \*\*non-failure shipments only\*\* (i.e., trained as a one-class/normal-behavior model).

\- \*\*Inference signal:\*\* At inference time, reconstruction error on a new shipment's sensor stream is computed; elevated reconstruction error (above a validation-derived percentile threshold, e.g., 95th percentile of non-failure reconstruction error) is emitted as an \*\*Anomaly / Reconstruction Signal\*\*, fused as an additional input feature into the Tier 4 ensemble's final probability output (late-fusion architecture) rather than used as a standalone classifier.

\- \*\*Comparative benchmark — Xie et al. (2025) eiForest.\*\* The academic base paper proposes an improved Isolation Forest (eiForest) using subsampling and a cross-grouping factor (optimal r = 0.6) for multi-sensor anomaly detection, reporting average Precision = 0.8784, Recall = 0.8731, F1 = 0.8639, and AUC = 0.9064 on proprietary vehicle-mounted sensor data. The Conv1D Autoencoder is benchmarked head-to-head against a re-implemented eiForest baseline (using the same r = 0.6 cross factor configuration) on the temporal slice of the Kaggle benchmark dataset, with the explicit product goal of \*\*matching or exceeding eiForest's F1/AUC while additionally providing reconstruction-error-based localized explainability\*\* (i.e., identifying \*which time window\* within the shipment drove the anomaly score) — a capability the base paper's iForest approach does not natively provide.



\*\*Tier 4 Output Contract:\*\* A per-shipment \*Failure Probability\* (calibrated, threshold-annotated) and an \*Anomaly / Reconstruction Signal\*, both passed to Tier 5.



\---



\## 6. Explainable AI \& Audit Compliance (Tier 5)



\### 6.1 Rationale



Gradient-boosted ensembles are high-performing but are, by default, opaque. In a GxP-regulated pharmaceutical environment, a model that flags a shipment as high-risk \*\*must\*\* be able to state \*why\* in terms a QA reviewer can act on and a regulator can audit. Tier 5 is therefore architecturally mandatory, not optional, before any Tier 6 alert can be issued.



\### 6.2 Global Explainability



\- \*\*Method:\*\* TreeSHAP (exact, model-specific Shapley value computation for tree ensembles), computed dataset-wide on the validation set at the end of every training cycle.

\- \*\*Output:\*\* A ranked, dataset-wide feature importance report (mirroring the reference EDA's Table III correlation ranking, but causally grounded in Shapley values rather than raw Pearson correlation), used for (a) model documentation/validation sign-off, and (b) detecting feature-importance drift between retraining cycles as an early warning of data/concept drift.



\### 6.3 Local (Per-Shipment) Explainability



\- \*\*Primary method:\*\* TreeSHAP local explanations, generated for \*\*every\*\* shipment scored above the Medium-risk threshold (Section 7.2), decomposing the predicted probability into additive per-feature contributions.

\- \*\*Secondary/corroborating method:\*\* LIME, run as a model-agnostic cross-check on a sampled subset of high-risk predictions, to validate that TreeSHAP attributions are not an artifact of tree-structure quirks (e.g., surrogate-model agreement rate is tracked as a model-quality KPI; target agreement > 90% on top-3 attributed features).

\- \*\*Output format:\*\* A structured \*\*Attribution Payload\*\* per shipment:



```json

{

&#x20; "shipment\_id": "SHP-00481293",

&#x20; "failure\_probability": 0.81,

&#x20; "risk\_tier": "High",

&#x20; "top\_contributing\_features": \[

&#x20;   {"feature": "excursion\_intensity", "shap\_value": 0.24, "direction": "increases\_risk"},

&#x20;   {"feature": "door\_opens",          "shap\_value": 0.11, "direction": "increases\_risk"},

&#x20;   {"feature": "vibration\_rate",      "shap\_value": -0.06, "direction": "decreases\_risk"}

&#x20; ],

&#x20; "anomaly\_reconstruction\_score": 0.93,

&#x20; "model\_version": "xgb-ensemble-v2.4.1",

&#x20; "threshold\_used": 0.22,

&#x20; "explanation\_method": "TreeSHAP",

&#x20; "corroboration\_method": "LIME",

&#x20; "corroboration\_agreement": 0.94

}

```



\### 6.4 Regulatory \& Audit Compliance Mapping



| Requirement | Regulatory Basis | SentinelCold Implementation |

|---|---|---|

| Electronic records must be attributable, legible, contemporaneous, and accurate | FDA 21 CFR Part 11 | Every Attribution Payload is immutably logged with `shipment\_id`, `model\_version`, and `timestamp`, write-once to an append-only audit store |

| Computerized systems must have documented validation and change control | EU GxP Annex 11 | Every model promotion (Tier 4 retrain → production) is versioned; threshold, feature set, and training data snapshot hash are logged alongside the model artifact |

| Deviations must be investigatable with documented root cause | GxP Deviation Management | The Attribution Payload's `top\_contributing\_features` field serves as the auto-generated first-pass root-cause hypothesis for QA deviation investigations, reducing manual triage time |

| Audit trails must be tamper-evident | 21 CFR Part 11 §11.10(e) | Audit log store is hash-chained (each record includes the hash of the prior record) to detect post-hoc tampering |



\*\*Tier 5 Output Contract:\*\* A structured \*Attribution Payload\* (Section 6.3 schema) passed to Tier 6, and a persisted, immutable audit-log entry.



\---



\## 7. Decision Support \& Alerting Engine (Tier 6)



\### 7.1 Risk-Tiering Logic



Risk tiers are derived from the calibrated `failure\_probability` output of Tier 4, using tier boundaries that are \*\*re-calibrated per retraining cycle\*\* against the asymmetric-cost-optimized threshold (`θ\*`, Section 5.3) rather than fixed a priori:



| Risk Tier | Probability Band | Operational Meaning |

|---|---|---|

| \*\*Low\*\* | `p < θ\*` | Shipment proceeds without intervention; logged for model monitoring only |

| \*\*Medium\*\* | `θ\* ≤ p < θ\_high` (default `θ\_high = θ\* + 0.35`) | Shipment flagged for QA review at destination; no in-transit action required |

| \*\*High\*\* | `p ≥ θ\_high` | Immediate in-transit intervention evaluated (reroute / re-ice / expedite) |



\### 7.2 SHAP-to-Action Mapping Matrix



The Decision Support Engine does not merely report a risk tier — it inspects the \*\*dominant SHAP attribution(s)\*\* from the Tier 5 payload to select among differentiated recommended responses, ensuring the recommendation is causally grounded rather than a generic "high risk" alert.



| Dominant Attribution Pattern | Risk Tier | Recommended Response |

|---|---|---|

| `excursion\_intensity` and/or `temp\_range\_c` dominant | High | \*\*Reroute / Re-cool\*\* — active thermal intervention; cumulative unmonitored thermal exposure is the primary driver |

| `vibration\_index` / `vibration\_rate` dominant | High | \*\*QA Flag — Physical Handling Stress\*\* — route to physical inspection for packaging integrity, not thermal re-icing |

| `door\_opens` dominant | Medium | \*\*QA Flag — Handling Protocol Review\*\* — flag origin/handling facility for door-discipline audit |

| `rh\_range` / `rh\_max` dominant | Medium–High | \*\*QA Flag — Humidity Exposure\*\* — route to potency/stability re-test prior to release, independent of temperature status |

| `sensor\_gap\_hours` dominant (with low `temp\_range\_c`) | Medium | \*\*Sensor/Telemetry Audit\*\* — flag hardware/connectivity fault rather than a genuine product-condition failure, preventing unnecessary product quarantine |

| No single feature > 0.15 absolute SHAP value (diffuse attribution) | Low–Medium | \*\*Monitor / Log Only\*\* — probability elevated by combination of weak signals; insufficient causal concentration to justify intervention cost |

| Anomaly/Reconstruction Signal high, tabular probability moderate | Medium–High | \*\*Escalate for Manual Review\*\* — discordance between temporal and tabular models is itself flagged as a distinct, higher-priority review category |



\### 7.3 Alert Delivery \& Human-in-the-Loop Requirement



\- All \*\*High\*\*-tier alerts require human QA acknowledgment before shipment disposition is finalized; the system is decision-\*\*support\*\*, not decision-\*\*replacement\*\*, consistent with regulatory expectations for AI-assisted GxP decision-making.

\- \*\*Medium\*\*-tier alerts are queued for standard destination QA review workflow without in-transit escalation.

\- \*\*Low\*\*-tier shipments generate no operator-facing alert but are retained in the monitoring log for retrospective model-performance auditing (e.g., to catch false negatives once lab potency assay ground truth becomes available post-delivery).



\*\*Tier 6 Output Contract:\*\* A finalized \*Recommended Response\* record, delivered to the logistics/QA operations system of record (e.g., WMS/TMS integration), carrying the full risk tier, dominant attribution, and recommended action.



\---



\## 8. Technical, Performance \& Non-Functional Requirements



\### 8.1 Model Performance Targets



| Metric | Target | Rationale |

|---|---|---|

| F1-Score | > 0.90 | Exceeds both the KNN baseline (0.4922) and the Xie et al. (2025) eiForest academic baseline (0.8639) by a clear, defensible margin |

| ROC-AUC | > 0.92 | Exceeds the KNN baseline (0.8729) and the eiForest baseline (0.9064) |

| Recall (Sensitivity) | > 0.95 | Reflects the asymmetric cost matrix (Section 5.3); a missed silent failure is categorically less acceptable than an unnecessary QA flag |

| Precision | Reported, not gated | Monitored to ensure the recall-first threshold does not degrade operational usability below a documented floor (target: Precision > 0.55 at the deployed threshold, to bound QA review workload) |

| PR-AUC | Reported per retrain | Primary metric for tracking model quality across retraining cycles given persistent class imbalance; more informative than ROC-AUC under skewed priors |



\### 8.2 Latency \& Scalability



\- \*\*Inference latency:\*\* < 100 ms per shipment for real-time/edge scoring (single-shipment tabular ensemble inference), enabling in-transit re-scoring as new telemetry batches arrive rather than only at shipment completion.

\- \*\*Temporal model latency:\*\* Conv1D Autoencoder inference is budgeted separately at < 250 ms per shipment sensor-stream window, run asynchronously and fused into the final probability without blocking the primary tabular inference path.

\- \*\*Throughput:\*\* The batch scoring path must sustain ≥ 500 shipments/second on reference production hardware to support fleet-scale, near-real-time monitoring.

\- \*\*Horizontal scalability:\*\* Tiers 2–5 are stateless per-shipment and horizontally scalable behind a load balancer; only Tier 4 model training (not inference) requires GPU/high-memory batch infrastructure.



\### 8.3 Deployment \& Offline Capability



\- \*\*Edge deployment:\*\* The Tier 4 tabular ensemble must be exportable to a lightweight runtime (e.g., ONNX) capable of running on constrained in-vehicle/edge gateway hardware without requiring a live connection to cloud infrastructure, given that cold-chain vehicles frequently traverse network-degraded routes (tunnels, rural corridors, sea transit).

\- \*\*Offline-first design:\*\* Edge nodes buffer scored predictions and Attribution Payloads locally and sync to the central audit store upon reconnection; no shipment-level decision is blocked on network availability.

\- \*\*Model versioning:\*\* Every deployed model artifact (tabular ensemble + autoencoder) is immutably versioned and hash-referenced in the Attribution Payload (Section 6.3), enabling exact reproducibility of any historical prediction for audit purposes.



\### 8.4 Data Privacy \& Security



\- Shipment and routing metadata (`carrier\_id`, `origin\_zone`, `dest\_zone`) are treated as commercially sensitive and are access-controlled separately from raw sensor telemetry.

\- No patient-identifying or personally identifiable information is present in or required by the modeling pipeline; the system operates exclusively on shipment- and sensor-level data.

\- All audit-log and Attribution Payload storage is encrypted at rest and in transit, consistent with GxP computerized-system security expectations (Section 6.4).



\### 8.5 Monitoring \& Retraining Cadence



\- \*\*Drift monitoring:\*\* Tier 1's target-distribution audit (Section 3.1, FR-1.5) and Tier 5's global SHAP importance tracking (Section 6.2) jointly serve as the primary concept-drift early-warning system.

\- \*\*Retraining cadence:\*\* Full pipeline retraining (Tiers 2–5) is scheduled quarterly at minimum, or immediately upon a drift alert, with all retraining runs subject to the same Stratified k-Fold validation and asymmetric-threshold recalibration procedures defined in Section 5.



\---



\## 9. Appendix: Glossary, Data Dictionary \& References



\### 9.1 Glossary



\- \*\*Silent Failure:\*\* A cold-chain excursion event that does not breach a static alarm threshold but nonetheless compromises product potency, detectable only through cumulative, multi-sensor, interaction-based analysis.

\- \*\*Excursion:\*\* Any deviation of a monitored environmental variable (temperature, humidity) from its validated setpoint range.

\- \*\*Winsorization:\*\* A statistical technique that caps extreme values at a defined percentile/IQR fence rather than removing them, preserving record count and directional signal.

\- \*\*CGAN:\*\* Conditional Generative Adversarial Network; used here to synthesize realistic minority-class training examples conditioned on key correlated features.

\- \*\*TreeSHAP:\*\* An exact, polynomial-time algorithm for computing Shapley additive feature attributions specific to tree-ensemble models.

\- \*\*eiForest:\*\* Enhanced/Improved Isolation Forest, per Xie et al. (2025), using subsampling and a cross-grouping factor to overcome swamping/masking limitations of classical Isolation Forest in high-dimensional anomaly detection.



\### 9.2 Core Data Dictionary (Reference Benchmark Dataset)



| Group | Representative Variables |

|---|---|

| Identifier | `shipment\_id` |

| Transit / Handling | `transit\_days`, `door\_opens`, `leg\_count` |

| Temperature | `temp\_mean\_c`, `temp\_max\_c`, `temp\_min\_c`, `temp\_std\_c`, `temp\_recovery\_rate` |

| Humidity | `rh\_mean`, `rh\_std`, `rh\_max` |

| Packaging / Product | `package\_type`, `product\_volume\_l`, `fill\_ratio` |

| Routing | `carrier\_id`, `origin\_zone`, `dest\_zone` |

| Sensor / Motion | `sensor\_gap\_hours`, `vibration\_index` |

| Anonymized Covariates | `feature\_x1`, `feature\_x2`, `feature\_x3` |

| Target | `silent\_failure` (binary: 0 = safe, 1 = failure) |



\### 9.3 References



1\. World Health Organization — vaccine wastage estimate (as cited in Patheon/Thermo Fisher Scientific industry brief).

2\. European Pharmaceutical Manufacturer — annual pharmaceutical cold-chain revenue loss estimate.

3\. Ribeiro, M.T., Singh, S., \& Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. \*KDD 2016\*.

4\. Lundberg, S.M., \& Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. \*NeurIPS 2017\*.

5\. Xie, Z., Long, H., Ling, C., Zhou, Y., \& Luo, Y. (2025). An Anomaly Detection Scheme for Data Stream in Cold Chain Logistics. \*PLOS ONE\*, 20(3), e0315322.

6\. Kaggle — Cold Chain Shipment Silent Failure Dataset.

7\. Dhandia, U. (2026). Predictive Cold Chain Disruption Modeling for Pharmaceutical Logistics: An Exploratory Data Analysis Report \& Literature Review. Christ University.



\---



\*End of Document — SentinelCold PRD v1.0\*

