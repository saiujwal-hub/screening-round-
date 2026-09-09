# CPRI State-Level Hackathon — Screening Round Methodology Note
**Challenge**: The Black-Box Test Bench Challenge | **Team**: og | **Date**: September 2026  
**Team Members**: Sai Ujwal Meesala (Team Leader), Yuvan Reddy Vadde, Samartha Dayananda, Akash Kulkarni, Kaavya Janagan

---

## 1. Executive Summary & Approach Overview
In electrical apparatus testing (transformers, switchgear, busbars), predicting critical hotspot temperature rise while filtering measurement anomalies is essential for insulation life evaluation. We deployed a **Physics-Informed Machine Learning (PIML)** framework integrating electrical conduction physics, multi-tier anomaly detection, and ensemble regression. Task 1 deploys a deterministic multi-tier anomaly engine separating high-load operational transitions from sensor defects and duplicate logs, validated via leakage-free 5-fold cross-validation. Task 2 reconstructs corrupted sensor channels via thermodynamic baselines and trains a blended Gradient Boosting, XGBoost, and LightGBM ensemble on Joule dissipation (P ∝ I²), achieving cross-validated R² = 0.9928 and RMSE = 0.896 °C.

---

## 2. Parameter Importance & Physical Relationships
On verified valid records, parameter contributions to hotspot rise (`Reference_Parameter`) were characterized across 8 raw inputs:
- **Load_Current_A (I)**: Dominant thermal driver (r = +0.8767, 90.03% Random Forest importance) governing quadratic Joule heating losses (P ∝ I², r = +0.927).
- **Sensor_S2**: Direct conductive thermal coupling to hotspot (r = +0.7953, 5.99% RF importance), acting as the primary load-side conduction path.
- **Ambient_Temperature_C (T_amb)**: Reference thermal boundary condition (r = +0.2377, 3.19% RF importance).
- **Sensor_S1 & Sensor_S3**: Terminal and critical body rises providing spatial conduction constraints (r = +0.5914 / +0.5012, 0.22% / 0.18% RF importance).
- **Applied_Voltage_kV (V) & Test_Duration_min (t)**: Secondary dielectric loss and transient timing (r = +0.2666 / -0.0042, 0.12% / 0.13% RF importance).
- **Auxiliary Sensor_S4**: Both the negligible linear Pearson correlation (r = -0.0086) and bottom-tier nonlinear Random Forest importance (0.13%, tied for lowest across all sensors) provide joint empirical evidence that S4 carries no physical signal or predictive utility regarding hotspot rise, justifying its deliberate pruning to eliminate noise variance.

---

## 3. Abnormal Data Detection Methodology (Task 1)
We distinguish operational regime shifts (e.g. non-linear heating under I > 85 A, V > 25 kV) from equipment defects using a **Multi-Tier Physics-Grounded Anomaly Engine**:

<div class="equation-box">Invalid = M<sub>MissingOp</sub> ∪ M<sub>MissingSensor</sub> ∪ M<sub>Duplicate</sub> ∪ M<sub>Negative</sub> ∪ M<sub>Cross</sub> ∪ M<sub>Spike</sub></div>

1. **Operating Parameter Missingness (M_MissingOp)**: Missing operating inputs (V, I, T_amb, t) render tests unprovable per challenge rules; they are imputed with training medians for safe inference and flagged `Invalid`.
2. **Sensor Dropout (M_MissingSensor)**: Missing entries (NaN) in terminal probes S1, S2, S3.
3. **Duplicate Test Runs (M_Duplicate)**: Identical operating vectors [V, I, T_amb, t] arising from data-logger collisions or test repetitions.
4. **Physical Plausibility Rule (M_Negative)**: Active load testing cannot yield negative temperature rise above ambient (S_i < 0), catching unphysical readings (e.g. `TST-0213`, S2 = -1.55 °C).
5. **Cross-Sensor Consistency (M_Cross)**: Gross deviations from linear thermodynamic coupling between terminal probes (S3 ≈ 1.27 · S1, R² = 0.983; τ_cross = 1.25 × max_res).
6. **Conduction Residuals (M_Spike)**: Residual deviations exceeding 1.25× baseline conduction tolerances from S_pred(V, I, T_amb, t) (τ_S1 = 0.86 °C, τ_S2 = 0.67 °C, τ_S3 = 1.30 °C).

![Thermal Response and Conduction Residuals vs. Load Current](task1_regime_vs_anomaly.png)
*Figure 1: High operating load (I > 85 A) follows the deterministic physical heating curve rather than an anomaly, whereas true anomalies (sensor spikes, dropouts, duplicates) are scattered completely independent of load level.*

<div style="page-break-before: always;"></div>

### Cross-Validation & Robustness Verification
- In strict leakage-free 5-fold CV evaluated out-of-fold without cross-batch history matching (the exact deployed setting, as zero of 350 `Test_Data` records share operating conditions with training data), the detector achieves **Accuracy: 98.0%, Precision: 100.0%, Recall: 84.7%, and F1: 0.9165**. On `Test_Data`, it identified **46 abnormal records (13.14%)**, matching historical defect rates (13.40%). The two new detection tiers (physical plausibility, cross-sensor consistency) did not change any labels on the actual 350-record `Test_Data` submission versus the original single-tier check — they exist as defensive depth for the hidden/second dataset, not as a demonstrated improvement on this specific submission.
- **Defensive Robustness Stress Test**: Under synthetic fault injection (5% NaNs per operating column + 5% duplicates; 367 records), the pipeline achieved **8/8 PASS status**, ensuring zero NaN predictions and 100% correct invalid flagging.

### Known Limitations & Next Steps
Detailed failure analysis reveals that **100% of false negatives (20/20 missed records across 5 CV folds) are duplicate test runs whose twins were partitioned into the training fold**; in isolated slices, these solitary twins exhibit physically valid telemetry (|S_i - S_pred_i| ≤ 0.36 °C) with zero single-record anomalies. Concretely testing persistent historical streaming across the 1000 records confirms recall jumps from 84.7% to 91.04% at 100% precision (catching the recurrence of every duplicate pair). On `Test_Data`, our batch detector already captures 100% of internal duplicates (8/8 records), while cross-file logging adds 0 detections due to zero operating overlap. Future work on isolated single-record deployments will evaluate supervised classifiers or non-linear polynomial baselines for complex cooling regimes.

---

## 4. Key Engineering Assumptions & Boundary Conditions
1. **Quasi-Steady Thermal Equilibrium**: Test durations (15–60 min) allow compact busbar/terminal assemblies to approach steady-state temperature rises relative to electrical transient time constants.
2. **Piecewise Linear Thermal Conduction**: Heat transfer from internal hotspot to external terminals adheres to Fourier's conduction law with effectively constant thermal conductivity over 20°C ≤ T_amb ≤ 55°C.
3. **Reference Instrumentation Calibration**: Historical reference measurements reflect calibrated ground truth; uninstrumented or incomplete operating parameters violate test compliance and indicate invalid test runs.

---

## 5. Digital Twin Automation Roadmap
To transition this black-box laboratory test bench into a continuous, real-time **Automated Digital Twin**:
1. **Edge Telemetry Ingestion & Persistent History Quality Gate**: Micro-controllers stream telemetry (OPC-UA/MQTT) through inline filters maintaining a persistent cache of seen operating vectors [V, I, T_amb, t]. Concretely validated in testing, persistent sequential logging elevates anomaly recall from 84.7% to 91.04% at 100% precision (and 100% with retrospective back-tagging).
2. **Virtual Sensor Imputation & State Estimation**: When a physical probe experiences contact degradation or fails, steady-state thermal regression estimators reconstruct virtual sensor values (S_pred), enabling uninterrupted test continuity.
3. **Physics-Informed Real-Time Thermal Tracking**: The trained ensemble executes sub-millisecond inference to track instantaneous hotspot rise, feeding Arrhenius loss-of-life equations (V = 2^((T_hs - 110)/6)) to log dynamic insulation degradation.
4. **Closed-Loop Supervisory Protection & CPRI Certification**: Automated compliance reporting compiles certified test sheets, while real-time thermal limits trigger automated PLC current derating or safety trip interlocks if hotspot temperatures exceed safety margins (T_hs > 110 °C).
