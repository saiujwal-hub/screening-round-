# CPRI State-Level Hackathon — Screening Round Methodology Note
**Challenge**: The Black-Box Test Bench Challenge: Can You Discover What the Data Is Telling You?  
**Organizers**: Central Power Research Institute (CPRI) & MIT Bengaluru  
**Team Name**: PowerNext Alpha  
**Date**: September 2026  

---

## 1. Executive Summary & Approach Overview
In electrical apparatus testing (transformers, switchgear, busbars), accurate characterization of critical hotspot temperature rise is vital for thermal insulation life estimation and rating validation. The test bench dataset presents realistic experimental challenges: high-voltage/high-current regime transitions, duplicate runs, random measurement noise, probe dropouts, and intermittent sensor spikes.

Rather than treating the test bench as a naive statistical black box, our team deployed a **Physics-Informed Machine Learning (PIML)** methodology consisting of three interconnected stages:
1. **Tiered Deterministic Anomaly Engine (Task 1)**: Formulated physical boundaries for thermal conduction across electrical terminals ($S_1, S_2, S_3$), separating genuine high-load operational transitions from measurement errors and duplicated runs.
2. **Virtual Sensor Reconstruction & Hybrid Ensemble (Task 2)**: For records with corrupted/missing sensor channels, virtual sensor values were estimated via steady-state thermal conduction mappings. A blended ensemble of Gradient Boosting Regressors, XGBoost, and LightGBM was trained incorporating Joule heating physics ($P \propto I^2$) to predict hotspot temperature rise (`Reference_Parameter`) with $R^2 > 0.992$.
3. **Automated Fleet-Level Telemetry & Risk Scoring (Task 3)**: Developed an automated diagnostics module that extracts fleet statistics, classifies failure modes, and flags high-risk units requiring urgent inspection.

---

## 2. Parameter Importance & Physical Relationships

Through empirical correlation and thermodynamic sensitivity analysis on verified tests, parameters were classified into fundamental thermal drivers, local state indicators, and uninformative signals. To rigorously evaluate predictive utility without assuming strict linearity, an independent Random Forest regressor was trained on all 8 raw features (`Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`, and `Sensor_S1` through `Sensor_S4`) against `Reference_Parameter` using only Valid training records:

```
+---------------------------------------------------------------------------------------------+
| Feature (Raw 8 Inputs)  | Physical Interpretation                 | Pearson r | RF Importance|
+---------------------------------------------------------------------------------------------+
| Load_Current_A (I)      | Dominant heat source (Joule losses I²R) |  +0.8767  | 90.03%       |
| Sensor_S2               | Load-side terminal temperature rise     |  +0.7953  |  5.99%       |
| Ambient_Temperature_C   | Thermal boundary / reference temp       |  +0.2377  |  3.19%       |
| Sensor_S1               | Incoming terminal temperature rise      |  +0.5914  |  0.22%       |
| Sensor_S3               | Critical body temperature rise          |  +0.5012  |  0.18%       |
| Test_Duration_min       | Thermal transient duration (time to t_ss)| -0.0042  |  0.13%       |
| Sensor_S4 (auxiliary)   | Auxiliary channel (uncorrelated noise)  |  -0.0086  |  0.13%       |
| Applied_Voltage_kV (V)  | Dielectric stress / core loss component |  +0.2666  |  0.12%       |
+---------------------------------------------------------------------------------------------+
```

### Key Physical Insights:
1. **Joule Dissipation Dominance**: Thermal dissipation is fundamentally quadratic in current ($P = I^2 R$). Current alone exhibits an $87.7\%$ linear correlation with hotspot rise, jumping to $92.7\%$ when transformed to $I^2$.
2. **Thermal Conduction to Load Terminal ($S_2$)**: Sensor $S_2$ shows the closest thermal coupling to the internal hotspot ($r = 0.795$, $5.99\%$ RF importance). As current flows to the load side, $S_2$ acts as the direct thermal conduction pathway from the internal winding hotspot.
3. **Irrelevance and Pruning of Auxiliary Sensor $S_4$**: Both the negligible linear Pearson correlation ($r = -0.0086$) and the bottom-tier nonlinear Random Forest feature importance ($0.13\%$, indistinguishable from baseline noise and tied for lowest across all sensors) provide joint empirical evidence that $S_4$ carries no physical signal or predictive utility regarding hotspot temperature rise, justifying its deliberate exclusion to eliminate noise variance and prevent overfitting on hidden test distributions.

---

## 3. Abnormal Data Detection Methodology (Task 1)

A central objective of the challenge is **distinguishing physical regime shifts from measurement anomalies**. When equipment is subjected to heavy loading ($I > 85\text{ A}$, $V > 25\text{ kV}$), temperatures rise non-linearly; this is genuine equipment behaviour, not an anomaly. Conversely, an anomaly represents unphysical readings, hardware loss, or duplicate data.

We established a **Five-Tier Physics-Grounded Anomaly Engine**:
$$\text{Invalid} = \mathcal{M}_{\text{Missing}} \cup \mathcal{M}_{\text{Duplicate}} \cup \mathcal{M}_{\text{Negative}} \cup \mathcal{M}_{\text{Cross}} \cup \mathcal{M}_{\text{Spike}}$$

1. **Tier 1 — Essential Channel Dropout ($\mathcal{M}_{\text{Missing}}$)**: Records where $S_1, S_2,$ or $S_3$ contain missing ($NaN$) entries. An uninstrumented terminal cannot validate equipment compliance.
2. **Tier 2 — Duplicate Test Runs ($\mathcal{M}_{\text{Duplicate}}$)**: Records possessing identical operating input vectors $[V, I, T_{amb}, t]$. Historical audit revealed 100% of duplicate runs were erroneous test repetitions or data-logger logging collisions.
3. **Tier 3 — Physical Plausibility Rule ($\mathcal{M}_{\text{Negative}}$)**: In an active electrical test with continuous current injection, heat dissipation is strictly non-negative; therefore, temperature rise above ambient cannot be negative ($S_1 < 0 \lor S_2 < 0 \lor S_3 < 0$). This rule caught anomalous negative measurements (e.g. `TST-0213`, where $S_2 = -1.55^\circ\text{C}$).
4. **Tier 4 — Cross-Sensor Thermodynamic Coupling ($\mathcal{M}_{\text{Cross}}$)**: Under quasi-steady heat transfer, adjacent terminal sensors exhibit tight physical coupling ($S_3 \approx 1.27 S_1$, $R^2 = 0.983$). We fit cross-sensor linear models ($\hat{S}_3(S_1), \hat{S}_2(S_1)$) on valid training records; deviations exceeding $1.25\times$ baseline maximum cross-residuals indicate localized probe faults.
5. **Tier 5 — Physical Conduction Residuals ($\mathcal{M}_{\text{Spike}}$)**: Under thermal conduction laws, terminal temperature rises follow deterministic relations with operating conditions:
   $$\hat{S}_i = \beta_{i,0} + \beta_{i,V} V + \beta_{i,I} I + \beta_{i,T} T_{amb} + \beta_{i,t} t$$
   Residual errors $|S_i - \hat{S}_i|$ on valid data are strictly bounded by normal measurement tolerances: $\tau_{S1} = 0.71^\circ\text{C}$, $\tau_{S2} = 0.55^\circ\text{C}$, $\tau_{S3} = 1.04^\circ\text{C}$. Any record where residual error exceeds $1.25 \times \tau_i$ represents a sensor spike/loose thermocouple, without penalizing high-current operational regime shifts.

> **Leakage-Free 5-Fold Cross-Validation Evaluation**:
> In each fold, single-sensor baselines ($\beta$), cross-sensor models, and residual thresholds ($\tau$) were established **strictly on the training fold's Valid records** (80% split), and then evaluated blindly on the **held-out validation fold** (20% split, containing both unseen Valid and Invalid records).
>
> **Mode A: Deployment Setting (Batch & Test History Tracking)**:
> | Fold | Accuracy | Precision (Invalid) | Recall (Invalid) | F1-Score | Threshold $\tau_{S1}$ | Threshold $\tau_{S2}$ | Threshold $\tau_{S3}$ |
> | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
> | **Fold 1** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8744°C | 0.6507°C | 1.2642°C |
> | **Fold 2** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8906°C | 0.6787°C | 1.2996°C |
> | **Fold 3** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8911°C | 0.6807°C | 1.3021°C |
> | **Fold 4** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7601°C | 0.6857°C | 1.3204°C |
> | **Fold 5** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8690°C | 0.6778°C | 1.3101°C |
> | **Mean ± Std** | **1.0000 ± 0.00** | **1.0000 ± 0.00** | **1.0000 ± 0.00** | **1.0000 ± 0.00** | 0.8570°C | 0.6747°C | 1.2993°C |
>
> **Mode B: Isolated Slice Setting (strictly within 200-row validation slice without history matching)**:
> - **Mean Accuracy**: `0.9800 ± 0.0055`
> - **Mean Precision**: `1.0000 ± 0.0000` (Zero false alarms on valid regime shifts)
> - **Mean Recall**: `0.8471 ± 0.0467`
> - **Mean F1-Score**: `0.9165 ± 0.0268`
>
> *Diagnostic Note on Recall*: In Mode B, the 15.3% unflagged invalid records correspond precisely to the 20 duplicate measurement pairs whose twin fell into the other fold. Because their physical sensor values are unspiked ($r < \tau$), they can only be identified when checked against prior test history (Mode A), where recall reaches 100.0%.
>
> Following validation, the **final deployed production model** was fitted on the full training dataset to perform inference on unlabelled data, flagging **46 abnormal records (13.14%)** in `Test_Data` (closely matching the historical failure rate of $13.40\%$).

### 3.1 Defensive Ingestion, Parameter Imputation & Synthetic Robustness Stress Testing
To guarantee robust operation on hidden or schema-drifted evaluation datasets:
1. **Defensive Schema Validation**: The ingestion module strictly verifies workbook structure against required sheets (`Training_Data`, `Test_Data`) and mandatory feature columns. If a sheet or column is missing, the pipeline raises a loud, descriptive `ValueError` detailing the exact mismatch, preventing silent crashes or garbage predictions.
2. **Missing Operating Parameter Handling**: If any operating parameter ($V, I, T_{amb}, t$) is missing, the engine:
   - Computes column medians strictly from training-fold Valid records.
   - Imputes missing entries so downstream linear and tree regressors receive clean inputs without generating NaNs.
   - Flags the record as `Invalid` (missing critical operating parameters render test validity unprovable per CPRI standards).
3. **Synthetic Robustness Stress Testing**: We verified the pipeline under severe fault injection by randomly nulling out 5% of entries across each operating parameter column and injecting 5% duplicate operating condition rows (expanding the test set from 350 to 367 records). The pipeline achieved **8/8 PASS status**:
   - 100% completion without crashes or unhandled exceptions.
   - Zero NaN predictions or missing validity labels.
   - 100% of records with missing operating parameters flagged as `Invalid` (62/62).
   - 100% of injected duplicate records flagged as `Invalid` (17/17).
   - Physical prediction plausibility preserved across all stressed records ($12.95^\circ\text{C} \le \Delta T_{pred} \le 56.35^\circ\text{C}$).

---

## 4. Key Engineering Assumptions & Boundary Conditions
1. **Steady-State Approximations**: Test durations in the dataset ($15\text{--}60\text{ min}$) are sufficient for compact terminal assemblies to approach quasi-steady thermal equilibrium relative to rapid current changes.
2. **Linear Conduction Coupling**: Heat flow between internal hotspot and external terminals ($S_1, S_2$) adheres to Fourier's law with approximately constant thermal conductivity over the operating range ($20^\circ\text{C} \le T_{amb} \le 55^\circ\text{C}$).
3. **Reference Calibration Integrity**: Historical Reference Parameters were gathered with calibrated laboratory instruments and represent ground truth for valid thermal models.

---

## 5. Digital Twin Automation Roadmap

To transition this black-box laboratory test bench into a continuous, real-time **Automated Digital Twin**, the following four-stage architecture must be deployed:

```
[Physical Test Bench] ---> (Edge Ingestion & Validation) ---> (Dynamic Thermal Twin Engine) ---> (Closed-Loop Supervisory)
  - Thermocouples S1-S3       - MQTT / OPC-UA Pub/Sub          - State-Space Kalman Filter       - Auto Derating / Trip
  - CTs & PTs (V, I)          - Tier 1-3 Anomaly Gate          - Ensemble Hotspot Predictor      - Predictive Maintenance
  - Weather Station (Tamb)    - Virtual Sensor Imputation      - Arrhenius Life Loss Tracker     - CPRI Test Certificate Gen
```

### Step 1: Real-Time Edge Data Ingestion & Signal Quality Gate
* **Protocol & Sampling**: Instrument test benches with edge micro-controllers running OPC-UA / MQTT protocols streaming telemetry at 1–5 Hz.
* **Inline Anomaly Filtering**: Embed the Tier 1–3 anomaly logic at the edge gateway. Real-time sensor spikes trigger automatic sensor health alarms, and missing packets trigger automatic communication re-sync before data enters the model.

### Step 2: Virtual Sensor Imputation & State Estimation
* **Kalman Filtering / Physics Estimator**: When a thermocouple experiences contact resistance degradation or disconnects, the digital twin activates the trained virtual sensor estimator ($\hat{S}_i$), maintaining continuous monitoring without interrupting high-voltage test sequences.

### Step 3: Physics-Informed Hotspot Thermal Tracking & Insulation Aging
* **Continuous Thermal Tracking**: The digital twin executes the trained ensemble model in microsecond inference time to compute instantaneous hotspot temperature $T_{hs}(t) = T_{amb}(t) + \Delta T_{pred}(t)$.
* **Arrhenius Degradation Modeling**: Hotspot rise is fed into IEEE/IEC thermal aging equations ($V = 2^{(T_{hs} - 110)/6}$) to dynamically compute instantaneous loss-of-life and insulation degradation rate.

### Step 4: Closed-Loop Automation, Dashboarding & Certification
* **Automated Test Certification**: Upon test cycle completion, the digital twin automatically compiles the summary telemetry (validity audit, peak hotspot, loss of life) and outputs verified test reports conforming to CPRI standards.
* **Supervisory Overheat Protection**: If predicted hotspot temperature exceeds design thermal thresholds ($T_{hs} > 110^\circ\text{C}$), the digital twin automatically signals PLC interlocks to derate current or trigger safety trip relays.
