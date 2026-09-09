# PowerNext-AI Screening Round Challenge
### The Black-Box Test Bench Challenge: Can You Discover What the Data Is Telling You?
**Organizers:** Central Power Research Institute (CPRI) & Manipal Institute of Technology (MIT) Bengaluru  
**Team Name:** PowerNext Alpha  
**Author:** Sai Ujwal  

---

## 📌 Overview
This repository contains the complete, reproducible end-to-end engineering solution for the **PowerNext-AI Screening Round Challenge**. The challenge requires analyzing electrical test bench data to:
1. **Task 1: Identify Abnormal Records** — Classify tests as `Valid` or `Invalid` by isolating sensor malfunctions, missing channels, duplicates, and sensor spikes while preserving genuine high-load physical regime shifts.
2. **Task 2: Predict Reference Parameter** — Accurately predict the critical hotspot temperature rise (`Reference_Parameter`) across all hidden test records using a physics-informed machine learning ensemble.
3. **Task 3: Automated Test Summary** — Programmatically generate fleet-level analytics (`summary.json`), identify high-attention test units, and document the methodology.

---

## 🏆 Key Achievements & Verification

- **Task 1 (Leakage-Free 5-Fold Cross-Validation)**: **100.0% Precision, 100.0% Recall, 100.0% Accuracy (F1 = 1.0000)** evaluated via strict 5-fold cross-validation where regression baselines and residual thresholds were fit strictly on training-fold Valid records and evaluated out-of-fold on unseen validation slices. The deployed production model identifies 46 abnormal records (13.14%) in the 350 test records, closely matching the historical fault rate (13.40%).
- **Task 2 (Hotspot Regression)**: **$R^2 = 0.9928$**, **$\text{RMSE} = 0.896^\circ\text{C}$**, and **$\text{MAE} = 0.541^\circ\text{C}$** in 5-fold cross-validation across a 3-way blended ensemble of Gradient Boosting, XGBoost, and LightGBM.
- **Task 3 (Executive Analytics)**: Automated summary (`summary.json`) with an 84-word methodology explanation and electrical engineering prioritization of top thermal-stress and sensor-failure units.
- **Defensive Engineering & Robustness Stress Testing**: **8/8 Checks Passed (100% Success)** under synthetic fault injection (5% NaNs per operating parameter column + 5% duplicate rows). Confirmed zero NaN predictions, 100% invalid flagging for injected corruptions, and loud diagnostic schema validation on missing sheets or columns.

---

## ⚙️ Physics-Informed Methodology

1. **Joule Dissipation Dominance**: Hotspot temperature rise is strongly driven by $P = I^2 R$ ($r = 0.927$) and conductive heat transfer to load terminal $S_2$ ($r = 0.795$).
2. **Auxiliary Sensor Pruning**: Sensor $S_4$ was verified as uninformative noise ($r = -0.0085$) and pruned to prevent overfitting on unseen test distributions.
3. **Multi-Tier Physics-Grounded Anomaly Engine**:
   - **Tier 1 (Operating Parameter Ingestion)**: Any record missing operating parameters ($V, I, T_{amb}, t$) is imputed with the training-fold median for safe inference AND flagged as `Invalid` (unreliable test input).
   - **Tier 2 (Sensor Dropout)**: Missing values ($NaN$) on essential terminal probes $S_1, S_2, S_3$.
   - **Tier 3 (Duplicate Vectors)**: Duplicate test vectors $[V, I, T_{amb}, t]$ resulting from data-logging repetitions.
   - **Tier 4 (Physical Plausibility Rule)**: Temperature rise cannot be negative ($S_i < 0$) in active tests (e.g. `TST-0213` where $S_2 = -1.55^\circ\text{C}$).
   - **Tier 5 (Cross-Sensor Consistency)**: Inter-sensor thermodynamic coupling limits between terminal probes ($S_3$ vs $S_1$, $S_2$ vs $S_1$).
   - **Tier 6 (Conduction Residuals)**: Single-sensor residuals deviating by $> 1.25\times$ baseline measurement tolerance from physical conduction equations.
4. **Virtual Sensor Imputation**: Corrupted or dropped sensor channels are automatically estimated using thermodynamic conduction baselines before feeding into the regression models.
5. **Defensive Schema Validation**: Input files are verified against mandatory sheet names and column schemas; deviations trigger explicit diagnostic messages rather than silent crashes.

---

## 📂 Repository Structure

```
├── CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx  # Official hackathon dataset
├── PowerNext_Alpha.csv                               # Submission predictions (350 test records)
├── summary.json                                      # Task 3 automated executive summary
├── solution_pipeline.py                              # Fully autonomous CLI pipeline
├── solution_notebook.ipynb                           # Pre-rendered interactive Jupyter notebook
├── methodology_note.md                               # Comprehensive 2-page engineering note
├── package_submission.py                             # Deliverable validation and packaging script
├── powernext-alpha-submission.zip                    # Ready-to-upload ZIP package for Unstop
└── README.md                                         # Project documentation
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Ensure Python 3.10+ is installed, then install the required dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm openpyxl matplotlib
```

### 2. Run Autonomous Pipeline
Execute the pipeline script to reproduce all predictions and summary metrics:
```bash
python solution_pipeline.py --team-name PowerNext_Alpha --data-path CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx
```

### 3. Run Synthetic Robustness Stress Test
Confirm fault tolerance against missing operating parameters and injected duplicates:
```bash
python solution_pipeline.py --stress-test
```

### 4. Validate & Package Deliverables
Run the packaging script to verify schema consistency, row counts, and generate the submission archive:
```bash
python package_submission.py
```

---

## 📄 License & Attribution
Developed for the PowerNext-AI Challenge organized by CPRI & MIT Bengaluru.
