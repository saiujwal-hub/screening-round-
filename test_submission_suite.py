"""
PowerNext-AI 2026 Screening Round - Automated Test Suite
Team: og

Executes 10 comprehensive robustness and failure-mode scenarios:
  1. Standard valid test inputs (schema conformity, finite non-NaN predictions, [0, 150]°C bounds)
  2. Missing sensor readings (partial sensor drop -> flagged Invalid, robust fallback)
  3. Duplicate test records (identical operating parameters -> Tier 3 flags duplicates)
  4. Extreme high-load regime shifts (I > 85 A -> 0 false positives, valid physical scaling)
  5. Corrupted sensor values (unphysical readings -> Tier 4/5/6 flags Invalid, bounded predictions)
  6. Variable row counts (50 rows, 200 rows, 500 rows -> no hardcoded 350 assumptions)
  7. Shuffled / non-consecutive IDs (exact 1-to-1 Test_ID preservation and alignment)
  8. Extreme ambient conditions (Tamb = 10°C to 50°C -> robust thermal bounds)
  9. Column aliasing & naming variations (lowercase, missing units -> successfully normalized)
 10. Malformed / corrupted schema (missing mandatory columns -> loud diagnostic error, no silent failure)
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Import functions from solution_pipeline
from solution_pipeline import (
    DEFAULT_DATASET_PATH,
    standardize_columns,
    validate_schema,
    load_data,
    detect_anomalies,
    predict_reference_parameter,
    train_and_predict_hotspot,
    generate_summary,
    REQUIRED_TRAIN_COLUMNS,
    REQUIRED_TEST_COLUMNS,
)


class TestPowerNextSubmissionSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load baseline training and test data
        cls.df_train_base, cls.df_test_base = load_data(DEFAULT_DATASET_PATH)

    def test_01_standard_valid_inputs(self):
        """Scenario 1: Verify standard valid pipeline behavior and output bounds."""
        test_df = self.df_test_base.copy()
        test_flags, _ = detect_anomalies(self.df_train_base, test_df)
        preds = predict_reference_parameter(self.df_train_base, test_df)

        self.assertEqual(len(preds), len(test_df), "Predictions length must match test set length.")
        self.assertEqual(len(test_flags), len(test_df), "Validity flags length must match test set length.")
        self.assertFalse(np.isnan(preds).any(), "Predictions must not contain null/NaN values.")
        self.assertFalse(np.isinf(preds).any(), "Predictions must not contain infinite values.")
        
        # Physical plausibility: Temperature rise must be > 0 and < 150 °C
        self.assertTrue((preds > 0.0).all(), "All predicted temperatures must be strictly positive (> 0°C).")
        self.assertTrue((preds < 150.0).all(), "All predicted temperatures must be strictly within physical limits (< 150°C).")
        
        # Valid labels must be strictly 'Valid' or 'Invalid'
        unique_labels = set(np.unique(test_flags))
        self.assertTrue(unique_labels.issubset({'Valid', 'Invalid'}), f"Unexpected labels found: {unique_labels}")

    def test_02_missing_sensor_readings(self):
        """Scenario 2: Missing sensor readings must be flagged Invalid and cleanly imputed."""
        test_df = self.df_test_base.copy().head(20).reset_index(drop=True)
        # Drop S1 on row 0, S2 on row 1, S3 on row 2, and all sensors on row 3
        test_df.loc[0, 'Sensor_S1'] = np.nan
        test_df.loc[1, 'Sensor_S2'] = np.nan
        test_df.loc[2, 'Sensor_S3'] = np.nan
        test_df.loc[3, ['Sensor_S1', 'Sensor_S2', 'Sensor_S3']] = np.nan

        test_flags, _ = detect_anomalies(self.df_train_base, test_df)
        preds = predict_reference_parameter(self.df_train_base, test_df)

        self.assertEqual(test_flags[0], 'Invalid', "Row with missing S1 must be flagged Invalid.")
        self.assertEqual(test_flags[1], 'Invalid', "Row with missing S2 must be flagged Invalid.")
        self.assertEqual(test_flags[2], 'Invalid', "Row with missing S3 must be flagged Invalid.")
        self.assertEqual(test_flags[3], 'Invalid', "Row with all missing sensors must be flagged Invalid.")
        
        # Model must fall back to thermal reconstruction without producing NaNs
        self.assertFalse(np.isnan(preds).any(), "Missing sensors must be imputed without producing NaNs.")
        self.assertTrue((preds > 0.0).all(), "Imputed predictions must be positive.")

    def test_03_duplicate_records(self):
        """Scenario 3: Exact duplicate test records must be flagged Invalid in Tier 3."""
        test_df = self.df_test_base.copy().head(10).reset_index(drop=True)
        # Create an exact duplicate of row 0 as row 5
        test_df.iloc[5] = test_df.iloc[0]
        test_df.loc[5, 'Test_ID'] = 'Test_DUP_05'

        test_flags, _ = detect_anomalies(self.df_train_base, test_df)
        self.assertEqual(test_flags[5], 'Invalid', "Duplicate operating test run must be flagged Invalid.")

    def test_04_high_load_regime_shifts(self):
        """Scenario 4: High current (I > 85 A) genuine physical runs must NOT be falsely flagged."""
        # Find high load runs in training set that are Valid ground-truth
        high_load_valid = self.df_train_base[
            (self.df_train_base['Load_Current_A'] > 85.0) & 
            (self.df_train_base['Validity_Label'] == 'Valid')
        ].copy().head(20).reset_index(drop=True)

        test_input = high_load_valid.drop(columns=['Reference_Parameter', 'Validity_Label'])
        test_flags, _ = detect_anomalies(self.df_train_base, test_input)
        preds = predict_reference_parameter(self.df_train_base, test_input)

        invalid_count = (test_flags == 'Invalid').sum()
        self.assertEqual(invalid_count, 0, f"High-load valid runs must not have false alarms. Got {invalid_count} Invalid flags.")
        self.assertTrue((preds > 0.0).all(), "High-load predictions must be positive.")
        self.assertTrue((preds < 150.0).all(), "High-load predictions must respect upper physical ceiling.")

    def test_05_corrupted_sensor_values(self):
        """Scenario 5: Unphysical and corrupted sensor values must be flagged Invalid."""
        test_df = self.df_test_base.copy().head(15).reset_index(drop=True)
        # Negative temperature rise (violates thermodynamics)
        test_df.loc[0, 'Sensor_S1'] = -15.0
        # Severely uncoupled reading (S3 8x higher than S1)
        test_df.loc[1, 'Sensor_S1'] = 25.0
        test_df.loc[1, 'Sensor_S3'] = 220.0

        test_flags, _ = detect_anomalies(self.df_train_base, test_df)
        preds = predict_reference_parameter(self.df_train_base, test_df)

        self.assertEqual(test_flags[0], 'Invalid', "Negative sensor reading must be flagged Invalid.")
        self.assertEqual(test_flags[1], 'Invalid', "Thermodynamically uncoupled sensor reading must be flagged Invalid.")
        self.assertFalse(np.isnan(preds).any(), "Predictions on corrupted records must remain finite.")

    def test_06_variable_row_counts(self):
        """Scenario 6: Verify arbitrary test set sizes (50, 200, 500 rows) without hardcoded 350 assumptions."""
        for n_rows in [50, 200, 500]:
            if n_rows <= len(self.df_test_base):
                subset = self.df_test_base.copy().iloc[:n_rows].reset_index(drop=True)
            else:
                # Synthesize 500 rows by repeating baseline with modified IDs
                repeat_count = int(np.ceil(n_rows / len(self.df_test_base)))
                subset = pd.concat([self.df_test_base] * repeat_count, ignore_index=True).iloc[:n_rows].copy()
                subset['Test_ID'] = [f"SYN_{i:04d}" for i in range(n_rows)]

            test_flags, _ = detect_anomalies(self.df_train_base, subset)
            preds = predict_reference_parameter(self.df_train_base, subset)

            self.assertEqual(len(test_flags), n_rows, f"Validity flags count ({len(test_flags)}) != expected {n_rows}")
            self.assertEqual(len(preds), n_rows, f"Predictions count ({len(preds)}) != expected {n_rows}")
            self.assertFalse(np.isnan(preds).any(), f"NaNs found in predictions for size {n_rows}")

    def test_07_shuffled_and_non_consecutive_ids(self):
        """Scenario 7: Ensure arbitrary, shuffled Test_IDs are preserved exactly 1-to-1."""
        test_df = self.df_test_base.copy().sample(frac=1.0, random_state=123).reset_index(drop=True)
        arbitrary_ids = [f"UNSEEN_RUN_{i*7+13}" for i in range(len(test_df))]
        test_df['Test_ID'] = arbitrary_ids

        test_flags, _ = detect_anomalies(self.df_train_base, test_df)
        preds = predict_reference_parameter(self.df_train_base, test_df)

        out_df = pd.DataFrame({
            'Test_ID': test_df['Test_ID'],
            'Validity_Label': test_flags,
            'Predicted_Reference_Parameter': preds
        })

        self.assertListEqual(list(out_df['Test_ID']), arbitrary_ids, "Test_ID order and identity must match input exactly.")
        self.assertEqual(len(out_df), len(test_df))

    def test_08_extreme_ambient_conditions(self):
        """Scenario 8: Extreme low and high ambient conditions must remain numerically stable."""
        test_df = self.df_test_base.copy().head(10).reset_index(drop=True)
        # Extreme cold ambient
        test_df.loc[0, 'Ambient_Temperature_C'] = 10.0
        # Extreme hot ambient
        test_df.loc[1, 'Ambient_Temperature_C'] = 52.0

        test_flags, _ = detect_anomalies(self.df_train_base, test_df)
        preds = predict_reference_parameter(self.df_train_base, test_df)

        self.assertFalse(np.isnan(preds).any(), "Predictions must not be null under extreme ambient.")
        self.assertTrue((preds > 0.0).all(), "Predictions must stay positive.")
        self.assertTrue((preds < 150.0).all(), "Predictions must stay below 150°C.")
        self.assertTrue((preds > 0.0).all(), "Predictions must stay positive.")
        self.assertTrue((preds < 150.0).all(), "Predictions must stay below 150°C.")

    def test_09_column_aliasing_and_variations(self):
        """Scenario 9: Column name variations (lowercase, units) must be parsed automatically."""
        test_df = self.df_test_base.copy().head(10)
        renamed_df = test_df.rename(columns={
            'Applied_Voltage_kV': 'applied_voltage',
            'Load_Current_A': 'load_current_in_a',
            'Ambient_Temperature_C': 'ambient_temp_c',
            'Test_Duration_min': 'test_duration',
            'Sensor_S1': 'sensor_s1',
            'Sensor_S2': 's2',
            'Sensor_S3': 'sensor_s3'
        })

        std_df = standardize_columns(renamed_df)
        # Verify all canonical columns exist
        for req_col in REQUIRED_TEST_COLUMNS:
            self.assertIn(req_col, std_df.columns, f"Canonical column '{req_col}' not resolved from alias.")

        test_flags, _ = detect_anomalies(self.df_train_base, std_df)
        preds = predict_reference_parameter(self.df_train_base, std_df)
        self.assertEqual(len(preds), 10)

    def test_10_malformed_input_loud_error(self):
        """Scenario 10: Missing mandatory columns must raise loud, diagnostic error without silent failure."""
        malformed_df = pd.DataFrame({
            'Test_ID': ['T_01', 'T_02'],
            'Applied_Voltage_kV': [11.0, 11.2],
            # Load_Current_A is missing!
            'Ambient_Temperature_C': [25.0, 26.0],
            'Test_Duration_min': [60.0, 60.0],
            'Sensor_S1': [45.0, 48.0],
            'Sensor_S2': [44.0, 47.0],
            'Sensor_S3': [46.0, 49.0]
        })

        with self.assertRaises(ValueError) as ctx:
            validate_schema(self.df_train_base, malformed_df, source_desc="Malformed_Test_Data")
        
        self.assertIn("Missing required column", str(ctx.exception))
        self.assertIn("Load_Current_A", str(ctx.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
