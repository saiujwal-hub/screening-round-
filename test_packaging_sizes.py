"""
Verification script: Tests packaging with 50, 200, 300, 350, and 500 rows.
Ensures package_submission.py is completely dataset-agnostic.
"""

import os
import shutil
import tempfile
import json
import pandas as pd
from package_submission import create_submission_package


def test_packaging_sizes():
    print("=" * 80)
    print("    TESTING SUBMISSION PACKAGING ACROSS VARIABLE DATASET SIZES")
    print("=" * 80)

    # Base deliverable files to copy into temporary environments
    shared_files = [
        "solution_pipeline.py",
        "solution_notebook.ipynb",
        "methodology_note.md",
        "methodology_note.pdf",
        "task1_regime_vs_anomaly.png"
    ]

    for size in [50, 200, 300, 350, 500]:
        print(f"\n--- Testing Dataset Size: {size} Rows ---")
        temp_dir = tempfile.mkdtemp(prefix=f"test_pkg_{size}_")
        try:
            # 1. Copy common assets
            for fname in shared_files:
                if os.path.exists(fname):
                    shutil.copy(fname, os.path.join(temp_dir, fname))

            # 2. Synthesize test data and predictions of exact size
            test_ids = [f"TST-{i+1:04d}" for i in range(size)]
            df_test_ref = pd.DataFrame({"Test_ID": test_ids})

            df_sub = pd.DataFrame({
                "Test_ID": test_ids,
                "Predicted_Reference_Parameter": [25.0 + (i % 15) * 0.5 for i in range(size)],
                "Valid_Invalid": ["Valid" if i % 7 != 0 else "Invalid" for i in range(size)]
            })
            df_sub.to_csv(os.path.join(temp_dir, "og.csv"), index=False)

            # 3. Create matching summary.json
            summary_data = {
                "number_of_records_analysed": size,
                "number_of_valid_records": int((df_sub['Valid_Invalid'] == 'Valid').sum()),
                "number_of_abnormal_invalid_records_identified": int((df_sub['Valid_Invalid'] == 'Invalid').sum()),
                "percentage_abnormal": round(((df_sub['Valid_Invalid'] == 'Invalid').sum() / size) * 100, 2),
                "minimum_predicted_reference_parameter": round(float(df_sub['Predicted_Reference_Parameter'].min()), 4),
                "maximum_predicted_reference_parameter": round(float(df_sub['Predicted_Reference_Parameter'].max()), 4),
                "average_predicted_reference_parameter": round(float(df_sub['Predicted_Reference_Parameter'].mean()), 4),
                "three_test_ids_requiring_highest_attention": test_ids[:3],
                "approach_explanation": "A verified physics-grounded ML approach with dynamic validation."
            }
            with open(os.path.join(temp_dir, "summary.json"), "w") as f:
                json.dump(summary_data, f, indent=4)

            # 4. Create matching summary.csv
            summary_rows = [
                {"Metric": "number_of_records_analysed", "Value": size},
                {"Metric": "approach_explanation", "Value": summary_data["approach_explanation"]}
            ]
            pd.DataFrame(summary_rows).to_csv(os.path.join(temp_dir, "summary.csv"), index=False)

            # 5. Execute packaging
            create_submission_package(team_name="og", output_dir=temp_dir, test_df=df_test_ref)
            
            # 6. Verify zip exists
            zip_path = os.path.join(temp_dir, "og-submission.zip")
            assert os.path.exists(zip_path), f"ZIP was not created for size {size}!"
            print(f"  [PASS] Successfully packaged size {size} (Archive size: {os.path.getsize(zip_path)/1024:.1f} KB)")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("ALL 5 DATASET SIZES (50, 200, 300, 350, 500) PACKAGED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_packaging_sizes()
