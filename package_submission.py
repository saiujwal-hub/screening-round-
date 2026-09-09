"""
CPRI & MIT Bengaluru - PowerNext-AI Screening Round
Submission Packaging Script

Validates all required deliverables and creates the final submission ZIP archive:
<teamname>-submission.zip
"""

import os
import zipfile
import json
import numpy as np
import pandas as pd


def create_submission_package(team_name="og", output_dir=".", dataset_path=None):
    csv_file = f"{team_name}.csv" if os.path.exists(os.path.join(output_dir, f"{team_name}.csv")) else "og.csv"
    summary_file = "summary.json"
    script_file = "solution_pipeline.py"
    notebook_file = "solution_notebook.ipynb"
    methodology_file = "methodology_note.md"
    files_to_pack = [csv_file, summary_file, script_file, notebook_file, methodology_file]
    if os.path.exists(os.path.join(output_dir, "summary.csv")):
        files_to_pack.append("summary.csv")
    if os.path.exists(os.path.join(output_dir, "task1_regime_vs_anomaly.png")):
        files_to_pack.append("task1_regime_vs_anomaly.png")
    if os.path.exists(os.path.join(output_dir, "methodology_note.pdf")):
        files_to_pack.append("methodology_note.pdf")

    print(f"=== Validating Submission Deliverables for '{team_name}' ===")
    all_present = True
    for fname in files_to_pack:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"  [MISSING] {fname}")
            all_present = False
        else:
            size_kb = os.path.getsize(path) / 1024.0
            print(f"  [OK] {fname} ({size_kb:.1f} KB)")

    if not all_present:
        raise FileNotFoundError("One or more required deliverable files are missing!")

    # Validate CSV schema and dataset-independent row validity
    df_sub = pd.read_csv(os.path.join(output_dir, csv_file))
    print(f"\nValidating CSV format ({csv_file}):")
    print(f"  Rows count: {len(df_sub)}")
    print(f"  Columns: {list(df_sub.columns)}")
    assert len(df_sub) > 0, f"CSV {csv_file} is empty!"
    assert list(df_sub.columns) == ['Test_ID', 'Predicted_Reference_Parameter', 'Validity_Label'], f"CSV column headers mismatch! Found: {list(df_sub.columns)}"
    assert df_sub['Test_ID'].isnull().sum() == 0, "Test_ID contains null values!"
    assert (df_sub['Test_ID'].astype(str).str.strip() == '').sum() == 0, "Test_ID contains empty string values!"
    assert df_sub['Test_ID'].duplicated().sum() == 0, f"Test_ID contains {df_sub['Test_ID'].duplicated().sum()} unexpected duplicate IDs!"
    assert df_sub['Predicted_Reference_Parameter'].isnull().sum() == 0, "Predicted_Reference_Parameter contains NaN values!"
    assert np.all(np.isfinite(df_sub['Predicted_Reference_Parameter'])), "Predicted_Reference_Parameter contains infinite values!"
    assert (df_sub['Predicted_Reference_Parameter'] > 0).all(), "Predictions contain non-positive temperature rise values!"
    assert (df_sub['Predicted_Reference_Parameter'] < 150).all(), "Predictions contain implausibly high temperature rise values (> 150°C)!"
    assert df_sub['Validity_Label'].isin(['Valid', 'Invalid']).all(), f"Validity_Label contains invalid classes! Found: {set(df_sub['Validity_Label']) - {'Valid', 'Invalid'}}"

    # Dynamic cross-validation against supplied test dataset
    actual_test_path = dataset_path or ("CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx" if os.path.exists("CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx") else None)
    if actual_test_path and os.path.exists(actual_test_path):
        try:
            if actual_test_path.endswith('.xlsx') or actual_test_path.endswith('.xls'):
                with pd.ExcelFile(actual_test_path) as xl:
                    test_sheets = [s for s in xl.sheet_names if 'test' in s.lower()]
                    sheet_to_use = test_sheets[0] if test_sheets else xl.sheet_names[0]
                    df_test_ref = pd.read_excel(xl, sheet_name=sheet_to_use)
            else:
                df_test_ref = pd.read_csv(actual_test_path)

            if 'Test_ID' in df_test_ref.columns:
                print(f"  Checking alignment against actual test dataset ({actual_test_path}):")
                print(f"    Expected rows: {len(df_test_ref)} | Prediction rows: {len(df_sub)}")
                assert len(df_sub) == len(df_test_ref), (
                    f"Prediction row count ({len(df_sub)}) does not match actual test dataset row count ({len(df_test_ref)})!"
                )
                assert set(df_sub['Test_ID']) == set(df_test_ref['Test_ID']), (
                    "Every test record must receive exactly one prediction, and Test_ID set must match test dataset exactly!"
                )
                assert list(df_sub['Test_ID']) == list(df_test_ref['Test_ID']), (
                    "Predictions order must match test dataset order 1-to-1!"
                )
                print("    1-to-1 Test_ID matching and row count validation PASSED!")
        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            print(f"  Note: Could not parse test dataset reference for alignment check ({e}). Skipping cross-dataset check.")

    print("  CSV validation passed!")

    # Validate JSON
    with open(os.path.join(output_dir, summary_file), 'r') as f:
        s_data = json.load(f)
    print(f"\nValidating Summary JSON ({summary_file}):")
    assert "number_of_records_analysed" in s_data
    assert s_data["number_of_records_analysed"] == len(df_sub), (
        f"Summary analysed records ({s_data['number_of_records_analysed']}) does not match prediction CSV row count ({len(df_sub)})!"
    )
    assert "three_test_ids_requiring_highest_attention" in s_data
    assert len(s_data["three_test_ids_requiring_highest_attention"]) == min(3, len(df_sub))
    for tid in s_data["three_test_ids_requiring_highest_attention"]:
        assert tid in df_sub['Test_ID'].values, f"Top attention ID '{tid}' is not present in prediction CSV!"
    assert "approach_explanation" in s_data
    words = len(s_data['approach_explanation'].split())
    print(f"  Approach explanation length: {words} words (<= 100 words requirement)")
    assert words <= 100, f"Approach explanation has {words} words, exceeds 100 word limit!"
    print("  Summary JSON validation passed!")

    # Validate Summary CSV if present
    if os.path.exists(os.path.join(output_dir, "summary.csv")):
        df_sum_csv = pd.read_csv(os.path.join(output_dir, "summary.csv"))
        metric_map = dict(zip(df_sum_csv['Metric'], df_sum_csv['Value']))
        assert int(metric_map['number_of_records_analysed']) == len(df_sub), (
            f"Summary CSV analysed records ({metric_map['number_of_records_analysed']}) does not match CSV row count ({len(df_sub)})!"
        )
        print("  Summary CSV validation passed!")

    # Create ZIP archive with deliverables enclosed in a team-named subfolder
    subfolder_name = team_name
    zip_filename = os.path.join(output_dir, f"{team_name}-submission.zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fname in files_to_pack:
            file_path = os.path.join(output_dir, fname)
            arcname = f"{subfolder_name}/{fname}"
            zipf.write(file_path, arcname=arcname)

    zip_size_kb = os.path.getsize(zip_filename) / 1024.0
    print(f"\n=== Submission Package Created Successfully ===")
    print(f"Archive: {zip_filename} ({zip_size_kb:.1f} KB)")
    print(f"Enclosed directory: '{subfolder_name}/'")
    print("Contents:")
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        for item in zipf.infolist():
            print(f"  - {item.filename:35s} ({item.file_size / 1024.0:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Package PowerNext-AI screening submission archive.")
    parser.add_argument("--team-name", default="og", help="Registered team name (default: 'og')")
    parser.add_argument("--output-dir", default=".", help="Output directory containing deliverables")
    parser.add_argument("--dataset-path", default=None, help="Path to input test dataset for verification")
    args = parser.parse_args()
    create_submission_package(team_name=args.team_name, output_dir=args.output_dir, dataset_path=args.dataset_path)
