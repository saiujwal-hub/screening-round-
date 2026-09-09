"""
CPRI & MIT Bengaluru - PowerNext-AI Screening Round
Submission Packaging Script

Validates all required deliverables and creates the final submission ZIP archive:
<teamname>-submission.zip
"""

import os
import zipfile
import json
import pandas as pd


def create_submission_package(team_name="og", output_dir="."):
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

    # Validate CSV schema and row count
    df_sub = pd.read_csv(os.path.join(output_dir, csv_file))
    print(f"\nValidating CSV format ({csv_file}):")
    print(f"  Rows count: {len(df_sub)} (expected: 350)")
    print(f"  Columns: {list(df_sub.columns)}")
    assert len(df_sub) == 350, "CSV does not have 350 rows!"
    assert list(df_sub.columns) == ['Test_ID', 'Predicted_Reference_Parameter', 'Validity_Label'], "CSV column headers mismatch!"
    assert df_sub.isnull().sum().sum() == 0, "CSV contains NaN values!"
    print("  CSV validation passed!")

    # Validate JSON
    with open(os.path.join(output_dir, summary_file), 'r') as f:
        s_data = json.load(f)
    print(f"\nValidating Summary JSON ({summary_file}):")
    assert "number_of_records_analysed" in s_data
    assert "three_test_ids_requiring_highest_attention" in s_data
    assert "approach_explanation" in s_data
    words = len(s_data['approach_explanation'].split())
    print(f"  Approach explanation length: {words} words (<= 100 words requirement)")
    assert words <= 100, f"Approach explanation has {words} words, exceeds 100 word limit!"
    print("  Summary JSON validation passed!")

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
    create_submission_package()
