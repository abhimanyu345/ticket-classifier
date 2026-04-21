import pandas as pd
import os
from evidently.future.report import Report
from evidently.future.presets import DataDriftPreset

REFERENCE_PATH = "data/processed/train.csv"
REPORTS_DIR    = "monitoring/reports"

def run_sample_drift_check():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    reference_full = pd.read_csv(REFERENCE_PATH)
    test           = pd.read_csv("data/processed/test.csv")

    # Sample equal sizes so chi-square test is fair
    sample_size = 200
    reference = reference_full.sample(n=sample_size, random_state=0).copy()
    current   = test.sample(n=sample_size, random_state=42).copy()

    print(f"Reference size: {len(reference)}")
    print(f"Current size:   {len(current)}")
    print(f"\nReference label distribution:\n{reference['label'].value_counts()}")
    print(f"\nCurrent label distribution:\n{current['label'].value_counts()}")

    report = Report([DataDriftPreset()])
    result = report.run(reference[["label"]], current[["label"]])

    report_path = f"{REPORTS_DIR}/drift_report.html"
    result.save_html(report_path)

    print(f"\nDrift report saved → {report_path}")
    print("Open it in your browser to view the full analysis")

    return report_path

if __name__ == "__main__":
    run_sample_drift_check()