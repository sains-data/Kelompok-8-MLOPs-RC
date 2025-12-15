import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import DataDriftMetric
from evidently.report import Report
import json

def run_monitoring():
    baseline = pd.read_csv("data/dataset MLOps.csv", sep=";")
    prod = pd.read_csv("monitoring/logs/prediction_logs.csv")

    # Pilih fitur saja (tanpa target)
    features = [
        "Mahasiswa Angkatan",
        "Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?",
        "Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?",
        "Bagaimana tujuan utama penggunaan gadget Kamu?",
        "Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget mu?",
        "Bagaimana presepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik  ?",
        "Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain ?",
        "Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?"
    ]

    baseline = baseline[features]
    prod = prod[features]

    report = Report(metrics=[DataDriftMetric()])
    report.run(reference_data=baseline, current_data=prod)

    # Save JSON drift report
    drift_json = report.as_dict()
    with open("monitoring/reports/drift.json", "w") as f:
        json.dump(drift_json, f, indent=4)

    print("Drift report saved → monitoring/reports/drift.json")
