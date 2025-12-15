# src/utils.py
import csv
import os
from datetime import datetime

LOG_PATH = "monitoring/logs/prediction_logs.csv"

def log_prediction(input_data, prediction):
    os.makedirs("monitoring/logs", exist_ok=True)
    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Jika file kosong → tulis header
        if not file_exists:
            writer.writerow([
                "Mahasiswa Angkatan",
                "Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?",
                "Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?",
                "Bagaimana tujuan utama penggunaan gadget Kamu?",
                "Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget mu?",
                "Bagaimana presepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik  ?",
                "Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain ?",
                "Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?",
                "prediction",
                "timestamp"
            ])

        # Tulis baris data baru
        writer.writerow([
            input_data["Mahasiswa Angkatan"],
            input_data["Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?"],
            input_data["Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?"],
            input_data["Bagaimana tujuan utama penggunaan gadget Kamu?"],
            input_data["Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget mu?"],
            input_data["Bagaimana presepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik  ?"],
            input_data["Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain ?"],
            input_data["Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?"],
            prediction,
            datetime.now()
        ])
