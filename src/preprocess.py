# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess(path="data/dataset MLOps.csv"):

    df = pd.read_csv(path, sep=";")

    # Rename target
    df.rename(columns={
        "Bagaimana menurut Kamu, tingkat distraksi akibat penggunaan gadget?": "distraksi"
    }, inplace=True)

    # Drop timestamp
    df = df.drop(columns=["Timestamp"])

    # Mapping target menjadi 3 kelas
    def map_class(x):
        if x <= 3:
            return "low"
        elif x <= 6:
            return "medium"
        else:
            return "high"

    df["label"] = df["distraksi"].apply(map_class)
    df = df.drop(columns=["distraksi"])

    # Pilih kolom fitur yang ADA DI DATASET
    X = df[[
        "Mahasiswa Angkatan",
        "Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?",
        "Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?",
        "Bagaimana tujuan utama penggunaan gadget Kamu?",
        "Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget mu?",
        "Bagaimana presepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik  ?",
        "Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain ?",
        "Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?"
    ]]

    y = df["label"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
