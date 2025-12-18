import pandas as pd
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Monitoring Model", layout="wide")

# =============================
# FastAPI URL
# =============================
API_URL = "http://localhost:8000/predict"  # ganti jika deploy API ke cloud


# =====================================
#   FUNGSI MENGHITUNG PSI
# =====================================
def calculate_psi(expected, actual, buckets=10):
    def scale_range(col):
        return (col - col.min()) / (col.max() - col.min() + 1e-9)

    expected = scale_range(expected)
    actual = scale_range(actual)

    breaks = np.linspace(0, 1, buckets + 1)

    expected_counts = np.histogram(expected, bins=breaks)[0]
    actual_counts = np.histogram(actual, bins=breaks)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    psi = 0
    for e, a in zip(expected_perc, actual_perc):
        if e == 0 or a == 0:
            continue
        psi += (e - a) * np.log(e / a)

    return psi


# =====================================
#   SIDEBAR - FORM INPUT DATA
# =====================================
st.sidebar.header("📝 Input Data Baru untuk Prediksi")

with st.sidebar.form("input_form"):
    Mahasiswa_Angkatan = st.number_input("Angkatan", min_value=2015, max_value=2025, value=2022)
    frekuensi = st.slider("Frekuensi penggunaan gadget", 1, 8, 3)
    durasi = st.slider("Durasi penggunaan gadget", 1, 8, 3)
    tujuan = st.slider("Tujuan penggunaan gadget", 1, 8, 3)
    sulit_kontrol = st.slider("Sulit kontrol penggunaan gadget", 1, 8, 3)
    persepsi = st.slider("Persepsi pengaruh gadget", 1, 8, 3)
    kemampuan_waktu = st.slider("Kemampuan atur waktu", 1, 8, 3)
    upaya = st.slider("Upaya mengurangi penggunaan gadget", 1, 8, 3)

    submit_btn = st.form_submit_button("Kirim ke API untuk Prediksi 🚀")

if submit_btn:
    input_payload = {
        "Mahasiswa_Angkatan": Mahasiswa_Angkatan,
        "frekuensi": frekuensi,
        "durasi": durasi,
        "tujuan": tujuan,
        "sulit_kontrol": sulit_kontrol,
        "persepsi": persepsi,
        "kemampuan_waktu": kemampuan_waktu,
        "upaya": upaya
    }

    try:
        response = requests.post(API_URL, json=input_payload)
        pred_result = response.json()

        st.sidebar.success(f"Prediksi Model: **{pred_result['prediction']}** 🎉")

    except Exception as e:
        st.sidebar.error(f"Gagal menghubungi API: {e}")


# =====================================
#   DASHBOARD MONITORING
# =====================================
st.title("📊 Dashboard Monitoring Model — Data Drift & Predictions")

baseline_path = "data/dataset MLOps.csv"
production_path = "monitoring/logs/prediction_logs.csv"

# ---- LOAD BASELINE ----
try:
    baseline_raw = pd.read_csv(baseline_path, sep=";")
except Exception as e:
    st.error(f"❌ Gagal memuat baseline dataset: {e}")
    st.stop()


# =====================================
#   DEFINISI FITUR SESUAI DATASET
# =====================================
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

baseline = baseline_raw[features]


# =====================================
#   LOAD PRODUCTION LOGS
# =====================================
if os.path.exists(production_path):
    try:
        production_raw = pd.read_csv(production_path)
    except Exception as e:
        st.error(f"❌ Gagal membaca production log: {e}")
        st.stop()

    # Mapping ke baseline
    column_map = {
        "Mahasiswa_Angkatan": "Mahasiswa Angkatan",
        "frekuensi": "Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?",
        "durasi": "Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?",
        "tujuan": "Bagaimana tujuan utama penggunaan gadget Kamu?",
        "sulit_kontrol": "Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget mu?",
        "persepsi": "Bagaimana presepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik  ?",
        "kemampuan_waktu": "Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain ?",
        "upaya": "Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?"
    }

    production_raw.rename(columns=column_map, inplace=True)
    production = production_raw[features]

else:
    st.warning("⚠ Belum ada data produksi. Lakukan prediksi melalui sidebar.")
    production = None


# =====================================
#   TAMPILKAN TABEL PRODUKSI
# =====================================
st.subheader("📌 Data Produksi Terbaru")

if production is not None:
    st.dataframe(production.tail(10))

# =====================================
#   PSI DRIFT MONITORING
# =====================================
st.subheader("📈 Data Drift Monitoring (PSI)")

if production is not None:
    psi_results = {}
    for col in features:
        psi_results[col] = calculate_psi(baseline[col], production[col])

    df_psi = pd.DataFrame(psi_results.items(), columns=["Feature", "PSI"])
    st.write(df_psi)

    st.subheader("📌 Interpretasi PSI")

    for f, psi_val in psi_results.items():
        if psi_val < 0.1:
            status = "🟢 Stabil"
        elif psi_val < 0.25:
            status = "🟡 Warning"
        else:
            status = "🔴 Drift"

        st.write(f"**{f}** → PSI = `{psi_val:.4f}` → {status}")
else:
    st.info("PSI akan muncul setelah ada data prediksi.")


# =====================================
#   VISUALISASI DISTRIBUSI
# =====================================
st.subheader("📊 Distribusi Fitur")

selected_feature = st.selectbox("Pilih fitur:", features)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(baseline[selected_feature], alpha=0.5, label="Baseline")

if production is not None:
    ax.hist(production[selected_feature], alpha=0.5, label="Production")

ax.legend()
st.pyplot(fig)
