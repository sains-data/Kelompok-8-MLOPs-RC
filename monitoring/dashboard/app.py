import pandas as pd
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Monitoring Model", layout="wide")

st.write("DEBUG: APP.PY SEDANG DIJALANKAN")
st.write("Working directory:", os.getcwd())
st.write("Baseline exists:", os.path.exists("data/dataset MLOps.csv"))



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
#   LOAD BASELINE DATA
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

    # ---- Mapping kolom production → baseline ----
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

    # Ambil hanya fitur relevan (jika ada)
    try:
        production = production_raw[features]
    except KeyError:
        st.error("❌ Kolom pada production log tidak sesuai dengan baseline dataset.")
        st.write("Kolom production:", production_raw.columns.tolist())
        st.write("Kolom diperlukan:", features)
        st.stop()

else:
    st.warning("⚠ Belum ada data produksi (prediction_logs.csv). Jalankan API & lakukan prediksi.")
    production = None


# =====================================
#   TAMPILKAN DATA PRODUKSI
# =====================================
st.subheader("📌 Data Produksi Terbaru")
if production is not None:
    st.dataframe(production.tail(10))
else:
    st.info("Tidak ada data prediksi untuk ditampilkan.")


# =====================================
#   HITUNG PSI (DATA DRIFT)
# =====================================
st.subheader("📈 Data Drift Monitoring (PSI)")

psi_results = {}

if production is not None:
    for col in features:
        psi_results[col] = calculate_psi(baseline[col], production[col])

    df_psi = pd.DataFrame(psi_results.items(), columns=["Feature", "PSI"])
    st.write(df_psi)

    # Interpretasi otomatis
    st.subheader("📌 Interpretasi PSI per Fitur")

    for feature, psi_val in psi_results.items():
        if psi_val < 0.1:
            status = "🟢 Stabil"
        elif psi_val < 0.25:
            status = "🟡 Warning"
        else:
            status = "🔴 Drift Terdeteksi"

        st.write(f"**{feature}** → PSI = `{psi_val:.4f}` → {status}")

else:
    st.info("Menunggu data produksi untuk menghitung PSI.")


# =====================================
#   VISUALISASI DISTRIBUSI FITUR
# =====================================
st.subheader("📊 Distribusi Fitur (Baseline vs Production)")

selected_feature = st.selectbox("Pilih fitur untuk visualisasi:", features)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(baseline[selected_feature], alpha=0.5, label="Baseline")

if production is not None:
    ax.hist(production[selected_feature], alpha=0.5, label="Production")

ax.legend()
ax.set_title(f"Distribusi {selected_feature}")
st.pyplot(fig)
