# Student Smartphone Distraction Prediction System (MLOps)-Kelompok-8-MLOPs-RC
![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Docker](https://img.shields.io/badge/Docker-Available-blue)
![MLOps](https://img.shields.io/badge/MLOps-End--to--End-orange)

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan model klasifikasi untuk memprediksi tingkat distraksi mahasiswa (Rendah, Sedang, Tinggi) berdasarkan pola penggunaan gadget mereka. Sistem dilengkapi dengan:
- **Experiment Tracking** menggunakan MLflow
- **Model Monitoring** dengan Evidently AI untuk deteksi data drift
- **Dashboard Monitoring** real-time dengan Streamlit menggunakan PSI (Population Stability Index)
- **Prediction Logging** untuk tracking prediksi production

## 🎯 Fitur Input

Model menerima 8 input:

1. **Mahasiswa Angkatan** - Tahun angkatan mahasiswa
2. **Frekuensi Penggunaan Gadget** (Skala 1-8) - Seberapa sering menggunakan gadget setiap hari
3. **Durasi Penggunaan** (Skala 1-8) - Berapa lama menggunakan gadget di luar jam kuliah
4. **Tujuan Penggunaan** (Skala 1-8) - Untuk apa gadget digunakan
5. **Kesulitan Kontrol Waktu** (Skala 1-8) - Seberapa sulit mengontrol waktu penggunaan
6. **Persepsi Pengaruh Akademik** (Skala 1-8) - Bagaimana persepsi pengaruh gadget terhadap akademik
7. **Kemampuan Mengatur Waktu** (Skala 1-8) - Kemampuan mengatur waktu antara gadget dan akademik
8. **Upaya Mengurangi Intensitas** (Skala 1-8) - Upaya untuk mengurangi penggunaan gadget

## 📊 Output Klasifikasi

- **Low (Rendah)**: Skor distraksi 1-3
- **Medium (Sedang)**: Skor distraksi 4-6
- **High (Tinggi)**: Skor distraksi 7-10

## 🏗️ Struktur Proyek

```
.
├── data/
│   └── dataset MLOps.csv          # Dataset training (baseline)
├── src/
│   ├── preprocess.py              # Preprocessing dan split data
│   ├── train.py                   # Training model dengan MLflow
│   ├── predict.py                 # Fungsi load model dan prediksi
│   └── utils.py                   # Logging prediksi ke CSV
├── monitoring/
│   ├── monitor.py                 # Data drift detection dengan Evidently
│   ├── dashboard/
│   │   └── app.py                 # Streamlit dashboard (PSI monitoring)
│   ├── logs/
│   │   └── prediction_logs.csv    # Log prediksi production
│   └── reports/
│       └── drift.json             # Laporan drift dari Evidently
├── models/
│   └── distraction_model/         # Model tersimpan (MLflow format)
├── requirements.txt               # Dependencies Python
├── conda.yaml                     # Conda environment
├── python_env.yaml               # Python environment config
├── run_api.bat                   # Script untuk menjalankan API
└── run_dashboard.bat             # Script untuk menjalankan dashboard
```

## 🚀 Instalasi

### 1. Clone Repository
```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Setup Environment

**Menggunakan Conda:**
```bash
conda env create -f conda.yaml
conda activate mlflow-env
```

**Atau menggunakan pip:**
```bash
pip install -r requirements.txt
```

### 3. Install Dependencies Tambahan

Untuk monitoring dan dashboard:
```bash
pip install evidently streamlit matplotlib
```

### 4. Verifikasi Instalasi
```bash
python --version  # Should be 3.11.5
pip list | grep mlflow
pip list | grep streamlit
```

## 📚 Cara Penggunaan

### 1️⃣ Training Model

Jalankan training dengan MLflow tracking:

```bash
python src/train.py
```

**Fitur Training:**
- Menggunakan Random Forest Classifier
- Automatic logging metrics (accuracy) dan parameters ke MLflow
- Model tersimpan otomatis di `models/distraction_model/`
- Experiment name: `distraction-classification`
- Hyperparameter yang digunakan: `n_estimators=200`, `max_depth=10`, `random_state=42`

**Melihat Experiment di MLflow UI:**
```bash
mlflow ui
```
Buka browser: `http://localhost:5000`

### 2️⃣ Melakukan Prediksi

Gunakan model untuk prediksi data baru:

```python
from src.predict import load_model, predict
from src.utils import log_prediction

# Load model
model = load_model("models/distraction_model")

# Input data baru
input_data = {
    "Mahasiswa Angkatan": 2021,
    "Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?": 7,
    "Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?": 6,
    "Bagaimana tujuan utama penggunaan gadget Kamu?": 5,
    "Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget mu?": 7,
    "Bagaimana presepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik  ?": 6,
    "Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain ?": 4,
    "Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?": 3
}

# Prediksi
result = predict(model, input_data)
print(f"Tingkat Distraksi: {result}")

# Log prediksi (PENTING untuk monitoring)
log_prediction(input_data, result)
```

**Catatan:** Setiap prediksi harus di-log menggunakan `log_prediction()` agar data masuk ke `prediction_logs.csv` untuk monitoring drift.

### 3️⃣ Hyperparameter Tuning

Untuk melakukan hyperparameter tuning, edit parameter di file `src/train.py`
```

**Workflow Tuning:**
```bash
# Run 1 - baseline
python src/train.py

# Edit params di train.py (misal: n_estimators=100, max_depth=5)
# Run 2
python src/train.py

# Edit params lagi (misal: n_estimators=300, max_depth=15)
# Run 3
python src/train.py
```

Bandingkan hasil di **MLflow UI** (`mlflow ui`) untuk memilih kombinasi hyperparameter terbaik berdasarkan accuracy.

### 4️⃣ Monitoring Data Drift dengan Evidently

Jalankan monitoring untuk mendeteksi data drift menggunakan Evidently AI:

```bash
python monitoring/monitor.py
```

**Proses Monitoring:**
- Membandingkan baseline data (`data/dataset MLOps.csv`) dengan production data (`monitoring/logs/prediction_logs.csv`)
- Mendeteksi drift menggunakan Evidently AI DataDriftMetric
- Hasil disimpan di: `monitoring/reports/drift.json`

**Catatan:** Pastikan sudah ada data prediksi di `prediction_logs.csv` sebelum menjalankan monitoring.

### 5️⃣ Dashboard Monitoring Real-time

Jalankan dashboard Streamlit untuk monitoring visual:

**Windows:**
```bash
run_dashboard.bat
```

**Linux/Mac:**
```bash
streamlit run monitoring/dashboard/app.py
```

Dashboard akan terbuka di: `http://localhost:8501`

**Fitur Dashboard:**
- 📊 **Visualisasi Data Produksi Terbaru** - 10 prediksi terakhir
- 📈 **PSI (Population Stability Index) per Fitur** - Mengukur pergeseran distribusi data
- 🚦 **Status Drift Otomatis**:
  - 🟢 **Stabil** (PSI < 0.1) - Tidak ada drift
  - 🟡 **Warning** (0.1 ≤ PSI < 0.25) - Drift ringan, monitor lebih ketat
  - 🔴 **Drift Terdeteksi** (PSI ≥ 0.25) - Pertimbangkan retraining
- 📉 **Histogram Distribusi** - Perbandingan baseline vs production per fitur

**Cara Kerja Dashboard:**
- Dashboard membaca data dari `data/dataset MLOps.csv` (baseline) dan `monitoring/logs/prediction_logs.csv` (production)
- Menghitung PSI untuk setiap fitur secara real-time
- Menampilkan visualisasi distribusi untuk analisis drift

## 🔄 Production Workflow

```
1. TRAINING (Satu kali / Periodic)
   └─> python src/train.py
   └─> Model disimpan di models/distraction_model/
   └─> Track experiment di MLflow

2. DEPLOYMENT & PREDICTION (Continuous)
   └─> Load model: load_model()
   └─> Terima input baru (dari API/form/etc)
   └─> Prediksi: predict(model, input_data)
   └─> Log prediksi: log_prediction() → prediction_logs.csv

3. MONITORING (Berkala: Harian/Mingguan)
   
   A. Evidently Drift Detection:
      └─> python monitoring/monitor.py
      └─> Review drift.json
   
   B. Dashboard Visual Monitoring:
      └─> streamlit run monitoring/dashboard/app.py
      └─> Monitor PSI per fitur
      └─> Cek status drift (hijau/kuning/merah)

4. RETRAINING (Jika Drift Terdeteksi)
   └─> Jika PSI ≥ 0.25 atau drift.json menunjukkan drift
   └─> Gabungkan data baseline + production
   └─> python src/train.py (dengan data gabungan)
   └─> Deploy model baru
```

## 📊 MLflow Experiment Tracking

MLflow otomatis mencatat setiap training run:

**Yang di-track:**
- ✅ **Parameters**: n_estimators, max_depth, random_state
- ✅ **Metrics**: accuracy
- ✅ **Model**: Disimpan dalam MLflow format
- ✅ **Metadata**: Timestamp, duration, status

**Cara Melihat Experiments:**
```bash
mlflow ui --port 5000
```

Buka browser ke `http://localhost:5000` untuk:
- Compare runs side-by-side
- Visualisasi metrics
- Download model artifacts
- Track experiment history

## 📈 Model Monitoring dengan Evidently AI

### Evidently - Data Drift Detection

File `monitoring/monitor.py` menggunakan Evidently untuk deteksi drift statistik.

**Cara Kerja:**
1. Load baseline data (training dataset)
2. Load production data (prediction logs)
3. Hitung drift menggunakan DataDriftMetric
4. Generate JSON report

**Interpretasi Drift Report (`drift.json`):**
- `drift_detected = true`: Ada perubahan distribusi signifikan
- `drift_score`: Score per fitur (0-1, semakin tinggi semakin besar drift)
- `drift_by_columns`: Detail drift per fitur

### Streamlit Dashboard - PSI Monitoring

File `monitoring/dashboard/app.py` menggunakan PSI (Population Stability Index) untuk monitoring visual.

**PSI (Population Stability Index):**
- Mengukur pergeseran distribusi data antara baseline dan production
- Formula: PSI = Σ (% expected - % actual) × ln(% expected / % actual)

**Interpretasi PSI:**
- **PSI < 0.1**: 🟢 Tidak ada drift signifikan (populasi stabil)
- **0.1 ≤ PSI < 0.25**: 🟡 Drift moderat (monitor lebih ketat)
- **PSI ≥ 0.25**: 🔴 Drift signifikan (retraining direkomendasikan)

**Fitur Dashboard:**
- Tabel PSI per fitur dengan status warna
- Histogram distribusi baseline vs production
- Data produksi terbaru (10 rows terakhir)
- Pilih fitur untuk visualisasi detail

## 🔧 Hyperparameter Tuning

Untuk melakukan hyperparameter tuning, edit parameter di file `src/train.py`:

```python
params = {
    "n_estimators": 200,      # Coba: 100, 200, 300, 500
    "max_depth": 10,          # Coba: 5, 10, 15, None
    "min_samples_split": 2,   # Tambahkan: 2, 5, 10
    "min_samples_leaf": 1,    # Tambahkan: 1, 2, 4
    "random_state": 42
}
```

**Workflow Tuning:**
```bash
# Experiment 1
python src/train.py

# Edit params
# Experiment 2
python src/train.py

# Edit params
# Experiment 3
python src/train.py
```

Bandingkan hasil di MLflow UI untuk memilih model terbaik.

## 📦 Dependencies

### Core Libraries
- `mlflow==3.7.0` - Experiment tracking & model registry
- `scikit-learn==1.6.1` - Machine learning framework
- `pandas==2.2.3` - Data manipulation
- `numpy==1.25.2` - Numerical computing

### Monitoring & Dashboard
- `evidently` - Statistical drift detection
- `streamlit` - Interactive dashboard
- `matplotlib` - Visualisasi

### MLflow Components
- `cloudpickle==3.1.2` - Model serialization
- `psutil==5.9.5` - System monitoring
- `scipy==1.15.2` - Scientific computing

**Install Monitoring Dependencies:**
```bash
pip install evidently streamlit matplotlib
```

## 🛠️ Troubleshooting

### Error: Module 'evidently' or 'streamlit' not found
```bash
pip install evidently streamlit matplotlib
```

### Dashboard error: Baseline file not found
Edit path di `monitoring/dashboard/app.py`:
```python
baseline_path = "data/dataset MLOps.csv"  # Sesuaikan path
production_path = "monitoring/logs/prediction_logs.csv"
```

### MLflow UI tidak muncul
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

### Error saat monitoring: File not found
- Pastikan sudah ada `monitoring/logs/prediction_logs.csv`
- Jalankan beberapa prediksi dengan `log_prediction()` terlebih dahulu

### Dashboard menunjukkan "Belum ada data produksi"
- Jalankan prediksi minimal 5-10 kali
- Pastikan setiap prediksi di-log dengan `log_prediction()`

### Model tidak ditemukan saat predict
```bash
# Pastikan model sudah di-train
ls models/distraction_model/
# Jika kosong, jalankan:
python src/train.py
```

### Error saat training: Column not found
- Pastikan `data/dataset MLOps.csv` menggunakan separator `;` (semicolon)
- Cek nama kolom sesuai dengan yang ada di `preprocess.py`

### Dashboard PSI calculation error
- Pastikan kolom production log

-----

## 👥 Anggota Kelompok

1. Sesilia Putri Subandi (122450012)
2. Andre Hadiman Rotua Parhusip (122450108)
3. Syalaisha Andina Putriansyah (122450121)
4. Anam

-----

**Catatan:** Proyek ini dikembangkan untuk memenuhi Tugas Besar Mata Kuliah MLOps.

```

