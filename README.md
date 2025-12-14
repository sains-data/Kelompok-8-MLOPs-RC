# Student Smartphone Distraction Prediction System (MLOps)-Kelompok-8-MLOPs-RC

![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Docker](https://img.shields.io/badge/Docker-Available-blue)
![MLOps](https://img.shields.io/badge/MLOps-End--to--End-orange)

## 📋 Deskripsi Proyek
Proyek ini adalah implementasi **End-to-End MLOps** untuk memprediksi tingkat distraksi mahasiswa akibat penggunaan smartphone. Sistem ini mencakup pipeline data otomatis, pelatihan model (SVM vs Random Forest), deployment berbasis API (FastAPI) & Docker, serta mekanisme **Continuous Learning** melalui *feedback loop* dari pengguna.

Input prediksi menggunakan skala 1-8 pada 7 indikator perilaku (Frekuensi, Durasi, Tujuan, Kontrol, Persepsi, Manajemen, Upaya). Output berupa klasifikasi tingkat distraksi: **Low, Medium, atau High**.

---

## 🏗️ Arsitektur Sistem & Struktur Folder

````markdown
gadget-distraction-classifier/
├── .dockerignore          # Konfigurasi ignore Docker
├── Dockerfile             # Resep image Docker
├── requirements.txt       # Daftar dependensi Python
├── README.md              # Dokumentasi Proyek
│
├── data/
│   ├── raw/               # Dataset mentah (dataset tubes dl.csv)
│   ├── processed/         # Data hasil preprocessing (.npy)
│   └── feedback/          # Data baru dari input user (user_feedback.csv)
│
├── models/
│   ├── saved_models/      # Artifacts (production_model.pkl, scaler.pkl, encoder.pkl)
│   └── evaluation_reports/# Laporan kinerja model (Confusion Matrix)
│
├── src/
│   ├── api/               # Backend API (main.py)
│   ├── data/              # Pipeline Data (ingestion.py, preprocessing.py)
│   ├── models/            # Pipeline Model (train.py, evaluate.py)
│   └── monitoring/        # Pipeline Monitoring (drift_monitor.py)
│
└── web/
    └── index.html         # Frontend Interface User
````

-----

## 🚀 Cara Menjalankan (Local Development)

### 1\. Setup Environment

Pastikan Python 3.9+ terinstall.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2\. Pipeline Data & Training

Jalankan perintah berikut secara berurutan untuk menyiapkan model:

**Langkah A: Data Ingestion & Preprocessing**
Script ini membaca CSV, melakukan cleaning, binning target (1-8 -\> Low/Med/High), dan scaling.

```bash
python src/data/preprocessing.py
```

*Output: File `.npy` di `data/processed/` dan `.pkl` di `models/saved_models/`.*

**Langkah B: Model Training & Selection**
Script ini melatih **Random Forest** dan **SVM**, mencatat eksperimen (MLflow), dan otomatis menyimpan model dengan akurasi terbaik.

```bash
python src/models/train.py
```

*Output: `production_model.pkl` tersimpan.*

**Langkah C: Evaluasi Model**
Menghasilkan Confusion Matrix untuk laporan.

```bash
python src/models/evaluate.py
```

*Output: Cek gambar di folder `models/evaluation_reports/`.*

### 3\. Menjalankan Aplikasi (Deployment)

Jalankan server FastAPI:

```bash
uvicorn src.api.main:app --reload
```

Akses web interface di browser:

  * **Frontend:** Buka file `web/index.html` (atau akses via localhost jika sudah di-mount).
  * **Swagger UI:** `http://127.0.0.1:8000/docs`

-----

## 🐳 Cara Menjalankan dengan Docker

Untuk memastikan aplikasi berjalan konsisten di mana saja (Production Ready):

1.  **Build Image:**

    ```bash
    docker build -t distraction-app .
    ```

2.  **Run Container:**

    ```bash
    docker run -p 8000:8000 distraction-app
    ```

    Aplikasi dapat diakses di `http://localhost:8000`.

-----

## 🔄 Fitur MLOps (Monitoring & Feedback Loop)

Sistem ini memiliki kemampuan **Continuous Learning**:

1.  **Feedback Loop:** Saat user menggunakan Web App, mereka dapat memberikan validasi ("Apakah prediksi ini benar?"). Data validasi disimpan otomatis ke `data/feedback/user_feedback.csv`.

2.  **Drift Detection & Monitoring:**
    Jalankan script monitoring untuk mengecek apakah pola perilaku user berubah dibandingkan data training (Data Drift) dan mengukur akurasi real-time.

    ```bash
    python src/monitoring/drift_monitor.py
    ```

    *Fitur:* - Mendeteksi penurunan akurasi pada data real-world.

      - Memberikan peringatan ("ALERT") jika rata-rata input user bergeser signifikan (Drift).

-----

## 📊 Hasil Evaluasi Model

Model terbaik dipilih berdasarkan akurasi tertinggi pada data test (20% split).

  - **Algoritma:** Random Forest / SVM (Otomatis terpilih)
  - **Metrik Utama:** Accuracy, Precision, Recall
  - **Laporan Visual:** Lihat `models/evaluation_reports/confusion_matrix.png`

-----

## 👥 Anggota Kelompok

1. Sesilia Putri Subandi (122450012)
2. Andre Hadiman Rotua Parhusip (122450108)
3. Syalaisha Andina Putriansyah (122450121)
4. Anam

-----

**Catatan:** Proyek ini dikembangkan untuk memenuhi Tugas Besar Mata Kuliah MLOps.

```
