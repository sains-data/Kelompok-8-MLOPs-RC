# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict
from src.utils import log_prediction

app = FastAPI()
model = load_model()

class InputData(BaseModel):
    Mahasiswa_Angkatan: int
    frekuensi: int
    durasi: int
    tujuan: int
    sulit_kontrol: int
    persepsi: int
    kemampuan_waktu: int
    upaya: int

@app.post("/predict")
def predict_distraction(data: InputData):

    input_dict = {
        "Mahasiswa Angkatan": data.Mahasiswa_Angkatan,
        "Seberapa sering (frekuensi) penggunaan gadget Kamu setiap hari?": data.frekuensi,
        "Berapa durasi penggunaan gadget Kamu, di luar jam perkuliahan?": data.durasi,
        "Bagaimana tujuan utama penggunaan gadget Kamu?": data.tujuan,
        "Seberapa sulit Kamu dalam mengontrol waktu penggunaan gadget mu?": data.sulit_kontrol,
        "Bagaimana presepsi Kamu terhadap pengaruh penggunaan gadget pada kondisi Akademik  ?": data.persepsi,
        "Bagaimana kemampuan Kamu dalam mengatur waktu antara menggunakan gadget dengan aktivitas akademik lain ?": data.kemampuan_waktu,
        "Bagaimana upaya Kamu mengurangi intensitas penggunaan gadget?": data.upaya
    }

    # PREDIKSI
    result = predict(model, input_dict)

    # LOGGING PREDIKSI (format CSV benar)
    log_prediction(input_dict, result)

    return {"prediction": result}
