import cv2
import numpy as np
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor import predict_frame

app = FastAPI(title="Sign Language Recognition API")

# ==========================================================
# CORS — izinkan frontend (localhost) akses backend
# ==========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Ganti dengan domain spesifik saat production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ==========================================================
# REQUEST SCHEMA
# ==========================================================
class PredictRequest(BaseModel):
    image: str   # Base64-encoded JPEG frame dari kamera
    mode: str    # "huruf" | "angka" | "kata"

# ==========================================================
# ENDPOINTS
# ==========================================================
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sign Language API berjalan"}

@app.post("/predict")
def predict(req: PredictRequest):
    # Validasi mode
    if req.mode not in ["huruf", "angka", "kata"]:
        raise HTTPException(status_code=400, detail="Mode tidak valid. Gunakan: huruf, angka, atau kata")

    try:
        # Decode base64 → bytes → numpy array → frame OpenCV
        image_data = base64.b64decode(req.image)
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Gagal decode gambar")

        # Jalankan prediksi
        result = predict_frame(frame, req.mode)
        return {"result": result, "mode": req.mode}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))