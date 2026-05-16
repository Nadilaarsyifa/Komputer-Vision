import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import predict_frame

app = FastAPI(title="Sign Language Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image: str
    mode: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sign Language API berjalan"}

@app.post("/predict")
def predict(req: PredictRequest):
    if req.mode not in ["huruf", "angka", "kata"]:
        raise HTTPException(status_code=400, detail="Mode tidak valid. Gunakan huruf, angka, atau kata.")

    try:
        image_data = req.image
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Gagal membaca gambar dari frontend")

        result = predict_frame(frame, req.mode)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
