CARA PAKAI PROJECT INI

1. Pakai struktur ini:
   sign-language-app/
   ├── backend/
   │   ├── main.py
   │   ├── predictor.py
   │   ├── requirements.txt
   │   └── models/
   │       ├── label_encoder_alfabet.pkl
   │       ├── label_encoder_kata.pkl
   │       ├── label_encoder_number.pkl
   │       ├── model_bisindo_alfabet.pkl
   │       ├── model_kata_mlp.pkl
   │       ├── model_number_xgboost.pkl
   │       └── scaler_kata_mlp.pkl
   └── frontend/
       ├── index.html
       ├── app.js
       ├── style.css
       └── images/

2. Jangan pakai app.py Flask lama dan jangan pakai templates/index.html lama.
   Untuk versi ini yang dipakai adalah FastAPI + frontend HTML/JS.

3. Jalankan backend:
   cd sign-language-app/backend
   pip install -r requirements.txt
   uvicorn main:app --reload

4. Buka frontend:
   Buka file frontend/index.html di browser.
   Kalau kamera tidak mau aktif karena dibuka sebagai file, pakai Live Server di VS Code.

5. Test API:
   buka http://127.0.0.1:8000
   harus muncul status ok.
