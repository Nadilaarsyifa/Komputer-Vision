import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque, Counter

# =========================
# PATH MODEL
# =========================
MODEL_PATH = "model_angka_video_laptop.h5"
ENCODER_PATH = "label_encoder_angka_video.pkl"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 63
CONFIDENCE_THRESHOLD = 0.70

# Agar realtime tidak terlalu berat
PREDICT_EVERY = 5

# Kalau tangan hilang beberapa frame, reset prediksi
NO_HAND_LIMIT = 10

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH, compile=False)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Classes:", label_encoder.classes_)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# NORMALISASI LANDMARK
# =========================
def normalize_landmarks(hand_landmarks):
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    scale = np.linalg.norm(landmarks[9])

    if scale < 1e-6:
        scale = 1.0

    landmarks = landmarks / scale

    return landmarks.flatten()

# =========================
# REALTIME
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak terbuka. Coba ganti VideoCapture(0) menjadi VideoCapture(1).")
    exit()

sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions_buffer = deque(maxlen=10)

final_prediction = "-"
final_confidence = 0.0
frame_counter = 0
no_hand_counter = 0

print("Tekan Q untuk keluar")
print("Tekan R untuk reset prediksi")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame kamera tidak terbaca.")
        break

    frame_counter += 1
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        no_hand_counter = 0

        hand_landmarks = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        landmark_data = normalize_landmarks(hand_landmarks)
        sequence.append(landmark_data)

    else:
        no_hand_counter += 1

    # Kalau tangan hilang beberapa frame, reset hasil
    if no_hand_counter >= NO_HAND_LIMIT:
        sequence.clear()
        predictions_buffer.clear()
        final_prediction = "-"
        final_confidence = 0.0

    # Prediksi hanya kalau tangan terdeteksi, sequence penuh, dan tiap beberapa frame
    if (
        hand_detected
        and len(sequence) == SEQUENCE_LENGTH
        and frame_counter % PREDICT_EVERY == 0
    ):
        input_data = np.expand_dims(np.array(sequence, dtype=np.float32), axis=0)

        pred = model.predict(input_data, verbose=0)

        pred_index = int(np.argmax(pred))
        confidence = float(np.max(pred))

        pred_label = label_encoder.inverse_transform([pred_index])[0]

        if confidence >= CONFIDENCE_THRESHOLD:
            predictions_buffer.append(pred_label)

            final_prediction = Counter(predictions_buffer).most_common(1)[0][0]
            final_confidence = confidence

    status_text = "Hand detected" if hand_detected else "No hand"

    cv2.putText(frame, f"Prediksi: {final_prediction}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(frame, f"Confidence: {final_confidence:.2f}", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Status: {status_text}", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, "Q: keluar | R: reset", (30, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Realtime Angka Video", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        sequence.clear()
        predictions_buffer.clear()
        final_prediction = "-"
        final_confidence = 0.0
        frame_counter = 0
        no_hand_counter = 0
        print("Sequence dan prediksi di-reset.")

cap.release()
cv2.destroyAllWindows()