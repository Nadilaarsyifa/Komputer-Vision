import os
import warnings
from collections import Counter, deque

import cv2
import joblib
import pickle
import mediapipe as mp
import numpy as np

warnings.filterwarnings("ignore")

# ==========================================================
# PATH MODEL
# ==========================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Nama file disesuaikan dengan folder models kamu
model_huruf = joblib.load(os.path.join(MODEL_DIR, "model_bisindo_alfabet.pkl"))
label_encoder_huruf = joblib.load(os.path.join(MODEL_DIR, "label_encoder_alfabet.pkl"))

from tensorflow.keras.models import load_model

# Model angka revisi: video sequence / BiLSTM
model_angka_video = load_model(os.path.join(MODEL_DIR, "model_angka_video_laptop.h5"), compile=False)
with open(os.path.join(MODEL_DIR, "label_encoder_angka_video.pkl"), "rb") as f:
    label_encoder_angka_video = pickle.load(f)

model_kata = joblib.load(os.path.join(MODEL_DIR, "model_kata_mlp.pkl"))
scaler_kata = joblib.load(os.path.join(MODEL_DIR, "scaler_kata_mlp.pkl"))
label_encoder_kata = joblib.load(os.path.join(MODEL_DIR, "label_encoder_kata.pkl"))

# ==========================================================
# THRESHOLD & SMOOTHING
# ==========================================================
CONFIDENCE_THRESHOLD = 0.5
SMOOTH_WINDOW = 10
pred_history = deque(maxlen=SMOOTH_WINDOW)
last_mode = None

# ==========================================================
# KONFIGURASI KHUSUS MODE ANGKA VIDEO
# ==========================================================
SEQUENCE_LENGTH_ANGKA = 30
FEATURE_SIZE_ANGKA = 63
PREDICT_EVERY_ANGKA = 5
NO_HAND_LIMIT_ANGKA = 10
CONF_THRESHOLD_ANGKA = 0.70

sequence_angka = deque(maxlen=SEQUENCE_LENGTH_ANGKA)
pred_buffer_angka = deque(maxlen=10)
final_pred_angka = "-"
final_conf_angka = 0.0
frame_counter_angka = 0
no_hand_counter_angka = 0

# ==========================================================
# MEDIAPIPE
# ==========================================================
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

FACE_POINTS = [1, 33, 61, 199, 263]
POSE_POINTS = [11, 12, 13, 14, 15]
CHIN = 152
NOSE = 1
FOREHEAD = 10
CHEEK_RIGHT = 234
CHEEK_LEFT = 454
MOUTH = 13

# ==========================================================
# HELPER
# ==========================================================
def get_smooth_pred(new_pred):
    pred_history.append(new_pred)
    count = Counter(pred_history)
    return count.most_common(1)[0][0]


def reset_history_if_mode_changed(mode):
    global last_mode
    if last_mode != mode:
        pred_history.clear()
        if last_mode == "angka" or mode == "angka":
            reset_angka_video()
        last_mode = mode


def extract_one_hand(hand_landmarks):
    lm = hand_landmarks.landmark
    pts = np.array([[p.x, p.y] for p in lm])

    base = pts[0]
    pts = pts - base

    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val

    data = pts.flatten().tolist()

    def dist(a, b):
        return ((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2) ** 0.5

    data.append(dist(4, 8))
    data.append(dist(8, 12))
    data.append(dist(12, 16))
    data.append(dist(16, 20))

    return data


def normalize_landmarks_angka_video(hand_landmarks):
    """Ekstraksi fitur angka sesuai model video BiLSTM: 21 landmark x,y,z = 63 fitur."""
    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32,
    )

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Skala berdasarkan jarak wrist ke MCP jari tengah agar ukuran tangan lebih stabil.
    scale = np.linalg.norm(landmarks[9])
    if scale < 1e-6:
        scale = 1.0

    landmarks = landmarks / scale
    return landmarks.flatten()


def reset_angka_video():
    global final_pred_angka, final_conf_angka, frame_counter_angka, no_hand_counter_angka
    sequence_angka.clear()
    pred_buffer_angka.clear()
    final_pred_angka = "-"
    final_conf_angka = 0.0
    frame_counter_angka = 0
    no_hand_counter_angka = 0


def extract_kata_features(result):
    data = []

    def normalize_landmarks(landmarks):
        pts = np.array([[lm.x, lm.y] for lm in landmarks])
        base = pts[0]
        pts = pts - base
        mv = np.max(np.abs(pts))
        if mv != 0:
            pts = pts / mv
        return pts.flatten().tolist()

    def pairwise_distances(landmarks):
        pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
        return [
            ((landmarks[a].x - landmarks[b].x) ** 2 +
             (landmarks[a].y - landmarks[b].y) ** 2) ** 0.5
            for a, b in pairs
        ]

    def dist_xy(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    # LEFT HAND: 46 fitur
    if result.left_hand_landmarks:
        left = result.left_hand_landmarks.landmark
        data += normalize_landmarks(left)
        data += pairwise_distances(left)
    else:
        data += [0.0] * 46

    # RIGHT HAND: 46 fitur
    if result.right_hand_landmarks:
        right = result.right_hand_landmarks.landmark
        data += normalize_landmarks(right)
        data += pairwise_distances(right)
    else:
        data += [0.0] * 46

    # FACE: 10 fitur
    if result.face_landmarks:
        face = result.face_landmarks.landmark
        base = np.array([face[1].x, face[1].y])
        for i in FACE_POINTS:
            lm = face[i]
            pt = np.array([lm.x, lm.y]) - base
            data += pt.tolist()
    else:
        data += [0.0] * 10

    # POSE: 10 fitur
    if result.pose_landmarks:
        pose = result.pose_landmarks.landmark
        base = np.array([pose[11].x, pose[11].y])
        for i in POSE_POINTS:
            lm = pose[i]
            pt = np.array([lm.x, lm.y]) - base
            data += pt.tolist()
    else:
        data += [0.0] * 10

    extra = []

    # CROSS FEATURES ASLI: 2 fitur
    if result.right_hand_landmarks and result.face_landmarks:
        hand = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        extra.append(dist_xy(hand[8], face[NOSE]))
        extra.append(dist_xy(hand[4], face[NOSE]))
    else:
        extra += [0.0, 0.0]

    # FITUR TAMBAHAN BARU: 9 fitur
    if result.right_hand_landmarks and result.face_landmarks and result.pose_landmarks:
        rh = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        pose = result.pose_landmarks.landmark
        extra.append(dist_xy(rh[8], face[CHIN]))
        extra.append(dist_xy(rh[8], face[NOSE]))
        extra.append(dist_xy(rh[8], pose[12]))
        extra.append(dist_xy(rh[4], face[CHIN]))
    else:
        extra += [0.0] * 4

    if result.left_hand_landmarks and result.face_landmarks and result.pose_landmarks:
        lh = result.left_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        pose = result.pose_landmarks.landmark
        extra.append(dist_xy(lh[8], face[CHIN]))
        extra.append(dist_xy(lh[8], pose[11]))
        extra.append(dist_xy(lh[4], face[CHIN]))
    else:
        extra += [0.0] * 3

    both_hands = 1.0 if (result.right_hand_landmarks and result.left_hand_landmarks) else 0.0
    extra.append(both_hands)

    if result.right_hand_landmarks and result.face_landmarks:
        rh = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        y_rel = rh[8].y - face[NOSE].y
        extra.append(y_rel)
    else:
        extra.append(0.0)

    # FITUR TAMBAHAN BARU 2: 6 fitur
    if result.right_hand_landmarks and result.face_landmarks:
        rh = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        extra.append(dist_xy(rh[8], face[FOREHEAD]))
        extra.append(dist_xy(rh[8], face[CHEEK_RIGHT]))
        extra.append(dist_xy(rh[8], face[MOUTH]))
    else:
        extra += [0.0] * 3

    if result.left_hand_landmarks and result.face_landmarks:
        lh = result.left_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        extra.append(dist_xy(lh[8], face[FOREHEAD]))
        extra.append(dist_xy(lh[8], face[CHEEK_LEFT]))
        extra.append(dist_xy(lh[8], face[MOUTH]))
    else:
        extra += [0.0] * 3

    data += extra
    return data  # 129 fitur


# ==========================================================
# FUNGSI UTAMA UNTUK API
# ==========================================================
def predict_frame(frame: np.ndarray, mode: str) -> dict:
    reset_history_if_mode_changed(mode)

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pred = "-"
    confidence = 0.0

    if mode == "huruf":
        result = hands.process(rgb)
        empty_hand = [0.0] * 46
        left_data = empty_hand.copy()
        right_data = empty_hand.copy()

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label
                if label == "Left":
                    left_data = extract_one_hand(hand_landmarks)
                elif label == "Right":
                    right_data = extract_one_hand(hand_landmarks)

            data = left_data + right_data
            proba = model_huruf.predict_proba(np.array(data).reshape(1, -1))[0]
            confidence = float(np.max(proba))

            if confidence >= CONFIDENCE_THRESHOLD:
                pred_encoded = int(np.argmax(proba))
                raw_pred = label_encoder_huruf.inverse_transform([pred_encoded])[0]
                pred = get_smooth_pred(str(raw_pred))
            else:
                pred_history.clear()
        else:
            pred_history.clear()

    elif mode == "angka":
        global final_pred_angka, final_conf_angka, frame_counter_angka, no_hand_counter_angka

        result = hands.process(rgb)
        hand_detected = False
        frame_counter_angka += 1

        if result.multi_hand_landmarks:
            hand_detected = True
            no_hand_counter_angka = 0

            # Ambil 1 tangan pertama, sama seperti kode training/testing video.
            hand = result.multi_hand_landmarks[0]
            landmark_data = normalize_landmarks_angka_video(hand)

            if landmark_data.shape[0] == FEATURE_SIZE_ANGKA:
                sequence_angka.append(landmark_data)
        else:
            no_hand_counter_angka += 1

        if no_hand_counter_angka >= NO_HAND_LIMIT_ANGKA:
            reset_angka_video()
        elif (
            hand_detected
            and len(sequence_angka) == SEQUENCE_LENGTH_ANGKA
            and frame_counter_angka % PREDICT_EVERY_ANGKA == 0
        ):
            input_data = np.expand_dims(
                np.array(sequence_angka, dtype=np.float32),
                axis=0,
            )

            proba = model_angka_video.predict(input_data, verbose=0)[0]
            confidence = float(np.max(proba))

            if confidence >= CONF_THRESHOLD_ANGKA:
                pred_encoded = int(np.argmax(proba))
                raw_pred = label_encoder_angka_video.inverse_transform([pred_encoded])[0]

                pred_buffer_angka.append(str(raw_pred))
                final_pred_angka = Counter(pred_buffer_angka).most_common(1)[0][0]
                final_conf_angka = confidence
            else:
                pred_buffer_angka.clear()
                final_pred_angka = "-"
                final_conf_angka = 0.0

        pred = final_pred_angka
        confidence = final_conf_angka

    elif mode == "kata":
        result = holistic.process(rgb)
        data = extract_kata_features(result)

        if len(data) != 129:
            return {"result": f"err:{len(data)}", "confidence": 0.0}

        hand_detected = bool(result.left_hand_landmarks or result.right_hand_landmarks)
        if hand_detected:
            data_scaled = scaler_kata.transform(np.array(data).reshape(1, -1))
            proba = model_kata.predict_proba(data_scaled)[0]
            confidence = float(np.max(proba))

            if confidence >= CONFIDENCE_THRESHOLD:
                pred_encoded = int(np.argmax(proba))
                raw_pred = label_encoder_kata.inverse_transform([pred_encoded])[0]
                pred = get_smooth_pred(str(raw_pred))
            else:
                pred_history.clear()
        else:
            pred_history.clear()

    return {
        "result": pred,
        "confidence": round(confidence * 100, 2),
    }
