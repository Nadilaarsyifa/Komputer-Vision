import os
import warnings
from collections import Counter, deque

import cv2
import joblib
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

model_angka = joblib.load(os.path.join(MODEL_DIR, "model_number_xgboost.pkl"))
label_encoder_angka = joblib.load(os.path.join(MODEL_DIR, "label_encoder_number.pkl"))

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
        result = hands.process(rgb)
        hand = result.multi_hand_landmarks[0] if result.multi_hand_landmarks else None

        if hand:
            lm_list = hand.landmark
            data = []
            base_x = lm_list[0].x
            base_y = lm_list[0].y

            for lm in lm_list:
                data.append(lm.x - base_x)
                data.append(lm.y - base_y)

            data = np.array(data)
            data = data - np.min(data)
            if np.max(data) != 0:
                data = data / np.max(data)
            data = data.tolist()

            def dist(a, b):
                return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

            data.append(dist(lm_list[4], lm_list[8]))
            data.append(dist(lm_list[8], lm_list[12]))
            data.append(dist(lm_list[12], lm_list[16]))
            data.append(dist(lm_list[16], lm_list[20]))

            proba = model_angka.predict_proba(np.array(data).reshape(1, -1))[0]
            confidence = float(np.max(proba))

            if confidence >= CONFIDENCE_THRESHOLD:
                pred_encoded = int(np.argmax(proba))
                raw_pred = label_encoder_angka.inverse_transform([pred_encoded])[0]
                pred = get_smooth_pred(str(raw_pred))
            else:
                pred_history.clear()
        else:
            pred_history.clear()

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
