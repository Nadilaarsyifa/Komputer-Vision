import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# LABEL KATA
# ==========================================================
LABEL_KATA = {
    0: "berdoa",
    1: "berjalan",
    2: "berpikir",
    3: "makan",
    4: "mandi",
    5: "saya",
    6: "tidur",
}

FACE_POINTS = [1, 33, 61, 199, 263]
POSE_POINTS = [11, 12, 13, 14, 15]

# ==========================================================
# Load MediaPipe Holistic (sekali saja saat startup)
# ==========================================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================================
# Load Model (sekali saja saat startup)
# ==========================================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_huruf = joblib.load(os.path.join(MODEL_DIR, "model_alfabet.pkl"))
model_angka = joblib.load(os.path.join(MODEL_DIR, "model_angka.pkl"))
model_kata  = joblib.load(os.path.join(MODEL_DIR, "model_kata.pkl"))

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def normalize_landmarks(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks])
    base = pts[0]
    pts = pts - base
    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val
    return pts.flatten().tolist()

def pairwise_distances(landmarks):
    pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    dist = []
    for a, b in pairs:
        dx = landmarks[a].x - landmarks[b].x
        dy = landmarks[a].y - landmarks[b].y
        dist.append((dx**2 + dy**2) ** 0.5)
    return dist

def dist_xy(a, b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

# ==========================================================
# FUNGSI UTAMA: predict_frame
# Input  : frame (numpy array BGR dari OpenCV), mode (str)
# Output : string hasil prediksi
# ==========================================================
def predict_frame(frame: np.ndarray, mode: str) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)
    pred = "-"

    # ----------------------------------------------------------
    # MODE HURUF
    # ----------------------------------------------------------
    if mode == "huruf":
        hand = result.right_hand_landmarks or result.left_hand_landmarks
        if hand:
            data = []
            for lm in hand.landmark:
                data.append(lm.x)
                data.append(lm.y)
            if len(data) == 42:
                data = np.array(data).reshape(1, -1)
                pred = str(model_huruf.predict(data)[0])

    # ----------------------------------------------------------
    # MODE ANGKA
    # ----------------------------------------------------------
    elif mode == "angka":
        hand = result.right_hand_landmarks or result.left_hand_landmarks
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
                return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

            data.append(dist(lm_list[4], lm_list[8]))
            data.append(dist(lm_list[8], lm_list[12]))
            data.append(dist(lm_list[12], lm_list[16]))
            data.append(dist(lm_list[16], lm_list[20]))

            data = np.array(data).reshape(1, -1)
            pred = str(model_angka.predict(data)[0])

    # ----------------------------------------------------------
    # MODE KATA
    # ----------------------------------------------------------
    elif mode == "kata":
        data = []

        if result.left_hand_landmarks:
            left = result.left_hand_landmarks.landmark
            data += normalize_landmarks(left)
            data += pairwise_distances(left)
        else:
            data += [0.0] * (42 + 4)

        if result.right_hand_landmarks:
            right = result.right_hand_landmarks.landmark
            data += normalize_landmarks(right)
            data += pairwise_distances(right)
        else:
            data += [0.0] * (42 + 4)

        if result.face_landmarks:
            face = result.face_landmarks.landmark
            base = np.array([face[1].x, face[1].y])
            for i in FACE_POINTS:
                lm = face[i]
                pt = np.array([lm.x, lm.y]) - base
                data += pt.tolist()
        else:
            data += [0.0] * 10

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
        if result.right_hand_landmarks and result.face_landmarks:
            hand = result.right_hand_landmarks.landmark
            face = result.face_landmarks.landmark
            extra.append(dist_xy(hand[8], face[1]))
            extra.append(dist_xy(hand[4], face[1]))
        else:
            extra += [0.0, 0.0]

        data += extra

        if len(data) == 114:
            data = np.array(data).reshape(1, -1)
            pred_index = int(model_kata.predict(data)[0])
            pred = LABEL_KATA.get(pred_index, f"?{pred_index}")
        else:
            pred = f"err:{len(data)}"

    return pred