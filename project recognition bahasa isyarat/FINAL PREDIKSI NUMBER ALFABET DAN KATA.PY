import cv2
import mediapipe as mp
import numpy as np
import joblib
import pickle
import warnings
from collections import deque, Counter
from tensorflow.keras.models import load_model
warnings.filterwarnings("ignore")

# ==========================================================
# LOAD MODEL
# ==========================================================
model_huruf        = joblib.load("model_bisindo_alfabet.pkl")
label_encoder_huruf = joblib.load("label_encoder_alfabet.pkl")

# Model angka sekarang memakai video sequence / BiLSTM
model_angka_video = load_model("model_angka_video_laptop.h5", compile=False)
with open("label_encoder_angka_video.pkl", "rb") as f:
    label_encoder_angka_video = pickle.load(f)
model_kata         = joblib.load("model_kata_mlp.pkl")
scaler_kata        = joblib.load("scaler_kata_mlp.pkl")
label_encoder_kata = joblib.load("label_encoder_kata.pkl")

# ==========================================================
# THRESHOLD & SMOOTHING
# Naikkan CONFIDENCE_THRESHOLD kalau prediksi masih goyang
# ==========================================================
CONFIDENCE_THRESHOLD = 0.5   # di bawah ini tampilkan "-"
SMOOTH_WINDOW        = 10    # ambil prediksi mayoritas dari N frame terakhir

pred_history = deque(maxlen=SMOOTH_WINDOW)

# ==========================================================
# KONFIGURASI KHUSUS MODE ANGKA VIDEO
# ==========================================================
SEQUENCE_LENGTH_ANGKA = 30
FEATURE_SIZE_ANGKA    = 63
PREDICT_EVERY_ANGKA   = 5
NO_HAND_LIMIT_ANGKA   = 10
CONF_THRESHOLD_ANGKA  = 0.70

sequence_angka        = deque(maxlen=SEQUENCE_LENGTH_ANGKA)
pred_buffer_angka     = deque(maxlen=10)
final_pred_angka      = "-"
final_conf_angka      = 0.0
frame_counter_angka   = 0
no_hand_counter_angka = 0

# ==========================================================
# MEDIAPIPE
# ==========================================================
mp_hands    = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FACE_POINTS = [1, 33, 61, 199, 263]
POSE_POINTS = [11, 12, 13, 14, 15]
CHIN        = 152
NOSE        = 1
FOREHEAD    = 10    # dahi → Berpikir
CHEEK_RIGHT = 234   # pipi kanan → Tolong
CHEEK_LEFT  = 454   # pipi kiri
MOUTH       = 13    # mulut → Berdoa

mode = "huruf"
cap  = cv2.VideoCapture(0)

# ==========================================================
# FUNGSI SMOOTHING
# ==========================================================
def get_smooth_pred(new_pred):
    pred_history.append(new_pred)
    # ambil prediksi yang paling sering muncul
    from collections import Counter
    count = Counter(pred_history)
    return count.most_common(1)[0][0]

# ==========================================================
# FUNGSI EKSTRAK 1 TANGAN — huruf (46 fitur)
# ==========================================================
def extract_one_hand(hand_landmarks):
    lm  = hand_landmarks.landmark
    pts = np.array([[p.x, p.y] for p in lm])

    base = pts[0]
    pts  = pts - base

    max_val = np.max(np.abs(pts))
    if max_val != 0:
        pts = pts / max_val

    data = pts.flatten().tolist()

    def dist(a, b):
        return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2)**0.5

    data.append(dist(4, 8))
    data.append(dist(8, 12))
    data.append(dist(12, 16))
    data.append(dist(16, 20))

    return data

# ==========================================================
# FUNGSI EKSTRAK ANGKA VIDEO — 1 TANGAN, 63 FITUR
# ==========================================================
def normalize_landmarks_angka_video(hand_landmarks):
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


def reset_angka_video():
    global final_pred_angka, final_conf_angka, frame_counter_angka, no_hand_counter_angka
    sequence_angka.clear()
    pred_buffer_angka.clear()
    final_pred_angka = "-"
    final_conf_angka = 0.0
    frame_counter_angka = 0
    no_hand_counter_angka = 0


# ==========================================================
# FUNGSI EKSTRAK KATA — 123 fitur (sama persis dengan CSV baru)
# ==========================================================
def extract_kata_features(result):
    data = []

    def normalize_landmarks(landmarks):
        pts = np.array([[lm.x, lm.y] for lm in landmarks])
        base = pts[0]
        pts  = pts - base
        mv   = np.max(np.abs(pts))
        if mv != 0:
            pts = pts / mv
        return pts.flatten().tolist()

    def pairwise_distances(landmarks):
        pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
        return [((landmarks[a].x - landmarks[b].x)**2 +
                 (landmarks[a].y - landmarks[b].y)**2)**0.5
                for a, b in pairs]

    def dist_xy(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5

    # LEFT HAND
    if result.left_hand_landmarks:
        left = result.left_hand_landmarks.landmark
        data += normalize_landmarks(left)
        data += pairwise_distances(left)
    else:
        data += [0.0] * 46

    # RIGHT HAND
    if result.right_hand_landmarks:
        right = result.right_hand_landmarks.landmark
        data += normalize_landmarks(right)
        data += pairwise_distances(right)
    else:
        data += [0.0] * 46

    # FACE
    if result.face_landmarks:
        face = result.face_landmarks.landmark
        base = np.array([face[1].x, face[1].y])
        for i in FACE_POINTS:
            lm = face[i]
            pt = np.array([lm.x, lm.y]) - base
            data += pt.tolist()
    else:
        data += [0.0] * 10

    # POSE
    if result.pose_landmarks:
        pose = result.pose_landmarks.landmark
        base = np.array([pose[11].x, pose[11].y])
        for i in POSE_POINTS:
            lm = pose[i]
            pt = np.array([lm.x, lm.y]) - base
            data += pt.tolist()
    else:
        data += [0.0] * 10

    # CROSS FEATURES ASLI (2 fitur)
    extra = []
    if result.right_hand_landmarks and result.face_landmarks:
        hand = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        extra.append(dist_xy(hand[8], face[NOSE]))
        extra.append(dist_xy(hand[4], face[NOSE]))
    else:
        extra += [0.0, 0.0]

    # FITUR TAMBAHAN BARU (9 fitur)
    if result.right_hand_landmarks and result.face_landmarks and result.pose_landmarks:
        rh   = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        pose = result.pose_landmarks.landmark
        extra.append(dist_xy(rh[8], face[CHIN]))
        extra.append(dist_xy(rh[8], face[NOSE]))
        extra.append(dist_xy(rh[8], pose[12]))
        extra.append(dist_xy(rh[4], face[CHIN]))
    else:
        extra += [0.0] * 4

    if result.left_hand_landmarks and result.face_landmarks and result.pose_landmarks:
        lh   = result.left_hand_landmarks.landmark
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
        rh   = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        y_rel = rh[8].y - face[NOSE].y
        extra.append(y_rel)
    else:
        extra.append(0.0)

    # FITUR TAMBAHAN BARU 2 (6 fitur)
    # Bedakan Tolong vs Berpikir, Apa vs Berdoa

    # Tangan kanan ke dahi, pipi, mulut
    if result.right_hand_landmarks and result.face_landmarks:
        rh   = result.right_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        extra.append(dist_xy(rh[8], face[FOREHEAD]))     # telunjuk → dahi
        extra.append(dist_xy(rh[8], face[CHEEK_RIGHT]))  # telunjuk → pipi kanan
        extra.append(dist_xy(rh[8], face[MOUTH]))        # telunjuk → mulut
    else:
        extra += [0.0] * 3

    # Tangan kiri ke dahi, pipi, mulut
    if result.left_hand_landmarks and result.face_landmarks:
        lh   = result.left_hand_landmarks.landmark
        face = result.face_landmarks.landmark
        extra.append(dist_xy(lh[8], face[FOREHEAD]))
        extra.append(dist_xy(lh[8], face[CHEEK_LEFT]))
        extra.append(dist_xy(lh[8], face[MOUTH]))
    else:
        extra += [0.0] * 3

    data += extra
    return data  # 129 fitur

# ==========================================================
# LOOP CAMERA
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    raw_pred  = "-"
    pred      = "-"
    conf_text = ""

    # ======================================================
    # MODE HURUF
    # ======================================================
    if mode == "huruf":
        result     = hands.process(rgb)
        empty_hand = [0.0] * 46
        left_data  = empty_hand.copy()
        right_data = empty_hand.copy()

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness
            ):
                # Tampilkan landmark tangan agar tampilan konsisten dengan mode angka
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                label = handedness.classification[0].label
                if label == "Left":
                    left_data  = extract_one_hand(hand_landmarks)
                elif label == "Right":
                    right_data = extract_one_hand(hand_landmarks)

        data = left_data + right_data

        if result.multi_hand_landmarks:
            proba    = model_huruf.predict_proba(np.array(data).reshape(1, -1))[0]
            max_conf = np.max(proba)
            conf_text = f"{max_conf*100:.0f}%"
        
            if max_conf >= CONFIDENCE_THRESHOLD:
                pred_encoded = np.argmax(proba)
                raw_pred = label_encoder_huruf.inverse_transform([pred_encoded])[0]
                pred = get_smooth_pred(raw_pred)
            else:
                pred_history.clear()
                pred = "-"
        else:
            pred_history.clear()
            pred = "-"
    # ======================================================
    # MODE ANGKA — VIDEO SEQUENCE / BiLSTM
    # ======================================================
    elif mode == "angka":
        result = hands.process(rgb)
        hand_detected = False

        frame_counter_angka += 1

        if result.multi_hand_landmarks:
            hand_detected = True
            no_hand_counter_angka = 0

            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            landmark_data = normalize_landmarks_angka_video(hand)
            sequence_angka.append(landmark_data)

        else:
            no_hand_counter_angka += 1

        # Kalau tangan hilang beberapa frame, kosongkan prediksi
        if no_hand_counter_angka >= NO_HAND_LIMIT_ANGKA:
            reset_angka_video()
            pred = "-"
            conf_text = ""

        # Prediksi hanya kalau tangan terdeteksi, sequence penuh, dan tidak setiap frame
        if (
            hand_detected
            and len(sequence_angka) == SEQUENCE_LENGTH_ANGKA
            and frame_counter_angka % PREDICT_EVERY_ANGKA == 0
        ):
            input_data = np.expand_dims(
                np.array(sequence_angka, dtype=np.float32),
                axis=0
            )

            proba = model_angka_video.predict(input_data, verbose=0)[0]
            max_conf = float(np.max(proba))
            conf_text = f"{max_conf*100:.0f}%"

            if max_conf >= CONF_THRESHOLD_ANGKA:
                pred_encoded = int(np.argmax(proba))
                raw_pred = label_encoder_angka_video.inverse_transform([pred_encoded])[0]

                pred_buffer_angka.append(raw_pred)
                final_pred_angka = Counter(pred_buffer_angka).most_common(1)[0][0]
                final_conf_angka = max_conf
            else:
                pred_buffer_angka.clear()
                final_pred_angka = "-"
                final_conf_angka = 0.0

        pred = final_pred_angka
        if final_conf_angka > 0:
            conf_text = f"{final_conf_angka*100:.0f}%"

    # ======================================================
    # MODE KATA — MLP + scaler + label encoder
    # ======================================================
    elif mode == "kata":
        result = holistic.process(rgb)
        data   = extract_kata_features(result)

        if len(data) == 129:
            # Tampilkan landmark tangan saja agar layar tidak terlalu penuh
            if result.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    result.left_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            if result.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    result.right_hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            hand_detected = (result.left_hand_landmarks or result.right_hand_landmarks)

            if hand_detected:
                # WAJIB: scale dulu sebelum masuk MLP
                data_scaled = scaler_kata.transform(np.array(data).reshape(1, -1))
                proba        = model_kata.predict_proba(data_scaled)[0]
                max_conf     = np.max(proba)
                conf_text    = f"{max_conf*100:.0f}%"

                if max_conf >= CONFIDENCE_THRESHOLD:
                    pred_encoded = np.argmax(proba)
                    raw_pred     = label_encoder_kata.inverse_transform([pred_encoded])[0]
                    pred         = get_smooth_pred(raw_pred)
                else:
                    pred_history.clear()
                    pred = "-"
            else:
                pred_history.clear()
                pred = "-"
        else:
            pred = f"err:{len(data)}"

    # ======================================================
    # TAMPILAN
    # ======================================================
    cv2.putText(frame, f"MODE  : {mode.upper()}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"HASIL : {pred}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    if conf_text:
        cv2.putText(frame, f"CONF  : {conf_text}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(frame, "A=Huruf  N=Angka  K=Kata  Q=Keluar",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Final Project Sign Language", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        mode = "huruf"
        pred_history.clear()
        reset_angka_video()
    elif key == ord('n'):
        mode = "angka"
        pred_history.clear()
        reset_angka_video()
    elif key == ord('k'):
        mode = "kata"
        pred_history.clear()
        reset_angka_video()
    elif key == ord('q'):
        break

cap.release()
hands.close()
holistic.close()
cv2.destroyAllWindows()