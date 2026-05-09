from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import joblib

app = Flask(__name__)

# ==========================================================
# LOAD MODEL
# ==========================================================
model_huruf = joblib.load("model_alfabet.pkl")
model_angka = joblib.load("model_angka.pkl")
model_kata  = joblib.load("model_kata.pkl")

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

# ==========================================================
# MEDIAPIPE
# ==========================================================
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================================
# LANDMARK
# ==========================================================
FACE_POINTS = [1, 33, 61, 199, 263]
POSE_POINTS = [11, 12, 13, 14, 15]

# ==========================================================
# MODE
# ==========================================================
mode = "huruf"

# ==========================================================
# CAMERA
# ==========================================================
camera = cv2.VideoCapture(0)

# ==========================================================
# GENERATE FRAME
# ==========================================================
def generate_frames():

    global mode

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = holistic.process(rgb)

        pred = "-"

        # ======================================================
        # MODE HURUF / ANGKA
        # ======================================================
        if mode in ["huruf", "angka"]:

            hand = result.right_hand_landmarks or result.left_hand_landmarks

            if hand:

                # =========================
                # HURUF
                # =========================
                if mode == "huruf":

                    data = []

                    for lm in hand.landmark:
                        data.append(lm.x)
                        data.append(lm.y)

                    if len(data) == 42:

                        data = np.array(data).reshape(1, -1)

                        pred = str(model_huruf.predict(data)[0])

                # =========================
                # ANGKA
                # =========================
                elif mode == "angka":

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

        # ======================================================
        # MODE KATA
        # ======================================================
        elif mode == "kata":

            data = []

            def normalize_landmarks(landmarks):

                pts = np.array([[lm.x, lm.y] for lm in landmarks])

                base = pts[0]

                pts = pts - base

                max_val = np.max(np.abs(pts))

                if max_val != 0:
                    pts = pts / max_val

                return pts.flatten().tolist()

            def pairwise_distances(landmarks):

                pairs = [(4,8), (8,12), (12,16), (16,20)]

                dist = []

                for a,b in pairs:

                    dx = landmarks[a].x - landmarks[b].x
                    dy = landmarks[a].y - landmarks[b].y

                    dist.append((dx**2 + dy**2)**0.5)

                return dist

            # LEFT HAND
            if result.left_hand_landmarks:

                left = result.left_hand_landmarks.landmark

                data += normalize_landmarks(left)

                data += pairwise_distances(left)

            else:
                data += [0.0] * (42 + 4)

            # RIGHT HAND
            if result.right_hand_landmarks:

                right = result.right_hand_landmarks.landmark

                data += normalize_landmarks(right)

                data += pairwise_distances(right)

            else:
                data += [0.0] * (42 + 4)

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

            # EXTRA
            extra = []

            def dist_xy(a, b):
                return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

            if result.right_hand_landmarks and result.face_landmarks:

                hand = result.right_hand_landmarks.landmark
                face = result.face_landmarks.landmark

                extra.append(dist_xy(hand[8], face[1]))
                extra.append(dist_xy(hand[4], face[1]))

            else:
                extra += [0.0, 0.0]

            data += extra

            # PREDICT
            if len(data) == 114:

                data = np.array(data).reshape(1, -1)

                pred_index = int(model_kata.predict(data)[0])

                pred = LABEL_KATA.get(pred_index, f"?{pred_index}")

            else:
                pred = f"err:{len(data)}"

        # ======================================================
        # DISPLAY
        # ======================================================
        cv2.putText(frame,
                    f"MODE : {mode.upper()}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.putText(frame,
                    f"HASIL : {pred}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    2)

        # ======================================================
        # STREAM FRAME
        # ======================================================
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ==========================================================
# HOME
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================================
# VIDEO FEED
# ==========================================================
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================================
# CHANGE MODE
# ==========================================================
@app.route('/set_mode/<new_mode>')
def set_mode(new_mode):

    global mode

    mode = new_mode

    print("MODE DIGANTI:", mode)

    return f"Mode changed to {mode}"
def set_mode(new_mode):

    global mode

    mode = new_mode

    return f"Mode changed to {mode}"

# ==========================================================
# RUN
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)