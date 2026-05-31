import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# KONFIGURASI
# ==========================================================
DATASET_PATH = "dataset_angka_video"
OUTPUT_PATH  = "extracted_number_data"

SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3
FEATURE_SIZE = NUM_LANDMARKS * FEATURES_PER_LANDMARK

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("MediaPipe Version :", mp.__version__)
print("Dataset ditemukan :", os.path.exists(DATASET_PATH))

# ==========================================================
# MEDIAPIPE HANDS
# ==========================================================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ==========================================================
# NORMALISASI LANDMARK
# ==========================================================
def normalize_landmarks(hand_landmarks):

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
        dtype=np.float32
    )

    # Wrist jadi pusat
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale normalisasi
    scale = np.linalg.norm(landmarks[9])

    if scale < 1e-6:
        scale = 1.0

    landmarks = landmarks / scale

    return landmarks.flatten()

# ==========================================================
# AMBIL FRAME MERATA
# ==========================================================
def extract_frames_evenly(video_path, sequence_length=30):

    cap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return frames

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length
    ).astype(int)

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)

    cap.release()

    return frames

# ==========================================================
# EKSTRAK SEQUENCE LANDMARK
# ==========================================================
def extract_landmark_sequence(video_path):

    frames = extract_frames_evenly(
        video_path,
        SEQUENCE_LENGTH
    )

    sequence = []

    last_valid_landmark = np.zeros(
        FEATURE_SIZE,
        dtype=np.float32
    )

    for frame in frames:

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:

            hand_landmarks = result.multi_hand_landmarks[0]

            landmark = normalize_landmarks(
                hand_landmarks
            )

            last_valid_landmark = landmark

        else:
            landmark = last_valid_landmark

        sequence.append(landmark)

    # Padding kalau frame kurang
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(last_valid_landmark)

    return np.array(sequence, dtype=np.float32)

# ==========================================================
# MAIN
# ==========================================================
X = []
y = []

labels = sorted(
    [
        folder for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    ],
    key=lambda x: int(x)
)

print("\nLabel ditemukan:")
print(labels)

for label in labels:

    label_path = os.path.join(DATASET_PATH, label)

    video_files = [
        file for file in os.listdir(label_path)
        if file.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        )
    ]

    print(f"\nMemproses label {label}")
    print(f"Jumlah video: {len(video_files)}")

    for i, video_file in enumerate(video_files, start=1):

        video_path = os.path.join(
            label_path,
            video_file
        )

        try:

            sequence = extract_landmark_sequence(
                video_path
            )

            if sequence.shape == (
                SEQUENCE_LENGTH,
                FEATURE_SIZE
            ):

                X.append(sequence)
                y.append(label)

                print(
                    f"[{i}/{len(video_files)}] OK: {video_file}"
                )

            else:
                print(
                    f"[{i}/{len(video_files)}] SKIP: "
                    f"{video_file} shape={sequence.shape}"
                )

        except Exception as e:

            print(
                f"[{i}/{len(video_files)}] ERROR: "
                f"{video_file} -> {e}"
            )

# ==========================================================
# CONVERT NUMPY
# ==========================================================
X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\n==============================")
print("HASIL EKSTRAKSI")
print("==============================")
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================================
# LABEL ENCODER
# ==========================================================
label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# ==========================================================
# SAVE
# ==========================================================
np.save(
    os.path.join(OUTPUT_PATH, "X.npy"),
    X
)

np.save(
    os.path.join(OUTPUT_PATH, "y.npy"),
    y_encoded
)

with open(
    os.path.join(OUTPUT_PATH, "label_encoder.pkl"),
    "wb"
) as f:

    pickle.dump(label_encoder, f)

print("\n==============================")
print("FILE BERHASIL DISIMPAN")
print("==============================")

print("X.npy")
print("y.npy")
print("label_encoder.pkl")

print("\nClasses:")
print(label_encoder.classes_)

hands.close()