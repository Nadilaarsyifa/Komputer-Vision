import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

# =========================
# KONFIGURASI
# =========================
DATASET_PATH = "dataset_video_kata"
OUTPUT_PATH = "extracted_kata_video_data"

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 162

os.makedirs(OUTPUT_PATH, exist_ok=True)

mp_holistic = mp.solutions.holistic

POSE_POINTS = [11, 12, 13, 14, 15, 16]
FACE_POINTS = [1, 10, 13, 152, 234, 454]


# =========================
# NORMALISASI TANGAN
# =========================
def normalize_hand(hand_landmarks):
    if hand_landmarks is None:
        return np.zeros(63, dtype=np.float32)

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
# EKSTRAK POSE
# =========================
def extract_pose(pose_landmarks):
    if pose_landmarks is None:
        return np.zeros(18, dtype=np.float32)

    lm = pose_landmarks.landmark

    # pusat = tengah bahu kiri dan kanan
    base = np.array([
        (lm[11].x + lm[12].x) / 2,
        (lm[11].y + lm[12].y) / 2,
        (lm[11].z + lm[12].z) / 2
    ], dtype=np.float32)

    points = []

    for idx in POSE_POINTS:
        p = np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=np.float32)
        points.extend((p - base).tolist())

    return np.array(points, dtype=np.float32)


# =========================
# EKSTRAK FACE
# =========================
def extract_face(face_landmarks):
    if face_landmarks is None:
        return np.zeros(18, dtype=np.float32)

    lm = face_landmarks.landmark

    # pusat = hidung
    base = np.array([lm[1].x, lm[1].y, lm[1].z], dtype=np.float32)

    points = []

    for idx in FACE_POINTS:
        p = np.array([lm[idx].x, lm[idx].y, lm[idx].z], dtype=np.float32)
        points.extend((p - base).tolist())

    return np.array(points, dtype=np.float32)


# =========================
# AMBIL FRAME MERATA
# =========================
def extract_frames_evenly(video_path, sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return frames

    frame_indices = np.linspace(0, total_frames - 1, sequence_length).astype(int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)

    cap.release()
    return frames


# =========================
# EKSTRAK 1 VIDEO
# =========================
def extract_video_sequence(video_path):
    frames = extract_frames_evenly(video_path, SEQUENCE_LENGTH)

    sequence = []

    last_valid = np.zeros(FEATURE_SIZE, dtype=np.float32)

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            left_hand = normalize_hand(result.left_hand_landmarks)
            right_hand = normalize_hand(result.right_hand_landmarks)
            pose = extract_pose(result.pose_landmarks)
            face = extract_face(result.face_landmarks)

            features = np.concatenate([
                left_hand,
                right_hand,
                pose,
                face
            ])

            if features.shape[0] == FEATURE_SIZE:
                last_valid = features
            else:
                features = last_valid

            sequence.append(features)

    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(last_valid)

    return np.array(sequence, dtype=np.float32)


# =========================
# MAIN
# =========================
def main():
    X = []
    y = []

    labels = sorted([
        folder for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    ])

    print("Label ditemukan:")
    print(labels)

    for label in labels:
        label_path = os.path.join(DATASET_PATH, label)

        video_files = [
            file for file in os.listdir(label_path)
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]

        print(f"\nMemproses label: {label}")
        print(f"Jumlah video: {len(video_files)}")

        for i, video_file in enumerate(video_files, start=1):
            video_path = os.path.join(label_path, video_file)

            try:
                sequence = extract_video_sequence(video_path)

                if sequence.shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
                    X.append(sequence)
                    y.append(label)
                    print(f"[{i}/{len(video_files)}] OK: {video_file}")
                else:
                    print(f"[{i}/{len(video_files)}] SKIP shape salah: {video_file}, {sequence.shape}")

            except Exception as e:
                print(f"[{i}/{len(video_files)}] ERROR: {video_file} -> {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print("\n========================")
    print("HASIL EKSTRAKSI")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("========================")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    np.save(os.path.join(OUTPUT_PATH, "X_kata_video.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y_kata_video.npy"), y_encoded)

    with open(os.path.join(OUTPUT_PATH, "label_encoder_kata_video.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nFile berhasil disimpan:")
    print(f"- {OUTPUT_PATH}/X_kata_video.npy")
    print(f"- {OUTPUT_PATH}/y_kata_video.npy")
    print(f"- {OUTPUT_PATH}/label_encoder_kata_video.pkl")
    print("\nClasses:", label_encoder.classes_)


if __name__ == "__main__":
    main()