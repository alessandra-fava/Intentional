"""
STEP 1 - Test di base: RealSense D435i + MediaPipe Pose
=========================================================
Obiettivo di questo script:
- Aprire lo stream RGB + Depth della D435i
- Allineare depth al colore (così ogni pixel RGB ha anche un valore di depth)
- Far girare MediaPipe Pose sul frame RGB per trovare i landmark del corpo
- Per il polso (e la spalla, che ci servirà come origine del riferimento
  corporeo nello Step 2) calcolare la posizione 3D REALE in metri,
  usando gli intrinsics della camera (deprojection pixel -> punto 3D)
- Visualizzare tutto a schermo e stampare le coordinate in tempo reale

Uso:
    python step1_pose_stream.py

Premi 'q' per uscire.

Prerequisiti (vedi requirements.txt):
    pip install pyrealsense2 mediapipe opencv-python numpy
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import mediapipe as mp

try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

# ---------------------------------------------------------------------------
# 1) Configurazione RealSense
# ---------------------------------------------------------------------------
WIDTH, HEIGHT, FPS = 640, 480, 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

profile = pipeline.start(config)

# Allinea i frame di depth al frame di colore, pixel per pixel
align = rs.align(rs.stream.color)

# Intrinsics della camera colore: servono per la deprojection 2D -> 3D
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

# ---------------------------------------------------------------------------
# 2) Configurazione MediaPipe Pose
# ---------------------------------------------------------------------------
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Landmark che ci interessano per ora (indici MediaPipe Pose):
# 11 = spalla sinistra, 12 = spalla destra
# 15 = polso sinistro,  16 = polso destro
LANDMARKS_OF_INTEREST = {
    "right_shoulder": 12,
    "right_wrist": 16,
    "left_shoulder": 11,
    "left_wrist": 15,
}


def get_3d_point(depth_frame, px, py, intr):
    """
    Converte un pixel (px, py) del frame colore + la sua depth
    in un punto 3D (x, y, z) in metri, nel riferimento della camera.
    Ritorna None se non c'è un valore di depth valido in quel punto.
    """
    if px < 0 or py < 0 or px >= depth_frame.get_width() or py >= depth_frame.get_height():
        return None

    depth_value = depth_frame.get_distance(px, py)
    if depth_value <= 0:
        return None  # nessun dato di depth valido (es. superficie riflettente, fuori range)

    point_3d = rs.rs2_deproject_pixel_to_point(intr, [px, py], depth_value)
    return np.array(point_3d)  # [x, y, z] in metri, origine = camera


def main():
    print("Avvio stream... premi 'q' sulla finestra video per uscire.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # MediaPipe vuole immagini RGB
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)

            points_3d = {}

            if results.pose_landmarks:
                # Disegniamo lo scheletro 2D sopra il frame per verifica visiva
                mp_drawing.draw_landmarks(
                    color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                landmarks = results.pose_landmarks.landmark
                for name, idx in LANDMARKS_OF_INTEREST.items():
                    lm = landmarks[idx]
                    # MediaPipe da coordinate normalizzate [0,1] -> le portiamo in pixel
                    px, py = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                    point = get_3d_point(depth_frame, px, py, intrinsics)
                    if point is not None:
                        points_3d[name] = point
                        cv2.circle(color_image, (px, py), 6, (0, 255, 0), -1)

            # Stampa a schermo le coordinate del polso destro, se disponibili
            if "right_wrist" in points_3d:
                x, y, z = points_3d["right_wrist"]
                text = f"Polso DX: x={x:.3f} y={y:.3f} z={z:.3f} m"
                cv2.putText(
                    color_image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                print(text)

            cv2.imshow("RealSense + MediaPipe Pose (step 1)", color_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
