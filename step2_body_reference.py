"""
STEP 2 - Body reference calibration
==================================
This script extends the pose pipeline by computing a body-centered reference
frame and expressing the wrist position relative to that frame.

The first approximation uses:
- origin: midpoint between the two shoulders
- x-axis: from left shoulder to right shoulder
- y-axis: from the torso origin toward the nose
- z-axis: orthogonal to the previous two axes

The resulting coordinates are expressed in meters in the body frame.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp

try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

WIDTH, HEIGHT, FPS = 640, 480, 30

LANDMARKS_OF_INTEREST = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "nose": 0,
}


def get_3d_point(depth_frame, px, py, intr):
    if px < 0 or py < 0 or px >= depth_frame.get_width() or py >= depth_frame.get_height():
        return None

    depth_value = depth_frame.get_distance(px, py)
    if depth_value <= 0:
        return None

    point_3d = rs.rs2_deproject_pixel_to_point(intr, [px, py], depth_value)
    return np.array(point_3d, dtype=np.float32)


def transform_to_body_frame(origin, point, x_axis, y_axis, z_axis):
    vector = point - origin
    return np.array(
        [
            np.dot(vector, x_axis),
            np.dot(vector, y_axis),
            np.dot(vector, z_axis),
        ],
        dtype=np.float32,
    )


def build_body_axes(origin, left_shoulder, right_shoulder, nose):
    if origin is None or left_shoulder is None or right_shoulder is None or nose is None:
        return None

    x_axis = right_shoulder - left_shoulder
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis = x_axis / x_norm

    y_axis = nose - origin
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        y_axis = y_axis / y_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        z_axis = z_axis / z_norm

    return x_axis, y_axis, z_axis


def main():
    print("Avvio Step 2: calibrazione riferimento corporeo...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                px_points = {}
                point_3d = {}

                for name, idx in LANDMARKS_OF_INTEREST.items():
                    lm = landmarks[idx]
                    px, py = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                    px_points[name] = (px, py)
                    point = get_3d_point(depth_frame, px, py, intrinsics)
                    if point is not None:
                        point_3d[name] = point
                        cv2.circle(color_image, (px, py), 6, (0, 255, 0), -1)

                if {"left_shoulder", "right_shoulder", "right_wrist", "nose"}.issubset(point_3d.keys()):
                    left_shoulder = point_3d["left_shoulder"]
                    right_shoulder = point_3d["right_shoulder"]
                    right_wrist = point_3d["right_wrist"]
                    nose = point_3d["nose"]
                    origin = (left_shoulder + right_shoulder) / 2.0

                    axes = build_body_axes(origin, left_shoulder, right_shoulder, nose)
                    if axes is not None:
                        x_axis, y_axis, z_axis = axes
                        body_coords = transform_to_body_frame(origin, right_wrist, x_axis, y_axis, z_axis)
                        text = (
                            f"Polso body-frame: x={body_coords[0]:.3f} "
                            f"y={body_coords[1]:.3f} z={body_coords[2]:.3f} m"
                        )
                        cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        print(text)

                    if "left_shoulder" in px_points and "right_shoulder" in px_points:
                        origin_px = (
                            (px_points["left_shoulder"][0] + px_points["right_shoulder"][0]) // 2,
                            (px_points["left_shoulder"][1] + px_points["right_shoulder"][1]) // 2,
                        )
                        cv2.circle(color_image, origin_px, 8, (255, 0, 0), -1)

            cv2.imshow("RealSense + MediaPipe Pose (step 2)", color_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
