"""
STEP 3 - Record reachable wrist points with smoothing
====================================================
This script extends Step 2 by smoothing the body-frame wrist coordinates and
recording them to a CSV file for later replay and workspace analysis.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from websockets.sync.client import connect as ws_connect

from step2_body_reference import (
    LANDMARKS_OF_INTEREST,
    build_body_axes,
    get_3d_point,
    transform_to_body_frame,
)

try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing

WIDTH, HEIGHT, FPS = 640, 480, 30


class ExponentialMovingAverage:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self._initialized = False
        self._value = None

    def update(self, point):
        point = np.asarray(point, dtype=np.float32)
        if not self._initialized:
            self._value = point.copy()
            self._initialized = True
            return self._value

        self._value = self.alpha * point + (1.0 - self.alpha) * self._value
        return self._value


def prepare_output_path(output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class PoseWebSocketBroadcaster:
    def __init__(self, uri="ws://127.0.0.1:8000/ws"):
        self.uri = uri
        self._socket = None

    def connect(self):
        try:
            self._socket = ws_connect(self.uri)
            print(f"Connected to avatar server at {self.uri}")
        except Exception as exc:
            print(f"Avatar server unavailable: {exc}")
            self._socket = None

    def send(self, payload):
        if self._socket is None:
            return
        try:
            self._socket.send(json.dumps(payload))
        except Exception as exc:
            print(f"Could not send to avatar server: {exc}")
            self._socket = None

    def close(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None


def build_pose_payload(timestamp, raw_point, smoothed_point, joints=None, valid=True):
    raw = np.asarray(raw_point, dtype=np.float32)
    smoothed = np.asarray(smoothed_point, dtype=np.float32)

    payload = {
        "timestamp": timestamp,
        "valid": valid,
        "wrist": {
            "x": float(smoothed[0]),
            "y": float(smoothed[1]),
            "z": float(smoothed[2]),
        },
        "raw": {
            "x": float(raw[0]),
            "y": float(raw[1]),
            "z": float(raw[2]),
        },
    }

    if joints:
        payload["joints"] = {}
        for name, point in joints.items():
            if point is None:
                continue
            point_array = np.asarray(point, dtype=np.float32)
            payload["joints"][name] = {
                "x": float(point_array[0]),
                "y": float(point_array[1]),
                "z": float(point_array[2]),
            }

    return payload


def record_session(output_path, alpha=0.25, broadcast_url=None):
    output_path = prepare_output_path(output_path)
    print(f"Recording to {output_path}")

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
    smoother = ExponentialMovingAverage(alpha=alpha)
    broadcaster = PoseWebSocketBroadcaster(broadcast_url or "ws://127.0.0.1:8000/ws")
    broadcaster.connect()

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "timestamp",
                "raw_x",
                "raw_y",
                "raw_z",
                "smoothed_x",
                "smoothed_y",
                "smoothed_z",
                "valid",
            ]
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
                    point_3d = {}
                    for name, idx in LANDMARKS_OF_INTEREST.items():
                        lm = landmarks[idx]
                        px, py = int(lm.x * WIDTH), int(lm.y * HEIGHT)
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
                            raw_point = transform_to_body_frame(origin, right_wrist, x_axis, y_axis, z_axis)
                            smoothed_point = smoother.update(raw_point)

                            right_elbow = None
                            if "right_elbow" in point_3d:
                                right_elbow = point_3d["right_elbow"]

                            joints = {
                                "right_shoulder": right_shoulder,
                                "right_elbow": right_elbow,
                                "right_wrist": right_wrist,
                            }

                            timestamp = time.time()
                            writer.writerow(
                                [
                                    timestamp,
                                    f"{raw_point[0]:.6f}",
                                    f"{raw_point[1]:.6f}",
                                    f"{raw_point[2]:.6f}",
                                    f"{smoothed_point[0]:.6f}",
                                    f"{smoothed_point[1]:.6f}",
                                    f"{smoothed_point[2]:.6f}",
                                    1,
                                ]
                            )
                            csv_file.flush()

                            payload = build_pose_payload(timestamp, raw_point, smoothed_point, joints=joints, valid=True)
                            broadcaster.send(payload)

                            text = (
                                f"raw: x={raw_point[0]:.3f} y={raw_point[1]:.3f} z={raw_point[2]:.3f} | "
                                f"smoothed: x={smoothed_point[0]:.3f} y={smoothed_point[1]:.3f} z={smoothed_point[2]:.3f}"
                            )
                            cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            print(text)

                cv2.imshow("RealSense + MediaPipe Pose (step 3)", color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            broadcaster.close()
            pipeline.stop()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Smooth and record body-frame wrist positions")
    parser.add_argument("--output", default="recordings/reachability.csv", help="Path to the CSV output file")
    parser.add_argument("--alpha", type=float, default=0.25, help="Exponential smoothing factor between 0 and 1")
    parser.add_argument("--broadcast-url", default="ws://127.0.0.1:8000/ws", help="WebSocket URL of the avatar server")
    args = parser.parse_args()
    record_session(args.output, alpha=args.alpha, broadcast_url=args.broadcast_url)


if __name__ == "__main__":
    main()
