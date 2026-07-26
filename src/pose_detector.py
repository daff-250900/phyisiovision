from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.config import settings
from src.schemas import PoseResult


class PoseDetector:
    LANDMARK_NAMES = {
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
    }

    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> PoseResult:
        if frame is None or frame.size == 0:
            raise ValueError("El frame recibido está vacío.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        annotated = frame.copy()

        if not results.pose_landmarks:
            return PoseResult({}, annotated, False)

        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
        )

        landmarks: dict[str, tuple[float, float, float, float]] = {}
        for index, name in self.LANDMARK_NAMES.items():
            landmark = results.pose_landmarks.landmark[index]
            landmarks[name] = (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                float(landmark.visibility),
            )

        return PoseResult(landmarks, annotated, True)

    def close(self) -> None:
        self.pose.close()
