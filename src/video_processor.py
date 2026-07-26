from __future__ import annotations

import time
from pathlib import Path

import cv2

from src.config import settings
from src.feature_engineering import MotionAccumulator
from src.pose_detector import PoseDetector
from src.schemas import VideoProcessingResult


class VideoProcessor:
    def process(self, video_path: str, exercise_id: str) -> VideoProcessingResult:
        input_path = Path(video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"No se encontró el video: {video_path}")

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("No fue posible abrir el video.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("El video no tiene dimensiones válidas.")

        output_path = settings.processed_dir / f"processed_{int(time.time() * 1000)}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        detector = PoseDetector()
        accumulator = MotionAccumulator()
        total_frames = 0
        valid_frames = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                total_frames += 1
                pose_result = detector.process_frame(frame)
                annotated = pose_result.annotated_frame

                if pose_result.pose_detected:
                    metrics = accumulator.update(pose_result.landmarks)
                    if metrics:
                        valid_frames += 1
                        cv2.putText(
                            annotated,
                            f"Hombro: {metrics['shoulder_angle']:.1f} deg",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            annotated,
                            f"Repeticiones: {accumulator.repetitions}",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                writer.write(annotated)
        finally:
            capture.release()
            writer.release()
            detector.close()

        if total_frames == 0:
            raise ValueError("El video no contiene frames legibles.")
        if valid_frames < max(5, int(total_frames * 0.1)):
            raise ValueError("No se detectó una postura válida durante suficiente tiempo.")

        features = accumulator.summarize(fps)
        max_rom = features["max_rom"]
        model_features = {
            "shoulder_angle": features["shoulder_angle"],
            "elbow_angle": features["elbow_angle"],
            "trunk_inclination": features["trunk_inclination"],
            "movement_speed": features["movement_speed"],
        }

        warnings: list[str] = []
        if valid_frames / total_frames < 0.5:
            warnings.append("La postura fue visible en menos de la mitad del video.")

        return VideoProcessingResult(
            output_video_path=str(output_path),
            features=model_features,
            max_rom=max_rom,
            repetitions=accumulator.repetitions,
            frame_count=total_frames,
            warnings=warnings,
        )
