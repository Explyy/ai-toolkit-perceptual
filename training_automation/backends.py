from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class InsightFaceCPUBackend:
    """Optional adapter requiring explicit, already-present InsightFace weights."""

    def __init__(self, *, model_dir: str, det_size: int = 640):
        path = Path(model_dir).resolve()
        if not path.is_dir():
            raise ValueError("model_dir must point to user-supplied InsightFace weights")
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("install insightface and onnxruntime to use this optional backend") from exc
        self._app = FaceAnalysis(
            name=path.name,
            root=str(path.parent.parent),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(int(det_size), int(det_size)))

    def embeddings(self, image: np.ndarray) -> list[np.ndarray]:
        bgr = image[:, :, ::-1]
        return [np.asarray(face.embedding) for face in self._app.get(bgr)]


def _pose_metrics(points: list[Any], width: int, height: int, min_visibility: float) -> dict[str, Any]:
    required = [points[index] for index in (11, 12, 23, 24)]
    if any(float(point.visibility) < min_visibility for point in required):
        return {
            "status": "occluded",
            "values": None,
            "reason": "shoulder or hip landmarks are below the configured visibility threshold",
        }
    left_shoulder, right_shoulder, left_hip, right_hip = required

    def pixel_distance(left: Any, right: Any) -> float:
        return float(
            np.hypot((left.x - right.x) * width, (left.y - right.y) * height)
        )

    shoulder_width = pixel_distance(left_shoulder, right_shoulder)
    hip_width = pixel_distance(left_hip, right_hip)
    return {
        "status": "available",
        "values": {
            "landmark_count": len(points),
            "mean_visibility": float(np.mean([float(point.visibility) for point in points])),
            "shoulder_to_hip_width_image_plane_proxy": shoulder_width / hip_width if hip_width > 1e-8 else None,
        },
        "reason": "pixel-corrected 2D image-plane proxy; perspective, crop, pose, and occlusion limit comparison",
    }


class MediaPipePoseCPUBackend:
    """Optional MediaPipe Tasks Pose Landmarker using an explicit local model."""

    def __init__(
        self,
        *,
        model_path: str,
        min_detection_confidence: float = 0.5,
        min_visibility: float = 0.5,
    ):
        model = Path(model_path).resolve()
        if not model.is_file():
            raise ValueError("model_path must point to a user-supplied MediaPipe Pose Landmarker model")
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError("install the documented MediaPipe Tasks dependency") from exc
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=float(min_detection_confidence),
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._min_visibility = float(min_visibility)

    def metrics(self, image: np.ndarray) -> dict[str, Any]:
        media = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image)
        result = self._landmarker.detect(media)
        if not result.pose_landmarks:
            return {"status": "missing", "values": None, "reason": "no pose landmarks detected"}
        if len(result.pose_landmarks) != 1:
            return {"status": "ambiguous", "values": None, "reason": "multiple poses detected"}
        height, width = image.shape[:2]
        return _pose_metrics(result.pose_landmarks[0], width, height, self._min_visibility)
