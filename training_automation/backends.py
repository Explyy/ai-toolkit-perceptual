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


class MediaPipePoseCPUBackend:
    """Optional CPU image-plane pose proxy using the installed MediaPipe package."""

    def __init__(self, *, min_detection_confidence: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("install mediapipe to use this optional landmark backend") from exc
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=float(min_detection_confidence),
        )

    def metrics(self, image: np.ndarray) -> dict[str, Any]:
        result = self._pose.process(image)
        if result.pose_landmarks is None:
            return {"status": "missing", "values": None, "reason": "no pose landmarks detected"}
        points = result.pose_landmarks.landmark
        visibility = [float(point.visibility) for point in points]
        left_shoulder, right_shoulder = points[11], points[12]
        left_hip, right_hip = points[23], points[24]
        shoulder_width = float(np.hypot(left_shoulder.x - right_shoulder.x, left_shoulder.y - right_shoulder.y))
        hip_width = float(np.hypot(left_hip.x - right_hip.x, left_hip.y - right_hip.y))
        return {
            "status": "available",
            "values": {
                "landmark_count": len(points),
                "mean_visibility": float(np.mean(visibility)),
                "shoulder_to_hip_width_image_plane_proxy": shoulder_width / hip_width if hip_width > 1e-8 else None,
            },
            "reason": "2D image-plane proxy; perspective, crop, pose, and occlusion limit comparison",
        }
