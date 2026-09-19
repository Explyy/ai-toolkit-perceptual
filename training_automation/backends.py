from __future__ import annotations

import hashlib
import importlib.metadata
import math
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

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": f"{type(self).__module__}:{type(self).__name__}",
            "package_version": importlib.metadata.version("insightface"),
            "device": "cpu",
        }


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    left, right = a - b, c - b
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-8:
        return None
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def _coco_pose_metrics(
    xy: np.ndarray, confidence: np.ndarray, *, width: int, height: int, min_confidence: float
) -> dict[str, Any]:
    xy = np.asarray(xy, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if xy.shape != (17, 2) or confidence.shape != (17,):
        return {"status": "degenerate", "values": None, "reason": "pose output is not COCO-17"}
    required = list(range(5, 17))
    if not np.isfinite(xy[required]).all() or not np.isfinite(confidence[required]).all():
        return {"status": "degenerate", "values": None, "reason": "pose output contains non-finite coordinates"}
    low = [index for index in required if confidence[index] < min_confidence]
    if low:
        return {
            "status": "occluded", "values": None,
            "reason": f"required COCO joints below confidence threshold: {low}",
        }
    points = xy * np.array([width, height], dtype=np.float64)
    distance = lambda a, b: float(np.linalg.norm(points[a] - points[b]))
    shoulder_width, hip_width = distance(5, 6), distance(11, 12)
    shoulder_mid = (points[5] + points[6]) / 2
    hip_mid = (points[11] + points[12]) / 2
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    if hip_width <= 1e-8 or torso <= 1e-8:
        return {"status": "degenerate", "values": None, "reason": "hip width or torso length is zero"}
    ratios = {
        "shoulder_to_hip_width": shoulder_width / hip_width,
        "left_upper_arm_to_torso": distance(5, 7) / torso,
        "right_upper_arm_to_torso": distance(6, 8) / torso,
        "left_forearm_to_torso": distance(7, 9) / torso,
        "right_forearm_to_torso": distance(8, 10) / torso,
        "left_thigh_to_torso": distance(11, 13) / torso,
        "right_thigh_to_torso": distance(12, 14) / torso,
        "left_shin_to_torso": distance(13, 15) / torso,
        "right_shin_to_torso": distance(14, 16) / torso,
    }
    angles = {
        "left_elbow_degrees": _angle(points[5], points[7], points[9]),
        "right_elbow_degrees": _angle(points[6], points[8], points[10]),
        "left_knee_degrees": _angle(points[11], points[13], points[15]),
        "right_knee_degrees": _angle(points[12], points[14], points[16]),
    }
    return {
        "status": "available",
        "values": {
            "coordinate_space": "pixel-corrected 2D image plane",
            "mean_required_joint_confidence": float(np.mean(confidence[required])),
            "ratios": ratios,
            "joint_angles": angles,
        },
        "reason": "2D proxies are viewpoint-, crop-, pose-, and occlusion-sensitive; they are not 3D anatomy",
    }


class UltralyticsPoseCPUBackend:
    """Deterministic, single-person COCO-17 pose evaluation on CPU."""

    def __init__(
        self, *, model_path: str, expected_sha256: str,
        min_detection_confidence: float = 0.25, min_keypoint_confidence: float = 0.5,
        iou_threshold: float = 0.7, image_size: int = 640,
    ):
        path = Path(model_path).resolve()
        if not path.is_file():
            raise ValueError("model_path must point to the pinned Ultralytics pose weight")
        actual = _sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"pose model SHA-256 mismatch: {actual}")
        from ultralytics import YOLO

        self._model = YOLO(str(path))
        self._path = path
        self._sha256 = actual
        self._min_detection_confidence = float(min_detection_confidence)
        self._min_keypoint_confidence = float(min_keypoint_confidence)
        self._iou_threshold = float(iou_threshold)
        self._image_size = int(image_size)

    def metrics(self, image: np.ndarray) -> dict[str, Any]:
        bgr = np.ascontiguousarray(image[:, :, ::-1])
        results = self._model.predict(
            source=bgr, device="cpu", verbose=False, imgsz=self._image_size,
            conf=self._min_detection_confidence, iou=self._iou_threshold,
            max_det=2, augment=False, deterministic=True, save=False,
        )
        if not results:
            return {"status": "missing", "values": None, "reason": "no pose detected"}
        result = results[0]
        xy = result.keypoints.xyn.cpu().numpy() if result.keypoints is not None else np.empty((0, 17, 2))
        conf = result.keypoints.conf.cpu().numpy() if result.keypoints is not None and result.keypoints.conf is not None else np.empty((0, 17))
        if len(xy) == 0:
            return {"status": "missing", "values": None, "reason": "no pose detected"}
        if len(xy) != 1:
            return {"status": "ambiguous", "values": None, "reason": "multiple people detected"}
        height, width = image.shape[:2]
        return _coco_pose_metrics(
            xy[0], conf[0], width=width, height=height,
            min_confidence=self._min_keypoint_confidence,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": f"{type(self).__module__}:{type(self).__name__}",
            "package_version": importlib.metadata.version("ultralytics"),
            "model_path": str(self._path), "model_sha256": self._sha256,
            "device": "cpu", "image_size": self._image_size,
            "min_detection_confidence": self._min_detection_confidence,
            "min_keypoint_confidence": self._min_keypoint_confidence,
            "iou_threshold": self._iou_threshold,
        }
