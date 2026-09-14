from types import SimpleNamespace

import numpy as np
import pytest

from training_automation.backends import (
    MediaPipePoseCPUBackend, UltralyticsPoseCPUBackend,
    _coco_pose_metrics, _pose_metrics,
)


def point(x=0.0, y=0.0, visibility=1.0):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def test_pose_ratio_uses_pixel_dimensions_for_non_square_images():
    points = [point() for _ in range(25)]
    points[11] = point(0.2, 0.2)
    points[12] = point(0.4, 0.3)
    points[23] = point(0.2, 0.6)
    points[24] = point(0.4, 0.6)
    result = _pose_metrics(points, width=200, height=100, min_visibility=0.5)
    assert result["status"] == "available"
    assert result["values"]["shoulder_to_hip_width_image_plane_proxy"] == pytest.approx(
        (40**2 + 10**2) ** 0.5 / 40
    )


def test_pose_visibility_gate_reports_occlusion():
    points = [point() for _ in range(25)]
    points[23].visibility = 0.2
    result = _pose_metrics(points, width=100, height=100, min_visibility=0.5)
    assert result["status"] == "occluded"
    assert result["values"] is None


def test_pose_backend_requires_explicit_existing_model(tmp_path):
    with pytest.raises(ValueError, match="user-supplied MediaPipe"):
        MediaPipePoseCPUBackend(model_path=str(tmp_path / "missing.task"))


def test_coco_pose_geometry_is_pixel_corrected_and_reports_angles():
    xy = np.zeros((17, 2), dtype=float)
    xy[5], xy[6] = (0.2, 0.2), (0.4, 0.3)
    xy[7], xy[8] = (0.2, 0.4), (0.4, 0.4)
    xy[9], xy[10] = (0.3, 0.4), (0.5, 0.4)
    xy[11], xy[12] = (0.2, 0.6), (0.4, 0.6)
    xy[13], xy[14] = (0.2, 0.8), (0.4, 0.8)
    xy[15], xy[16] = (0.3, 0.9), (0.5, 0.9)
    result = _coco_pose_metrics(
        xy, np.ones(17), width=200, height=100, min_confidence=0.5
    )
    assert result["status"] == "available"
    assert result["values"]["ratios"]["shoulder_to_hip_width"] == pytest.approx(
        (40**2 + 10**2) ** 0.5 / 40
    )
    assert result["values"]["joint_angles"]["left_elbow_degrees"] is not None


def test_coco_pose_reports_occluded_and_degenerate_without_metrics():
    xy = np.ones((17, 2), dtype=float)
    confidence = np.ones(17)
    confidence[11] = 0.1
    assert _coco_pose_metrics(xy, confidence, width=100, height=100, min_confidence=0.5)["status"] == "occluded"
    assert _coco_pose_metrics(xy, np.ones(17), width=100, height=100, min_confidence=0.5)["status"] == "degenerate"


def test_ultralytics_backend_rejects_unpinned_weight_bytes(tmp_path):
    model = tmp_path / "pose.pt"
    model.write_bytes(b"not the pinned model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        UltralyticsPoseCPUBackend(model_path=str(model), expected_sha256="0" * 64)


class ArrayTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakePoseModel:
    def __init__(self, count):
        self.count = count

    def predict(self, **kwargs):
        xy = np.zeros((self.count, 17, 2))
        confidence = np.ones((self.count, 17))
        keypoints = SimpleNamespace(xyn=ArrayTensor(xy), conf=ArrayTensor(confidence))
        return [SimpleNamespace(keypoints=keypoints)]


@pytest.mark.parametrize("count,status", [(0, "missing"), (2, "ambiguous")])
def test_ultralytics_backend_does_not_choose_from_missing_or_multiple_people(count, status):
    backend = object.__new__(UltralyticsPoseCPUBackend)
    backend._model = FakePoseModel(count)
    backend._image_size = 640
    backend._min_detection_confidence = 0.25
    backend._min_keypoint_confidence = 0.5
    backend._iou_threshold = 0.7
    result = backend.metrics(np.zeros((100, 200, 3), dtype=np.uint8))
    assert result["status"] == status
    assert result["values"] is None
