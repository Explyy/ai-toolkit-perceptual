from types import SimpleNamespace

import pytest

from training_automation.backends import MediaPipePoseCPUBackend, _pose_metrics


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
