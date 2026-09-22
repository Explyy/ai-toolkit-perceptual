from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


SOURCE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
EXPECTED_ARCHIVE_SIZE = 288_621_354
EXPECTED_ARCHIVE_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
REQUIRED = {
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
}
ROOT = Path("/opt/training-automation-models")
TARGET = ROOT / "insightface" / "models" / "buffalo_l"
POSE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt"
POSE_SIZE = 6_255_593
POSE_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"
POSE_TARGET = ROOT / "ultralytics" / "yolo11n-pose.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_face() -> dict[str, object]:
    TARGET.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        archive = Path(temporary_dir) / "buffalo_l.zip"
        urllib.request.urlretrieve(SOURCE_URL, archive)
        if archive.stat().st_size != EXPECTED_ARCHIVE_SIZE:
            raise RuntimeError(
                f"Buffalo L archive size changed: {archive.stat().st_size} != {EXPECTED_ARCHIVE_SIZE}"
            )
        archive_sha256 = sha256(archive)
        if archive_sha256 != EXPECTED_ARCHIVE_SHA256:
            raise RuntimeError(
                f"Buffalo L archive SHA-256 changed: {archive_sha256} != {EXPECTED_ARCHIVE_SHA256}"
            )
        with zipfile.ZipFile(archive) as bundle:
            members = {Path(info.filename).name: info for info in bundle.infolist() if not info.is_dir()}
            missing = REQUIRED - set(members)
            if missing:
                raise RuntimeError(f"Buffalo L release is missing expected files: {sorted(missing)}")
            for name in sorted(REQUIRED):
                with bundle.open(members[name]) as source, (TARGET / name).open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
        return {
            "source_url": SOURCE_URL,
            "release": "deepinsight/insightface v0.7 buffalo_l",
            "archive_size": archive.stat().st_size,
            "archive_sha256": archive_sha256,
            "files": {
                name: {"size": (TARGET / name).stat().st_size, "sha256": sha256(TARGET / name)}
                for name in sorted(REQUIRED)
            },
            "license_notice": "InsightFace pretrained model packs are noncommercial research only unless separately licensed.",
        }


def install_pose() -> dict[str, object]:
    POSE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_dir:
        download = Path(temporary_dir) / POSE_TARGET.name
        urllib.request.urlretrieve(POSE_URL, download)
        if download.stat().st_size != POSE_SIZE:
            raise RuntimeError(f"pose weight size changed: {download.stat().st_size} != {POSE_SIZE}")
        actual = sha256(download)
        if actual != POSE_SHA256:
            raise RuntimeError(f"pose weight SHA-256 changed: {actual} != {POSE_SHA256}")
        with download.open("rb") as source, POSE_TARGET.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    return {
        "source_url": POSE_URL, "release": "ultralytics assets v8.4.0 yolo11n-pose.pt",
        "size": POSE_SIZE, "sha256": POSE_SHA256,
        "path": str(POSE_TARGET),
        "license_notice": "Ultralytics software and model use must comply with AGPL-3.0 or an applicable Enterprise license.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-only", action="store_true")
    args = parser.parse_args()
    sources_path = ROOT / "SOURCES.json"
    record = json.loads(sources_path.read_text()) if sources_path.is_file() else {}
    if not args.pose_only:
        record.update(install_face())
        print(f"buffalo_l archive sha256={record['archive_sha256']}")
    record["pose"] = install_pose()
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"yolo11n-pose sha256={record['pose']['sha256']}")


if __name__ == "__main__":
    main()
