from __future__ import annotations

import hashlib
import json
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
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
        record = {
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
    (ROOT / "SOURCES.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"buffalo_l archive sha256={record['archive_sha256']}")


if __name__ == "__main__":
    main()
