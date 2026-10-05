"""Non-ML artifact contracts. Provisioning is an explicit separate cloud command."""
from __future__ import annotations
import hashlib
import json
import os
import uuid
from pathlib import Path

VERSION = 'arcface-depth-pose-v1'
MANIFEST = Path(__file__).with_name('models.json')

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def configuration() -> tuple[dict, str]:
    value = json.loads(MANIFEST.read_text())
    canonical = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return value, hashlib.sha256(canonical.encode()).hexdigest()

def safe_file(root: Path, relative: str, missing=False) -> Path:
    if not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Artifact path escapes owned storage')
    base = root.resolve(strict=True)
    current = base
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink artifact refused')
    if not missing:
        current.resolve(strict=True).relative_to(base)
    else:
        current.resolve().relative_to(base)
    return current

def verified_models(root: Path) -> dict[str, Path]:
    manifest, _ = configuration()
    result = {}
    for key, model in manifest['models'].items():
        folder = root / key
        for name, expected in model['files'].items():
            file = safe_file(root, key + '/' + name)
            if file.stat().st_size != expected['size'] or digest(file) != expected['sha256']:
                raise ValueError('Model checksum mismatch: ' + key + '/' + name)
        result[key] = folder
    return result

def atomic_json(path: Path, value):
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('x') as f:
            json.dump(value, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
