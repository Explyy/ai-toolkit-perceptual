from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "model_get.py"
SPEC = importlib.util.spec_from_file_location("model_get", SCRIPT)
assert SPEC and SPEC.loader
model_get = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model_get
SPEC.loader.exec_module(model_get)


class Response:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None, *, max_chunk: int | None = None):
        self.body = io.BytesIO(body)
        self.headers = headers or {}
        self.max_chunk = max_chunk

    def read(self, size: int = -1) -> bytes:
        if self.max_chunk is not None and (size < 0 or size > self.max_chunk):
            size = self.max_chunk
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class Opener:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def open(self, request, timeout=0):
        del timeout
        self.requests.append(request)
        route = self.routes.get(request.full_url)
        if route is None:
            raise AssertionError(f"unexpected request: {request.full_url}")
        if isinstance(route, Exception):
            raise route
        if isinstance(route, Response):
            return route
        if callable(route):
            return route(request)
        return Response(route)


def hf_metadata(payload: bytes, *, files=None):
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "sha": "a" * 40,
        "siblings": files
        or [
            {
                "rfilename": "weights/model.safetensors",
                "lfs": {"sha256": digest, "size": len(payload)},
            }
        ],
    }


def hf_routes(payload: bytes, *, metadata=None, repo_type="models"):
    metadata = metadata or hf_metadata(payload)
    api = f"https://huggingface.co/api/{repo_type}/owner/repo/revision/main?blobs=true"
    download = f"https://huggingface.co/owner/repo/resolve/{'a' * 40}/weights/model.safetensors?download=true"
    if repo_type == "datasets":
        download = f"https://huggingface.co/datasets/owner/repo/resolve/{'a' * 40}/weights/model.safetensors?download=true"
    return {api: json.dumps(metadata).encode(), download: Response(payload, {"Content-Length": str(len(payload)), "Content-Type": "application/octet-stream"})}


def run_cli(args, opener, *, stdin=None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = model_get.main(args, opener=opener, stdin=stdin or io.StringIO(), stdout=stdout, stderr=stderr)
    return code, stdout.getvalue(), stderr.getvalue()


def test_hf_blob_downloads_resolved_bytes_with_remote_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-hf")
    payload = b"model-bytes"
    opener = Opener(hf_routes(payload))
    code, stdout, _ = run_cli(
        ["https://huggingface.co/owner/repo/blob/main/weights/model.safetensors", "--dir", str(tmp_path)], opener
    )
    assert code == 0
    assert (tmp_path / "model.safetensors").read_bytes() == payload
    assert "Completato e verificato" in stdout
    assert all(req.get_header("Authorization") == "Bearer secret-hf" for req in opener.requests)
    assert all("secret-hf" not in req.full_url for req in opener.requests)


def test_hf_resolve_dataset_url_uses_dataset_api_and_pinned_download(tmp_path):
    payload = b"dataset-model"
    opener = Opener(hf_routes(payload, repo_type="datasets"))
    code, _, _ = run_cli(
        ["https://huggingface.co/datasets/owner/repo/resolve/main/weights/model.safetensors", "--dir", str(tmp_path)], opener
    )
    assert code == 0
    assert (tmp_path / "model.safetensors").read_bytes() == payload
    assert "/datasets/owner/repo/resolve/" in opener.requests[-1].full_url


def test_hf_repo_dry_run_resolves_without_downloading_model(tmp_path):
    payload = b"unused"
    routes = hf_routes(payload)
    routes.pop(next(url for url in routes if "/resolve/" in url))
    opener = Opener(routes)
    code, stdout, _ = run_cli(["https://huggingface.co/owner/repo", "--dir", str(tmp_path), "--dry-run"], opener)
    assert code == 0
    assert "Dry run" in stdout
    assert "model.safetensors" in stdout
    assert len(opener.requests) == 1
    assert not list(tmp_path.iterdir())


def test_hf_tree_filters_candidates_and_explicit_file(tmp_path):
    payload = b"chosen"
    digest = hashlib.sha256(payload).hexdigest()
    metadata = hf_metadata(
        payload,
        files=[
            {"rfilename": "other/no.bin", "lfs": {"sha256": "1" * 64, "size": 2}},
            {"rfilename": "weights/a.safetensors", "lfs": {"sha256": "2" * 64, "size": 2}},
            {"rfilename": "weights/b.gguf", "lfs": {"sha256": digest, "size": len(payload)}},
        ],
    )
    routes = hf_routes(payload, metadata=metadata)
    old_download = next(url for url in routes if "/resolve/" in url)
    routes[f"https://huggingface.co/owner/repo/resolve/{'a' * 40}/weights/b.gguf?download=true"] = routes.pop(old_download)
    opener = Opener(routes)
    code, _, _ = run_cli(
        ["https://huggingface.co/owner/repo/tree/main/weights", "--file", "weights/b.gguf", "--dir", str(tmp_path)], opener
    )
    assert code == 0
    assert (tmp_path / "b.gguf").read_bytes() == payload


def test_ambiguous_repo_fails_noninteractive_with_choices(tmp_path):
    metadata = hf_metadata(
        b"x",
        files=[{"rfilename": "a.safetensors"}, {"rfilename": "b.gguf"}],
    )
    opener = Opener(hf_routes(b"x", metadata=metadata))
    code, _, stderr = run_cli(["https://huggingface.co/owner/repo", "--dir", str(tmp_path)], opener)
    assert code == 2
    assert "a.safetensors" in stderr and "b.gguf" in stderr
    assert "--file" in stderr


def test_ambiguous_repo_offers_numbered_interactive_choice(tmp_path):
    payload = b"second"
    digest = hashlib.sha256(payload).hexdigest()
    metadata = hf_metadata(
        payload,
        files=[
            {"rfilename": "a.safetensors", "lfs": {"sha256": "1" * 64, "size": 1}},
            {"rfilename": "b.gguf", "lfs": {"sha256": digest, "size": len(payload)}},
        ],
    )
    routes = hf_routes(payload, metadata=metadata)
    old = next(url for url in routes if "/resolve/" in url)
    routes[f"https://huggingface.co/owner/repo/resolve/{'a' * 40}/b.gguf?download=true"] = routes.pop(old)
    stdin = io.StringIO("2\n")
    stdin.isatty = lambda: True
    code, _, stderr = run_cli(["https://huggingface.co/owner/repo", "--dir", str(tmp_path)], Opener(routes), stdin=stdin)
    assert code == 0
    assert "2. b.gguf" in stderr
    assert (tmp_path / "b.gguf").read_bytes() == payload


def civitai_version(payload: bytes):
    return {
        "id": 290640,
        "files": [
            {
                "name": "full/model-fp16.safetensors",
                "type": "Model",
                "primary": True,
                "downloadUrl": "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor&size=full&fp=fp16",
                "hashes": {"SHA256": hashlib.sha256(payload).hexdigest().upper()},
                "metadata": {"format": "SafeTensor", "size": "full", "fp": "fp16"},
            },
            {
                "name": "training-data.zip",
                "type": "Training Data",
                "downloadUrl": "https://civitai.com/api/download/models/999",
            },
        ],
    }


def test_civitai_direct_query_honors_version_variant_and_name(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_API_TOKEN", "secret-civitai")
    payload = b"civitai-model"
    source = "https://civitai.com/api/download/models/290640?type=Model&format=SafeTensor&size=full&fp=fp16"
    routes = {
        "https://civitai.com/api/v1/model-versions/290640": json.dumps(civitai_version(payload)).encode(),
        source: Response(payload, {"Content-Length": str(len(payload)), "Content-Type": "application/octet-stream"}),
    }
    opener = Opener(routes)
    code, stdout, _ = run_cli([source, "--dir", str(tmp_path)], opener)
    assert code == 0
    assert "revisione 290640" in stdout
    assert (tmp_path / "model-fp16.safetensors").read_bytes() == payload
    assert all(req.get_header("Authorization") == "Bearer secret-civitai" for req in opener.requests)


def test_civitai_model_page_honors_explicit_version_dry_run(tmp_path):
    payload = b"unused"
    opener = Opener({"https://civitai.com/api/v1/model-versions/290640": json.dumps(civitai_version(payload)).encode()})
    code, stdout, _ = run_cli(
        ["https://civitai.com/models/257749/example?modelVersionId=290640", "--dir", str(tmp_path), "--dry-run"], opener
    )
    assert code == 0
    assert "model-fp16.safetensors" in stdout
    assert len(opener.requests) == 1


def test_civitai_primary_model_is_only_automatic_choice(tmp_path):
    payload = b"primary"
    version = civitai_version(payload)
    version["files"].append(
        {
            "name": "alternate.gguf",
            "type": "Model",
            "primary": False,
            "downloadUrl": "https://civitai.com/api/download/models/290640?format=GGUF",
            "hashes": {"SHA256": "1" * 64},
            "metadata": {"format": "GGUF"},
        }
    )
    source = version["files"][0]["downloadUrl"]
    opener = Opener(
        {
            "https://civitai.com/api/v1/model-versions/290640": json.dumps(version).encode(),
            source: Response(payload, {"Content-Length": str(len(payload))}),
        }
    )
    code, _, _ = run_cli(["https://civitai.com/model-versions/290640", "--dir", str(tmp_path)], opener)
    assert code == 0
    assert (tmp_path / "model-fp16.safetensors").exists()


def test_existing_matching_sha_is_reused_without_download(tmp_path):
    payload = b"already-here"
    target = tmp_path / "model.safetensors"
    target.write_bytes(payload)
    routes = hf_routes(payload)
    routes.pop(next(url for url in routes if "/resolve/" in url))
    code, stdout, stderr = run_cli(
        ["https://huggingface.co/owner/repo/blob/main/weights/model.safetensors", "--dir", str(tmp_path)], Opener(routes)
    )
    assert code == 0
    assert "Verificato:" in stdout and "Già presente e verificato" in stderr
    assert target.read_bytes() == payload


def test_existing_conflicting_file_is_never_overwritten(tmp_path):
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"keep-me")
    opener = Opener(hf_routes(b"new"))
    code, _, stderr = run_cli(
        ["https://huggingface.co/owner/repo/blob/main/weights/model.safetensors", "--dir", str(tmp_path)], opener
    )
    assert code == 2 and "non coincide" in stderr
    assert target.read_bytes() == b"keep-me"
    assert len(opener.requests) == 1


@pytest.mark.parametrize(("racing_bytes", "expected_code"), [(b"new", 0), (b"other", 2)])
def test_atomic_publication_handles_same_and_conflicting_concurrent_writer(tmp_path, monkeypatch, racing_bytes, expected_code):
    payload = b"new"
    destination = tmp_path / "model.safetensors"

    def concurrent_link(_source, target):
        Path(target).write_bytes(racing_bytes)
        raise FileExistsError(target)

    monkeypatch.setattr(model_get.os, "link", concurrent_link)
    code, stdout, stderr = run_cli(
        ["https://huggingface.co/owner/repo/blob/main/weights/model.safetensors", "--dir", str(tmp_path)],
        Opener(hf_routes(payload)),
    )
    assert code == expected_code
    assert destination.read_bytes() == racing_bytes
    if expected_code == 0:
        assert "Pubblicato da un altro processo e verificato" in stderr
        assert "Verificato:" in stdout
    else:
        assert "Conflitto" in stderr


@pytest.mark.parametrize(
    ("headers", "body", "message"),
    [
        ({"Content-Length": "20", "Content-Type": "application/octet-stream"}, b"short", "Download incompleto"),
        ({"Content-Length": "2", "Content-Type": "text/html"}, b"{}", "text/html"),
        ({"Content-Length": "2", "Content-Type": "application/json"}, b"{}", "application/json"),
        ({"Content-Length": "35", "Content-Type": "application/octet-stream"}, b'{"error":"gated model unavailable"}', "pagina di errore"),
    ],
)
def test_bad_download_is_discarded(tmp_path, headers, body, message):
    metadata = hf_metadata(body)
    routes = hf_routes(body, metadata=metadata)
    download_url = next(url for url in routes if "/resolve/" in url)
    routes[download_url] = Response(body, headers)
    code, _, stderr = run_cli(
        ["https://huggingface.co/owner/repo/blob/main/weights/model.safetensors", "--dir", str(tmp_path)], Opener(routes)
    )
    assert code == 2 and message in stderr
    assert not (tmp_path / "model.safetensors").exists()
    assert not list(tmp_path.glob("*.part"))


def test_wrong_expected_sha_is_discarded(tmp_path):
    payload = b"content"
    metadata = hf_metadata(payload)
    metadata["siblings"][0]["lfs"]["sha256"] = "0" * 64
    opener = Opener(hf_routes(payload, metadata=metadata))
    code, _, stderr = run_cli(
        ["https://huggingface.co/owner/repo/blob/main/weights/model.safetensors", "--dir", str(tmp_path)], opener
    )
    assert code == 2 and "SHA256" in stderr
    assert not (tmp_path / "model.safetensors").exists()


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/owner/repo",
        "https://evil.example/model.safetensors",
        "https://user:pass@huggingface.co/owner/repo",
        "https://huggingface.co/owner/repo?token=secret",
    ],
)
def test_unsafe_source_urls_are_rejected_without_network(url, tmp_path):
    opener = Opener({})
    code, _, stderr = run_cli([url, "--dir", str(tmp_path)], opener)
    assert code == 2 and "Errore:" in stderr
    assert opener.requests == []


@pytest.mark.parametrize("name", ["../evil.safetensors", "dir/evil.safetensors", "evil.sh", "bad\x00.safetensors"])
def test_unsafe_destination_names_are_rejected(name, tmp_path):
    routes = hf_routes(b"unused")
    routes.pop(next(url for url in routes if "/resolve/" in url))
    code, _, _ = run_cli(
        ["https://huggingface.co/owner/repo/blob/main/weights/model.safetensors", "--dir", str(tmp_path), "--name", name, "--dry-run"],
        Opener(routes),
    )
    assert code == 2
    assert not list(tmp_path.iterdir())


def test_cross_host_redirect_strips_authorization_and_query_secret():
    handler = model_get.SecureRedirectHandler()
    request = urllib.request.Request("https://huggingface.co/file", headers={"Authorization": "Bearer secret"})
    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://cdn.example/file?token=secret&signature=keep",
    )
    assert redirected.get_header("Authorization") is None
    assert "token=" not in redirected.full_url
    assert "signature=keep" in redirected.full_url


def test_cross_host_redirect_preserves_signed_cdn_query_byte_for_byte():
    handler = model_get.SecureRedirectHandler()
    request = urllib.request.Request("https://huggingface.co/file", headers={"Authorization": "Bearer secret"})
    signed = "https://cdn.example/file?X-Amz-Credential=a%2Fb%2Bc&X-Amz-Signature=abc%2F123"
    redirected = handler.redirect_request(request, None, 302, "Found", {}, signed)
    assert redirected.full_url == signed
    assert redirected.get_header("Authorization") is None


def test_same_host_redirect_keeps_authorization():
    handler = model_get.SecureRedirectHandler()
    request = urllib.request.Request("https://huggingface.co/file", headers={"Authorization": "Bearer secret"})
    redirected = handler.redirect_request(request, None, 302, "Found", {}, "https://huggingface.co/pinned")
    assert redirected.get_header("Authorization") == "Bearer secret"


def test_non_https_redirect_is_rejected():
    handler = model_get.SecureRedirectHandler()
    request = urllib.request.Request("https://civitai.com/file")
    with pytest.raises(model_get.ModelGetError, match="HTTPS"):
        handler.redirect_request(request, None, 302, "Found", {}, "http://cdn.example/file")


def test_gated_error_names_token_environment_without_leaking_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "do-not-print")
    api = "https://huggingface.co/api/models/owner/repo/revision/main?blobs=true"
    error = urllib.error.HTTPError(api, 403, "Forbidden", {}, None)
    code, _, stderr = run_cli(["https://huggingface.co/owner/repo", "--dir", str(tmp_path)], Opener({api: error}))
    assert code == 2 and "HF_TOKEN" in stderr
    assert "do-not-print" not in stderr


def test_shell_entrypoint_is_portable_and_executable():
    launcher = SCRIPT.with_name("model-get")
    text = launcher.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert 'exec python3 "$script_dir/model_get.py" "$@"' in text
    assert os.access(launcher, os.X_OK)
