#!/usr/bin/env python3
"""Scarica in modo verificato un singolo file modello da Hugging Face o Civitai."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, TextIO


MODEL_SUFFIXES = {".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin"}
PROVIDER_HOSTS = {"huggingface.co": "hf", "www.huggingface.co": "hf", "civitai.com": "civitai", "www.civitai.com": "civitai"}
SENSITIVE_QUERY_KEYS = {"token", "api_key", "apikey", "authorization", "access_token"}
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ModelGetError(RuntimeError):
    """Errore mostrabile all'utente senza traceback."""


@dataclass(frozen=True)
class Candidate:
    name: str
    download_url: str
    sha256: str | None = None
    size: int | None = None
    primary: bool = False
    kind: str | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class ResolvedFile:
    provider: str
    name: str
    download_url: str
    sha256: str | None
    size: int | None
    revision: str


def _host(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def _sensitive_query(url: str) -> bool:
    return any(key.lower() in SENSITIVE_QUERY_KEYS for key, _ in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True))


def _validate_https_url(url: str, *, initial: bool = False) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise ModelGetError("Sono accettati solo URL HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ModelGetError("L'URL non può contenere credenziali.")
    if not parsed.hostname:
        raise ModelGetError("URL non valido: host mancante.")
    if initial and _host(url) not in PROVIDER_HOSTS:
        raise ModelGetError("Provider non supportato: usa un URL Hugging Face o Civitai.")
    if initial and _sensitive_query(url):
        raise ModelGetError("Non inserire token nell'URL: usa HF_TOKEN o CIVITAI_API_TOKEN.")
    return parsed


def _strip_sensitive_query(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key.lower() in SENSITIVE_QUERY_KEYS for key, _ in pairs):
        return url
    clean = [(key, value) for key, value in pairs if key.lower() not in SENSITIVE_QUERY_KEYS]
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(clean), parsed.fragment))


class SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Non inoltra segreti a un host di redirect diverso."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        _validate_https_url(newurl)
        if _host(newurl) != _host(req.full_url):
            newurl = _strip_sensitive_query(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _host(newurl) != _host(req.full_url):
            redirected.remove_header("Authorization")
            redirected.remove_header("Proxy-Authorization")
        return redirected


def _read_cached_hf_token() -> str | None:
    direct = os.environ.get("HF_TOKEN", "").strip()
    if direct:
        return direct
    configured = os.environ.get("HF_TOKEN_PATH", "").strip()
    if configured:
        paths = [Path(configured).expanduser()]
    else:
        hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        paths = [hf_home / "token"]
    for path in paths:
        try:
            token = path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            continue
        if token:
            return token
    return None


class HttpClient:
    def __init__(self, opener=None):  # noqa: ANN001
        self.opener = opener or urllib.request.build_opener(SecureRedirectHandler())
        self.tokens = {
            "hf": _read_cached_hf_token(),
            "civitai": (os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_TOKEN") or "").strip() or None,
        }

    def open(self, url: str, *, provider: str, headers: Mapping[str, str] | None = None):  # noqa: ANN201
        _validate_https_url(url)
        request_headers = {"User-Agent": "model-get/1", "Accept-Encoding": "identity"}
        if headers:
            request_headers.update(headers)
        provider_hosts = {"hf": {"huggingface.co", "www.huggingface.co"}, "civitai": {"civitai.com", "www.civitai.com"}}
        token = self.tokens.get(provider)
        if token and _host(url) in provider_hosts[provider]:
            request_headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=request_headers)
        try:
            return self.opener.open(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                env_name = "HF_TOKEN" if provider == "hf" else "CIVITAI_API_TOKEN"
                raise ModelGetError(f"Accesso negato dal provider ({exc.code}). Configura {env_name} se il file è privato o gated.") from exc
            raise ModelGetError(f"Il provider ha risposto con errore HTTP {exc.code}.") from exc
        except urllib.error.URLError as exc:
            raise ModelGetError(f"Connessione al provider non riuscita: {exc.reason}.") from exc

    def json(self, url: str, *, provider: str) -> Mapping[str, object]:
        with self.open(url, provider=provider, headers={"Accept": "application/json"}) as response:
            payload = response.read(16 * 1024 * 1024 + 1)
        if len(payload) > 16 * 1024 * 1024:
            raise ModelGetError("Risposta metadati troppo grande.")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelGetError("Il provider ha restituito metadati non validi.") from exc
        if not isinstance(value, dict):
            raise ModelGetError("Il provider ha restituito metadati inattesi.")
        return value


def _safe_remote_path(value: str) -> str:
    decoded = urllib.parse.unquote(value)
    pure = Path(decoded)
    if not decoded or decoded.startswith("/") or "\\" in decoded or ".." in pure.parts or CONTROL_RE.search(decoded):
        raise ModelGetError("Percorso file remoto non sicuro.")
    return decoded


def _safe_filename(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value or CONTROL_RE.search(value):
        raise ModelGetError("Il nome destinazione deve essere un nome file sicuro, senza cartelle.")
    if Path(value).suffix.lower() not in MODEL_SUFFIXES:
        allowed = ", ".join(sorted(MODEL_SUFFIXES))
        raise ModelGetError(f"Estensione modello non supportata. Usa una di: {allowed}.")
    return value


def _sha_from_mapping(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.removeprefix("sha256:")
    return candidate.lower() if SHA256_RE.fullmatch(candidate) else None


def _choose_candidate(
    candidates: Iterable[Candidate],
    requested_file: str | None,
    *,
    stdin: TextIO,
    stderr: TextIO,
) -> Candidate:
    options = list(candidates)
    if requested_file:
        requested = _safe_remote_path(requested_file)
        exact = [item for item in options if item.name == requested]
        if not exact:
            by_basename = [item for item in options if Path(item.name).name == requested]
            exact = by_basename if len(by_basename) == 1 else []
        if len(exact) != 1:
            raise ModelGetError(f"File remoto non trovato o ambiguo: {requested_file}")
        return exact[0]
    if not options:
        raise ModelGetError("Nessun file modello supportato trovato.")
    if len(options) == 1:
        return options[0]
    primary = [item for item in options if item.primary]
    if len(primary) == 1:
        return primary[0]
    lines = ["La sorgente contiene più file modello:"]
    lines.extend(f"  {index}. {item.name}" for index, item in enumerate(options, 1))
    if not getattr(stdin, "isatty", lambda: False)():
        raise ModelGetError("\n".join(lines + ["Specifica --file NOME per scegliere senza interazione."]))
    print("\n".join(lines), file=stderr)
    print("Numero del file: ", end="", file=stderr, flush=True)
    answer = stdin.readline().strip()
    try:
        index = int(answer)
    except ValueError as exc:
        raise ModelGetError("Selezione non valida.") from exc
    if index < 1 or index > len(options):
        raise ModelGetError("Selezione fuori intervallo.")
    return options[index - 1]


def _hf_parts(url: str) -> tuple[str, str, str, str | None, str | None]:
    parsed = _validate_https_url(url, initial=True)
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    repo_type = "model"
    if parts and parts[0] == "datasets":
        repo_type = "dataset"
        parts = parts[1:]
    if len(parts) < 2:
        raise ModelGetError("URL Hugging Face incompleto: manca owner/repository.")
    repo_id = f"{parts[0]}/{parts[1]}"
    action = parts[2] if len(parts) > 2 else "root"
    if action not in {"root", "blob", "resolve", "tree"}:
        raise ModelGetError("URL Hugging Face non riconosciuto.")
    revision = parts[3] if len(parts) > 3 and action != "root" else "main"
    remote_path = "/".join(parts[4:]) if len(parts) > 4 else None
    if action in {"blob", "resolve"} and not remote_path:
        raise ModelGetError("URL Hugging Face incompleto: manca il file.")
    return repo_type, repo_id, revision, action, remote_path


def _resolve_hf(url: str, requested_file: str | None, client: HttpClient, *, stdin: TextIO, stderr: TextIO) -> ResolvedFile:
    repo_type, repo_id, revision, action, remote_path = _hf_parts(url)
    api_kind = "datasets" if repo_type == "dataset" else "models"
    api_url = f"https://huggingface.co/api/{api_kind}/{repo_id}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    metadata = client.json(api_url, provider="hf")
    immutable = metadata.get("sha")
    if not isinstance(immutable, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", immutable):
        raise ModelGetError("Hugging Face non ha restituito una revisione immutabile valida.")
    prefix = remote_path.rstrip("/") + "/" if action == "tree" and remote_path else ""
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise ModelGetError("Elenco file Hugging Face assente.")
    candidates: list[Candidate] = []
    for sibling in siblings:
        if not isinstance(sibling, dict) or not isinstance(sibling.get("rfilename"), str):
            continue
        name = _safe_remote_path(sibling["rfilename"])
        if prefix and not name.startswith(prefix):
            continue
        if Path(name).suffix.lower() not in MODEL_SUFFIXES:
            continue
        lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else {}
        digest = _sha_from_mapping(lfs.get("sha256") or lfs.get("oid") or sibling.get("blobId"))
        raw_size = lfs.get("size", sibling.get("size"))
        size = raw_size if isinstance(raw_size, int) and raw_size >= 0 else None
        base = f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"
        download_url = f"{base}/resolve/{immutable}/{urllib.parse.quote(name, safe='/')}?download=true"
        candidates.append(Candidate(name=name, download_url=download_url, sha256=digest, size=size))
    if action in {"blob", "resolve"}:
        requested_file = remote_path
    selected = _choose_candidate(candidates, requested_file, stdin=stdin, stderr=stderr)
    return ResolvedFile("hf", Path(selected.name).name, selected.download_url, selected.sha256, selected.size, immutable.lower())


def _query_first(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def _resolve_civitai(url: str, requested_file: str | None, client: HttpClient, *, stdin: TextIO, stderr: TextIO) -> ResolvedFile:
    parsed = _validate_https_url(url, initial=True)
    parts = [part for part in parsed.path.split("/") if part]
    query = urllib.parse.parse_qs(parsed.query)
    version_id: str | None = None
    model_id: str | None = None
    if len(parts) >= 4 and parts[:3] == ["api", "download", "models"]:
        version_id = parts[3]
    elif len(parts) >= 2 and parts[0] == "model-versions":
        version_id = parts[1]
    elif len(parts) >= 2 and parts[0] == "models":
        model_id = parts[1]
        version_id = _query_first(query, "modelVersionId")
    else:
        raise ModelGetError("URL Civitai non riconosciuto.")
    if version_id and not version_id.isdigit():
        raise ModelGetError("ID versione Civitai non valido.")
    if model_id and not model_id.isdigit():
        raise ModelGetError("ID modello Civitai non valido.")
    if version_id:
        version = client.json(f"https://civitai.com/api/v1/model-versions/{version_id}", provider="civitai")
    else:
        model = client.json(f"https://civitai.com/api/v1/models/{model_id}", provider="civitai")
        versions = model.get("modelVersions")
        if not isinstance(versions, list) or not versions or not isinstance(versions[0], dict):
            raise ModelGetError("Il modello Civitai non contiene versioni scaricabili.")
        version = versions[0]
        version_id = str(version.get("id", ""))
    if str(version.get("id", "")) != version_id:
        raise ModelGetError("I metadati Civitai non corrispondono alla versione richiesta.")
    files = version.get("files")
    if not isinstance(files, list):
        raise ModelGetError("Elenco file Civitai assente.")
    variant_keys = {key.lower(): value[0].lower() for key, value in query.items() if value and key.lower() in {"type", "format", "size", "fp"}}
    candidates: list[Candidate] = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("downloadUrl"), str):
            continue
        name = _safe_remote_path(item["name"])
        if Path(name).suffix.lower() not in MODEL_SUFFIXES:
            continue
        item_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if any(
            str(item.get("type", "") if key == "type" else item_metadata.get(key, "")).lower() != wanted
            for key, wanted in variant_keys.items()
        ):
            continue
        download_url = item["downloadUrl"]
        _validate_https_url(download_url)
        if _host(download_url) not in {"civitai.com", "www.civitai.com"}:
            raise ModelGetError("Civitai ha restituito un host di download inatteso.")
        hashes = item.get("hashes") if isinstance(item.get("hashes"), dict) else {}
        digest = _sha_from_mapping(hashes.get("SHA256") or hashes.get("sha256"))
        candidates.append(
            Candidate(
                name=name,
                download_url=download_url,
                sha256=digest,
                primary=item.get("primary") is True and str(item.get("type", "")).lower() == "model",
                kind=str(item.get("type", "")),
                metadata=item_metadata,
            )
        )
    selected = _choose_candidate(candidates, requested_file, stdin=stdin, stderr=stderr)
    return ResolvedFile("civitai", Path(selected.name).name, selected.download_url, selected.sha256, selected.size, version_id)


def resolve(url: str, requested_file: str | None, client: HttpClient, *, stdin: TextIO, stderr: TextIO) -> ResolvedFile:
    parsed = _validate_https_url(url, initial=True)
    provider = PROVIDER_HOSTS[_host(url)]
    if provider == "hf":
        return _resolve_hf(url, requested_file, client, stdin=stdin, stderr=stderr)
    return _resolve_civitai(url, requested_file, client, stdin=stdin, stderr=stderr)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header(response, name: str) -> str | None:  # noqa: ANN001
    headers = getattr(response, "headers", None)
    return headers.get(name) if headers is not None else None


def download(resolved: ResolvedFile, destination: Path, client: HttpClient, *, stderr: TextIO) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and resolved.sha256 and _file_sha256(destination) == resolved.sha256:
            print(f"Già presente e verificato: {destination}", file=stderr)
            return "reused"
        raise ModelGetError(f"Il file esiste già e non coincide con la sorgente: {destination}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=destination.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    expected_header: int | None = None
    try:
        with os.fdopen(fd, "wb") as output:
            with client.open(resolved.download_url, provider=resolved.provider, headers={"Accept": "application/octet-stream"}) as response:
                content_type = (_header(response, "Content-Type") or "").split(";", 1)[0].strip().lower()
                if content_type in {"text/html", "application/json", "text/json"} or content_type.endswith("+json"):
                    raise ModelGetError(f"Il provider ha restituito {content_type or 'contenuto non modello'} invece del modello.")
                raw_length = _header(response, "Content-Length")
                if raw_length:
                    try:
                        expected_header = int(raw_length)
                    except ValueError as exc:
                        raise ModelGetError("Content-Length non valido.") from exc
                    if expected_header < 0:
                        raise ModelGetError("Content-Length non valido.")
                next_report = 64 * 1024 * 1024
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if count == 0:
                        leading = block.lstrip()[:64].lower()
                        if leading.startswith((b"<html", b"<!doctype html", b'{"error"', b'{"message"', b'[{"error"')):
                            raise ModelGetError("Il provider ha restituito una pagina di errore invece del modello.")
                    output.write(block)
                    digest.update(block)
                    count += len(block)
                    if count >= next_report:
                        if expected_header:
                            print(f"Scaricati {count / 1048576:.0f}/{expected_header / 1048576:.0f} MiB", file=stderr)
                        else:
                            print(f"Scaricati {count / 1048576:.0f} MiB", file=stderr)
                        next_report += 64 * 1024 * 1024
            output.flush()
            os.fsync(output.fileno())
        if expected_header is not None and count != expected_header:
            raise ModelGetError(f"Download incompleto: ricevuti {count} byte, attesi {expected_header}.")
        if resolved.size is not None and count != resolved.size:
            raise ModelGetError(f"Dimensione non valida: ricevuti {count} byte, attesi {resolved.size}.")
        actual_sha = digest.hexdigest()
        if resolved.sha256 and actual_sha != resolved.sha256:
            raise ModelGetError("SHA256 non corrispondente: il file scaricato è stato scartato.")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_file() and _file_sha256(destination) == actual_sha:
                print(f"Pubblicato da un altro processo e verificato: {destination}", file=stderr)
                return "reused"
            raise ModelGetError(f"Conflitto: un altro processo ha creato {destination}.")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError:
                # Alcuni filesystem remoti non supportano fsync sulle directory;
                # il file è già stato pubblicato atomicamente con link(2).
                pass
        finally:
            os.close(directory_fd)
        return "downloaded"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-get",
        description="Scarica un singolo modello verificato da Hugging Face o Civitai.",
        epilog=(
            "Esempi:\n"
            "  model-get 'https://huggingface.co/owner/repo/blob/main/model.safetensors'\n"
            "  model-get 'https://civitai.com/models/123?modelVersionId=456' --dir /storage/ComfyUI/models/loras\n"
            "  model-get URL --file variante.safetensors --name mio-modello.safetensors --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", help="URL Hugging Face o Civitai")
    parser.add_argument("--dir", default=".", dest="directory", help="cartella destinazione (default: cartella corrente)")
    parser.add_argument("--name", help="nome locale esplicito, con estensione modello valida")
    parser.add_argument("--file", dest="remote_file", help="file remoto da scegliere quando la sorgente è ambigua")
    parser.add_argument("--dry-run", action="store_true", help="risolve metadati e destinazione senza scaricare i byte")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    opener=None,  # noqa: ANN001
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = _parser().parse_args(argv)
    try:
        client = HttpClient(opener)
        selected = resolve(args.url, args.remote_file, client, stdin=stdin, stderr=stderr)
        filename = _safe_filename(args.name or selected.name)
        directory = Path(args.directory).expanduser().resolve()
        destination = directory / filename
        print(f"Sorgente: {selected.provider} · revisione {selected.revision}", file=stdout)
        print(f"File: {filename}", file=stdout)
        print(f"Destinazione: {destination}", file=stdout)
        if args.dry_run:
            print("Dry run: nessun byte modello scaricato.", file=stdout)
            return 0
        print(f"Download di {filename}…", file=stderr)
        result = download(selected, destination, client, stderr=stderr)
        if result == "downloaded":
            print(f"Completato e verificato: {destination}", file=stdout)
        else:
            print(f"Verificato: {destination}", file=stdout)
        return 0
    except ModelGetError as exc:
        print(f"Errore: {exc}", file=stderr)
        return 2
    except OSError as exc:
        print(f"Errore filesystem: {exc}", file=stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
