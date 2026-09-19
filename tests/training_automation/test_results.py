import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from training_automation.backup import BackupError
from training_automation.gallery import render_gallery
import training_automation.results as results
from training_automation.results import (
    BODY_FONT_SIZE,
    CARD_WIDTH,
    GRID_GAP,
    SAMPLE_HEADING_FONT_SIZE,
    SHEET_HEADING_FONT_SIZE,
    _scalable_font,
    _sample_text_rows,
    _text_width,
    _wrap_text_pixels,
    publish_archived_run_results,
    publish_ranked_results,
    refresh_archived_run_reports,
    render_contact_sheet,
)


FAKE_SPEC = importlib.util.spec_from_file_location(
    "result_export_fakes", Path(__file__).with_name("fakes.py")
)
FAKES = importlib.util.module_from_spec(FAKE_SPEC)
assert FAKE_SPEC.loader is not None
FAKE_SPEC.loader.exec_module(FAKES)
FakeHubClient = FAKES.FakeHubClient
ARCHIVE_COMPLETION_REVISION = "a" * 40
ARCHIVE_EVIDENCE_REVISION = "b" * 40


def _write_samples(root: Path, steps=(100, 200, 300)) -> dict[int, list[dict]]:
    root.mkdir(parents=True)
    result = {}
    for step in steps:
        samples = []
        for index, size in enumerate(((18, 9), (9, 18), (12, 12))):
            name = f"123__{step:09d}_{index}.png"
            Image.new("RGB", size, (step // 2, 20 + index, 80)).save(root / name)
            samples.append({
                "path": f"/deleted/private/pod/{name}",
                "prompt_index": index,
                "prompt": "<script>private prompt</script>" if index == 0 else f"prompt {index}",
                "seed": 42,
                "metrics": {
                    "clipping_fraction_proxy": 0.01 * index,
                    "sharpness_laplacian_variance_proxy": 0.2 + index,
                },
                "identity": {
                    "status": "missing" if index == 2 else "available",
                    "cosine_similarity": None if index == 2 else 0.5 + step / 1000 + index / 100,
                    "reason": "no face detected" if index == 2 else None,
                },
                "pose_body_landmarks": {
                    "status": "available" if index != 2 else "occluded",
                    "values": {
                        "ratios": {"shoulder_to_hip_width": 1.2 + index / 10},
                        "joint_angles": {"left_elbow_degrees": 120.0},
                    } if index != 2 else None,
                    "reason": "2D image-plane proxy" if index != 2 else "joint hidden",
                },
            })
        result[step] = samples
    return result


def _report(samples: dict[int, list[dict]], shortlist_status="available") -> dict:
    order = [200, 100, 300]
    identity_aggregates = [
        {
            "step": step,
            "mean_face_cosine_similarity": {200: 0.81, 100: 0.79, 300: 0.75}[step],
            "prompt_indices": [0, 1],
        }
        for step in order
    ]
    shortlist = [
        {
            **item,
            "mean_clipping_fraction_proxy": sum(
                sample["metrics"]["clipping_fraction_proxy"]
                for sample in samples[item["step"]]
            ) / len(samples[item["step"]]),
        }
        for item in identity_aggregates
    ]
    return {
        "schema_version": 1,
        "job_config": "/storage/generated/subject-job.yaml",
        "checkpoints": [
            {
                "step": step,
                "final": step == 300,
                "catalog_checkpoint_id": f"checkpoint-{step}",
                "remote_checkpoint": {"commit_id": "r0"},
                "remote_association": {"status": "unique", "reason": None},
                "sample_run_status": "complete",
                "samples": samples[step],
            }
            for step in (100, 200, 300)
        ],
        "identity_ranking": {
            "status": "available",
            "reason": None,
            "method": "raw common-face cosine",
            "items": identity_aggregates,
        },
        "ranking": {
            "status": "available",
            "reason": None,
        },
        "automatic_shortlist": {
            "status": shortlist_status,
            "reason": None if shortlist_status == "available" else "fewer than three common faces",
            "method": "face cosine descending, clipping ascending, step ascending",
            "items": shortlist if shortlist_status == "available" else [],
        },
    }


def _catalog_and_client(report: dict) -> tuple[FakeHubClient, bytes]:
    client = FakeHubClient()
    checkpoints = []
    for checkpoint in report["checkpoints"]:
        step = checkpoint["step"]
        payload = f"immutable-weight-{step}".encode()
        remote = f"training-backups/models/0001-subject/checkpoints/checkpoint-{step}/weights/model-{step}.safetensors"
        client.remote[remote] = payload
        checkpoints.append({
            "checkpoint_id": f"checkpoint-{step}",
            "step": step,
            "final": step == 300,
            "revision": "r0",
            "weights": [{
                "remote_path": remote,
                "generation_relative_path": f"0001-subject/model-{step}.safetensors",
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }],
            "resume_artifacts": [],
        })
    catalog = json.dumps({
        "schema_version": 2,
        "models": [{
            "id": 1,
            "name": "Subject",
            "folder": "0001-subject",
            "base_arch": "flux2_klein_9b",
            "base_model": "black-forest-labs/FLUX.2-klein-base-9B",
            "trigger_word": "TOKEN",
            "destination_kind": "loras",
            "checkpoints": checkpoints,
            "selected_checkpoint_id": "checkpoint-300",
            "selection": {"checkpoint_id": "checkpoint-300"},
        }],
    }, sort_keys=True).encode()
    client.remote["training-backups/catalog.json"] = catalog
    client.snapshots["r0"] = dict(client.remote)
    return client, catalog


def test_gallery_is_self_contained_ranked_and_escapes_untrusted_text(tmp_path):
    samples = _write_samples(tmp_path / "samples")
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(_report(samples)), encoding="utf-8")
    output = render_gallery(
        report_path,
        sample_root=tmp_path / "samples",
        output_path=tmp_path / "gallery.html",
        title="<img src=x onerror=alert(1)>",
    )
    content = output.read_text(encoding="utf-8")
    first_payload = (tmp_path / "samples" / "123__000000100_0.png").read_bytes()
    assert "&lt;img src=x onerror=alert(1)&gt;" in content
    assert "<script>private prompt</script>" not in content
    assert "&lt;script&gt;private prompt&lt;/script&gt;" in content
    assert f"data:image/png;base64,{base64.b64encode(first_payload).decode()}" in content
    assert content.index("Step 200") < content.index("Step 100") < content.index("Step 300")
    assert "raw face cosine</dt><dd>missing: no face detected" in content
    assert "Raw cosine is not a percentage" in content
    assert "object-fit" not in content and "height:150px" not in content


def test_gallery_rejects_duplicate_basename_remapping(tmp_path):
    samples = _write_samples(tmp_path / "samples", steps=(100,))
    report = _report({100: samples[100], 200: samples[100], 300: samples[100]})
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(BackupError, match="duplicate sample basename"):
        render_gallery(
            report_path,
            sample_root=tmp_path / "samples",
            output_path=tmp_path / "gallery.html",
        )


def test_contact_sheet_uses_readable_measured_type_and_preserves_every_image(
    tmp_path,
):
    samples = _write_samples(tmp_path / "samples", steps=(100,))
    checkpoint = {"step": 100, "samples": samples[100]}
    aggregate = {
        "step": 100,
        "mean_face_cosine_similarity": 0.75,
        "mean_clipping_fraction_proxy": 0.01,
        "prompt_indices": [0, 1],
    }
    output = render_contact_sheet(
        checkpoint,
        sample_root=tmp_path / "samples",
        output_path=tmp_path / "contact-sheet.png",
        heading="0001-subject · top-1 · step 100",
        aggregate=aggregate,
    )

    assert BODY_FONT_SIZE >= 18
    assert SAMPLE_HEADING_FONT_SIZE >= 24
    assert SHEET_HEADING_FONT_SIZE >= 24
    with Image.open(output) as sheet:
        sheet.load()
        assert sheet.width == 2 * CARD_WIDTH + 3 * GRID_GAP
        colors = set(sheet.getdata())
    for index in range(3):
        assert (50, 20 + index, 80) in colors

    font = _scalable_font(BODY_FONT_SIZE)
    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    wrapped = _wrap_text_pixels(
        draw,
        "one " + "unbroken-token-" * 80,
        font=font,
        max_width=260,
    )
    assert len(wrapped) > 2
    assert all(_text_width(draw, line, font) <= 260 for line in wrapped)
    visible_rows = _sample_text_rows(samples[100][0])
    assert any("Shoulder / hip width" in text for _, text in visible_rows)
    assert not any("left_elbow_degrees" in text for _, text in visible_rows)


def test_publisher_creates_verified_top_three_without_changing_selection(tmp_path):
    samples = _write_samples(tmp_path / "samples")
    report = _report(samples)
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    client, original_catalog = _catalog_and_client(report)

    record, evidence = publish_ranked_results(
        client=client,
        repo_id="owner/private",
        repo_type="dataset",
        run_id="run-1",
        job_id="subject-job",
        report_path=report_path,
        sample_root=tmp_path / "samples",
        work_dir=tmp_path / "results",
        sleep=lambda _: None,
        completed_at="2026-09-15T12:00:00+00:00",
    )

    assert record["status"] == "available"
    assert [item["step"] for item in record["candidates"]] == [200, 100, 300]
    assert record["canonical_selected_checkpoint_id"] == "checkpoint-300"
    assert client.remote["training-backups/catalog.json"] == original_catalog
    for rank, step in enumerate((200, 100, 300), 1):
        root = f"training-results/run-1/0001-subject/top-{rank}"
        assert f"{root}/contact-sheet.png" in client.remote
        assert f"{root}/report-index.json" in client.remote
        assert f"{root}/pages/page-01.png" in client.remote
        assert f"{root}/selection.json" in client.remote
        copied = client.remote[f"{root}/weights/model-{step}.safetensors"]
        assert copied == f"immutable-weight-{step}".encode()
        manifest = json.loads(client.remote[f"{root}/selection.json"])
        assert manifest["rank"] == rank
        assert manifest["catalog_checkpoint_id"] == f"checkpoint-{step}"
        assert len(manifest["samples"]) == 3
        assert manifest["report"]["pages"] == ["pages/page-01.png"]
        assert (
            manifest["samples"][0]["pose_body_landmarks"]["values"]
            ["joint_angles"]["left_elbow_degrees"]
            == 120.0
        )
        assert manifest["canonical_selection_unchanged"] is True
        assert manifest["restore"]["argv"][-2:] == [
            "--comfy-root", "/workspace/ComfyUI"
        ]
        assert f"--checkpoint-id checkpoint-{step}" in manifest["restore"]["command"]
    assert "training-results/run-1/0001-subject/overview.png" in client.remote
    pointer = json.loads(client.remote["training-results/latest/0001-subject.json"])
    assert pointer["status"] == "completed"
    assert pointer["run_id"] == "run-1"
    assert pointer["completed_at"] == "2026-09-15T12:00:00+00:00"
    assert pointer["evaluation_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert "training-results/run-1/0001-subject/index.json" in client.remote
    assert "training-results/run-1/0001-subject/evaluation-gallery.html" in client.remote
    assert {relative for _, relative in evidence} >= {
        "results/0001-subject/index.json",
        "results/0001-subject/top-1/contact-sheet.png",
    }


def test_unavailable_shortlist_publishes_reason_without_top_folder(tmp_path):
    samples = _write_samples(tmp_path / "samples")
    report = _report(samples, shortlist_status="unavailable")
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    client, _ = _catalog_and_client(report)

    record, _ = publish_ranked_results(
        client=client,
        repo_id="owner/private",
        repo_type="dataset",
        run_id="run-1",
        job_id="subject-job",
        report_path=report_path,
        sample_root=tmp_path / "samples",
        work_dir=tmp_path / "results",
        sleep=lambda _: None,
    )

    assert record["status"] == "unavailable"
    assert record["latest_pointer"] is None
    assert "training-results/latest/0001-subject.json" not in client.remote
    assert record["exported_candidate_count"] == 0
    assert "common faces" in record["reason"]
    assert not any("/top-" in path for path in client.remote)


def test_publisher_rejects_catalog_weight_hash_mismatch(tmp_path):
    samples = _write_samples(tmp_path / "samples")
    report = _report(samples)
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    client, _ = _catalog_and_client(report)
    client.corrupt_metadata = True

    with pytest.raises(BackupError, match="bounded retries"):
        publish_ranked_results(
            client=client,
            repo_id="owner/private",
            repo_type="dataset",
            run_id="run-1",
            job_id="subject-job",
            report_path=report_path,
            sample_root=tmp_path / "samples",
            work_dir=tmp_path / "results",
            max_attempts=1,
        )
    assert not any(path.startswith("training-results/") for path in client.remote)


def test_publisher_refuses_to_replace_different_existing_result_evidence(tmp_path):
    samples = _write_samples(tmp_path / "samples")
    report = _report(samples)
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    client, _ = _catalog_and_client(report)
    index_path = "training-results/run-1/0001-subject/index.json"
    original = b'{"run_id":"another-run","job_id":"another-job"}'
    client.remote[index_path] = original
    client.snapshots["r0"][index_path] = original

    with pytest.raises(BackupError, match="bounded retries"):
        publish_ranked_results(
            client=client,
            repo_id="owner/private",
            repo_type="dataset",
            run_id="run-1",
            job_id="subject-job",
            report_path=report_path,
            sample_root=tmp_path / "samples",
            work_dir=tmp_path / "results",
            max_attempts=1,
        )
    assert client.remote[index_path] == original


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["ranking"].update(status="unavailable"),
        lambda report: report["checkpoints"][0].update(sample_run_status="incomplete"),
        lambda report: report["checkpoints"][0]["remote_association"].update(
            status="ambiguous"
        ),
        lambda report: report["automatic_shortlist"]["items"].reverse(),
    ],
)
def test_available_shortlist_must_retain_complete_deterministic_gates(
    tmp_path, mutation
):
    samples = _write_samples(tmp_path / "samples")
    report = _report(samples)
    mutation(report)
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    client, _ = _catalog_and_client(report)

    with pytest.raises(BackupError, match="bounded retries"):
        publish_ranked_results(
            client=client,
            repo_id="owner/private",
            repo_type="dataset",
            run_id="run-1",
            job_id="subject-job",
            report_path=report_path,
            sample_root=tmp_path / "samples",
            work_dir=tmp_path / "results",
            max_attempts=1,
        )
    assert not any(path.startswith("training-results/") for path in client.remote)


def test_archived_run_backfill_discovers_every_job_without_manual_choice(
    tmp_path, monkeypatch
):
    archive = tmp_path / "archive"
    called = []
    for job_id in ("job-b", "job-a"):
        job_root = archive / "jobs" / job_id
        (job_root / "samples").mkdir(parents=True)
        (job_root / "evaluation.json").write_text(
            json.dumps({"job_config": f"/old/generated/{job_id}.yaml"}),
            encoding="utf-8",
        )

    def publish(**kwargs):
        called.append(kwargs["job_id"])
        return {"job_id": kwargs["job_id"]}, []

    monkeypatch.setattr(results, "publish_ranked_results", publish)
    records = publish_archived_run_results(
        client=FakeHubClient(),
        repo_id="owner/private",
        repo_type="dataset",
        run_id="run-1",
        archive_root=archive,
        work_dir=tmp_path / "work",
    )

    assert called == ["job-a", "job-b"]
    assert [item["job_id"] for item in records] == called


def test_older_or_unknown_completion_never_replaces_newer_latest_pointer(tmp_path):
    samples = _write_samples(tmp_path / "samples")
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(_report(samples)), encoding="utf-8")
    client, _ = _catalog_and_client(_report(samples))

    newest, _ = publish_ranked_results(
        client=client, repo_id="owner/private", repo_type="dataset",
        run_id="run-new", job_id="subject-job", report_path=report_path,
        sample_root=tmp_path / "samples", work_dir=tmp_path / "new",
        completed_at="2026-09-15T12:00:00+00:00", sleep=lambda _: None,
    )
    assert newest["latest_pointer"] == "training-results/latest/0001-subject.json"
    pointer_path = "training-results/latest/0001-subject.json"
    newest_pointer = client.remote[pointer_path]

    older, _ = publish_ranked_results(
        client=client, repo_id="owner/private", repo_type="dataset",
        run_id="run-old", job_id="subject-job", report_path=report_path,
        sample_root=tmp_path / "samples", work_dir=tmp_path / "old",
        completed_at="2025-09-15T12:00:00+00:00", sleep=lambda _: None,
    )
    assert older["latest_pointer"] is None
    assert client.remote[pointer_path] == newest_pointer

    unknown, _ = publish_ranked_results(
        client=client, repo_id="owner/private", repo_type="dataset",
        run_id="run-unknown", job_id="subject-job", report_path=report_path,
        sample_root=tmp_path / "samples", work_dir=tmp_path / "unknown",
        sleep=lambda _: None,
    )
    assert unknown["latest_pointer"] is None
    assert client.remote[pointer_path] == newest_pointer


def _install_archive(
    client, report, sample_root, *, corrupt=False,
    evidence_revision=ARCHIVE_EVIDENCE_REVISION,
):
    run_id = "run-archive"
    shard = "worker-a"
    job_id = "subject-job"
    root = f"training-archives/{run_id}/{shard}/evidence"
    entries = []
    payloads = {f"jobs/{job_id}/evaluation.json": json.dumps(report).encode()}
    for checkpoint in report["checkpoints"]:
        for sample in checkpoint["samples"]:
            name = Path(sample["path"]).name
            payloads[f"jobs/{job_id}/samples/{name}"] = (sample_root / name).read_bytes()
    for index, (relative, payload) in enumerate(sorted(payloads.items())):
        remote = f"{root}/{relative}"
        client.remote[remote] = payload + (b"corrupt" if corrupt and index == 0 else b"")
        entries.append({
            "remote_path": remote,
            "relative_path": relative,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    completion_path = f"training-archives/{run_id}/{shard}/completion.json"
    client.remote[completion_path] = json.dumps({
        "schema_version": 1,
        "run_id": run_id,
        "shard_id": shard,
        "status": "completed",
        "evidence_revision": evidence_revision,
        "evidence": entries,
        "completion": {"jobs": [{"job_id": job_id}]},
    }).encode()
    evidence_snapshot = dict(client.remote)
    evidence_snapshot.pop(completion_path)
    client.snapshots[evidence_revision] = evidence_snapshot
    client.revision = ARCHIVE_COMPLETION_REVISION
    client.snapshots[ARCHIVE_COMPLETION_REVISION] = dict(client.remote)


def test_refreshes_historical_reports_from_verified_hub_archive_without_weights(tmp_path):
    samples = _write_samples(tmp_path / "samples", steps=(100, 200, 300))
    report = _report(samples)
    client, _ = _catalog_and_client(report)
    _install_archive(client, report, tmp_path / "samples")

    result = refresh_archived_run_reports(
        client=client, repo_id="owner/private", repo_type="dataset",
        run_id="run-archive", source_revision=ARCHIVE_COMPLETION_REVISION,
        work_dir=tmp_path / "refresh", job_ids=("subject-job",),
    )

    assert result["status"] == "completed"
    assert result["model_weights_transferred"] is False
    assert [item["job_id"] for item in result["jobs"]] == ["subject-job"]
    assert result["jobs"][0]["archive_evidence_revision"] == ARCHIVE_EVIDENCE_REVISION
    assert any("/reports-v2/" in path for path in client.remote)
    assert not any(
        "/reports-v2/" in path and "/weights/" in path for path in client.remote
    )


def test_archive_refresh_rejects_size_equal_hash_mismatch(tmp_path):
    samples = _write_samples(tmp_path / "samples", steps=(100, 200, 300))
    report = _report(samples)
    client, _ = _catalog_and_client(report)
    _install_archive(client, report, tmp_path / "samples")
    evidence_path = next(
        path for path in client.remote if "/evidence/jobs/" in path
    )
    payload = client.remote[evidence_path]
    client.remote[evidence_path] = bytes([payload[0] ^ 1]) + payload[1:]
    client.snapshots[ARCHIVE_EVIDENCE_REVISION] = dict(client.remote)

    with pytest.raises(BackupError, match="hash verification failed"):
        refresh_archived_run_reports(
            client=client, repo_id="owner/private", repo_type="dataset",
            run_id="run-archive", source_revision=ARCHIVE_COMPLETION_REVISION,
            work_dir=tmp_path / "refresh",
        )


def test_archive_refresh_rejects_mutable_source_revision_before_download(tmp_path):
    class DownloadCountingClient(FakeHubClient):
        downloads = 0

        def download_file(self, *args, **kwargs):
            type(self).downloads += 1
            return super().download_file(*args, **kwargs)

    client = DownloadCountingClient()
    with pytest.raises(BackupError, match="immutable 40-character commit SHA"):
        refresh_archived_run_reports(
            client=client, repo_id="owner/private", repo_type="dataset",
            run_id="run-archive", source_revision="main",
            work_dir=tmp_path / "refresh",
        )
    assert DownloadCountingClient.downloads == 0


def test_archive_refresh_rejects_mutable_evidence_revision_before_evidence_download(
    tmp_path,
):
    samples = _write_samples(tmp_path / "samples", steps=(100, 200, 300))
    report = _report(samples)

    class DownloadRecordingClient(FakeHubClient):
        def __init__(self):
            super().__init__()
            self.downloads = []

        def download_file(self, repo_id, repo_type, path, revision, destination):
            self.downloads.append((path, revision))
            return super().download_file(repo_id, repo_type, path, revision, destination)

    client = DownloadRecordingClient()
    catalog_client, _ = _catalog_and_client(report)
    client.remote = dict(catalog_client.remote)
    client.snapshots = dict(catalog_client.snapshots)
    _install_archive(
        client, report, tmp_path / "samples", evidence_revision="main"
    )
    with pytest.raises(BackupError, match="evidence_revision.*immutable"):
        refresh_archived_run_reports(
            client=client, repo_id="owner/private", repo_type="dataset",
            run_id="run-archive", source_revision=ARCHIVE_COMPLETION_REVISION,
            work_dir=tmp_path / "refresh",
        )
    assert client.downloads == [(
        "training-archives/run-archive/worker-a/completion.json",
        ARCHIVE_COMPLETION_REVISION,
    )]


def test_extended_run_exports_its_ranked_checkpoints_despite_an_unassociated_step(
    tmp_path,
):
    """An extension resumes from a checkpoint of the previous job.

    That base step can appear in this job's evidence with samples but without a
    unique verified receipt of its own. It cannot be exported, and it must not
    take the whole top three down with it: the LaProfumosa refinement published
    `unavailable` with zero candidates while every other checkpoint of the same
    run was complete and uniquely associated.
    """
    samples = _write_samples(tmp_path / "samples", steps=(50, 100, 200, 300))
    report = _report(samples)
    client, _ = _catalog_and_client(report)
    report["checkpoints"].insert(0, {
        "step": 50,
        "final": False,
        "catalog_checkpoint_id": None,
        "remote_checkpoint": None,
        "remote_checkpoint_candidates": [],
        "remote_association": {
            "status": "unavailable",
            "reason": "no verified catalog checkpoint receipt for this job and step",
        },
        "sample_run_status": "complete",
        "samples": samples[50],
    })
    # The base step would win on identity if it were allowed to compete.
    report["identity_ranking"]["items"].append(
        {"step": 50, "mean_face_cosine_similarity": 0.99, "prompt_indices": [0, 1]}
    )
    report["automatic_shortlist"]["excluded"] = [{
        "step": 50,
        "remote_association_status": "unavailable",
        "reason": (
            "remote association is unavailable: no verified catalog checkpoint "
            "receipt for this job and step"
        ),
    }]
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    record, _ = publish_ranked_results(
        client=client,
        repo_id="owner/private",
        repo_type="dataset",
        run_id="run-1",
        job_id="subject-job",
        report_path=report_path,
        sample_root=tmp_path / "samples",
        work_dir=tmp_path / "results",
        sleep=lambda _: None,
        completed_at="2026-09-15T12:00:00+00:00",
    )

    assert record["status"] == "available"
    assert record["exported_candidate_count"] == 3
    assert [item["step"] for item in record["candidates"]] == [200, 100, 300]
    assert "training-results/latest/0001-subject.json" in client.remote
    assert not any("/top-4" in path for path in client.remote)
    # A partial ranking must say so where a human reads it: the published index
    # and the gallery, not only the evaluation report.
    index = json.loads(client.remote["training-results/run-1/0001-subject/index.json"])
    assert index["status"] == "available"
    assert index["excluded_candidates"] == [{
        "step": 50,
        "remote_association_status": "unavailable",
        "reason": (
            "remote association is unavailable: no verified catalog checkpoint "
            "receipt for this job and step"
        ),
    }]
    gallery = client.remote[
        "training-results/run-1/0001-subject/evaluation-gallery.html"
    ].decode()
    assert "Partial coverage" in gallery
    assert "step 50" in gallery
    assert "no verified catalog checkpoint receipt" in gallery


def test_shortlist_may_not_contain_a_checkpoint_without_a_unique_association(tmp_path):
    """The named per-step refusal, not the generic publisher wrapper.

    A fourth eligible checkpoint keeps the candidate count at three, so the
    count check passes and the shortlist reaches the refusal that actually
    protects the export.
    """
    samples = _write_samples(tmp_path / "samples", steps=(100, 200, 300, 400))
    report = _report(samples)
    report["checkpoints"].append({
        "step": 400,
        "final": False,
        "catalog_checkpoint_id": "checkpoint-400",
        "remote_checkpoint": {"commit_id": "r0"},
        "remote_association": {"status": "unique", "reason": None},
        "sample_run_status": "complete",
        "samples": samples[400],
    })
    report["identity_ranking"]["items"].append(
        {"step": 400, "mean_face_cosine_similarity": 0.10, "prompt_indices": [0, 1]}
    )
    client, _ = _catalog_and_client(report)
    report["checkpoints"][0]["remote_association"] = {
        "status": "ambiguous", "reason": "two verified catalog receipts"
    }

    with pytest.raises(
        BackupError, match="shortlisted step 100 has no unique verified remote association"
    ):
        results._shortlist_checkpoints(report)

    report_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(BackupError, match="bounded retries"):
        publish_ranked_results(
            client=client,
            repo_id="owner/private",
            repo_type="dataset",
            run_id="run-1",
            job_id="subject-job",
            report_path=report_path,
            sample_root=tmp_path / "samples",
            work_dir=tmp_path / "results",
            max_attempts=1,
        )
    assert not any(path.startswith("training-results/") for path in client.remote)
