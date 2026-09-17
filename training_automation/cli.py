from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .backup import BackupConfigurationError, BackupError, HuggingFaceBackupClient, sha256_file
from .catalog import CatalogStore, restore_generation, restore_training
from .evaluation import evaluate_job, persist_selection
from .dataset_storage import upload_dataset_folder
from .gallery import render_gallery
from .queue import TrainingQueue, dataset_content_fingerprint
from .results import (
    publish_archived_run_results,
    publish_refreshed_report,
    publish_ranked_results,
    refresh_archived_run_reports,
    report_job_id,
)
from .sync import sync_latest_loras, sync_ranked_loras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="training_automation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="materialize and run the sequential training queue")
    run.add_argument("config", type=Path)
    run.add_argument("--dry-run", action="store_true")
    unified = commands.add_parser(
        "unified", help="sync, discover, and run the opt-in GUI-adjacent workflow"
    )
    unified.add_argument("config", type=Path)
    unified.add_argument("--dry-run", action="store_true")
    prepare_unified = commands.add_parser(
        "prepare-unified", help="validate and initialize the opt-in GUI storage bridge"
    )
    prepare_unified.add_argument("config", type=Path)
    commands.add_parser(
        "parallel-run",
        help="run the private pinned parallel shard bootstrap and success-only self-delete",
    )
    fingerprint = commands.add_parser(
        "dataset-fingerprint",
        help="print the dataset content hash an extension must declare in extend_from",
    )
    fingerprint.add_argument("folder", type=Path)
    evaluate = commands.add_parser("evaluate", help="evaluate existing checkpoint samples without training")
    evaluate.add_argument("job_config", type=Path)
    evaluate.add_argument("output_dir", type=Path)
    evaluate.add_argument("--reference", action="append", type=Path, default=[])
    evaluate.add_argument("--config", type=Path, help="optional YAML evaluation settings")
    gallery = commands.add_parser(
        "gallery", help="render one portable HTML gallery from existing evaluation evidence"
    )
    gallery.add_argument("report", type=Path)
    gallery.add_argument("--samples-root", type=Path, required=True)
    gallery.add_argument("--output", type=Path, required=True)
    gallery.add_argument("--title")
    publish_results = commands.add_parser(
        "publish-results",
        help="publish deterministic top-checkpoint grids and verified weight copies",
    )
    _connection_args(publish_results)
    publish_results.add_argument("report", type=Path)
    publish_results.add_argument("--samples-root", type=Path, required=True)
    publish_results.add_argument("--run-id", required=True)
    publish_results.add_argument("--job-id")
    publish_results.add_argument("--work-dir", type=Path, required=True)
    publish_results.add_argument("--results-prefix", default="training-results")
    publish_results.add_argument("--completed-at")
    publish_run_results = commands.add_parser(
        "publish-run-results",
        help="publish every job from a reconstructed immutable evidence archive",
    )
    _connection_args(publish_run_results)
    publish_run_results.add_argument("archive_root", type=Path)
    publish_run_results.add_argument("--run-id", required=True)
    publish_run_results.add_argument("--work-dir", type=Path, required=True)
    publish_run_results.add_argument("--results-prefix", default="training-results")
    refresh_report = commands.add_parser(
        "refresh-report", help="publish additive paginated reports from immutable local evidence"
    )
    _connection_args(refresh_report)
    refresh_report.add_argument("report", type=Path)
    refresh_report.add_argument("--samples-root", type=Path, required=True)
    refresh_report.add_argument("--run-id", required=True)
    refresh_report.add_argument("--work-dir", type=Path, required=True)
    refresh_report.add_argument("--results-prefix", default="training-results")
    refresh_run = commands.add_parser(
        "refresh-archived-run",
        help="download verified immutable Hub evidence and publish additive reports",
    )
    _connection_args(refresh_run)
    refresh_run.add_argument("--run-id", required=True)
    refresh_run.add_argument("--source-revision")
    refresh_run.add_argument("--job-id", action="append", default=[])
    refresh_run.add_argument("--work-dir", type=Path, required=True)
    refresh_run.add_argument("--archive-prefix", default="training-archives")
    refresh_run.add_argument("--results-prefix", default="training-results")
    sync = commands.add_parser(
        "sync-loras", help="sync verified automatic LoRA ranks from one immutable Hub revision"
    )
    _connection_args(sync)
    sync.add_argument("--source-revision", required=True)
    sync.add_argument("--run-id")
    sync.add_argument("--loras-root", type=Path, required=True)
    sync.add_argument("--work-dir", type=Path, required=True)
    sync.add_argument("--rank", type=int, action="append", default=[])
    sync.add_argument("--model-id", type=int, action="append", default=[])
    sync.add_argument("--results-prefix", default="training-results")
    sync.add_argument("--dry-run", action="store_true")
    upload_dataset = commands.add_parser(
        "upload-dataset",
        help="upload one validated image/caption folder to private HF dataset storage",
    )
    _connection_args(upload_dataset)
    upload_dataset.set_defaults(repo_type="dataset")
    upload_dataset.add_argument("folder", type=Path)
    upload_dataset.add_argument("--remote-folder")
    upload_dataset.add_argument("--name")
    upload_dataset.add_argument("--trigger-word", default="Owhx")
    upload_dataset.add_argument("--work-dir", type=Path, required=True)
    upload_dataset.set_defaults(remote_prefix="datasets")
    select = commands.add_parser("select", help="persist a human checkpoint choice")
    select.add_argument("report", type=Path)
    select.add_argument("step", type=int)
    select.add_argument("--note", required=True)
    select.add_argument("--repo-id")
    select.add_argument("--repo-id-env", default="HF_REPO_ID")
    select.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    select.add_argument("--remote-prefix", default="training-backups")
    select.add_argument("--model")
    restore = commands.add_parser("restore", help="verified catalog restore for generation or training")
    _connection_args(restore)
    restore.add_argument("identifier")
    restore.add_argument("--checkpoint-id")
    restore.add_argument("--mode", choices=("generation", "training"), default="generation")
    restore.add_argument("--comfy-root", type=Path)
    restore.add_argument("--root-map", type=Path, help="JSON map of destination kind to local root")
    restore.add_argument("--training-root", type=Path)
    index = commands.add_parser("index-existing", help="index an existing remote weight without moving it")
    _connection_args(index)
    index.add_argument("remote_path")
    index.add_argument("--name", required=True)
    index.add_argument("--base-arch", required=True)
    index.add_argument("--base-model", required=True)
    index.add_argument("--trigger-word")
    index.add_argument("--destination-kind", choices=("loras", "diffusion_models", "vae", "text_encoders"), required=True)
    index.add_argument("--checkpoint-id", required=True)
    index.add_argument("--step", type=int, required=True)
    index.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = TrainingQueue(args.config).run(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        if not args.dry_run and any(
            item.get("status") in {
                "failed", "evidence_failed", "evaluation_failed", "publish_failed",
            }
            for item in result.get("jobs", {}).values()
        ):
            return 1
    elif args.command == "unified":
        from .unified import run_unified_workflow

        print(json.dumps(run_unified_workflow(args.config, dry_run=args.dry_run), indent=2))
    elif args.command == "prepare-unified":
        from .unified import prepare_unified_environment

        print(json.dumps(prepare_unified_environment(args.config), indent=2))
    elif args.command == "parallel-run":
        from .bootstrap import run_parallel_bootstrap

        print(json.dumps(run_parallel_bootstrap(), indent=2))
    elif args.command == "dataset-fingerprint":
        print(dataset_content_fingerprint(args.folder))
    elif args.command == "evaluate":
        import yaml

        config = {}
        if args.config:
            config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        print(evaluate_job(
            job_config_path=args.job_config,
            output_dir=args.output_dir,
            reference_images=args.reference,
            config=config,
        ))
    elif args.command == "gallery":
        print(render_gallery(
            args.report,
            sample_root=args.samples_root,
            output_path=args.output,
            title=args.title,
        ))
    elif args.command == "publish-results":
        store = _store(args)
        record, evidence_files = publish_ranked_results(
            client=store.client,
            repo_id=store.repo_id,
            repo_type=store.repo_type,
            run_id=args.run_id,
            job_id=args.job_id or report_job_id(args.report),
            report_path=args.report,
            sample_root=args.samples_root,
            work_dir=args.work_dir,
            catalog_prefix=args.remote_prefix,
            results_prefix=args.results_prefix,
            completed_at=args.completed_at,
        )
        print(json.dumps({
            **record,
            "local_evidence": [str(path) for path, _ in evidence_files],
        }, indent=2))
    elif args.command == "publish-run-results":
        store = _store(args)
        print(json.dumps(publish_archived_run_results(
            client=store.client,
            repo_id=store.repo_id,
            repo_type=store.repo_type,
            run_id=args.run_id,
            archive_root=args.archive_root,
            work_dir=args.work_dir,
            catalog_prefix=args.remote_prefix,
            results_prefix=args.results_prefix,
        ), indent=2))
    elif args.command == "refresh-report":
        store = _store(args)
        print(json.dumps(publish_refreshed_report(
            client=store.client,
            repo_id=store.repo_id,
            repo_type=store.repo_type,
            run_id=args.run_id,
            report_path=args.report,
            sample_root=args.samples_root,
            work_dir=args.work_dir,
            catalog_prefix=args.remote_prefix,
            results_prefix=args.results_prefix,
        ), indent=2))
    elif args.command == "refresh-archived-run":
        store = _store(args)
        print(json.dumps(refresh_archived_run_reports(
            client=store.client,
            repo_id=store.repo_id,
            repo_type=store.repo_type,
            run_id=args.run_id,
            source_revision=args.source_revision,
            job_ids=tuple(args.job_id),
            work_dir=args.work_dir,
            archive_prefix=args.archive_prefix,
            catalog_prefix=args.remote_prefix,
            results_prefix=args.results_prefix,
        ), indent=2))
    elif args.command == "sync-loras":
        store = _store(args)
        kwargs = dict(
            client=store.client,
            repo_id=store.repo_id,
            repo_type=store.repo_type,
            source_revision=args.source_revision,
            loras_root=args.loras_root,
            work_dir=args.work_dir,
            ranks=tuple(args.rank or [1]),
            model_ids=tuple(args.model_id),
            catalog_prefix=args.remote_prefix,
            results_prefix=args.results_prefix,
            dry_run=args.dry_run,
        )
        if args.run_id:
            result = sync_ranked_loras(run_id=args.run_id, **kwargs)
        else:
            result = sync_latest_loras(**kwargs)
        print(json.dumps(result, indent=2))
    elif args.command == "upload-dataset":
        repo_id = args.repo_id or os.environ.get(args.repo_id_env, "")
        token = os.environ.get(args.token_env)
        if not repo_id or not token:
            raise BackupConfigurationError(
                "private repository id and environment credential are required"
            )
        print(json.dumps(upload_dataset_folder(
            client=HuggingFaceBackupClient(token), repo_id=repo_id,
            repo_type=args.repo_type, folder=args.folder,
            remote_folder_name=args.remote_folder, catalog_name=args.name,
            trigger_word=args.trigger_word, remote_prefix=args.remote_prefix,
            work_dir=args.work_dir,
        ), indent=2))
    elif args.command == "select":
        if args.repo_id or os.environ.get(args.repo_id_env):
            if not args.model:
                raise BackupConfigurationError("--model is required when publishing selection")
            report = json.loads(args.report.read_text(encoding="utf-8"))
            checkpoint = next(item for item in report["checkpoints"] if int(item["step"]) == args.step)
            if checkpoint.get("remote_association", {}).get("status") == "ambiguous":
                raise BackupError(
                    "cannot publish selection: multiple remote checkpoint variants share this sample step"
                )
            store = _store(args)
            checkpoint_id = checkpoint.get("catalog_checkpoint_id")
            if not checkpoint_id:
                checkpoint_id = f"step-{args.step:09d}" + ("-final" if checkpoint.get("final") else "")
            store.select(
                args.model,
                checkpoint_id,
                evidence={
                    "report_sha256": sha256_file(args.report),
                    "selected_step": args.step,
                    "human_note": args.note,
                    "checkpoint_evidence": checkpoint,
                },
            )
        selection_path = persist_selection(args.report, args.step, args.note)
        print(selection_path)
    elif args.command == "restore":
        store = _store(args)
        catalog, _ = store.read()
        if args.mode == "generation":
            if args.root_map:
                roots = {key: Path(value) for key, value in json.loads(args.root_map.read_text(encoding="utf-8")).items()}
            elif args.comfy_root:
                roots = {
                    "loras": args.comfy_root / "models" / "loras",
                    "diffusion_models": args.comfy_root / "models" / "diffusion_models",
                    "vae": args.comfy_root / "models" / "vae",
                    "text_encoders": args.comfy_root / "models" / "text_encoders",
                }
            else:
                raise BackupConfigurationError("generation restore requires --comfy-root or --root-map")
            print(json.dumps(restore_generation(
                client=store.client, repo_id=store.repo_id, repo_type=store.repo_type,
                catalog=catalog, identifier=args.identifier, roots=roots,
                checkpoint_id=args.checkpoint_id,
            ), indent=2))
        else:
            if not args.training_root:
                raise BackupConfigurationError("training restore requires --training-root")
            print(json.dumps(restore_training(
                client=store.client, repo_id=store.repo_id, repo_type=store.repo_type,
                catalog=catalog, identifier=args.identifier, target_root=args.training_root,
                checkpoint_id=args.checkpoint_id,
            ), indent=2))
    else:
        store = _store(args)
        print(store.import_existing(
            metadata={
                "name": args.name, "base_arch": args.base_arch, "base_model": args.base_model,
                "trigger_word": args.trigger_word, "destination_kind": args.destination_kind,
            },
            remote_path=args.remote_path, checkpoint_id=args.checkpoint_id,
            step=args.step, final=args.final,
        ))
    return 0


def _connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-id")
    parser.add_argument("--repo-id-env", default="HF_REPO_ID")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--remote-prefix", default="training-backups")
    parser.add_argument("--token-env", default="HF_TOKEN")


def _store(args: argparse.Namespace) -> CatalogStore:
    repo_id = args.repo_id or os.environ.get(args.repo_id_env, "")
    token_env = getattr(args, "token_env", "HF_TOKEN")
    token = os.environ.get(token_env)
    if not repo_id or not token:
        raise BackupConfigurationError("private repository id and environment credential are required")
    client = HuggingFaceBackupClient(token)
    if not client.repo_is_private(repo_id, args.repo_type):
        raise BackupConfigurationError("refusing catalog operation on a non-private repository")
    return CatalogStore(
        client=client, repo_id=repo_id, repo_type=args.repo_type,
        catalog_path=f"{args.remote_prefix.strip('/')}/catalog.json",
        work_dir=Path(os.environ.get("TRAINING_AUTOMATION_ROOT", "/storage/automation")),
    )
