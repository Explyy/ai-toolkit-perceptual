from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .backup import BackupConfigurationError, BackupError, HuggingFaceBackupClient, sha256_file
from .catalog import CatalogStore, restore_generation, restore_training
from .evaluation import evaluate_job, persist_selection
from .queue import TrainingQueue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="training_automation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="materialize and run the sequential training queue")
    run.add_argument("config", type=Path)
    run.add_argument("--dry-run", action="store_true")
    evaluate = commands.add_parser("evaluate", help="evaluate existing checkpoint samples without training")
    evaluate.add_argument("job_config", type=Path)
    evaluate.add_argument("output_dir", type=Path)
    evaluate.add_argument("--reference", action="append", type=Path, default=[])
    evaluate.add_argument("--config", type=Path, help="optional YAML evaluation settings")
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
            item.get("status") in {"failed", "evaluation_failed"}
            for item in result.get("jobs", {}).values()
        ):
            return 1
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
