# Klein training automation

This opt-in pipeline materializes one AI Toolkit job per dataset, runs jobs sequentially, protects every completed checkpoint in an existing private Hugging Face repository before local retention, evaluates generated samples on CPU, and maintains a restore catalog. Existing training and Docker behavior is unchanged when `checkpoint_backup.enabled` is absent or false.

## Configure and run

Copy `config/examples/klein_automation` to persistent storage and edit only explicit placeholders. The trainer template contains 23 deterministic prompt/seed pairs; add, remove, or replace `sample.samples` for a custom count. Each dataset entry supplies its folder, trigger, reference images, catalog name, and generation destination kind.

Set credentials in the pod environment, never in YAML:

```bash
export HF_TOKEN='write-token-for-the-private-repository'
export HF_REPO_ID='owner/existing-private-repository'
python -m training_automation run /storage/config/klein/automation.yaml --dry-run
python -m training_automation run /storage/config/klein/automation.yaml
```

`repo_type` accepts `model` or `dataset`. The repository must already exist and report `private: true`; missing credentials, an inaccessible repository, or a public repository stops before training. The queue config remains schema 1. Queue state is schema 2 and migrates schema-1 entries by preserving completed training. Backup state is schema 2 and binds receipts to `repo_id`, `repo_type`, and `remote_prefix`. A schema-1 backup state with receipts is rejected because its destination cannot be proven; preserve it and choose a new `state_path`. All state writes use atomic replacement.

Stable job IDs include the template bytes, resolved dataset folder, trigger, references, effective `trainer_dataset` overrides, and optional `dataset_revision`. Increment `dataset_revision` when files or captions change without changing their folder. A process found in a training or evaluation `running` phase after restart returns to that phase's pending state. A queue lock refuses a second launcher. Successful training is recorded before evaluation starts, so retrying a failed evaluation never relaunches the GPU job. A failed training or evaluation makes the CLI exit nonzero.

The checkpoint hook runs after the model and available `optimizer.pt`, `config.yaml`, and learnable state are fully saved, but before retention. It writes a local pending manifest before upload, retries, verifies remote sizes at the returned commit and LFS SHA-256 when the Hub exposes it, then updates the catalog with a parent-commit guard. A failed upload, verification, or catalog update stops training; retention also independently refuses to delete bytes missing from verified state. Backup never waits for evaluation and never deletes local or remote data. Optimizer state reflects what AI Toolkit actually saves; it does not promise exact RNG or dataloader replay.

## Catalog and restore

The schema-2 remote catalog assigns a stable number and folder such as `0001-character-name` to each model. Multiple immutable checkpoints belong to that model. Entries record name, exact base model `name_or_path`, base architecture, trigger, destination kind, checkpoint step/final status, content hashes, sizes, artifact commit, and the human-selected checkpoint. A selected record also retains the immutable checkpoint revision and evaluation-report evidence. Optimistic `parent_commit` updates prevent silent concurrent catalog overwrites. Schema-1 catalogs are read compatibly and are upgraded on the next catalog write.

Persist a human choice locally and remotely after reviewing `evaluation.json`:

```bash
python -m training_automation select /storage/output/JOB/.automation/evaluation.json 1200 \
  --note 'Chosen after side-by-side review' --repo-type dataset \
  --model character-name
```

The downloader is always available as the `training_automation` CLI in this repository and in every overlay image built from `docker/automation/Dockerfile`. On a future generation pod, use the stable zero-padded numeric ID to restore its selected checkpoint. Generation restore downloads selected weight artifacts only and puts them under the numeric-name directory. It never places optimizer or configuration state in a generation model folder:

```bash
export HF_TOKEN='read-token-for-the-private-repository'
export HF_REPO_ID='owner/existing-private-repository'
python -m training_automation restore 0001 --repo-type dataset \
  --mode generation --comfy-root /workspace/ComfyUI
```

The default role mapping is `loras → models/loras`, `diffusion_models → models/diffusion_models`, `vae → models/vae`, and `text_encoders → models/text_encoders`. Pass `--root-map roots.json` to supply explicit absolute roots instead. Training resume is separate:

```bash
python -m training_automation restore 0001 --repo-type dataset --mode training \
  --training-root /storage/output/JOB_ID
```

`--training-root` must be the AI Toolkit job save root used by that job. Backup-generated entries restore the checkpoint weight, optimizer, and saved config to their original paths under this root. An imported pre-existing weight is cataloged as weights-only and cannot claim stateful training resume.

Every download goes to a temporary sibling, is checked against catalog size and SHA-256, and is atomically installed with a no-clobber filesystem operation. Matching existing bytes are accepted; conflicting or concurrently created files, unsafe relative paths, public repositories, and ambiguous IDs/names are rejected. Nothing is overwritten automatically.

Index an existing remote weight without moving or replacing it:

```bash
python -m training_automation index-existing existing/path/model.safetensors \
  --repo-type dataset --name character-name --base-arch flux2_klein_9b \
  --base-model black-forest-labs/FLUX.2-klein-base-9B \
  --destination-kind loras --checkpoint-id imported-original --step 0 --final
```

This explicit command downloads the remote artifact into temporary storage only to compute its SHA-256, then commits catalog metadata pointing to the original path.

## Evaluation limits

Evaluate an existing sample directory without launching training:

```bash
python -m training_automation evaluate /storage/config/generated/JOB_ID.yaml \
  /storage/output/JOB_ID --reference /storage/references/character-name/front.jpg
```

Sample association reads AI Toolkit names like `TIMESTAMP__000000100_0.png`, includes the unnumbered final checkpoint at the configured final step, and accepts a verified backup receipt when a retained checkpoint has already been pruned locally. If resumed sampling creates duplicates, evaluation selects the latest complete timestamp run and records how many older or partial samples it discarded. Clipping ranking is available only when every checkpoint has the same complete prompt/index/seed set. It uses negative mean near-black/near-white clipping fraction as a declared image proxy. Face identity is a separate ranking and is available only when every checkpoint has the same nonempty set of valid face prompt indices. Neither ranking is ground truth about quality or overtraining. Laplacian variance is reported as a sharpness proxy and is not scored.

Identity is `unavailable`, `missing`, or `ambiguous` unless exactly one face exists in every reference and generated image. The optional `InsightFaceCPUBackend` requires user-supplied local weights plus `insightface` and `onnxruntime`; no weights are bundled or downloaded. InsightFace pretrained model licensing can restrict use (the commonly used Buffalo models are described for noncommercial research), so supply weights whose license fits the project.

Run the pose backend in a separate CPU evaluation environment with the MediaPipe 0.10 Tasks API:

```bash
python -m pip install 'mediapipe>=0.10,<0.11'
```

Set `landmark_backend` to `training_automation.backends:MediaPipePoseCPUBackend` and `landmark_backend_options.model_path` to an already-present Pose Landmarker `.task` file. The backend never downloads a model. It uses image mode, pixel-corrects normalized coordinates with the actual width and height, and reports `occluded` when required shoulder or hip visibility is below `min_visibility`. The 2D width ratio remains dependent on perspective, crop, and pose. The API and explicit model-path contract follow the [official MediaPipe Tasks documentation](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker); actual model inference is a credentialed remote verification limit.

## Docker and remote checks

`docker/automation/Dockerfile` pins `explyy/ai-toolkit-perceptual` to digest `sha256:e604b849fdb6ea88a900f49b8f55dee30b6355d0d87fd7b307a7ae2d9e764b09`. It copies this checkout's automation package and patches four exact save/retention anchors. It does not replace the base image's older perceptual trainer or UI, and inherits its launch command. The build fails if those anchors drift. Mount persistent storage at `/storage`; keep the base image's existing `/workspace` volume when its UI needs it.

The `Training automation` GitHub Actions workflow runs CPU-only tests on this batch branch and on pull requests. An explicit `workflow_dispatch` runs those tests and the remote overlay build; `publish=true` also publishes the unique lowercase commit tag to GHCR. The pinned Docker Hub base image is public, but a newly published GHCR package may still require registry authentication until its package visibility is configured. The large base image needs substantial hosted-runner disk. Docker build smoke checks compile the patched trainer, load the CLI help, and materialize the shipped example with `--dry-run`. GPU training, real private Hub upload/restore, optional backend model inference, and SimplePod deployment remain credentialed remote checks rather than simulated metrics.
