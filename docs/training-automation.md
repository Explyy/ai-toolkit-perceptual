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

`repo_type` accepts `model` or `dataset`. The repository must already exist and report `private: true`; missing credentials, an inaccessible repository, or a public repository stops before training. Queue state and backup state use schema version 1 and atomic replacement. A process found in `running` state after restart returns to `pending`; completed stable job IDs are skipped.

The checkpoint hook runs after the model and available `optimizer.pt`, `config.yaml`, and learnable state are fully saved, but before retention. It writes a local pending manifest before upload, retries, verifies remote sizes at the returned commit and LFS SHA-256 when the Hub exposes it, then updates the catalog with a parent-commit guard. A failed upload, verification, or catalog update stops training; retention also independently refuses to delete bytes missing from verified state. Backup never waits for evaluation and never deletes local or remote data. Optimizer state reflects what AI Toolkit actually saves; it does not promise exact RNG or dataloader replay.

## Catalog and restore

The versioned remote catalog assigns a stable number and folder such as `0001-character-name` to each model. Multiple immutable checkpoints belong to that model. Entries record name, base architecture, trigger, destination kind, checkpoint step/final status, content hashes, sizes, artifact commit, and the human-selected checkpoint. Optimistic `parent_commit` updates prevent silent concurrent catalog overwrites.

Persist a human choice locally and remotely after reviewing `evaluation.json`:

```bash
python -m training_automation select /storage/output/JOB/.automation/evaluation.json 1200 \
  --note 'Chosen after side-by-side review' --repo-type dataset \
  --model character-name
```

On any future generation pod, copy this source package or use the overlay image and run the same CLI. Generation restore downloads selected weights only. It never places optimizer or configuration state in a model folder:

```bash
python -m training_automation restore character-name --repo-type dataset \
  --mode generation --comfy-root /workspace/ComfyUI
```

The default role mapping is `loras → models/loras`, `diffusion_models → models/diffusion_models`, `vae → models/vae`, and `text_encoders → models/text_encoders`. Pass `--root-map roots.json` to supply explicit absolute roots instead. Training resume is separate:

```bash
python -m training_automation restore 1 --repo-type dataset --mode training \
  --training-root /storage/output/character-name
```

Every download goes to a temporary sibling, is checked against catalog size and SHA-256, and is atomically renamed. Matching existing bytes are accepted; conflicting files, unsafe relative paths, public repositories, and ambiguous IDs/names are rejected. Nothing is overwritten automatically.

Index an existing remote weight without moving or replacing it:

```bash
python -m training_automation index-existing existing/path/model.safetensors \
  --repo-type dataset --name character-name --base-arch flux2_klein_9b \
  --destination-kind loras --checkpoint-id imported-original --step 0 --final
```

This explicit command downloads the remote artifact into temporary storage only to compute its SHA-256, then commits catalog metadata pointing to the original path.

## Evaluation limits

Sample association reads AI Toolkit names like `TIMESTAMP__000000100_0.png` and includes the unnumbered final checkpoint at the configured final step. Ranking is available only when every checkpoint has the complete same prompt/index/seed set. The score is a declared heuristic: mean face cosine similarity when available minus the near-black/near-white clipping proxy. It is not evidence of overtraining. Laplacian variance is reported as a sharpness proxy and is not scored.

Identity is `unavailable`, `missing`, or `ambiguous` unless exactly one face exists in every reference and generated image. The optional `InsightFaceCPUBackend` requires user-supplied local weights plus `insightface` and `onnxruntime`; no weights are bundled or downloaded. InsightFace pretrained model licensing can restrict use (the commonly used Buffalo models are described for noncommercial research), so supply weights whose license fits the project. Optional `MediaPipePoseCPUBackend` produces real 2D landmark visibility and image-plane width proxies; perspective, crop, pose, and occlusion limit them.

## Docker and remote checks

`docker/automation/Dockerfile` pins `explyy/ai-toolkit-perceptual` to digest `sha256:e604b849fdb6ea88a900f49b8f55dee30b6355d0d87fd7b307a7ae2d9e764b09`. It copies this checkout's automation package and patches four exact save/retention anchors. It does not replace the base image's older perceptual trainer or UI, and inherits its launch command. The build fails if those anchors drift. Mount persistent storage at `/storage`; keep the base image's existing `/workspace` volume when its UI needs it.

The `Training automation` GitHub Actions workflow runs CPU-only tests on branch pushes and pull requests, then builds the overlay remotely. An explicit `workflow_dispatch` with `publish=true` publishes the unique commit tag to GHCR. The large private base image must be accessible to the runner; image access, GPU training, real Hub upload, optional backend inference, and SimplePod deployment remain credentialed remote checks rather than simulated metrics.
