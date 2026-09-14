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

Sample association reads AI Toolkit names like `TIMESTAMP__000000100_0.png`, includes the unnumbered final checkpoint at the configured final step, and accepts a verified backup receipt when a retained checkpoint has already been pruned locally. If two immutable remote weight variants share one job and step, the report keeps both candidates but marks remote association ambiguous because sample filenames cannot identify the bytes that produced them. Such metrics remain reviewable, but remote `select` refuses to guess a catalog checkpoint. If resumed sampling creates duplicates, evaluation selects the latest complete timestamp run and records how many older or partial samples it discarded. Clipping ranking is available only when every checkpoint has the same complete prompt/index/seed set. It uses negative mean near-black/near-white clipping fraction as a declared image proxy. Face identity is a separate ranking and is available only when every checkpoint has the same nonempty set of valid face prompt indices. Neither ranking is ground truth about quality or overtraining. Laplacian variance is reported as a sharpness proxy and is not scored.

Identity is `unavailable`, `missing`, or `ambiguous` unless exactly one face exists in every reference and generated image. The optional `InsightFaceCPUBackend` requires local weights plus `insightface` and an ONNX Runtime provider. General installations must supply those assets. The dedicated parallel overlay described below bakes the versioned Buffalo L release during its remote image build. InsightFace pretrained model licensing can restrict use (the commonly used Buffalo models are described for noncommercial research), so use weights whose license fits the project.

Run the pose backend in a separate CPU evaluation environment with the MediaPipe 0.10 Tasks API:

```bash
python -m pip install 'mediapipe>=0.10,<0.11'
```

Set `landmark_backend` to `training_automation.backends:MediaPipePoseCPUBackend` and `landmark_backend_options.model_path` to an already-present Pose Landmarker `.task` file. The backend never downloads a model. It uses image mode, pixel-corrects normalized coordinates with the actual width and height, and reports `occluded` when required shoulder or hip visibility is below `min_visibility`. The 2D width ratio remains dependent on perspective, crop, and pose. The API and explicit model-path contract follow the [official MediaPipe Tasks documentation](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker); actual model inference is a credentialed remote verification limit.

## Docker and remote checks

`docker/automation/Dockerfile` pins `explyy/ai-toolkit-perceptual` to digest `sha256:e604b849fdb6ea88a900f49b8f55dee30b6355d0d87fd7b307a7ae2d9e764b09`. It copies this checkout's automation package and patches four exact save/retention anchors. It does not replace the base image's older perceptual trainer or UI, and inherits its launch command. The build fails if those anchors drift. Mount persistent storage at `/storage`; keep the base image's existing `/workspace` volume when its UI needs it.

The `Training automation` GitHub Actions workflow runs CPU-only tests on this batch branch and on pull requests. An explicit `workflow_dispatch` runs those tests and the remote overlay build; `publish=true` also publishes the unique lowercase commit tag to GHCR. The pinned Docker Hub base image is public, but a newly published GHCR package may still require registry authentication until its package visibility is configured. The large base image needs substantial hosted-runner disk. Docker build smoke checks compile the patched trainer, load the CLI help, and materialize the shipped example with `--dry-run`. GPU training, real private Hub upload/restore, optional backend model inference, and SimplePod deployment remain credentialed remote checks rather than simulated metrics.

## Parallel private deployment

The opt-in `/run-parallel-training` command runs one explicitly assigned shard and replaces the inherited GUI command for that container. The overlay does not set a new `CMD` or `ENTRYPOINT`, so normal launches continue to run the original GUI. The selected queue recipe is [trainer-subject-likeness-masked-klein-9b.yaml](../config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml), copied from `subject_likeness_masked_flux2_klein9b` at `origin/perceptual-port@ab5f146e9a30764a40bf29b0881d3a6dd7186c6f`. It retains 1200 steps, batch size 4, AdamW8bit at `5e-5`, LoKr 32, relative weight noise `0.0125`, resolution repeats `[16,4,1]`, masked depth loss `0.005`, and subject weights `background=0`, `clothing=1`, `body=1`. Its only content change is 23 generic, clothed identity prompts sampled every 100 steps with seed 42.

Keep the real deployment manifest private. Start from [parallel-manifest.example.json](../config/examples/klein_automation/parallel-manifest.example.json), replace every placeholder, upload it to the private dataset repository, and pin `HF_MANIFEST_REVISION` to the commit containing that manifest. `dataset_revision` separately pins the source dataset tree. Every staged file carries its source path, size, SHA-256, and safe relative destination. The pod stages only entries matching its shard and refuses missing files, hash failures, traversal, symlink traversal, and conflicting local bytes.

The required container environment is:

```text
HF_REPO_ID=<private repository>
HF_REPO_TYPE=dataset
HF_MANIFEST_PATH=<private path to manifest JSON>
HF_MANIFEST_REVISION=<exact 40-hex commit>
HF_TOKEN=<private read/write token>
TRAINING_RUN_ID=<manifest run_id>
TRAINING_SHARD_ID=a
SIMPLEPOD_API_TOKEN=<subaccount API token>
```

`TRAINING_STORAGE_ROOT` defaults to `/storage`, `TRAINING_REPO_ROOT` defaults to `/app/ai-toolkit`, and `TRAINING_RECIPE_PATH` defaults to the provenance-locked recipe above. Generated config, queue state, staging state, output, and archive state are isolated under `<storage>/{automation,output,datasets}/<run_id>/<shard_id>`. The generated dedicated-worker config disables the GUI logger and removes its SQLite path because this command does not start the GUI; normal GUI launches still inherit the original behavior. The manifest must assign exactly three datasets to each shard. Before paid work, bootstrap checks `storage.minimum_free_bytes`, verifies all expected numeric catalog IDs were reserved with their canonical Hub base model metadata, and stages the base and depth snapshots at exact revisions. The base-model `allow_patterns` intentionally exclude the duplicate root monolithic weight; snapshot targets share `/storage/models` and use a lock plus source marker so two shards can reuse them.

The image contains InsightFace Buffalo L from the versioned upstream v0.7 release and records the downloaded archive and individual file SHA-256 values in `/opt/training-automation-models/SOURCES.json` during the remote build. The parallel manifest configures `InsightFaceCPUBackend` against that directory and requires available comparable identity results before completion. These pretrained weights are restricted to noncommercial research unless separately licensed. Do not configure the pose backend for this deployment unless a compatible `.task` model and the documented MediaPipe dependency are also installed; an absent backend remains `unavailable` and never counts as identity success.

References listed in this manifest are training-set images. Reports and completion evidence label them `reference_provenance: training-set`; their cosine scores measure likeness to training examples and do not establish held-out quality or generalization.

### Exact self-binding and completion

Create each SimplePod instance once through management and never retry an uncertain creation request. Set an exact notes marker such as `training-run:<run_id>;shard:a`, then upload one private binding based on [parallel-binding.example.json](../config/examples/klein_automation/parallel-binding.example.json). Bootstrap polls only the configured binding path for a bounded time. It calls `GET /instances/{id}` and requires exact equality for numeric `id`, `hashId`, and `notes`; it never finds a pod by name or list search. This follows the [official SimplePod API](https://api.simplepod.ai/docs_ai.html), which documents `X-AUTH-TOKEN`, `GET /instances/{id}`, and `DELETE /instances/{id}`.

Successful training alone does not trigger deletion. Every assigned job must have completed training and evaluation, all checkpoint receipts including a final checkpoint must be verified and cataloged, the face backend and training-set reference identity must be available, and sample images, generated config, backup state, evaluation report, queue state, and staging state must be uploaded and verified. Missing or ambiguous faces in generated samples are an honest quality outcome: the report may record identity ranking as unavailable and still pass the operational completion gate, without inventing a score. A second commit publishes `training-runs/<run>/<shard>/completion.json` with the evidence commit and hashes. Bootstrap then re-fetches and re-verifies the same instance identity before issuing one `DELETE /instances/{id}`.

Any binding, disk, staging, model download, training, backup, evaluation, identity, archive, verification, or delete error writes `bootstrap-state.json` with `status: failed` on persistent storage and exits nonzero. Failures before the completion gate never call delete. The SimplePod template has exit-delete disabled, so a failed container can continue billing even after the command exits. An external management monitor must inspect both instances, alert on failure, and delete them manually; the in-container guard cannot guarantee billing termination when its API call or network fails.
