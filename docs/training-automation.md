# Klein training automation

This opt-in pipeline materializes one AI Toolkit job per dataset, runs jobs sequentially, protects every completed checkpoint in an existing private Hugging Face repository before local retention, evaluates generated samples on CPU, and maintains a restore catalog. Existing training and Docker behavior is unchanged when `checkpoint_backup.enabled` is absent or false.

## Configure and run

Copy `config/examples/klein_automation` to persistent storage and edit only explicit placeholders. Both masked Klein templates contain six deterministic prompt/seed pairs — two full figures in an articulated scene, two medium shots and two close-ups on the face — and save and sample every 100 steps; add, remove, or replace `sample.samples` for a custom count. Each dataset entry supplies its folder, trigger, reference images, catalog name, and generation destination kind.

Set credentials in the pod environment, never in YAML:

```bash
export HF_TOKEN='write-token-for-the-private-repository'
export HF_REPO_ID='owner/existing-private-repository'
python -m training_automation run /storage/config/klein/automation.yaml --dry-run
python -m training_automation run /storage/config/klein/automation.yaml
```

`repo_type` accepts `model` or `dataset`. The repository must already exist and report `private: true`; missing credentials, an inaccessible repository, or a public repository stops before training. The queue config remains schema 1. Queue state is schema 2 and migrates schema-1 entries by preserving completed training. Backup state is schema 2 and binds receipts to `repo_id`, `repo_type`, and `remote_prefix`. A schema-1 backup state with receipts is rejected because its destination cannot be proven; preserve it and choose a new `state_path`. All state writes use atomic replacement.

Stable job IDs include the template bytes, resolved dataset folder, trigger, references, effective `trainer_dataset` overrides, and optional `dataset_revision`. Increment `dataset_revision` when files or captions change without changing their folder. A process found in a training or evaluation `running` phase after restart returns to that phase's pending state. A queue lock refuses a second launcher. Successful training is recorded before evaluation starts, so retrying a failed evaluation never relaunches the GPU job. A failed training, checkpoint-evidence, evaluation or publication step makes the CLI exit nonzero.

The checkpoint hook runs after the model and available `optimizer.pt`, `config.yaml`, and learnable state are fully saved, but before retention. It writes a local pending manifest before upload, retries, verifies remote sizes at the returned commit and LFS SHA-256 when the Hub exposes it, then updates the catalog with a parent-commit guard. A failed upload, verification, or catalog update stops training; retention also independently refuses to delete bytes missing from verified state. Backup never waits for evaluation and never deletes local or remote data. Optimizer state reflects what AI Toolkit actually saves; it does not promise exact RNG or dataloader replay.

### Evidence for every checkpoint

A checkpoint whose images live only on the pod disk is worth nothing once the
machine is released: on 2026-09-17 LaProfumosa finished at 749 steps and lost
every sample that way, keeping only weights, optimizer and manifest. Each
checkpoint folder therefore also receives its own images and its own evaluation
record, next to `weights/`, `resume/` and `manifest.json`:

```text
training-backups/models/<numeric-name>/checkpoints/<checkpoint-id>/
├── manifest.json
├── weights/ ...
├── resume/ ...
├── evaluation.json
└── samples/<original sample filenames>
```

`evaluation.json` records the cohort, the step, the sample run status, every
sample with its prompt, seed, case id, identity, image and pose metrics, and one
checkpoint score: mean face cosine over the images of that checkpoint with
exactly one valid face, plus mean clipping proxy. It also carries the remote
path, size and SHA-256 of every uploaded image, so the checkpoint can be
reviewed and compared without regenerating anything.

The trainer saves a checkpoint and only afterwards samples that same step, so
the images of step N do not exist when step N is protected. Evidence is
published in a second pass: every checkpoint backup flushes the checkpoints
whose sample run has completed in the meantime, and the queue runs one strict
pass after the training process exits, which covers the last checkpoint. The
worst case is therefore one cadence of exposure on the newest checkpoint, not a
whole shard. A failure during training is recorded in the backup state and
retried by the next checkpoint; it never stops a paid run. The strict pass does
stop the job, because a checkpoint without evidence is not a delivered
checkpoint, and the shard completion gate refuses to self-delete while any
receipt written under this evidence contract lacks verified evidence. Every
receipt this version creates carries that contract marker; a receipt written
before the contract existed cannot be completed retroactively, so it is
tolerated and listed by step in the completion evidence as uncovered, never
counted as covered. The backup state schema stays at 2 on purpose: the marker is
an additive optional field, while a schema bump would make the state guard
refuse every existing state that already holds receipts, including one being
written by a pod that is training right now. Nothing already published is deleted
or overwritten, and the remote associations the evaluation code already verifies
stay unchanged: evidence adds files, it does not add catalog checkpoints.

Set `checkpoint_evidence.during_training: false` in the queue configuration to
keep the identity and pose models out of the live trainer process. The strict
pass after training still produces the same complete evidence; only the
protection against losing the machine mid-run is traded away.

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

## Extending an existing training

A model that only needs a longer run — more epochs on the same photographs with
the same recipe — continues from its last verified catalog checkpoint instead of
restarting from zero. Declare it on the dataset entry of the queue configuration
or of the private deployment manifest:

```yaml
extend_from:
  model: 'Exact catalog name or numeric id'
  dataset_fingerprint: '<64 hex content hash of the images and captions>'
  base_training_steps: 749
  checkpoint_id: 'optional exact catalog checkpoint id'
  reason: 'optional note'
```

Obtain the content hash from the dataset folder that will actually be trained:

```bash
python -m training_automation dataset-fingerprint /storage/datasets/RUN/SHARD/subject
```

The extension is a separate, explicit run identity: `extend_from` enters the job
id derivation, so the extended run has its own job id, its own checkpoint ids and
its own remote folders. It appends checkpoints to the same catalog model and
never rewrites or invalidates an existing entry. Jobs without `extend_from` keep
exactly the job ids they already had.

Before any paid step the queue refuses an incompatible base with a specific
error, instead of silently retraining or silently continuing:

- the recomputed dataset content hash must equal the declared one;
- the new duration must run past the base step;
- the chosen checkpoint must sit exactly at `base_training_steps` and must carry
  trainer resume state, so a weights-only imported entry is rejected;
- the catalog model's trigger word and base architecture must match the job;
- the base run's own `config.yaml`, downloaded and hash-verified from the
  immutable checkpoint revision, must match the new configuration on every
  training-relevant field. Only local paths, the automation block, the duration
  and the sampling and saving cadence are exempt; the sampling prompts are
  exempt too, because they never enter the trained weights. Any other difference
  — learning rate, network, weight noise, dataset repeats, masking, preservation
  — is reported by name with both values and refuses the extension, unless the
  continuation declares a refinement phase that names that field (see below).

Because `bound_norm` is part of the weight-noise configuration, extending one of
the six models trained before it was introduced requires pinning
`TRAINING_RECIPE_PATH` to a recipe whose weight-noise block matches that base
run. Adding the new regularization to an existing run is a recipe change, not a
duration change, and is refused by design.

The verified base weight and `optimizer.pt` are then installed into the extended
job's save root, with the weight renamed to the extended job's name so the
trainer actually finds and resumes it; the base `config.yaml` is kept in staging
as evidence and is not installed. The step recorded in the weight metadata makes
the trainer continue from it, and the schedule runs to the new duration. If no
base weight can be installed under the extended job's name, or if the base
checkpoint has no `optimizer.pt`, the extension is refused rather than started
from zero. Local retention never deletes the restored base file, because those
bytes are not in this job's own verified backup state.

### Declared refinement phase

A continuation that only needs more of the same is a duration change. A second,
short pass that deliberately concentrates capacity elsewhere is a **refinement
phase**: it is declared, it names the fields it changes, and it may change
nothing else.

```yaml
extend_from:
  model: 'Exact catalog name or numeric id'
  dataset_fingerprint: '<64 hex content hash of the images and captions>'
  base_training_steps: 1200
  phase:
    name: low-noise-1024
    changes:
      - train.content_or_style
      - train.lr
      - datasets.*.num_repeats
```

`changes` may only name these five fields, and `*` stands for exactly one path
segment, so `datasets.*.num_repeats` covers every configured dataset and nothing
else:

- `train.timestep_type`
- `train.content_or_style`
- `train.lr`
- `datasets.*.num_repeats`
- `datasets.*.resolution`

A field outside that set, a field that differs without being declared, and a
declaration that is not a `name` plus a non-empty `changes` list are all refused
before the first paid step, with the differing fields reported by name and with
both values. Without `phase` the continuation keeps exactly the duration-only
behavior described above: the declaration enters the job id derivation like the
rest of `extend_from`, so a refined run is again a separate run identity and
existing job ids do not move.

Why a refinement phase exists, measured on the current recipe: it trains 76% of
its exposures at 512px (`num_repeats: [16, 4, 1]` over `[512, 768, 1024]`) with
the default `content_or_style: balanced`, which spends a uniform 20% of the
budget in each noise band. Fine facial detail is learned in the low-noise band,
but at 512px a face inside a full-figure crop has too few pixels to carry it, so
the two settings compound. `content_or_style: style` switches the trainer to the
cubic sampling of `jobs/process/BaseSDTrainProcess.py`, which places 58.5% of the
steps below timestep 200; `timestep_type: linear` selects the timestep schedule
and does not disable that branch. Combined with the repeats reversed toward
1024px, that is the refinement pass. The `[16, 4, 1]` bias is documented in the
upstream quickstarts as a speed choice, not a quality one.

`config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b-refinement.yaml`
is that recipe: recipe v2 with `content_or_style: style`, `num_repeats:
[1, 4, 16]`, `lr: 0.00002` and `steps: 1500`, which is 300 steps past a
1200-step base. Every other training-relevant field is byte-identical to v2, so
the queue accepts it against a v2-trained base with the declaration above. Use it
as the `trainer_yaml` of the refinement queue configuration and validate it with
its own validator, which checks the same cohort, regularization and pinned-image
support as the full run and, for the phase fields, that the pass reuses the same
exposure budget in ascending order, so every higher resolution is weighted more
than the one below it, with a positive learning rate below the base one:

```bash
python -m training_automation.recipe --refinement \
  /app/ai-toolkit/config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b-refinement.yaml \
  /app/ai-toolkit
```

Because the archive gate requires the configured sample set at every scheduled
step past the base checkpoint and at the final step, and that verdict only
arrives when the paid run is already over, a declared phase also has its schedule
checked before the first GPU hour: the recipe must configure at least one
evaluation sample, `sample.sample_every` and `save.save_every` must be the same
cadence and one the archive gate itself supports (100 or 200), and the final step
must sit on that cadence. A refinement whose
completion evidence could never be produced is refused instead of started.

The lineage stays auditable in three places: the declaration is in the queue
state under `extend_from`, `<output>/<job>/.automation/extension.json` records
the declared phase together with the deviations actually observed against the
base configuration, and every checkpoint this run publishes carries
`refinement_phase` in its `evaluation.json` on the Hub, so a reader of the
catalog can tell a refined model from a plain continuation and see with which
fields. The per-checkpoint evidence contract stays at 1: the field is additive
and optional, nothing written by an earlier version becomes unreadable, and an
older image reading a newer queue configuration refuses the unknown
`extend_from.phase` by name instead of training something it cannot describe.

A phase is declared the same way in a queue configuration and in a private
deployment manifest: the manifest validates `extend_from` against the field set
the queue itself owns and validates the declaration with the queue's own
validator, so the two cannot drift apart on what a phase may declare, and the
whole mapping is copied into the generated queue configuration.

One manifest rule follows from the phase. A dataset's
`training_accounting.resolution_repeats` is normally pinned to `[16, 4, 1]`
exactly; a dataset whose phase declares `datasets.*.num_repeats` may instead
carry those same repeats in a different order, and nothing else. The budget
itself does not move — `[1, 4, 16]` and `[16, 4, 1]` both sum to 21 — so the
`original_image_exposures` identity is untouched, and a dataset that declares no
phase, or a phase that does not name `datasets.*.num_repeats`, is still refused
with the exact list.

## Evaluation limits

Evaluate an existing sample directory without launching training:

```bash
python -m training_automation evaluate /storage/config/generated/JOB_ID.yaml \
  /storage/output/JOB_ID --reference /storage/references/character-name/front.jpg
```

Sample association reads AI Toolkit names like `TIMESTAMP__000000100_0.png`, includes the unnumbered final checkpoint at the configured final step, and accepts a verified backup receipt when a retained checkpoint has already been pruned locally. If two immutable remote weight variants share one job and step, the report keeps both candidates but marks remote association ambiguous because sample filenames cannot identify the bytes that produced them. Such metrics remain reviewable, but remote `select` refuses to guess a catalog checkpoint. If resumed sampling creates duplicates, evaluation selects the latest complete timestamp run and records how many older or partial samples it discarded. Clipping ranking is available only when every checkpoint has the same complete prompt/index/seed set. Face identity compares the intersection of prompt indices with one valid face at every complete checkpoint; at least three shared prompts are required. The automatic shortlist is available only when every sample run is complete, the configured prompt/seed evidence is comparable, and every checkpoint has one unique verified remote association. It reports at most three evidence candidates ordered by mean face cosine descending, mean clipping ascending, then step ascending, and otherwise records an explicit unavailable reason. It never selects a checkpoint. These fixed-model CPU results are deterministic for the pinned code, inputs, and runtime, but are not guaranteed bitwise-identical across hardware and are not ground truth about quality or overtraining.

Identity is `unavailable`, `missing`, or `ambiguous` unless exactly one face exists in a generated image. References remain strict by default. An explicit `reference_identity_filter` may average only references containing exactly one valid face while recording every excluded filename and its missing, multiple, or invalid reason. The configured minimum must be at least three valid images and 50% coverage for this run; insufficient coverage blocks completion. The optional `InsightFaceCPUBackend` requires local weights plus `insightface` and an ONNX Runtime provider. General installations must supply those assets. The dedicated parallel overlay described below bakes the versioned Buffalo L release during its remote image build. InsightFace pretrained model licensing can restrict use (the commonly used Buffalo models are described for noncommercial research), so use weights whose license fits the project.

The parallel image pins `yolo11n-pose.pt` from Ultralytics assets v8.4.0 at SHA-256 `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`. `UltralyticsPoseCPUBackend` converts decoded RGB images to contiguous BGR arrays at the Ultralytics NumPy boundary and uses the image's existing dependency on CPU with fixed confidence, IoU, image size, disabled augmentation, and at most two detections. It requires exactly one person and reports missing, ambiguous, occluded, or degenerate outcomes without scores. Available results contain pixel-corrected 2D shoulder/hip, limb/torso ratios, and joint angles. They are viewpoint-sensitive proxies, not anatomical ground truth or proof that a text pose was followed. Ultralytics software and models require AGPL-3.0 compliance or an applicable Enterprise license.

Pinned Ultralytics 8.4.61 assigns an empty `CUDA_VISIBLE_DEVICES` value when CPU prediction selects its device. Evaluation snapshots whether this variable was absent or explicitly set, restores that exact state after backend preflight and every evaluation attempt, and also encloses the queue call so an exceptional or substituted backend cannot leak the mutation. The next training subprocess therefore inherits the queue's original authorized GPU visibility. This restoration does not turn a real CUDA failure into success; the fresh trainer still performs its own hardware checks.

For a generic installation, the older MediaPipe seam remains available in a separate CPU evaluation environment:

```bash
python -m pip install 'mediapipe>=0.10,<0.11'
```

Set `landmark_backend` to `training_automation.backends:MediaPipePoseCPUBackend` and `landmark_backend_options.model_path` to an already-present Pose Landmarker `.task` file. The backend never downloads a model. It uses image mode, pixel-corrects normalized coordinates with the actual width and height, and reports `occluded` when required shoulder or hip visibility is below `min_visibility`. The 2D width ratio remains dependent on perspective, crop, and pose. The API and explicit model-path contract follow the [official MediaPipe Tasks documentation](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker); actual model inference is a credentialed remote verification limit.

## Portable galleries and automatic result folders

Render a self-contained gallery from an existing report without running inference. `--samples-root` must be one flat directory containing the original sample files; report paths are remapped by basename and duplicate, missing, escaping, unreadable, or unsupported images are rejected. The HTML embeds both compact previews and original image bytes, uses no remote scripts or assets, and preserves each image's proportions:

```bash
python -m training_automation gallery \
  /storage/archive/jobs/SUBJECT_JOB/evaluation.json \
  --samples-root /storage/archive/jobs/SUBJECT_JOB/samples \
  --output /storage/gallery/SUBJECT_JOB.html
```

Publication happens at the end of **each job**, not at the finalization of the
shard: as soon as a job's evaluation completes, its ranked grids, top three and
latest pointer are published and recorded in the queue state together with the
local evidence manifest `published-evidence.json`. Interrupting a queue halfway
no longer costs the publication of the jobs that already finished. The shard
finalization reuses that exact publication for the archive instead of repeating
it, and only publishes a job that has none. Enable it with a `results` block
carrying `enabled`, `run_id` and `work_dir`; the parallel bootstrap writes that
block automatically, while the unified workflow keeps publishing its single job
itself.

The automatic publisher uses the report's existing shortlist only after rechecking complete comparable prompt/seed sets, available aggregate rankings, and unique verified checkpoint associations. It never makes a human selection or changes `selected_checkpoint_id`. For each available candidate, up to three, it downloads the checkpoint weight at the exact catalog revision, verifies the catalog size and SHA-256, and uploads a verified copy with its raster contact sheet and machine-readable evidence:

```text
training-results/<run-id>/<numeric-name>/
├── index.json
├── overview.png
├── evaluation-gallery.html
├── top-1/
│   ├── contact-sheet.png
│   ├── report-index.json
│   ├── pages/page-01.png ...
│   ├── selection.json
│   └── weights/<generation-weight>
├── top-2/...
└── top-3/...
```

The contact sheet includes all evaluated images for that checkpoint in prompt-index and seed order. Every image remains visible even when face or pose detection is missing or ambiguous. Each `selection.json` records the exact catalog checkpoint ID, source hashes/revision, automatic rank, metrics, and an executable `training_automation restore <numeric-id> --checkpoint-id <checkpoint-id>` command for the canonical ComfyUI path. This rank command bypasses the optional human-selected catalog default explicitly; it does not overwrite that default. Raw face cosine is not a percentage or probability; clipping, sharpness, and image-plane pose values are evidence proxies rather than quality or anatomical scores. If the shortlist is unavailable or contains fewer than three candidates, `index.json` records the reason and actual count without inventing folders or scores.

Publish one reconstructed archived job on a cloud host that holds its samples and has private Hub credentials:

```bash
python -m training_automation publish-results \
  /storage/archive/jobs/SUBJECT_JOB/evaluation.json \
  --samples-root /storage/archive/jobs/SUBJECT_JOB/samples \
  --run-id RUN_ID --work-dir /storage/results/SUBJECT_JOB \
  --repo-type dataset
```

To backfill every direct job in a reconstructed archive deterministically, use one command. The archive must use `jobs/<job-id>/evaluation.json` and `jobs/<job-id>/samples/`, and each report's stable job ID must match its directory:

```bash
python -m training_automation publish-run-results /storage/archive \
  --run-id RUN_ID --work-dir /storage/results \
  --repo-type dataset
```

Each successful publication with an available automatic top candidate can update `training-results/latest/<numeric-name>.json` in the same parent-guarded commit when it carries a verified completion time newer than the current pointer. Equal or older completions never replace it. Historical backfills with unknown chronology publish their reports but do not promote themselves; explicit historical seeds cover those runs. An unavailable shortlist leaves the prior usable pointer unchanged. The pointer names an explicit completed run, job, model, report hash, and completion time; no UUID, export wall time, or lexical path ordering is interpreted as chronology. The per-checkpoint pages contain at most six large uncropped images at 1920 pixels wide, while `contact-sheet.png` is the first concise page and `report-index.json` lists every page. `overview.png` compares the same prompt across the available top three with a raw cosine axis from -1 to 1.

Both publishers require `HF_TOKEN` and `HF_REPO_ID`, refuse public repositories, use parent-commit guards, verify every uploaded byte at the returned immutable revision, and leave original checkpoint paths and catalog history intact. Run them where the private samples and weights already exist; they do not download models to the workstation. Refresh old immutable evidence without regenerating samples or transferring LoRA weights:

```bash
python -m training_automation refresh-report \
  /storage/archive/jobs/SUBJECT_JOB/evaluation.json \
  --samples-root /storage/archive/jobs/SUBJECT_JOB/samples \
  --run-id RUN_ID --work-dir /storage/report-refresh/SUBJECT_JOB \
  --repo-type dataset
```

Refreshed files are additive under `training-results/<run>/<numeric-name>/reports-v2/<evaluation-hash>/`. Their manifest states that source evidence is unchanged and no model weights were transferred.

A completed remote archive can be refreshed without reconstructing deleted-pod paths manually. This command discovers completion manifests beneath the exact run, pins the current Hub revision, downloads only their declared evaluation JSON and sample images, verifies every declared size and SHA-256 against the archive's immutable evidence revision, and then invokes the same additive publisher:

```bash
python -m training_automation refresh-archived-run \
  --run-id RUN_ID --job-id OPTIONAL_JOB_ID \
  --work-dir /storage/report-refresh --repo-type dataset
```

Both the historical multi-job shard format and the new one-job completion format use `training-archive-completion-v1` and are supported. Missing completion manifests, undeclared samples, duplicate jobs, unsafe paths, or hash mismatches stop the refresh. Other archive layouts remain unsupported rather than inferred. No LoRA checkpoint is downloaded by this command.

## Unified GUI and automatic workflow

The overlay provides one additional opt-in command, `/run-unified-training`. It first runs `prepare-unified` to validate the shared roots and establish the explicit GUI setting, starts `training_automation.unified_supervisor`, and then gives the foreground process to the inherited `/start.sh`. The Dockerfile still declares no `CMD` or `ENTRYPOINT`, so an ordinary launch remains the original GUI. Use this command only on the existing Alternative Training template with its established `/storage` volume; do not create a second application or image family.

Copy [unified-workflow.yaml](../config/examples/klein_automation/unified-workflow.yaml) to `/storage/config/klein-unified.yaml`. The recommended environment is:

```text
HF_TOKEN=<private repository read/write token>
HF_REPO_ID=<owner/private repository>
TRAINING_UNIFIED_CONFIG=/storage/config/klein-unified.yaml
DATASETS_FOLDER=<the exact folder used by the GUI>
COMFYUI_ROOT=<the existing ComfyUI root, if loras_root is not explicit>
TRAINING_WORKER_ID=0
TRAINING_WORKER_COUNT=1
```

Set the template command to `/run-unified-training`. `TRAINING_GUI_START` may name another existing executable when the inherited image does not use `/start.sh`; it is never parsed as a shell command. The image build verifies that the pinned base still provides executable `/start.sh`. `TRAINING_WORKER_ID` and `TRAINING_WORKER_COUNT` take priority over the shared YAML values, so two instances of this same template can use IDs 0 and 1 with count 2. Each worker writes its actual state and log beneath `/storage/automation/unified/worker-<id>/`; explicit `TRAINING_UNIFIED_STATUS` and `TRAINING_UNIFIED_LOG` values may override those paths. An error after GUI launch is recorded as `held`; the GUI stays reachable and the supervisor polls again. A storage-bridge error stops the opt-in launch before the GUI can cache a divergent root. Persistent GUI sessions are never automatically deleted.

The GUI reads `DATASETS_FOLDER` from its SQLite `Settings` table at `/app/ai-toolkit/aitk_db.db`; setting a process environment variable alone does not configure the GUI. `prepare-unified` reads that table directly with Python's standard SQLite library. With `initialize_storage_directories: true`, first boot may create only the explicitly configured dataset, LoRA, and automation cache paths beneath an already-existing `storage_root`. With `initialize_gui_dataset_root: true`, it may create the exact native `Settings(id,key,value)` table on a clean first boot and insert that configured dataset root only when the setting is absent and the native GUI dataset folder contains no unseen files. It never overwrites a nonempty setting, follows a symlink, or moves data. YAML, environment, optional legacy JSON bridge, and SQLite values must all agree or startup holds. `HF_TOKEN` may come from the environment or the existing GUI Settings row and is never printed. The LoRA target is explicit through `loras_root`, `LORAS_ROOT`, `comfyui_root`, or `COMFYUI_ROOT`. When none is set, discovery is bounded to direct case-insensitive `ComfyUI` children of `/storage`; zero or multiple matching `models/loras` folders is actionable. No provider network volume is required: the private Hub is authoritative and these local directories are replaceable caches.

### Caricare un nuovo dataset su Hugging Face

Ogni dataset remoto sta in `datasets/<nome cartella>/` nel repository HF privato e contiene immagini supportate con una didascalia `.txt` omonima per ciascuna immagine. Il nome può contenere spazi, underscore, maiuscole e accenti. Il controller assegna automaticamente il successivo ID stabile e installa la copia locale come `0007-nome-cognome`; rinominare una cartella con gli stessi byte riusa lo stesso ID. Non serve modificare manifest o YAML per ogni persona.

Dal notebook remoto, imposta `UPLOAD_FOLDER`, `UPLOAD_REMOTE_FOLDER`, `UPLOAD_NAME` e usa `ACTION="upload"`. La stessa operazione è disponibile nella CLI dell'immagine:

```bash
python -m training_automation upload-dataset "/storage/uploads/Nome Cognome" \
  --repo-type dataset --remote-folder "Nome Cognome" --name "Nome Cognome" \
  --work-dir /storage/automation/unified/dataset-upload
```

Questo comando controlla prima la cartella locale e pubblica payload e manifest conclusivo in un unico commit con parent guard. Per un upload dal browser HF, carica invece tutti i file direttamente sotto `datasets/<nome cartella>/` e completa l'upload prima di avviare o attendere il prossimo ciclo. Il controller confronta due osservazioni complete separate da `discovery.quiet_seconds`; questa finestra rileva un contenuto cambiato durante l'intervallo, ma non può sapere se una sessione browser interrotta verrà ripresa più tardi. Una coppia immagine/didascalia mancante resta quindi in stato `remote-held` e non blocca gli altri dataset validi.

`ACTION="index"` nel notebook esegue lo stesso ciclo in modalità dry-run: osserva, verifica, registra e scarica le copie canoniche senza avviare il trainer. `ACTION="discover"` ha lo stesso comportamento; `ACTION="run"` prosegue con i soli dataset nuovi. Dopo la perdita del disco, il ciclo riscarica dal commit HF immutabile e consulta il ledger remoto prima di decidere: un'identità già completata non viene riaddestrata.

At each poll the controller performs these steps in order:

1. Observe every direct folder beneath the configured private-Hub `datasets` prefix at an immutable 40-hex revision. Each folder is validated independently, so one partial or malformed upload is reported and held without hiding valid datasets.
2. Sync every rank configured in `sync.ranks`, scan the current local cache, and reconcile the private completion ledger. The shipped configuration uses `[1,2,3]`; an explicit historical result seed is used only for a model that has no current latest pointer.
3. After the quiet interval, observe the remote dataset tree again. Stability compares per-file paths, sizes, and immutable object identities, rather than requiring the repository HEAD to stay equal while another worker uploads checkpoints. Verified payload bytes are staged outside the visible dataset root, checked against LFS SHA-256 or Git object identity, and atomically installed. Disk capacity is checked before downloads. A known remote source with changed bytes or a conflicting local canonical folder is held and cannot dispatch its stale local copy.
4. Scan the installed canonical folders and record complete image/caption filename, size, and SHA-256 identity in the private-Hub workflow ledger. A completed identity is never queued again after local disk replacement. An incomplete job remains held unless `retry_incomplete: true` is explicitly configured.
5. Reserve the same numeric ID in the dataset and model catalogs with one parent-guarded commit, persist a deterministic worker assignment, and order pending jobs by normalized model name unless the notebook supplies an explicit order. Canonical catalog metadata keeps `base_model: black-forest-labs/FLUX.2-klein-base-9B` even though the native trainer reads a local weight path. `worker.id` and `worker.count` permit two preassigned workers to divide any number of folders without a fixed six/three limit. Local download installation is serialized and every worker uses isolated mutable staging; Hub parent guards coordinate shared metadata.
6. For each pending dataset in order, persist a durable identity and phase before invoking the trainer. Materialize a one-dataset queue, complete training and evaluation, publish the automatic top three, sync all configured ranks, archive all evidence, and mark that dataset completed before moving to the next. A later job or export failure cannot erase a prior completion. Restarting after an export interruption reuses the same queue state and does not retrain completed work. If the private ledger says a dataset was dispatched but its local execution state disappeared with a disk replacement, automation holds rather than silently starting another paid run.

Loader accounting reads every image's dimensions, calls the same `toolkit/buckets.py` geometry with divisibility 16, freezes scale 1 with square/random crop disabled, applies resolution repeats `[16,4,1]`, and counts each unpadded partial bucket batch at batch size 4. Seven complete loader epochs produce exactly **147 exposures for every original photo**, the nearest complete-epoch schedule to the requested target of 150; 150 is not divisible by the 21 exposures contributed by one complete 16+4+1 epoch. Persisted steps equal exact bucket batches per epoch times seven. The six fresh dataset schedules are 1253, 1057, 2772, 749, 1855, and 1267 steps for AlexMucci, ElisaBrendani, Freya, LaProfumosa, Losciuto, and Mady respectively. These are new from-zero runs under a new run ID: historical runs and weights remain unchanged and are never used as resume inputs. Remote tests compare the accounting implementation directly to the standalone bucket function shipped in the pinned trainer checkout, and the Docker build validates the trainer anchors before use.

Existing completed datasets can be seeded without downloading source photos. Publish a private immutable JSON file and pin it in `discovery.legacy_completed_index`:

```json
{
  "schema_version": 1,
  "datasets": {
    "EXACT_EXISTING_FOLDER": {
      "fingerprint": "64_HEX_IMAGE_AND_CAPTION_CONTENT_HASH",
      "catalog_name": "Existing catalog name",
      "catalog_id": 1,
      "trigger_word": "Owhx",
      "run_id": "EXPLICIT_COMPLETED_RUN_ID",
      "source_dataset_revision": "40_HEX_REVISION"
    }
  }
}
```

The separate version-1 dataset catalog at `training-automation/dataset-catalog.json` maps that immutable content fingerprint to the same model ID, name, trigger and frozen canonical folder. Each source alias records its remote folder, immutable revision, and every payload file's relative path, remote path, size and true SHA-256. The initial six entries point to their existing `datasets/Training_Def_Owhx_*` bytes; migration adds the canonical cache folder and a ledger alias without copying remote payloads, renumbering models, or replaying training. Do not use a Git blob OID as a SHA-256: small non-LFS files are downloaded from the pinned revision and hashed before catalog publication.

Historical top-one results with no trustworthy completion timestamp are seeded separately through `sync.legacy_runs`, with an explicit run ID and model IDs. A configured revision is used exactly; when omitted, startup pins the same immutable current catalog revision used for that sync operation. Future publications use per-model latest pointers backed by persisted completion chronology. Neither path sorts UUIDs or guesses which historical run is newest.

The unified recipe [trainer-subject-likeness-masked-klein-9b-v2.yaml](../config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b-v2.yaml) preserves the masked Subject Likeness, Klein 9B, LoKr, and weight-noise settings and enables Differential Output Preservation from step zero with multiplier `1` and class `woman`. The pinned trainer replaces the dataset trigger with this class for the preservation target; it still requires a nonempty trigger, a network, and `train_text_encoder: false`. DOP adds computation and is an explicit regularization choice, not a guarantee of improved quality. Both masked recipes now enable `bound_norm` in `weight_noise` and share one fixed `subject-likeness-core-v3` evaluation cohort of six distinct seeds and cases: two full figures in an articulated scene (contrapposto in a sunlit courtyard, mid-stride walking in a station concourse), two medium shots (a three-quarter turn at a window, a bench in a workshop) and two close-ups on the face (frontal and three-quarter). Prompts open with `single adult subject [trigger]`, state visible limb geometry, vary wardrobe/location/light, and do not infer gender from a folder name. Case ID, category, seed, and cohort version are recorded in evaluation evidence. Pose values remain 2D image-plane proxies; they do not prove prompt adherence or anatomical correctness. The earlier legacy, articulated-v2 and core-v3 cohorts must never be compared as if they were the same evidence: the core cohort trades breadth for a shorter, denser cadence, and with six prompts a run whose generated images rarely contain exactly one valid face can leave the identity ranking and the automatic shortlist honestly unavailable.

Differential Output Preservation and cached text embeddings are mutually exclusive: the trainer refuses the combination in `validate_configs` (`Cannot use differential output preservation with caching text embeddings`), and the refusal happens at job construction, before the first step, so no code-level review can observe it. Both masked Klein recipes therefore ship `cache_text_embeddings: false`, `recipe.py` pins that value with the DOP fields so the pair cannot drift apart, and the patched masked Quickstart preset disables the cache whenever it enables preservation. `recipe.py` also requires the pinned trainer sources to mention `bound_norm`, so an image that does not support that weight-noise field fails the Docker build instead of the first paid step. The cost is real: the Qwen3-8B text encoder runs every step instead of serving cached embeddings. This was found on the first real GPU dispatch of image `d8bd669` on 2026-09-17, after the change had passed independent Standards and root Spec review as source.

The executable [Klein_Unified_Training.ipynb](../notebooks/Klein_Unified_Training.ipynb) is an optional interface to the same Python modules. Its automation source checkout is pinned to a concrete commit and every cell is clean. Training always invokes the validated native `/app/ai-toolkit/run.py`; the temporary checkout supplies the pinned automation package and recipe only. Edit the explicit repository, run ID, dataset, upload, ComfyUI, LoRA, worker, model/rank, historical-model, archive-job, and ordering fields. `ACTION="status"` is the safe Run All default: it reads durable state without cloning code, installing packages, or asking for a token. `upload` validates and atomically publishes one complete dataset folder; `index` and `discover` run the remote discovery/download path without training; `run` uses that same path and then trains only eligible work. `sync` uses latest pointers plus the explicit `RUN_ID` fallback for the original six models. `refresh` reconstructs the selected run directly from immutable private-Hub completion evidence, so deleted pod paths are unnecessary. Every action prints returned state; no cell simulates success or chooses an agent. The token comes from `HF_TOKEN` or a masked `getpass` prompt and is never stored in the notebook.

## Docker and remote checks

`docker/automation/Dockerfile` pins `explyy/ai-toolkit-perceptual` to digest `sha256:e604b849fdb6ea88a900f49b8f55dee30b6355d0d87fd7b307a7ae2d9e764b09`. It copies this checkout's automation package and patches four exact save/retention anchors. It does not replace the base image's older perceptual trainer or UI, and inherits its launch command. The build fails if those anchors drift. Mount persistent storage at `/storage`; keep the base image's existing `/workspace` volume when its UI needs it.

The `Training automation` GitHub Actions workflow runs CPU-only tests on this batch branch and on pull requests. An explicit `workflow_dispatch` normally runs those tests and the remote overlay build; `publish=true` also publishes the unique lowercase commit tag to GHCR. For a source snapshot already reviewed under an explicit no-tests instruction, `skip_automated_tests=true` skips only the automated suite and still permits the Docker build; its default remains false. The pinned Docker Hub base image is public, but a newly published GHCR package may still require registry authentication until its package visibility is configured. The large base image needs substantial hosted-runner disk. Docker build checks patch the masked UI preset without changing the unmasked preset, compile the patched trainer, validate DOP support and both recipes, load the CLI help, materialize the shipped example with `--dry-run`, verify both pinned model hashes, instantiate both evaluation backends, and run one synthetic CPU pose prediction. GPU training, real private Hub upload/restore, and SimplePod deployment remain credentialed remote checks.

## Parallel private deployment

The opt-in `/run-parallel-training` command runs one explicitly assigned shard and replaces the inherited GUI command for that container. Set the SimplePod template `argOptions` field to exactly `/run-parallel-training`, as one argument without shell metacharacters. The public provider API no longer honours that field: `POST` and `PUT` on `/instances/templates` answer `200` and store `null`, so a template maintained through the API silently starts the inherited GUI instead of the worker, which is what emptied the 2026-09-15 launch. Set `argOptions` through the provider UI, or dispatch the worker from the per-instance `startScript`, which the API does persist. The same original template can be switched temporarily for the two disposable launches and restored to `/run-unified-training` afterward; no second template or execution-mode wrapper is required. It writes the complete child output to both the console and `<storage>/automation/<run>/<shard>/worker.log`. A failed child is held in the same container for diagnosis instead of exiting into a provider restart loop; manual cleanup remains required. The overlay does not set a new `CMD` or `ENTRYPOINT`, so normal launches continue to run the original GUI. The selected queue recipe is [trainer-subject-likeness-masked-klein-9b.yaml](../config/examples/klein_automation/trainer-subject-likeness-masked-klein-9b.yaml), copied from `subject_likeness_masked_flux2_klein9b` at `origin/perceptual-port@ab5f146e9a30764a40bf29b0881d3a6dd7186c6f`. It retains batch size 4, AdamW8bit at `5e-5`, LoKr 32, relative weight noise `0.0125` with `bound_norm: true`, resolution repeats `[16,4,1]`, masked depth loss `0.005`, and subject weights `background=0`, `clothing=1`, `body=1`; DOP is enabled from the beginning with multiplier `1` and class `woman`. Fresh manifests supply each dataset's seven-epoch step count and accounting explicitly. The template's 1200-step value remains only the backward-compatible legacy fallback. Its six seed-pinned `subject-likeness-core-v3` prompts are saved and sampled every 100 steps.

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

`TRAINING_STORAGE_ROOT` defaults to `/storage`, `TRAINING_REPO_ROOT` defaults to `/app/ai-toolkit`, and `TRAINING_RECIPE_PATH` defaults to the provenance-locked recipe above. Set optional `TRAINING_LORAS_ROOT` to the existing shared ComfyUI LoRA directory when the disposable worker must install all three ranked results before shutdown. Generated config, queue state, staging state, output, and archive state are isolated under `<storage>/{automation,output,datasets}/<run_id>/<shard_id>`. The generated dedicated-worker config disables the GUI logger and removes its SQLite path because this command does not start the GUI; normal GUI launches still inherit the original behavior. The manifest must assign exactly three datasets to each shard. Before paid work, bootstrap checks `storage.minimum_free_bytes`, verifies all expected numeric catalog IDs were reserved with their canonical Hub base model metadata, and stages every model source at its exact revision.

The native Klein loader needs `flux-2-klein-base-9b.safetensors`; Diffusers transformer shards are not interchangeable with that file. The example pins that native file with `artifact_path`, stages Qwen3-8B as a text-encoder directory, stages `ae.safetensors` as an explicit VAE artifact, and keeps Depth Anything as a directory. Bootstrap injects those resolved paths into `model.name_or_path`, `model.te_name_or_path`, `model.vae_path`, and `depth_consistency.model_id`, while the catalog retains the canonical Hub base-model identity. `artifact_path` is a safe source-relative file selector supported for `base_model` and `vae`. Older two-role manifests containing only `base_model` and `depth_model` remain valid, but their staged base directory must contain the required native Klein file before the trainer can launch. Snapshot targets share `/storage/models` and use a lock plus source marker so shards can reuse them. Completion evidence records the repository, revision, patterns, local target, and selected artifact for every configured source.

The image contains InsightFace Buffalo L from the versioned upstream v0.7 release and the pinned Ultralytics pose weight. Their source URLs, sizes, hashes, and model paths are recorded in `/opt/training-automation-models/SOURCES.json`. The full remote image build runs `python docker/automation/install_evaluation_models.py`. To add only the pose weight to an existing reviewed pod without re-downloading Buffalo L, transfer that same script and run `python install_evaluation_models.py --pose-only`; it merges the pose record into the existing source manifest and does not install or replace Python packages. The parallel manifest configures both backends and bootstrap loads them before training. Buffalo pretrained weights are restricted to noncommercial research unless separately licensed.

The private manifest may give a dataset `training_steps` plus immutable `training_accounting`. Both fields must appear together. Accounting records the exact source-image count, actual un-padded loader batches per epoch, loader epochs, resolution repeats `[16,4,1]`, batch size `4`, source-image exposures, and `partial_bucket_batches: un-padded`; bootstrap verifies `steps = batches × epochs` and `exposures = 21 × epochs`. These values are injected after job identity derivation, so existing job IDs remain stable, and a changed duration is accepted only while the recorded training phase is pending. The run-level `checkpoint_policy` in the example saves and samples every 100 steps and retains the last five local step saves. Bootstrap accepts a 100-step or a 200-step policy and materializes matching save and sample cadences. A manifest without this policy uses the shipped 100-step recipe default. A legacy dataset without the seven-epoch schedule keeps the template's 1200-step duration.

References listed in this manifest are training-set images. Reports and completion evidence label them `reference_provenance: training-set`; their cosine scores measure likeness to training examples and do not establish held-out quality or generalization.

### Exact self-binding and completion

Create each SimplePod instance once through management and never retry an uncertain creation request. Set an exact notes marker such as `training-run:<run_id>;shard:a`, then upload one private binding based on [parallel-binding.example.json](../config/examples/klein_automation/parallel-binding.example.json). Bootstrap polls only the configured binding path for a bounded time. It calls `GET /instances/{id}` and requires exact equality for numeric `id`, `hashId`, and `notes`; it never finds a pod by name or list search. This follows the [official SimplePod API](https://api.simplepod.ai/docs_ai.html), which documents `X-AUTH-TOKEN`, `GET /instances/{id}`, and `DELETE /instances/{id}`.

Successful training alone does not trigger deletion. Every assigned job must have completed training and evaluation, every checkpoint scheduled by the job's configured 100- or 200-step cadence plus its final checkpoint must have the complete configured prompt set uniquely associated with a verified catalog receipt, every receipt written under the current evidence contract must carry its own verified per-checkpoint evidence, and all archived bytes must pass size plus SHA-256 verification. An extended job is scheduled only for the steps past its base checkpoint, so the gate requires exactly those; that reduction is authorized only by a completed extension record whose base step agrees with the declaration, and an ordinary job still requires its whole schedule. The completion evidence records the resumed step and the exact scheduled steps it checked. Before archiving, bootstrap reuses the gallery and deterministic top-candidate folders each job already published and records their immutable revision in completion evidence; it publishes only a job that has none. It persists one completion timestamp before publishing so the new result can safely advance the per-model latest pointer. When `TRAINING_LORAS_ROOT` names an existing shared LoRA directory, bootstrap also downloads and verifies ranks 1, 2, and 3 from the just-published immutable revision before archive or deletion; conflicts or an incomplete sync hold the instance, and the sync receipt is archived. Without that environment variable, legacy behavior remains unchanged. Archive metadata is fetched from the same immutable Hub revision in batches of at most 100 paths so large sample sets do not exceed the Hub request limit; any failed batch or missing entry fails the complete verification. When Hub metadata lacks a hash, verification downloads the exact immutable revision and hashes those bytes. Configured face and pose backends must load and run, and training-set reference identity must meet its coverage gate. Missing or ambiguous people or faces in generated images are honest evaluated quality outcomes and may make a ranking unavailable without fabricating a score. A second commit publishes `training-runs/<run>/<shard>/completion.json` with the evidence commit and hashes. Bootstrap then re-fetches and re-verifies the same instance identity before issuing one `DELETE /instances/{id}`.

Any binding, disk, staging, model download, training, backup, evaluation, identity, result export, archive, verification, or delete error writes `bootstrap-state.json` with `status: failed` on persistent storage. The dedicated supervisor automatically retries at most three times only when the persisted queue already proves that every assigned job completed both training and evaluation. It records each attempt and backoff in `worker-recovery-state.json`; a supervisor restart consumes rather than resets that budget. This archive-only path rechecks the immutable manifest, exact run/shard paths, Hub destination, model source markers, completion evidence, and bound instance; it bypasses dataset/model staging, backend preflight, and `TrainingQueue.run`. It then republishes deterministic result artifacts if needed, archives them, and requests deletion only after all verification succeeds. Incomplete queues and all other prior failures remain held for diagnosis rather than restarting training. A persisted prior delete request is treated as uncertain and is never reset or issued again automatically.

Failures before the completion gate never call delete. The `/run-parallel-training` supervisor records the actual traceback and exit status in `worker.log`, then holds the container rather than allowing an automatic restart to erase the console context. The SimplePod template has exit-delete disabled, so a failed held container continues billing. An external management monitor must inspect both instances, alert on failure, and delete them manually; the in-container guard cannot guarantee billing termination when its API call or network fails.
