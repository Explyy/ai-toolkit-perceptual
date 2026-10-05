# Dataset Studio dedicated cloud release

The approved native Studio UI runs over **upstream AI Toolkit 0.13.23**, linux/amd64 digest `sha256:26cb1eb1d8aae154c1c7399004265782124d2f0c1b29cab9b6886345538ed8d4`. Python, CUDA and caption implementations remain those of the pinned image. The repository's older `version.py` (0.13.3) and training automation image are not shipped. The UI reads the actual runtime version from the base; `studio-release.json` adds Studio revision, base digest and source-manifest SHA-256. `studio-source.json` records each shipped source file's SHA-256.

The Node build uses 22.22.3 and `npm ci` with the approved lockfile. The runtime adds nginx and copies Node/npm plus the built UI; no Python dependency upgrades or training overlays. The image compatibility gate imports the actual upstream Qwen3-VL captioner/Transformers classes and constructs a small exact caption configuration without fetching weights. Inference remains a live GPU check.

## Reproducible source and publication

Build `docker/dataset-studio/Dockerfile` from the repository root. Its Dockerfile-specific ignore file denies everything except UI `src`, `cron`, `public`, tests, Prisma schema, named manifests/configs, the cloud startup script, dedicated Docker folder and this release workflow (required by its publication-policy tests). `.env*`, receipts, DBs, node_modules, generated output, application data, `.brain`, `.scratch` and training/model files are excluded.

The root prepares an isolated release tree/commit with this allowlist, workflow and docs, preserving user/concurrent edits and index. After source approval, a push to **`codex/dataset-studio-cloud-live`** triggers checks and publishes `ghcr.io/explyy/ai-toolkit-perceptual:dataset-studio-<full-commit>`. Registry login and image publication both require exact ref `refs/heads/codex/dataset-studio-cloud-live`, including dispatch with `publish=true`; other refs never publish. Workflow dispatch supports a non-publishing build but requires the workflow to be registered on GitHub; the release branch push avoids merging into `main`.

The workflow runs Studio tests (including real disposable Prisma startup tests), build and then full TSC against generated route types. The image repeats source checks in that order and must pass the runtime import/config gate plus **real nginx HTTP smoke**: missing/wrong credentials deny all UI/API/media; valid credentials reach a private backend with Authorization stripped; cloud settings mutation is denied; Host/Origin/HTTPS forwarding survives; ambiguous protocols are rejected. A stopped local Docker daemon or absent nginx makes those image gates unavailable locally, never PASS. Deploy the published **digest**, retain workflow/commit/hash receipts privately and ensure provider registry pull access. Image-build success does not prove GPU or cloud persistence.

```sh
python3 scripts/dataset_studio_cloud.py manifest --toolkit "$PWD" > /tmp/studio-source.json
cd ui
npm run test:studio
npm run build
npx tsc --noEmit --incremental false
```

## Dedicated provider binding and authentication

Create a new Studio template/version. Do not edit template 31400 or install/restart on training nodes 168888/168824. Bind existing **DC1 volume 3489 at `/workspace`**. Independent provider receipt plus server readback proves that binding; startup's volume marker only checks consistency with the supplied identity.

Expose **8675 through HTTPS ingress**. Next 8676 is loopback-only and must never be exposed. Keep the NVIDIA entrypoint and image CMD; do not invoke upstream `/start.sh` or replace source by cloning upstream. Supply private server environment:

| Variable | Value |
| --- | --- |
| `AI_TOOLKIT_AUTH` | Fresh random URL-safe 32–256 character password; no default |
| `HF_TOKEN` | Existing authorized token, never public source/template/browser code |
| `DATASET_STUDIO_VOLUME_ID` | `3489`, independently checked against provider binding |
| `DATASET_STUDIO_MOUNT` | `/workspace` |
| `DATASET_STUDIO_ROOT` | `/workspace/dataset-studio` |
| `AI_TOOLKIT_DB_JOURNAL_MODE` | `DELETE` |

Browser login uses standard HTTP Basic, username **`studio`**, over HTTPS. The gateway covers every path, including legacy public media, and strips browser Authorization before proxying. Only the loopback Next child has `AI_TOOLKIT_AUTH` unset, disabling its conflicting native bearer prompt in this container. The generic app is unchanged. HF tokens remain server-side and the existing settings API masks them. Cloud `/api/settings` mutations return 403, keeping native worker roots/token aligned with startup; change them only through reviewed private configuration.

The gateway preserves external Host/Origin and accepts an absent forwarded protocol (transport scheme) or a single `http`/`https`; multiple/unknown values are rejected. Verify actual upload/save through the provider HTTPS URL because ingress behavior is not proven locally. Do not use direct HTTP with credentials.

## Persistent startup

`scripts/dataset_studio_cloud.py serve` refuses invalid credentials, preview mode, missing mount, symlink escapes, conflicting roots, insufficient reserve (default **24,000,000,000 bytes** free), a second owner, an image runtime DB, mismatched identity/schema or corrupt/incompatible persistent DB. It never falls back to ephemeral storage or silently migrates existing schema.

Only `/workspace/dataset-studio` is initialized:

| Child | Purpose |
| --- | --- |
| `datasets/` | Native originals/paired captions |
| `data/` | Studio metadata, selections, caption scopes and immutable exports |
| `output/` | Dedicated native job logs/output |
| `cache/` | Own HF/model cache |
| `aitk_db.db` | Private native Job/Queue/Settings DB |
| `volume.json`, `schema.json` | Persistent identity guards |
| `.instance.lock` | Ownership held by startup supervisor |

Fresh DB creation pre-creates the file without truncation and runs Prisma once. Each startup also creates a pristine reference database in a temporary directory using the shipped Prisma schema/engine. The existing database is opened read-only; normalized full table/index DDL must match the reference, including primary keys, UNIQUE/indexes, nullability and defaults, and integrity plus stored schema hash must match. Reference generation never pushes or repairs the existing database and leaves source schema unchanged. `/app/ai-toolkit/aitk_db.db` points to the same persistent inode for cron, Python and server Prisma. Startup seeds native root/token settings only after these guards pass. Schema changes require a migration decision; failed first initialization leaves a refused DB for private diagnosis rather than retrying a partial migration. Existing training DBs/datasets/checkpoints are not copied or moved.

`check` uses the same preparation and prints a secret-free release/inode/space receipt; it **can initialize a fresh namespace and seed settings**, so it is an operational action rather than a read-only probe.

CUDA 13/cu130 needs driver 580+. The base is approximately 6.57 GB compressed; allow at least 30 GB ephemeral disk for unpacked runtime. Before the first tiny Qwen3-VL-2B download, verify approximately 5 GB model cache plus the campaign's **24 GB reserve**, and reconcile actual model size/provider costs. Do not download many models or delete user files to make space. Revalidate host offer/prices before any paid POST.

## Required online proof after source approval

1. Reconcile new image/template/instance/driver/HTTPS URL and mounted provider binding; keep private receipts.
2. Check missing/wrong authentication denies UI/API/image/files. Log in through the browser and upload three unique synthetic images into a QA dataset.
3. Save caption/selection/drag and find/replace, reload the real page, and verify SHA-256 plus saved state from mounted server files.
4. Export a tiny immutable version to existing private **`daverave/Personal`**, unique QA prefix, through the existing sync flow. Verify exact committed revision and all readback hashes. Unknown commit outcomes reconcile before retrying.
5. Generate one tiny Qwen3-VL-2B caption job on the new supported GPU; verify exact native identity and nonempty completion/application. Never infer on the two training GPUs.
6. Restart only after the tiny job completes/quiesces, then read back DB inode/original SHA/caption/selection/export/HF revision. Unexpected mid-job restart can leave a stale native running row (existing advisory); verify process death and use native Mark stopped plus explicit reconciliation rather than claiming automatic resume.

Cloud storage, live HF, actual inference and restart proof remain pending until these checks execute on an authorized instance. Local synthetic checks do not prove them.
