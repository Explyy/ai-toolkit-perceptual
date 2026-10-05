# Dataset Studio in the native AI Toolkit UI

For a dedicated persistent Linux/GPU release, authentication gateway and online verification, see [Dataset Studio cloud deployment](dataset-studio-cloud.md). The cloud image ships this approved UI over pinned upstream 0.13.23 and does not replace existing training templates or datasets.

Open **Datasets → a dataset** (`/datasets/[datasetName]`) after the normal Toolkit UI startup. Image Studio is the default. **Native media view** retains the previous dataset browser and audio/video controls, with a return button. The separate private Dataset Studio Site remains legacy; this integration does not import or change its data.

Use Node20.17+ (existing sqlite3 requirement; validated with22.22.3). From `ui`, use the existing commands:

```sh
npm ci
# Fresh Mac setup: create the native SQLite file if absent, without truncation.
node -e 'const fs=require("node:fs");fs.closeSync(fs.openSync("../aitk_db.db","a"))'
npm run update_db
npm run build
npm run start
```

The normal start command includes the native queue worker. Opening Studio, saving metadata, approving a template or creating a training draft never starts training. The main Generate caption action can queue only its newly managed caption work on a supported Linux/GPU host. Inspect a stopped native job and explicitly use its existing Start control on the configured GPU host.

## Library and local captions

Native uploads accept the picker’s PNG/JPEG/GIF/BMP/WebP, audio/video and paired TXT/JSON formats. Upload limits are **24MB per file**, **100MB for the complete multipart batch**, and **100 files per request**; split larger batches. Existing filenames cannot be overwritten. Image Studio curates PNG/JPEG/WebP; use **Native media view** for GIF/BMP and other native media.

La schermata principale è **Immagini e caption**: carica più PNG/JPEG/WebP e TXT, scegli un unico **Modello locale** dal catalogo nativo e scrivi le istruzioni. Il nuovo dataset usa Qwen3-VL2B catalogato; modello e istruzioni si salvano senza template e ritornano dopo reload. Nessun OpenRouter, API key o classificatore simulato. GIF/BMP, audio/video e controlli precedenti restano in **Avanzate → Native media view**; il picker principale offre soltanto immagini visualizzabili e TXT. I limiti nativi restano24MB/file,100MB/richiesta,100file/richiesta; il main invia ciascun file separatamente, conferma i successi e nomina gli errori senza sovrascrivere collisioni.

Il checkbox accanto a un'immagine significa **inclusa**. Nuove immagini incluse automaticamente; Seleziona/Deseleziona tutte cambia soltanto le immagini visibili. La ricerca non modifica la selezione nascosta. Le non selezionate restano nella stessa griglia e posizione, con immagine attenuata e caption leggibile e modificabile; riclicca immagine o checkbox per riselezionarle. La selezione non elimina i file. Con il mouse scegli **Rettangolo mouse → Seleziona/Deseleziona** e trascina sulle foto: il rettangolo mostra i bersagli e la release salva un solo cambio multiplo, senza alterare altre immagini. Escape, perdita del puntatore, scorrimento o cambi di filtro/dataset annullano senza scritture; input e touch scroll restano nativi. Checkbox e comandi per tutte le visibili funzionano anche da tastiera/touch. Lo slider regola la griglia. Tag, categoria manuale, misura pixel, quote, bucket, test, template, HF e training sono raccolti in un solo pannello **Avanzate**, chiuso inizialmente; nessuna quota o categoria è richiesta per caption.

Modifica le caption sotto le immagini e usa **Salva caption**. Le bozze restano attraverso ricerca e Avanzate, sono protette separatamente sul server e si ripristinano al reload dopo conferma. Se una richiesta fallisce la bozza rimane aperta e l'errore è visibile; non chiudere finché la protezione non è confermata. Scarta bozza torna alla caption salvata. Le revisioni impediscono overwrite da un editor vecchio. I click durante un salvataggio automatico o un aggiornamento attendono la conferma e poi vengono eseguiti, senza doverli ripetere; cambiare dataset annulla gli intenti ancora in attesa del dataset precedente. Una modifica durante un salvataggio resta da salvare: la risposta conferma soltanto il testo inviato, senza cancellare il testo più recente. Modello/istruzioni usano lo stesso principio di snapshot delle richieste.

**Trova e sostituisci**, vicino a Salva caption, cerca testo letterale globale (senza regex o espansione `$`). Scegli le selezionate di tutto il dataset oppure tutte le visibili, comprese le attenuate. L’anteprima conta immagini e occorrenze sul testo corrente, comprese le bozze; cambi a testo, revisioni o scope richiedono una nuova anteprima. Trova vuoto e nessuna corrispondenza non applicano modifiche. **Applica alle bozze** usa la stessa protezione e lo stesso Salva caption manuale: non sostituisce subito i TXT finali. Annulla chiude senza applicare; Scarta bozza conserva la reversibilità esistente.

**Genera caption** prepara e accoda automaticamente un nuovo lavoro nativo per le sole immagini incluse al click, copiando gli originali in una cartella propria senza TXT baseline. È consenso a sostituire le caption correnti dello scope; modifiche manuali successive e bozze impediscono l'applicazione. GPU0 è l'unica coda ammessa: una coda già attiva non viene alterata; una propria coda nuova/inattiva può attivarsi solo senza altri job queued/running/stopping. Una coda in pausa con altri lavori resta ferma, con motivo e Riconcilia esplicito. Training e vecchi draft stopped non partono automaticamente.

Intento, scope, parametri e identità/config DB sono persistenti prima dell'accodamento. La pagina mostra progressi reali dal DB, riconcilia gli intenti propri anche al rientro e applica una sola volta soltanto native completed con step e totale esattamente uguali allo scope, input immutati e tutti i TXT generati non vuoti (massimo64KB ciascuno). Caption e originali vengono verificati prima dell'apply atomico. Timeout/esito sconosciuto riusa la stessa identità; stopped/error/risultati incompleti/conflitto non creano loop di retry o riavvii. Riconcilia recupera lo stesso lavoro; Mantieni le caption attuali ignora risultati falliti/in conflitto. Gli originali e i TXT originali restano invariati.

Mac/preview e host senza GPU0 NVIDIA non avviano modelli: il messaggio esplicito spiega il prerequisito. La normale coda nativa deve essere in esecuzione su un host Linux autorizzato. Il browser può chiudersi mentre il job già accodato prosegue; al rientro risultati completi vengono riconciliati. Non viene eseguita inferenza su GET. Il draft legacy fermo e l'applicazione esplicita restano solo in Avanzate, separati dagli intenti automatici.

Grid size and filters affect only the view. Pixel quality measurement records dHash, sharpness, exposure, clipping and original resolution; it does not claim aesthetic or face/pose model inference. Preview images are bounded thumbnails; source bytes, dimensions and JPEG EXIF orientation are preserved.

## Selection, immutable exports and HF

Choose N, a native bucket tier (512/768/1024), fit and a private Hugging Face repository. Face/body/variety quotas split N evenly with deterministic remainder order. Quality/similarity ordering is deterministic; pins override duplicate exclusion explicitly. Missing categories or conflicting pins produce visible deficits and block export.

Exports create an ordered `training/000001.jpg` and `training/000001.txt` sequence, using selected captions. The native nearest-area/div8 bucket rule permits only incidental rounding enlargement, with EXIF applied. Each finished snapshot has file SHA-256/size and a digest of the exact UTF-8 `manifest.json` bytes. Old finished snapshots are retained and cannot be overwritten. Before native training draft creation every file and the raw manifest are read back again.

**Test samples (optional)** is closed by default. A reference/prompt is for comparison, excluded from training; leave empty for normal training. Test files live outside the `training/` pairs.

Download ZIP includes the exact completed files and manifest. HF sync uses the existing native server HF_TOKEN setting, never returns the secret, and uses a unique `datasets/Training_Studio_<dataset-hash>_<manifest-digest>` destination. No existing source folder is overwritten. Upload and readback progress persist per file. One parent-bound commit is sent; an unknown response/conflict must reconcile the remote raw manifest before any file verification, and never repeats an unknown commit. Verified status requires all bytes' SHA/size at the exact Hub revision. Configure a server token using native Settings; a blank token field preserves the existing token and Remove explicitly clears it.

## Drafts, approvals and native job identity

**Save draft on server** persists the template name/subject/trigger, current local captioner/model/instructions and training JSON separately from approvals. Reload or switching datasets restores the saved draft. Unsaved edits remain drafts and must be saved explicitly.

**Approve template** creates a new immutable named approval: subject/trigger, local caption catalog pair, selection settings, category/tag/pin/membership inventory and reviewed native training config. It validates model/architecture against the authoritative native image model catalog. Apply restores compatible settings and curation only for matching image identity/SHA; it does not overwrite current captions. Duplicate/edit makes a draft, and a subsequent explicit approval makes a new version. Krea2 Raw is an unapproved initial draft, not a claimed campaign recipe approval.

Training draft preparation requires a completed immutable export and an explicit approved template with compatible N/tier/fit. It uses the approval's reviewed recipe and native GPU IDs, replaces dataset/input/output/SQLite paths with this exact version and the configured native roots, and preserves reviewed per-dataset options. The linked record stores exact snapshot and approval IDs. Creation intent is durable before writing the real native Job DB row. Deterministic unique names and config/ref comparison reconcile lost creation responses without a replacement job. The Training panel reads actual DB status, step/target/speed/info and links the native job controls. Existing campaign runs are not inferred as Studio jobs.

## Storage and access

Schema1 authority is `DATA_ROOT/dataset-studio/<SHA256(dataset-name)>/state.json`, with optional schema1 caption preferences, protected drafts and managed automatic intents, alongside immutable `versions/<plan-hash>/`, scoped caption copies and bounded HF operation staging. State is atomic and revision checked; serialized mutations use an exclusive directory lock. Original dataset files remain in the existing DATASETS_FOLDER. No selection/approval/job authority lives in browser localStorage. A leftover lock after server crash fails closed and requires an operator to confirm no live writer before reconciling it; do not delete it while an operation is live.

Dataset names with spaces are supported. New accesses require real containment and refuse symlink components. Existing AI_TOOLKIT_AUTH protection remains; public native media exemptions apply only to GET/HEAD. Mutations check the actual HTTP Host/protocol against browser Origin. Proxy deployments must preserve the correct Host and a single http/https forwarded protocol. Caption extensions cannot address paths or overwrite image/audio/video originals. Old native media reads retain their established behavior.

## Isolated verification

Run Next alone against synthetic dataset files and a disposable native-schema SQLite DB, never `npm run dev/start` (those start the queue worker):

```sh
DATASET_STUDIO_DATA_ROOT=/tmp/studio/data \
DATASET_STUDIO_DATASETS_ROOT=/tmp/studio/datasets \
DATASET_STUDIO_TRAINING_ROOT=/tmp/studio/output \
DATASET_STUDIO_DB_URL=file:/tmp/studio/aitk_db.db \
DATASET_STUDIO_PREVIEW=1 npx next dev --hostname 127.0.0.1 --port 5175
```

Preview disables native job Start and system/GPU monitoring with an explicit unavailable response. It does not fabricate GPU statistics. Normal startup keeps native monitoring unchanged. Deterministic checks: `npm run test:studio`; typecheck: `npx tsc --noEmit --incremental false`; native build: `npm run build`.

Synthetic checks exercise real CPU image decode/resize and storage; they do not prove local model inference, GPU training or live HF integration. Those remain unavailable until an authorized configured Linux/GPU/model/token environment exists. Prisma6.3.1 first creation of a missing SQLite file on this Mac returned a generic Schema engine error (P-60). Identical db push succeeded after pre-creating an empty file; existing databases are never truncated. Mac native system-monitor behavior is a carried advisory (P-59); the isolated preview intentionally bypasses it. No paid operation, model download or campaign change is part of verification.
