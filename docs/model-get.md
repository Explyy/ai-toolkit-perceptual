# Scaricare un modello dal terminale

`model-get` scarica un solo file modello da Hugging Face o Civitai, mantiene il nome originale e verifica i byte prima di pubblicarli nella cartella scelta. Accetta anche i link ufficiali `civitai.red` e `www.civitai.red`, risolvendoli tramite l'API canonica di Civitai.

```bash
cd /workspace/ComfyUI/models/loras
/workspace/bin/model-get "https://huggingface.co/owner/repo/blob/main/model.safetensors"
```

La cartella predefinita è quella corrente. Per indicarne una o cambiare il nome locale:

```bash
/workspace/bin/model-get "URL" \
  --dir /workspace/ComfyUI/models/diffusion_models \
  --name modello.safetensors
```

Un repository con più pesi chiede una scelta numerata nel terminale. In script o notebook la scelta deve essere esplicita:

```bash
/workspace/bin/model-get "URL_REPOSITORY" --file subfolder/modello.gguf
```

Per controllare risoluzione, revisione e destinazione senza scaricare i byte:

```bash
/workspace/bin/model-get "URL" --dry-run
```

Sono accettati file `.safetensors`, `.gguf`, `.ckpt`, `.pt`, `.pth` e `.bin`. Un file locale esistente non viene sovrascritto; viene riutilizzato soltanto quando il suo SHA256 coincide con quello dichiarato dalla sorgente.

Per repository privati o gated, impostare il token nell'ambiente senza inserirlo nell'URL:

```bash
export HF_TOKEN="..."
export CIVITAI_API_TOKEN="..."
```

Hugging Face usa anche il token già salvato in `HF_HOME/token` o `~/.cache/huggingface/token`. Il token Civitai è inviato solo agli host ufficiali `civitai.com`, `www.civitai.com`, `civitai.red` e `www.civitai.red`; il comando non stampa né salva i token.

La copia persistente può trovarsi in `/workspace/bin/model-get` oppure `/storage/bin/model-get`, secondo il volume montato. Usare quel percorso completo sulle nuove istanze: l'eventuale alias in `/usr/local/bin/model-get` appartiene alla singola istanza e può non sopravvivere alla sua sostituzione.
