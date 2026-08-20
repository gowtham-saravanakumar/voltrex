# Voltrex — API + Web App

A production-ready wrapper around the existing `VoltrexModel` (a tiny,
from-scratch decoder-only transformer): a FastAPI backend that loads the
checkpoint once and serves generations over HTTP, plus a themed static
frontend served from the same service. One Render web service, no separate
frontend host needed.

The model/training code itself (`model/architecture.py`, `scripts/train_v1.py`,
`scripts/generate.py`) is untouched — this only adds a service layer around it.

## Project layout

```
app/
  main.py            FastAPI app: /health, /api/info, /api/generate, static hosting
  model_service.py    Loads the checkpoint once, thread-safe generate()
model/
  architecture.py     Unmodified model definition
checkpoints/
  voltrex_v1_best.pt
tokenizer/vocab/       Byte-level BPE vocab/merges
static/                 Frontend (index.html, style.css, script.js)
scripts/                Original CLI scripts (generate.py, train_v1.py), unchanged
data/toy_corpus.txt     Original training corpus
requirements.txt
render.yaml             Render Blueprint (one web service)
```

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000 for the web app, or call the API directly:

```bash
curl localhost:8000/health

curl -X POST localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "the chef", "max_new_tokens": 40, "temperature": 0.8, "top_k": 20}'
```

## API

| Method | Path            | Description                                   |
|--------|-----------------|------------------------------------------------|
| GET    | `/health`       | `{"status": "ok", "model_loaded": true}`      |
| GET    | `/api/info`     | Checkpoint step, val loss, model config, param count |
| POST   | `/api/generate` | Generate text from a prompt (see below)       |

**POST `/api/generate`** body:

```json
{
  "prompt": "the chef",
  "max_new_tokens": 40,
  "temperature": 0.8,
  "top_k": 20
}
```

All fields are optional (defaults shown above). `max_new_tokens` is capped at
200, `temperature` between 0.05–2.0, `top_k` between 0–200 — invalid values
return a `422` with details; a missing/unavailable checkpoint returns `503`;
unexpected generation errors return `500` with a JSON `detail` message.

Response:

```json
{
  "prompt": "the chef",
  "generated_text": "the chef prepares some rice with a smile...",
  "meta": { "step": 900, "val_loss": 0.83 }
}
```

## Deploy to Render

This repo includes a `render.yaml` Blueprint, so deployment is push-and-go:

1. Push this repo to GitHub.
2. In the Render dashboard: **New +** → **Blueprint** → connect the repo.
   Render reads `render.yaml` and provisions a single web service
   (`voltrex`) with the correct build/start commands automatically.
3. Click **Apply**. Render will:
   - `pip install -r requirements.txt`
   - start with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - poll `/health` to confirm the service is up
4. Once deployed, your API + web app are live at the URL Render gives you
   (e.g. `https://voltrex.onrender.com`).

No separate frontend deploy step is needed — `static/` is served by the same
FastAPI process that serves the API.

### Manual setup (without the Blueprint)

If you'd rather click through the UI instead of using `render.yaml`:

- **New Web Service** → connect the repo
- Runtime: **Python 3**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

### Notes on the free plan

- Render's free web services spin down after inactivity; the first request
  after idling will be slow while the instance restarts and reloads the
  checkpoint (well under a minute for a model this size). Later requests are
  fast.
- The checkpoint (~6 MB) is small enough to commit directly to the repo —
  no Git LFS or external storage needed.

## Notes on the model

`VoltrexModel` is a small transformer (a few hundred K–low M params)
trained from scratch on `data/toy_corpus.txt`, a small placeholder corpus.
It's a demo of the architecture, not a production language model —
expect short, sometimes repetitive continuations within a narrow domain
rather than open-ended reasoning or chat.
