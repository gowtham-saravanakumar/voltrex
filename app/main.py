import mimetypes
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.model_service import ModelUnavailableError, service

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(APP_ROOT, "static")

# Not every platform's mimetypes DB knows .webmanifest / .svg by default.
mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("image/svg+xml", ".svg")

app = FastAPI(
    title="Voltrex API",
    description="Serves text generations from the from-scratch 'Voltrex' toy transformer.",
    version="1.0.0",
)

# Wide-open CORS: this is a small demo API meant to be called from its own
# bundled frontend as well as directly (curl, other tools). Tighten
# allow_origins if you deploy the frontend separately from a known domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="the chef", max_length=300, description="Seed text to continue.")
    max_new_tokens: int = Field(default=40, ge=1, le=200, description="How many tokens to generate.")
    temperature: float = Field(default=0.8, ge=0.05, le=2.0, description="Sampling temperature.")
    top_k: int = Field(default=20, ge=0, le=200, description="Top-k sampling cutoff (0 = disabled).")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    meta: dict


@app.on_event("startup")
def on_startup():
    # Try to warm the model at boot so the first real request is fast, but
    # never crash the process if it fails -- report it at request time
    # instead so /health still responds.
    try:
        service.load()
    except Exception as e:
        print(f"[startup] model load deferred/failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service.model is not None}


@app.get("/api/info")
def info():
    try:
        service.load()
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "step": service.meta.get("step"),
        "val_loss": service.meta.get("val_loss"),
        "config": service.meta.get("config"),
        "total_params": service.meta.get("total_params"),
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        text = service.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
    except ModelUnavailableError as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return GenerateResponse(
        prompt=req.prompt,
        generated_text=text,
        meta={"step": service.meta.get("step"), "val_loss": service.meta.get("val_loss")},
    )


# Mounted last so it never shadows /health or /api/*.
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
