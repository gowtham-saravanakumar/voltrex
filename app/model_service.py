"""
Thin service layer around the existing `model/architecture.py` + checkpoint.

This intentionally does NOT touch the training/model code. It loads the same
checkpoint the same way `scripts/generate.py` does, but once (at process
startup) instead of once-per-CLI-invocation, and exposes a thread-safe
`generate()` call for the API layer to use.
"""
import os
import sys
import threading
import time

# Make the repo root importable (same trick scripts/generate.py uses) so
# `from model.architecture import VoltrexModel` works regardless of cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from tokenizers import ByteLevelBPETokenizer  # noqa: E402
from model.architecture import VoltrexModel  # noqa: E402

CKPT_PATH = os.path.join(ROOT, "checkpoints", "voltrex_v1_best.pt")
TOK_DIR = os.path.join(ROOT, "tokenizer", "vocab")


class ModelUnavailableError(RuntimeError):
    """Raised when the checkpoint/tokenizer can't be loaded."""


class ModelService:
    """Loads the model + tokenizer once, guards generation with a lock.

    torch's autoregressive `generate()` mutates module state across steps
    (not thread-safe for concurrent calls on one model instance), so
    concurrent requests are serialized. The model is tiny (a few hundred K -
    low M params) and each request runs at most a couple hundred decode
    steps, so this is not a meaningful bottleneck for a demo-scale API.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.tokenizer = None
        self.model = None
        self.meta = {}

    def load(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return

            if not os.path.exists(CKPT_PATH):
                raise ModelUnavailableError(f"checkpoint not found at {CKPT_PATH}")
            if not os.path.exists(os.path.join(TOK_DIR, "vocab.json")):
                raise ModelUnavailableError(f"tokenizer vocab not found in {TOK_DIR}")

            start = time.time()
            tokenizer = ByteLevelBPETokenizer(
                os.path.join(TOK_DIR, "vocab.json"),
                os.path.join(TOK_DIR, "merges.txt"),
            )

            ckpt = torch.load(CKPT_PATH, map_location="cpu")
            model = VoltrexModel(**ckpt["config"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            self.tokenizer = tokenizer
            self.model = model
            self.meta = {
                "step": ckpt.get("step", "?"),
                "val_loss": ckpt.get("val_loss"),
                "config": ckpt["config"],
                "total_params": sum(p.numel() for p in model.parameters()),
                "load_seconds": round(time.time() - start, 3),
            }

    def generate(self, prompt: str, max_new_tokens: int = 40, temperature: float = 0.8, top_k: int | None = 20) -> str:
        self.load()

        prompt = (prompt or "").strip() or "the chef"
        block_size = self.meta["config"].get("block_size", 128)

        prompt_ids = self.tokenizer.encode(prompt).ids
        if not prompt_ids:
            prompt_ids = self.tokenizer.encode(" ").ids
        # Keep only what fits the model's context window, same as training/inference assumptions.
        prompt_ids = prompt_ids[-block_size:]

        idx = torch.tensor([prompt_ids], dtype=torch.long)

        with self._lock:
            out = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-4),
                top_k=top_k if top_k and top_k > 0 else None,
            )

        return self.tokenizer.decode(out[0].tolist())


# Singleton used by the API layer.
service = ModelService()
