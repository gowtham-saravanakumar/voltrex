"""
Load the trained checkpoint and generate text from a prompt.

Run: python generate.py "the chef"
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
from tokenizers import ByteLevelBPETokenizer
from model.architecture import VoltrexModel

CKPT_PATH = os.path.join(ROOT, "checkpoints", "voltrex_v1_best.pt")
tok_dir = os.path.join(ROOT, "tokenizer", "vocab")

tokenizer = ByteLevelBPETokenizer(
    os.path.join(tok_dir, "vocab.json"),
    os.path.join(tok_dir, "merges.txt"),
)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
model = VoltrexModel(**ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded checkpoint from step {ckpt.get('step', '?')} (val loss {ckpt.get('val_loss', float('nan')):.4f})")

prompt = sys.argv[1] if len(sys.argv) > 1 else "the chef"
prompt_ids = tokenizer.encode(prompt).ids
idx = torch.tensor([prompt_ids], dtype=torch.long)

out = model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=20)
print("\n" + tokenizer.decode(out[0].tolist()))
