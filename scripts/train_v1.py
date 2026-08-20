"""
"Voltrex" v1 — scaled-up training run.

Same from-scratch architecture, no pretrained weights anywhere.
Bumps capacity and training length within what's still comfortably
CPU-feasible, adds a warmup+cosine LR schedule and grad clipping,
and evaluates on a held-out val split throughout.

Run: python train_v1.py
"""
import sys
import os
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch
from tokenizers import ByteLevelBPETokenizer
from model.architecture import VoltrexModel

# ---- config (scaled up, still CPU-friendly) ----
BLOCK_SIZE = 96
N_LAYER = 4
N_HEAD = 4
N_EMBD = 160
BATCH_SIZE = 32
MAX_LR = 4e-4
MIN_LR = 4e-5
WARMUP_STEPS = 80
MAX_STEPS = 1200
EVAL_EVERY = 100
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
torch.manual_seed(1337)

# ---- load tokenizer trained from scratch ----
tok_dir = os.path.join(ROOT, "tokenizer", "vocab")
tokenizer = ByteLevelBPETokenizer(
    os.path.join(tok_dir, "vocab.json"),
    os.path.join(tok_dir, "merges.txt"),
)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

# ---- load + encode corpus ----
with open(os.path.join(ROOT, "data", "toy_corpus.txt")) as f:
    text = f.read()

ids = tokenizer.encode(text).ids
data = torch.tensor(ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(f"Corpus: {len(data)} tokens ({len(train_data)} train / {len(val_data)} val)")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, iters=20):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for k in range(iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


# ---- model, from-scratch random init ----
model = VoltrexModel(
    vocab_size=vocab_size, block_size=BLOCK_SIZE,
    n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1, betas=(0.9, 0.95))

# ---- train ----
print("\nTraining (v1 run)...")
best_val = float("inf")
for step in range(MAX_STEPS + 1):
    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    if step % EVAL_EVERY == 0:
        losses = estimate_loss(model)
        print(f"step {step:4d} | lr {lr:.2e} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": dict(vocab_size=vocab_size, block_size=BLOCK_SIZE,
                                n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD),
                "step": step,
                "val_loss": best_val,
            }, os.path.join(ROOT, "checkpoints", "voltrex_v1_best.pt"))

print(f"\nBest val loss: {best_val:.4f}")
print("Best checkpoint saved to " + os.path.join(ROOT, "checkpoints", "voltrex_v1_best.pt"))

# ---- generate samples ----
print("\nSample generations from the trained model:")
for prompt in ["the chef", "my grandmother", "a young cook", "the old kitchen"]:
    prompt_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long).to(DEVICE)
    out = model.generate(idx, max_new_tokens=25, temperature=0.8, top_k=20)
    print(" -", tokenizer.decode(out[0].tolist()).replace("\n", " / "))
