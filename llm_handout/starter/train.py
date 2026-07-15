"""Trainer. HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data data/train_corpus.txt --steps 2000 --out ckpt.pt

Changes vs the baseline starter (see RUNLOG.md for the ablations that
motivated each one): AdamW with decoupled weight decay (decay only on 2D+
weight matrices, not biases/LayerNorm), linear warmup + cosine decay LR
schedule instead of a constant LR, gradient-norm clipping, and a configurable
architecture (n_embd/n_layer/n_head/block_size/tie_weights/pos_encoding) so
the winning config from the sweep can be passed on the command line.
"""
import argparse
import hashlib
import math
import os
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def lr_at(step, total_steps, max_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def load_ids_cached(data_path, tok, cache_dir=".tok_cache"):
    """Encoding a multi-MB corpus with a BPE tokenizer is the slow part of
    each run's startup; cache the resulting id tensor per (file, vocab) so
    repeated experiments with the same tokenizer don't re-pay it."""
    text = open(data_path, encoding="utf-8").read()
    key = hashlib.md5(f"{data_path}:{tok.vocab_size}:{len(text)}".encode()).hexdigest()[:16]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"ids_{key}.pt")
    if os.path.exists(cache_path):
        ids = torch.load(cache_path)
        print(f"loaded cached tokenization: {cache_path} ({len(ids):,} tokens)")
        return ids, text
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    torch.save(ids, cache_path)
    return ids, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--min_lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--tie_weights", dest="tie_weights", action="store_true", default=None)
    ap.add_argument("--no_tie_weights", dest="tie_weights", action="store_false")
    ap.add_argument("--pos_encoding", choices=["learned", "rope"], default=None)
    ap.add_argument("--init_std", type=float, default=None)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    tok = tokenizer_mod.load()
    ids, text = load_ids_cached(args.data, tok)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    for field in ("block_size", "n_embd", "n_layer", "n_head", "dropout",
                  "tie_weights", "pos_encoding", "init_std"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  cfg: n_embd={cfg.n_embd} n_layer={cfg.n_layer} "
          f"n_head={cfg.n_head} block_size={cfg.block_size} "
          f"tie_weights={cfg.tie_weights} pos_encoding={cfg.pos_encoding}")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        lr = lr_at(step, args.steps, args.lr, args.min_lr, args.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        losses.append(loss.item())
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")

    # every public config attribute is saved -- if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
