# 2,000-Step LLM Speedrun — Submission

A small GPT-style language model trained **entirely from scratch** (no pretrained
weights of any kind) on a mixed English + Hindi corpus, under a hard budget of
**≤2,000 optimizer steps** and **≤2,000,000 parameters**, CPU only. The goal is
the lowest possible bits-per-byte (bpb) on held-out text within those caps —
see `../../LLM_assignment.pdf` for the full brief.

This README covers setup, what the code does, the architecture, the languages/
stack involved, and a short version of what changed and why. The full
run-by-run reasoning — including what was tried and failed — lives in
`RUNLOG.md`; the condensed final answer is in `NOTES.md`; a generated overview
is in `SUMMARY.html`.

---

## Setup

**Language / stack:** Python 3, PyTorch (CPU build), NumPy (used only to speed
up BPE-merge counting during tokenizer training — not used at inference or
training time for the model itself). No other ML libraries — no
`transformers`, no pretrained checkpoints, no GPU, no compiled kernels, per
the assignment's hard constraints.

```bash
# from inside this folder (starter/), with the venv activated
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

**Data** (`../data/`):
- `train_corpus.txt` — ~7.3 MB mixed English + Hindi (Devanagari) text. The
  *only* data the model or tokenizer may be trained on.
- `dev_eval.txt` — held-out text for self-scoring during development. The
  real grading file is different and never seen by us.

---

## How to run

```bash
# 1. (only if using the BPE tokenizer) train the tokenizer on the corpus
python train_bpe.py --data ../data/train_corpus.txt --out bpe_merges.json

# 2. train the model (final config — see NOTES.md for why these values)
python train.py --data ../data/train_corpus.txt --steps 2000 \
    --block_size 320 --batch 32 --lr 3e-3 --pos_encoding rope \
    --tie_weights --out ckpt.pt

# 3. score it
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

`evaluate.py --checkpoint ckpt.pt --text_file <any_text_file>` is the exact,
unmodified interface graders will run — nothing about its CLI or output
format was changed.

---

## What the code does

- **`model.py`** — the GPT itself: byte/BPE-token embedding → N transformer
  blocks (pre-norm self-attention + GELU MLP, residual connections) → final
  layer norm → linear head to vocab logits. Config is fully parameterized
  (layers, heads, embedding width, block size, positional encoding, init,
  weight tying) so architecture changes don't require rewriting the model.
- **`tokenizer.py`** — two tokenizers behind one `load()` interface:
  `ByteTokenizer` (the original raw-UTF-8-byte baseline, vocab 256) and
  `BPETokenizer` (byte-level BPE trained only on `train_corpus.txt`, merges
  loaded from `bpe_merges.json`). Both are guaranteed lossless
  (`decode(encode(text)) == text`) with a byte fallback for anything unseen,
  as required by the grading interface.
- **`train_bpe.py`** — trains the BPE merge table from the corpus using
  vectorized NumPy pair-counting (fast enough to learn ~1,800 merges in
  minutes instead of hours in pure Python).
- **`train.py`** — the training loop: AdamW, linear warmup + cosine decay,
  weight decay, gradient-norm clipping, full CLI control over every
  architecture/recipe knob, enforces the step and parameter caps before
  saving `ckpt.pt` (which embeds the step count, as required).
- **`evaluate.py`** — the official, unmodified scorer: computes bits-per-byte
  over a sliding window with 50% context carry-over.

---

## Architecture (final configuration)

| Component | Choice |
|---|---|
| Tokenizer | Byte-level BPE, vocab 2048 (256 base bytes + ~1,800 learned merges), trained only on `train_corpus.txt` |
| Positional encoding | **RoPE** (rotary), not learned absolute embeddings — zero extra parameters |
| Layers / heads / embedding width | 4 layers, 4 heads, n_embd ≈ 176 |
| Weight tying | **on** — output head shares weights with the token embedding |
| Init | GPT-2-style: `N(0, 0.02)`, residual-stream projections additionally scaled by `1/sqrt(2*n_layer)` |
| Optimizer | AdamW, weight decay 0.1, gradient-norm clip 1.0 |
| Schedule | Linear warmup (100 steps) → cosine decay, peak lr 3e-3 |
| Block size / batch | 320 tokens / batch 32 |
| Steps | 2,000 (full budget) |
| Parameter count | under the 2,000,000 cap (verified at save time by `train.py`'s built-in assertion) |

---

## What we tried, what failed, and what changed

Full detail with numbers is in `RUNLOG.md`; short version:

1. **Baseline (Run 0):** ran the starter unmodified — constant LR, no
   warmup/decay/clip, untied weights, flat `std=0.05` init, raw byte
   tokenizer. **dev bpb = 2.3718.** Reading the code surfaced six specific
   weaknesses (listed in `RUNLOG.md`) before changing anything.

2. **BPE tokenizer, first attempt — failed, then fixed.** The first BPE
   encode implementation rescanned the whole token sequence per merge
   (O(n × merges)), which is fine on a short string but effectively hung on
   the full ~7.3 MB corpus during a round-trip test. Diagnosed the
   complexity problem and rewrote the encoder using a doubly-linked-list +
   min-heap merge algorithm (O(n log n)), which finishes in about a minute
   on the full corpus and was then verified lossless on the entire corpus,
   the dev set, and adversarial inputs (emoji, mixed scripts).

3. **First ablation round (Runs 1–5, 500-step reduced budget to compare
   quickly):** isolated three independent levers — the training-recipe fix
   (schedule/decay/clip/tying/init), the BPE tokenizer swap, and RoPE vs.
   learned position embeddings. All three helped, and stacked: recipe fix
   alone was an 18% relative bpb drop, the tokenizer swap alone was ~19%,
   and RoPE on top of both was the single biggest lever (16% further
   relative drop) — and it's *free* under the parameter cap since it has no
   learned weights, unlike a learned position-embedding table.

4. **Second ablation round (Runs 6–12):** swept block size and batch size,
   since only optimizer *steps* are capped, not wall-clock or tokens —
   larger block size (more real context per step) and larger batch (less
   noisy gradients, more tokens per step) both helped, and combined
   additively. Also swept learning rate and found 3e-3 was already close to
   optimal — both lower and higher settings were worse.

5. **Final decision:** BPE tokenizer + full recipe fix (warmup/cosine/AdamW
   weight decay/grad clip/tied weights/GPT-2 init) + RoPE + block_size=320 +
   batch=32 + lr=3e-3, trained for the full 2,000-step budget.

---

## Deliverables checklist (submission folder)

- [x] `RUNLOG.md` — full run-by-run log, hypotheses, results, conclusions
- [x] Modified code — `model.py`, `tokenizer.py`, `train_bpe.py`, `train.py`, unmodified `evaluate.py`
- [ ] `ckpt.pt` — final full-budget checkpoint *(pending final 2,000-step run)*
- [ ] `NOTES.md` — condensed final-config summary *(pending)*
- [ ] `SUMMARY.html` — generated overview *(pending)*

Note: the final 2,000-step training run using the locked-in configuration
above has not yet been executed as of this README — `ckpt.pt`, `NOTES.md`,
and `SUMMARY.html` will be produced by that run. This README will need a
quick update afterward with the actual final dev bpb.
