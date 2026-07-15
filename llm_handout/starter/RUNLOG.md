# RUNLOG

One entry per training run. `steps` is the optimizer-step budget used for
*that* run (ablation runs use a reduced budget to iterate faster; the final
submission run uses the full 2000-step budget). `dev bpb` is
`evaluate.py --checkpoint <ckpt> --text_file data/dev_eval.txt`.

## Run 0 — baseline, as shipped

- **Hypothesis / purpose:** establish the reference score before touching anything.
- **Change:** none — starter `train.py` / `model.py` / `tokenizer.py` unmodified.
- **Config:** byte tokenizer (vocab 256), n_embd=160, n_layer=4, n_head=4, block_size=128, tie_weights=False, plain N(0, 0.05) init, Adam, constant lr=3e-4, no warmup/schedule/weight-decay/grad-clip, batch=8, steps=2000.
- **Result:** train loss 5.65 → 1.73 over 2000 steps (83 ms/step, 166s total). **dev bpb = 2.3718**, n_params = 1,339,840.
- **Conclusion / questionable things noted before changing anything:**
  1. Constant LR with no warmup — first ~50 steps take large steps into a randomly-initialized loss landscape; no decay means the model is still bouncing around late in training instead of settling.
  2. No weight decay, no grad clipping — nothing bounds weight growth or protects against occasional large-gradient steps.
  3. `tie_weights = False` spends ~41K params (out of a 1.34M budget) on a separate unembedding matrix that duplicates information already in the token embedding — wasteful under a tight param cap.
  4. Flat `std=0.05` init for every Linear/Embedding, including the residual-stream output projections (`attn.proj`, second MLP linear) — with 4 layers of residual adds, this lets activation variance grow with depth (no GPT-2-style 1/sqrt(2*n_layer) scaling).
  5. Byte-level tokenizer: Devanagari (Hindi) characters are 3 UTF-8 bytes each, and ~14% of corpus characters (by count) are Devanagari, so a large fraction of the corpus's *byte* stream is really compressible multi-byte units the model has to spend attention/params modeling one byte at a time. `block_size=128` therefore covers only 128 bytes of context — much less than 128 bytes' worth of *meaning*, especially in Hindi passages.
  6. `block_size=128` is small regardless of tokenizer; more context per step, within the same 2000-step budget, should help.

---

## Runs 1–5 — first ablation round (500-step reduced budget, isolate one factor at a time)

All five runs below use 500 optimizer steps (not the full 2000) purely so 5
configs could be compared quickly; only relative ranking matters here, not
the absolute numbers (which are worse than a full run would give).

| # | name | tokenizer | recipe | pos enc | block | batch | params | dev bpb |
|---|------|-----------|--------|---------|-------|-------|--------|---------|
| 1 | anchor_byte_oldrecipe | byte (256) | baseline-style (constant lr, no wd/clip, untied, std=0.05 init) | learned | 128 | 8 | 1,339,840 | **2.9294** |
| 2 | byte_newrecipe | byte (256) | warmup+cosine, AdamW wd=0.1, clip=1.0, tied, GPT-2 init | rope | 128 | 8 | 1,541,408 | **2.4059** |
| 3 | bpe_oldrecipe | BPE (2048) | baseline-style (same as #1) | learned | 128 tok (~434 bytes) | 8 | 1,913,280 | **2.3782** |
| 4 | bpe_newrecipe_learned | BPE (2048) | new recipe (same as #2) | learned | 256 tok (~870 bytes) | 24 | 1,901,856 | **2.2373** |
| 5 | bpe_newrecipe_rope | BPE (2048) | new recipe (same as #2) | **rope** | 256 tok | 24 | 1,856,800 | **1.8770** |

- **Hypothesis:** each of (a) the training-recipe fixes (schedule/decay/clip/tie/init), (b) the BPE tokenizer, and (c) RoPE vs learned position embeddings would independently improve dev bpb, and they'd stack.
- **What changed, run by run:** #1→#2 isolates the recipe fix (tokenizer/arch held at baseline-like settings). #1→#3 isolates the tokenizer swap (recipe held at old-style). #3→#4 adds the recipe fix on top of BPE, plus grows block_size/batch since BPE's ~3.4 bytes/token means block_size=256 tokens ≈ 870 bytes of real context (vs. 128 *bytes* for the byte tokenizer) — an apples-to-apples "more context" comparison would need block_size≈434 for BPE, but the whole point of switching tokenizers is that the *same nominal block_size* now buys far more real context for free, so we let it. #4→#5 isolates RoPE vs learned absolute position embeddings, everything else identical.
- **Result / conclusion:**
  1. **Recipe fix alone** (#1→#2): 2.9294 → 2.4059, a 18% relative bpb drop. Confirms the baseline's constant LR / no-decay / no-clip / untied / flat-init combination was leaving real performance on the table — and this is at only 500 steps, i.e. before the cosine schedule has even reached its later, more important decay phase.
  2. **Tokenizer swap alone** (#1→#3): 2.9294 → 2.3782, a 19% relative drop — roughly the *same size win* as the recipe fix, but from a completely orthogonal change. This matches the handout's hint: byte-level tokenization is a real weak point on Hindi-heavy text, and fixing it is worth as much as fixing the whole optimizer recipe.
  3. **They compound** (#3→#4): adding the recipe fix on top of BPE brings 2.3782 → 2.2373. Both effects are real and mostly additive, not redundant.
  4. **RoPE is the single biggest lever found so far** (#4→#5): 2.2373 → 1.8770, a further 16% relative drop, *and* RoPE has zero learned parameters (vs. `block_size * n_embd` = 256*176 = 45,056 params for the learned embedding table), so it's simultaneously better and cheaper under the param cap. Hypothesis for *why*: with only 500-2000 steps to train, a learned absolute position embedding table has to learn every one of its `block_size` rows from scratch from gradient signal alone; RoPE encodes relative position via a fixed (untrained) rotation, so there's nothing about position left to learn — all the step budget goes toward content modeling instead.
  5. **Decision:** carry forward BPE tokenizer + full recipe fix + RoPE as the base config for round 2 (block_size / batch / lr sweeps) and the final run.

---

## Runs 6–12 — second ablation round (block_size / batch / lr sweeps, 500 steps each)

Base config for all of these = run 5's winner (BPE, new recipe, RoPE, block_size=256, batch=24, lr=3e-3), varying one knob at a time, then testing the best two knobs combined.

| # | name | block_size | batch | lr | dev bpb | Δ vs base (1.8770) |
|---|------|-----------|-------|-----|---------|---------------------|
| 6 | bpe_block192 | 192 | 24 | 3e-3 | 1.9301 | worse |
| 7 | bpe_block320 | 320 | 24 | 3e-3 | 1.8539 | better |
| 8 | bpe_batch12 | 256 | 12 | 3e-3 | 2.2693 | much worse |
| 9 | bpe_batch32 | 256 | 32 | 3e-3 | 1.8343 | better |
| 10 | bpe_lr_low | 256 | 24 | 1.5e-3 | 1.9260 | worse |
| 11 | bpe_lr_high | 256 | 24 | 6e-3 | 2.1349 | much worse |
| 12 | bpe_block320_batch32 | **320** | **32** | 3e-3 | **1.8124** | **best** |

- **Hypothesis:** since only *optimizer steps* are capped (not compute/tokens), increasing tokens-per-step (via block_size or batch) should let the model see more data within the same 2000-step budget and improve bpb; lr=3e-3 (chosen as a reasonable default) might not be optimal.
- **Result / conclusion:**
  1. **block_size and batch both trade monotonically in the tested range**: 192→256→320 improves bpb (1.9301→1.8770→1.8539); 12→24→32 improves bpb even more sharply (2.2693→1.8770→1.8343). Smaller batch (12) hurts badly — noisier gradient estimates and fewer total tokens seen under a fixed 2000-step cap cost more than they save in wall-clock.
  2. **lr=3e-3 was already close to a local optimum**: both 1.5e-3 (undertrained-looking, higher bpb) and 6e-3 (unstable/overshooting, higher bpb) are worse. No further lr tuning done — diminishing returns for the time spent.
  3. **Combining the two winning knobs (block_size=320, batch=32) is close to additive**: base 1.8770, block-only Δ=-0.0231, batch-only Δ=-0.0427, sum≈-0.0658; observed combined Δ=-0.0646 (1.8124). This confirms they're capturing largely independent sources of improvement (more real context per prediction vs. less noisy gradients / more tokens per step), not double-counting the same effect.
  4. **Cost tradeoff:** block_size=320/batch=32 costs ~866ms/step (vs. 83ms/step for the original baseline shape) — about 10x slower per step, but since only *steps* are capped, not wall-clock, this is a legitimate way to spend the budget. A full 2000-step run at this config was estimated at ~29 minutes, judged acceptable for a one-time final run (no wall-clock cap in the hard caps list).
  5. **Decision:** lock in BPE tokenizer + full recipe (warmup+cosine, AdamW wd=0.1, clip=1.0, tied, GPT-2 init) + RoPE + block_size=320 + batch=32 + lr=3e-3 as the final configuration, and run it for the full 2000-step budget.

---
