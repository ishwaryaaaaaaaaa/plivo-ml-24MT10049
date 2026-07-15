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
