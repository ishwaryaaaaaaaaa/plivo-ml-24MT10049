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
