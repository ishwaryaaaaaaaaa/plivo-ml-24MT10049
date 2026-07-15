#!/bin/bash
# Ablation sweep: reduced-step runs to compare configs quickly, then score
# each with the official evaluate.py. Appends one line per run to results.jsonl.
set -e
cd "$(dirname "$0")"
source ~/speedrun/env/bin/activate 2>/dev/null || true

RESULTS=results.jsonl
mkdir -p runs

use_bpe()  { [ -f bpe_merges.json.bak ] && mv bpe_merges.json.bak bpe_merges.json; true; }
use_byte() { [ -f bpe_merges.json ] && mv bpe_merges.json bpe_merges.json.bak; true; }

run() {
  name="$1"; steps="$2"; shift 2
  echo "=== $name (steps=$steps) ==="
  t0=$(date +%s)
  python train.py --data data/train_corpus.txt --steps "$steps" --out "runs/${name}.pt" --log_every 200 "$@" > "runs/${name}.log" 2>&1
  tail -6 "runs/${name}.log"
  t1=$(date +%s)
  score=$(python evaluate.py --checkpoint "runs/${name}.pt" --text_file data/dev_eval.txt)
  line="{\"name\": \"$name\", \"steps\": $steps, \"wall_s\": $((t1-t0)), \"args\": \"$*\", \"eval\": $score}"
  echo "$line"
  echo "$line" >> "$RESULTS"
}

S=500

# 1. anchor: byte tokenizer, old-style recipe/arch (reproduces baseline through new train.py), reduced steps
use_byte
run anchor_byte_oldrecipe $S \
  --lr 3e-4 --min_lr 3e-4 --warmup_steps 0 --weight_decay 0 --grad_clip 0 \
  --no_tie_weights --pos_encoding learned --init_std 0.05 \
  --n_embd 160 --n_layer 4 --n_head 4 --block_size 128 --batch 8

# 2. byte tokenizer, new recipe (schedule+wd+clip+tie+rope+gpt2-init), same block/batch scale as baseline
use_byte
run byte_newrecipe $S \
  --block_size 128 --batch 8

# 3. BPE tokenizer, old-style recipe (isolates tokenizer effect)
use_bpe
run bpe_oldrecipe $S \
  --lr 3e-4 --min_lr 3e-4 --warmup_steps 0 --weight_decay 0 --grad_clip 0 \
  --no_tie_weights --pos_encoding learned --init_std 0.05 \
  --n_embd 160 --n_layer 4 --n_head 4 --block_size 128 --batch 8

# 4. BPE tokenizer, new recipe, learned pos (isolates rope vs learned)
use_bpe
run bpe_newrecipe_learned $S \
  --pos_encoding learned --block_size 256 --batch 24

# 5. BPE tokenizer, new recipe, rope (all defaults)
use_bpe
run bpe_newrecipe_rope $S

echo "=== sweep 1 done ==="
cat "$RESULTS"
