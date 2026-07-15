#!/bin/bash
set -e
cd "$(dirname "$0")"
source ~/speedrun/env/bin/activate 2>/dev/null || true

RESULTS=results2.jsonl
mkdir -p runs
[ -f bpe_merges.json.bak ] && mv bpe_merges.json.bak bpe_merges.json  # make sure BPE active

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
: > "$RESULTS"

# block_size sweep (batch=24 fixed, everything else = winning config defaults)
run bpe_block192 $S --block_size 192 --batch 24
run bpe_block320 $S --block_size 320 --batch 24
# (block256/batch24 result already have from round 1: bpe_newrecipe_rope, bpb=1.8770)

# batch sweep (block_size=256 fixed)
run bpe_batch12 $S --block_size 256 --batch 12
run bpe_batch32 $S --block_size 256 --batch 32

# lr sweep (block_size=256, batch=24 fixed)
run bpe_lr_low  $S --block_size 256 --batch 24 --lr 1.5e-3 --min_lr 1.5e-4
run bpe_lr_high $S --block_size 256 --batch 24 --lr 6e-3   --min_lr 6e-4

echo "=== sweep 2 done ==="
cat "$RESULTS"
