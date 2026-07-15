#!/bin/bash
# Ablation sweep: reduced-step runs to compare configs quickly, then score
# each with the official evaluate.py. Appends one line per run to results.jsonl.
set -e
cd "$(dirname "$0")"
source ~/speedrun/env/bin/activate 2>/dev/null || true

RESULTS=results.jsonl
STEPS=${STEPS:-500}

run() {
  name="$1"; shift
  echo "=== $name ==="
  t0=$(date +%s)
  python train.py --data data/train_corpus.txt --steps "$STEPS" --out "runs/${name}.pt" --log_every 100 "$@" 2>&1 | tail -8
  t1=$(date +%s)
  score=$(python evaluate.py --checkpoint "runs/${name}.pt" --text_file data/dev_eval.txt)
  echo "{\"name\": \"$name\", \"wall_s\": $((t1-t0)), \"args\": \"$*\", \"eval\": $score}" | tee -a "$RESULTS"
}

mkdir -p runs
: > "$RESULTS"

# Note: byte-tokenizer runs need bpe_merges.json temporarily out of the way.
