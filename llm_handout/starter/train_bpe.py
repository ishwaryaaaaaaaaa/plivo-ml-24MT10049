"""Train a byte-level BPE tokenizer on train_corpus.txt only, and save the
merge table to bpe_merges.json (loaded by tokenizer.py at runtime).

Byte-level BPE: base vocab is the 256 raw bytes (so it's lossless / has a
byte fallback for anything unseen, exactly like the starter's ByteTokenizer),
and merges combine frequent adjacent byte-pairs into new tokens. This
directly targets the Hindi/Devanagari problem called out in the handout:
each Devanagari character is 2-3 raw bytes, so common byte sequences (the
UTF-8 encoding of frequent Devanagari codepoints/conjuncts, and common
English subwords) collapse into single tokens, shrinking effective sequence
length -- especially for Hindi text, which otherwise inflates 2-3x relative
to English under the raw byte tokenizer.

Training is done on a random line-sample of the corpus (not the full
7.3MB) purely for speed of the merge-counting loop; this is still "trained
only on train_corpus.txt" per the rules, just not on every byte of it.
Uses numpy for vectorized pair counting so ~1800 merges finish in minutes,
not hours, in pure Python.
"""
import argparse
import json
import random
import time

import numpy as np

PAIR_BASE = 1 << 20  # > any token id we'll ever create (vocab stays << 2**20)


def get_most_frequent_pair(ids):
    combo = ids[:-1].astype(np.int64) * PAIR_BASE + ids[1:].astype(np.int64)
    vals, counts = np.unique(combo, return_counts=True)
    best = counts.argmax()
    if counts[best] < 2:
        return None, 0
    v = vals[best]
    a, b = int(v // PAIR_BASE), int(v % PAIR_BASE)
    return (a, b), int(counts[best])


def merge_pair(ids, pair, new_id):
    a, b = pair
    match = (ids[:-1] == a) & (ids[1:] == b)
    idx = np.nonzero(match)[0]
    if len(idx) == 0:
        return ids
    # greedy left-to-right, non-overlapping selection
    selected = [idx[0]]
    last_end = idx[0] + 1
    for i in idx[1:]:
        if i > last_end:
            selected.append(i)
            last_end = i + 1
    selected = np.array(selected)
    keep = np.ones(len(ids), dtype=bool)
    keep[selected + 1] = False
    ids = ids.copy()
    ids[selected] = new_id
    return ids[keep]


def train_bpe(text_bytes, vocab_size, log_every=100):
    assert vocab_size >= 256
    ids = np.frombuffer(text_bytes, dtype=np.uint8).astype(np.int64)
    merges = {}
    num_merges = vocab_size - 256
    t0 = time.time()
    for i in range(num_merges):
        pair, count = get_most_frequent_pair(ids)
        if pair is None:
            print(f"stopping early at {i} merges: no repeated pairs left")
            break
        new_id = 256 + i
        ids = merge_pair(ids, pair, new_id)
        merges[pair] = new_id
        if (i + 1) % log_every == 0:
            print(f"merge {i+1}/{num_merges}  pair={pair} -> {new_id}  "
                  f"count={count}  seq_len={len(ids):,}  "
                  f"({time.time()-t0:.0f}s)")
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--sample_bytes", type=int, default=2_000_000,
                     help="random line-sampled byte budget for training speed")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="bpe_merges.json")
    args = ap.parse_args()

    random.seed(args.seed)
    lines = open(args.data, encoding="utf-8").read().split("\n")
    total_corpus_bytes = sum(len(l.encode("utf-8")) + 1 for l in lines)
    random.shuffle(lines)
    budget = args.sample_bytes
    picked = []
    total = 0
    for ln in lines:
        b = len(ln.encode("utf-8")) + 1
        if total + b > budget and picked:
            break
        picked.append(ln)
        total += b
    sample_text = "\n".join(picked)
    sample_bytes = sample_text.encode("utf-8")
    print(f"training BPE on {len(sample_bytes):,} sampled bytes "
          f"(of {total_corpus_bytes:,} total) -> target vocab {args.vocab_size}")

    merges = train_bpe(sample_bytes, args.vocab_size)
    ordered = sorted(merges.items(), key=lambda kv: kv[1])
    merge_list = [[list(pair), new_id] for pair, new_id in ordered]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"type": "bpe", "vocab_size": 256 + len(merge_list),
                    "merges": merge_list}, f)
    print(f"saved {args.out}: {len(merge_list)} merges, "
          f"final vocab {256+len(merge_list)}")


if __name__ == "__main__":
    main()
