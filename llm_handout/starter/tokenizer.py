"""Tokenizer used by train.py / evaluate.py.

Two implementations:
  - ByteTokenizer: the original baseline, raw UTF-8 bytes, vocab 256.
  - BPETokenizer: byte-level BPE trained on train_corpus.txt only (see
    train_bpe.py), vocab configurable (default 2048). Base vocab is still
    the 256 raw bytes, so anything unseen just falls back to single-byte
    tokens -- it can encode ARBITRARY UTF-8 text and is always lossless:
    decode(encode(text)) == text, exactly, because merges only ever combine
    byte sequences and decode just concatenates the bytes each token stands
    for before the final utf-8 decode.

load() with no arguments returns whichever tokenizer this submission uses
(BPETokenizer if bpe_merges.json is present next to this file, else the
byte fallback). merges are loaded relative to __file__ so grading (cwd =
submission folder) works with no internet and no extra args.
"""
import heapq
import json
import os


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


class BPETokenizer:
    """Byte-level BPE. Base 256 byte tokens + learned merges on top."""

    def __init__(self, merges):
        # merges: list of [[a, b], new_id] in the order they were learned
        self.merge_ranks = {}          # (a, b) -> rank (lower = applied first)
        self.pair_to_id = {}           # (a, b) -> new_id
        id2bytes = {i: bytes([i]) for i in range(256)}
        for rank, (pair, new_id) in enumerate(merges):
            a, b = pair
            self.merge_ranks[(a, b)] = rank
            self.pair_to_id[(a, b)] = new_id
            id2bytes[new_id] = id2bytes[a] + id2bytes[b]
        self.id2bytes = id2bytes
        self.vocab_size = 256 + len(merges)

    def _bpe_ids(self, ids):
        # O(n log n) merge via a doubly linked list + min-heap of candidate
        # pairs keyed by merge rank (lower rank = learned earlier = applied
        # first) -- equivalent result to the naive "rescan whole sequence
        # per merge" approach, but the naive one is O(n * num_merges) and is
        # far too slow on a multi-megabyte corpus.
        n = len(ids)
        if n < 2:
            return list(ids)
        ids = list(ids)
        rank = self.merge_ranks
        pair_to_id = self.pair_to_id
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))          # nxt[n-1] == n (sentinel: no node)
        alive = [True] * n
        heap = []
        for i in range(n - 1):
            p = (ids[i], ids[i + 1])
            r = rank.get(p)
            if r is not None:
                heap.append((r, i))
        heapq.heapify(heap)
        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j >= n or not alive[j]:
                continue
            pair = (ids[i], ids[j])
            if rank.get(pair) != r:
                continue  # stale: one side changed since this entry was pushed
            ids[i] = pair_to_id[pair]
            alive[j] = False
            nj = nxt[j]
            nxt[i] = nj
            if nj < n:
                prev[nj] = i
            pi = prev[i]
            if pi != -1:
                newp = (ids[pi], ids[i])
                nr = rank.get(newp)
                if nr is not None:
                    heapq.heappush(heap, (nr, pi))
            if nj < n:
                newp = (ids[i], ids[nj])
                nr = rank.get(newp)
                if nr is not None:
                    heapq.heappush(heap, (nr, i))
        out = []
        i = 0
        while i < n:
            out.append(ids[i])
            i = nxt[i]
        return out

    def encode(self, text):
        byte_ids = list(text.encode("utf-8"))
        return self._bpe_ids(byte_ids)

    def decode(self, ids):
        b = b"".join(self.id2bytes[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "vocab_size": self.vocab_size}, f)


def _merges_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "bpe_merges.json")


def load(path=None):
    """Return the tokenizer used by train.py / evaluate.py.

    No-argument call (what train.py/evaluate.py do): looks for
    bpe_merges.json next to this file; if present, loads the trained BPE
    tokenizer, else falls back to raw bytes.
    """
    merges_path = path or _merges_path()
    if os.path.exists(merges_path):
        with open(merges_path, encoding="utf-8") as f:
            data = json.load(f)
        return BPETokenizer(data["merges"])
    return ByteTokenizer()
