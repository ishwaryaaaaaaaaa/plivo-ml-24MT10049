import sys
import tokenizer

t = tokenizer.load()
print("tokenizer:", type(t).__name__, "vocab_size=", t.vocab_size)

dev = open("data/dev_eval.txt", encoding="utf-8").read()
ids = t.encode(dev)
ok = t.decode(ids) == dev
nb = len(dev.encode("utf-8"))
print(f"dev_eval.txt: {nb} bytes -> {len(ids)} tokens, ratio={nb/len(ids):.3f}, roundtrip_ok={ok}")
if not ok:
    sys.exit("dev roundtrip FAILED")

train = open("data/train_corpus.txt", encoding="utf-8").read()
ids2 = t.encode(train)
ok2 = t.decode(ids2) == train
nb2 = len(train.encode("utf-8"))
print(f"train_corpus.txt: {nb2} bytes -> {len(ids2)} tokens, ratio={nb2/len(ids2):.3f}, roundtrip_ok={ok2}")
if not ok2:
    sys.exit("train roundtrip FAILED")

s = "Randomtext with emoji and unseen script mixed in, plus punctuation!?"
ok3 = t.decode(t.encode(s)) == s
print("arbitrary ascii roundtrip_ok=", ok3)
if not ok3:
    sys.exit("ascii roundtrip FAILED")

print("ALL OK")
