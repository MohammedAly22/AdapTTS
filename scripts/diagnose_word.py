"""Why does علم always choose code 0, and why is it absent from the eval table?

Run ON THE POD. Three candidate explanations, and the data can separate them:

  1. Class imbalance: علم is [عِلْمْ(63), عَلَمْ(7)], so always predicting
     science is 90% accurate on it. The model may simply have learned the prior
     for this word.
  2. Not in the held-out split at all, which is why it never appears in the
     per-word table -- 7 flag occurrences can easily all land in train.
  3. A labelling problem: the flag occurrences may not be where we think.

Also checks the same question for مصر, which collapsed to one reading.
"""
import collections
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from adaptts.text.diacritics import ReadingLexicon, strip_junk  # noqa: E402
from adaptts.utils.config import load_config  # noqa: E402

CONFIG = os.environ.get("CONFIG", "configs/exp1_egyptian.yaml")
cfg = load_config(CONFIG)
lex = ReadingLexicon.load(cfg.paths.reading_lexicon_path)

manifest = {}
for line in open(cfg.paths.manifest_path, encoding="utf-8"):
    r = json.loads(line)
    manifest[r["uid"]] = r.get("split", "train")

rows = [json.loads(l) for l in open(cfg.paths.diacritized_path, encoding="utf-8")]

for WORD in ["علم", "مصر", "عالم", "دول"]:
    e = lex.get(WORD)
    print("=" * 76)
    print(f"{WORD}: {e.n_codes if e else 0} reading(s)"
          + (f"  {e.examples}  counts {e.counts}  total {e.total}" if e else ""))
    print("=" * 76)
    if e is None:
        print("  not in the lexicon\n")
        continue

    per_split = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for r in rows:
        plain = r["text"].split()
        diac = strip_junk(r["diacritized"]).split()
        if len(plain) != len(diac):
            continue
        split = manifest.get(r["uid"], "train")
        for w, d in zip(plain, diac):
            if w != WORD:
                continue
            c = e.code_of(d)
            per_split[split][c] += 1
            if len(examples[c]) < 4:
                examples[c].append((d, r["text"][:70]))

    print(f"  {'split':<8}" + "".join(f"code{c:<7}" for c in range(e.n_codes))
          + "unmatched")
    for split in ("train", "dev", "test"):
        cnt = per_split.get(split)
        if not cnt:
            continue
        cells = "".join(f"{cnt.get(c, 0):<11}" for c in range(e.n_codes))
        print(f"  {split:<8}{cells}{cnt.get(-1, 0)}")

    print()
    for c in sorted(k for k in examples if k >= 0):
        label = e.examples[c] if c < len(e.examples) else "?"
        print(f"  --- code {c} ({label}) ---")
        for d, t in examples[c]:
            print(f"      {d:<14} {t}")
    print()

    # If a reading never appears in dev/test, the eval cannot score it.
    dev_test = collections.Counter()
    for s in ("dev", "test"):
        dev_test.update(per_split.get(s, {}))
    present = [c for c in range(e.n_codes) if dev_test.get(c, 0) > 0]
    if len(present) < 2:
        print(f"  NOTE: only code(s) {present} occur in dev/test, so held-out")
        print(f"        evaluation cannot measure this word's disambiguation.")
        print(f"        Its absence from the per-word table is expected.")
    print()
