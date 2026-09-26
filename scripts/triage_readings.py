"""Which of the 583 splits are real homographs and which are diacritizer noise?

Run this ON THE POD. It uses the context-agreement measurement: a genuine
homograph is determined by its context, so where the same neighbouring words
repeat, the reading agrees. Noise does not behave that way.

This is the same gate collect_readings already applies; here it is reported per
word so the split can be inspected rather than trusted.
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
rows = [json.loads(l) for l in open(cfg.paths.diacritized_path, encoding="utf-8")]

# (word, left, right) -> Counter of codes seen in that exact context
ctx = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for r in rows:
    plain = r["text"].split()
    diac = strip_junk(r["diacritized"]).split()
    if len(plain) != len(diac):
        continue
    for i, (w, d) in enumerate(zip(plain, diac)):
        e = lex.get(w)
        if e is None or e.n_codes < 2:
            continue
        c = e.code_of(d)
        if c < 0:
            continue
        left = plain[i - 1] if i else "<s>"
        right = plain[i + 1] if i + 1 < len(plain) else "</s>"
        ctx[w][(left, right)][c] += 1


def agreement(word):
    """Share of repeated contexts where the reading is consistent."""
    seen = ctx.get(word, {})
    repeated = [c for c in seen.values() if sum(c.values()) >= 2]
    if not repeated:
        return None, 0
    agree = sum(max(c.values()) for c in repeated)
    total = sum(sum(c.values()) for c in repeated)
    return agree / total, len(repeated)


amb = lex.ambiguous_words
print(f"{len(amb)} ambiguous word types\n")

rank = []
for w in amb:
    e = lex.get(w)
    a, n = agreement(w)
    rank.append((a if a is not None else -1.0, n, e.total, w, e))

# Words whose reading is predictable from context: the real homographs.
print("=" * 88)
print("CONTEXT-DETERMINED (a real homograph: same context -> same reading)")
print("=" * 88)
print(f"{'word':<16}{'agree':>7}{'ctxs':>6}{'uses':>7}  counts        examples")
strong = [r for r in rank if r[0] >= 0.95 and r[1] >= 3]
for a, n, total, w, e in sorted(strong, key=lambda r: -r[2])[:40]:
    print(f"{w:<16}{a:>7.2f}{n:>6}{total:>7}  {str(e.counts):<13} {' '.join(e.examples)}")
print(f"\n-> {len(strong)} words")

print()
print("=" * 88)
print("CONTEXT-INDEPENDENT (the same context takes both readings: likely noise)")
print("=" * 88)
print(f"{'word':<16}{'agree':>7}{'ctxs':>6}{'uses':>7}  counts        examples")
weak = [r for r in rank if 0 <= r[0] < 0.80 and r[1] >= 3]
for a, n, total, w, e in sorted(weak, key=lambda r: -r[2])[:40]:
    print(f"{w:<16}{a:>7.2f}{n:>6}{total:>7}  {str(e.counts):<13} {' '.join(e.examples)}")
print(f"\n-> {len(weak)} words")

print()
print("=" * 88)
print("THE WORDS FROM THE BRIEF")
print("=" * 88)
for w in ["علم", "مصر", "دول", "الدول", "عالم", "كتب", "ملك", "الميه", "قطر"]:
    e = lex.get(w)
    if e is None:
        print(f"  {w:<10} not in the lexicon at all")
        continue
    a, n = agreement(w)
    astr = f"{a:.2f} over {n} ctx" if a is not None else "no repeated context"
    print(f"  {w:<10} {e.n_codes} reading(s), {e.total:>4} uses, {astr:<22} "
          f"{str(e.counts):<12} {' '.join(e.examples)}")

# The sentence the user cares about.
print()
print("=" * 88)
print("THE USER'S SENTENCE: can مصر be disambiguated?")
print("=" * 88)
e = lex.get("مصر")
print(f"  مصر has {e.n_codes} reading(s): {e.patterns} {e.examples}")
if e.n_codes < 2:
    print("  -> NO. مُصِرّ and مَصْر are one class, so the insisting/Egypt")
    print("     distinction cannot be learned or corrected.")
    hits = collections.Counter()
    for r in rows:
        plain = r["text"].split()
        diac = strip_junk(r["diacritized"]).split()
        if len(plain) != len(diac):
            continue
        for w, d in zip(plain, diac):
            if w == "مصر":
                hits[d] += 1
    print("\n  raw diacritized forms CATT produced for مصر:")
    for form, k in hits.most_common(12):
        print(f"    {form:<14} {k}")

print()
print("=" * 88)
print(f"majority baseline printed by the pipeline: see the log (71.2%)")
print("=" * 88)
