# What the 150k-step run taught, and what changed

Everything here was measured, not estimated. The run cost $15 and five hours
and produced a model that had learned nothing useful. Three separate failures
were responsible, and a fourth is still open.

---

## 1. The labels encoded reading speed, not pronunciation

Unsupervised clustering of acoustic word spans returned these as the most
confident "homographs":

```
الجرس  التعليقات  لايك  الوصف  الرابط  البلاي  اكتبوه  عندكم
```

Every one is from the channel's subscribe pitch. They occur in exactly two
acoustic registers, the scripted promo read in a fixed fast cadence and
ordinary narration, and the clustering found that split. Meanwhile `علم`, `مصر`
and `دول` each received a single code.

**Why the gates did not catch it.** A mean-pooled self-supervised embedding
over a word span encodes speaking rate, energy, pitch and recording session far
more strongly than vowel identity. Bootstrap stability, silhouette margin and
centroid separation all filter *noise*. A confound that is reproducible and
well separated passes every one of them. A fourth gate on duration confound
helped and was not enough, because register differs in more than duration.

**What replaced it.** Labels now come from a diacritizer run once over the
transcripts. Verified with CATT-ECA on real corpus sentences:

| Word | Result |
|---|---|
| `علم` | عَلَم (flag) vs عِلْم (science) |
| `مصر` | مَصْر (Egypt) vs مُصِرّ (insisting) |
| `الدول` | إِلدِّوَل (countries) |
| `عالم` | عَالَم (world) vs عَالِم (scholar) |
| `الجرس`, `لايك`, `الوصف` | one reading each, correctly |

Token alignment is 100% and the pass runs at 13 sentences/s, about 20 minutes
for the corpus.

### The diacritizer is noisy, and that had to be handled

Raw patterns from 3000 real sentences gave **912** "ambiguous" word types. Most
were artifacts. Three rounds of suppression, each justified by phonology rather
than convenience:

| Round | Artifact | Example | Result |
|---|---|---|---|
| 1 | mark on the initial letter; sukun on a long vowel; marks inside `ال` | `فِيْ` / `فْيْ` | 912 → 145 |
| 2 | too many raw patterns; context disagreement | `بداوا` with 5 patterns | 145 → 134 |
| 3 | case ending before word-final `ه`; final shadda | `وَصْفَه` / `وَصْفُه` | 134 → **92** |

Round 2 rests on a measurement worth keeping. A genuine homograph is
*determined by its context*, so where the same neighbouring words repeat, the
reading agrees:

| Word | Uses | Patterns | Context agreement | |
|---|---|---|---|---|
| `مصر` | 58 | 2 | **100%** | real |
| `عالم` | 51 | 2 | **100%** | real |
| `دول` | 121 | 2 | **93%** | real |
| `علم` | 21 | 3 | **80%** | real |
| `الوشق` | 16 | 3 | 50% | noise |
| `بداوا` | 35 | 5 | no repeats | noise |

CATT is deterministic, verified on ten sentences that repeat verbatim, so the
variation on `بداوا` is model uncertainty rather than randomness.

---

## 2. CPU inference was ~10x slower than necessary

Profiled per component on a 10-second utterance:

| Component | Cost | Share |
|---|---|---|
| Text encoder (once) | 8.5 ms | — |
| Backbone step, depth 4 | 6.3 ms | 22% |
| **Depth transformer, per frame** | **21 ms** | **72-80%** |

The depth transformer ran `n_layers × n_quantizers` = 32 module calls per frame.

**The cost was dispatch, not arithmetic.** One block on a single `d=256` token:

| | Time |
|---|---|
| Raw matmuls | 67 µs |
| Through PyTorch modules | 453 µs |
| **Overhead factor** | **7x** |

A batch of 8 cost 614 µs against 465 µs for a batch of 1, which is the
signature of per-operator overhead rather than compute. A KV cache was tried
first and gave almost nothing, confirming the diagnosis before the rewrite.

**What replaced it.** `models/depth_head.py` predicts all RVQ levels from the
backbone state with a shared trunk and per-level heads: 2 calls per frame
instead of 32. Coarse-to-fine conditioning survives through a per-level
embedding of the code below.

| | Per-level transformer | Parallel head |
|---|---|---|
| Parameters | 11.9M | **5.3M** |
| Generation, per frame | 22.4 ms | **4.7 ms** |
| Training forward, 4000 frames | 1531 ms | **201 ms** |
| Causality tests | pass | pass |

Whole pipeline: real-time factor went from about **1.0** to **0.11**.

---

## 3. Three of five hours were spent overfitting

The run ended with `ce_exit0` = 4.14 **beating** `ce_exit2` = 4.41. Every exit
is distilled from the deepest, so a shallow exit winning means the model had
more capacity than the data supports. Eval loss had turned upward around step
100k and training continued to 150k regardless.

Three changes:

* **27M instead of 56M.** Measured RTF 0.11 against 0.14 as well.
* **Early stopping.** Replayed against the real loss curve, it halts at step
  **14000** and keeps the best checkpoint.
* **Overfit alarm** when a shallow exit beats the deepest, which nothing
  previously noticed.

---

## 4. Still open: is the label set learnable?

Trained end to end on 3000 real sentences with the round-1 labels, the context
encoder scored **67.9% held out against a 68.3% majority baseline**. It learned
nothing beyond the prior, because most of its training signal was noise.

Rounds 2 and 3 cut ambiguous words from 151 to 92 and the survivors look far
more credible (`كتب`, `قطر`, `فتح`). Retrained on the cleaned set, held-out
accuracy rose to **70.9% against a 70.0% baseline**. Better, but still within
noise of the prior.

### Why, and what would fix it

The diacritizer is not the limitation. On the probe sentences CATT separates
`علم` and `مصر` correctly every time. The limitation is the corpus: in 3000
sentences `علم` occurs 21 times and 18 of those are the science reading. There
is almost no contrast to learn from.

Two things follow:

1. **Diacritize all 15650 sentences, not 3000.** That is 20 minutes and costs
   nothing. Five times the data takes `علم` from 21 occurrences to roughly 100
   and makes the balance between readings measurable.
2. **If the baseline still is not beaten after that**, the honest conclusion is
   that 68 hours of single-speaker narration does not contain enough homograph
   contrast to learn from, whatever the labelling method. The fix would be more
   varied data rather than more code.

**The gating question before renting a GPU:** does held-out accuracy now exceed
the majority baseline by a clear margin? That is a local ten-minute test, not a
$15 one. Run it first.

```bash
python scripts/preprocess.py --config <cfg> --stage discover
head -40 cache/exp1/reading_report.txt
python scripts/train_context.py --config <cfg>
```

Watch `eval/code_acc` against the baseline printed in the reading report. If it
does not clear it, the labels still need work and the acoustic model will
inherit the problem.
