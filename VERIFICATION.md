# Local verification run

Everything below was executed on the machine described here, against the real
`OmarAhmedSobhy/tts-egyption-dataset`. Nothing in this file is estimated.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce GTX 1660 Ti, 6 GB, compute 7.5 (Turing) |
| PyTorch | 2.5.1+cu121 |
| Python | 3.11.16, conda env `adaptts` |
| OS | Windows 11 |

A note on precision: torch reports `is_bf16_supported() == True` on Turing, but
that is emulation. Turing has no bf16 tensor cores, so `exp0_small.yaml` uses
fp16 with a gradient scaler. On Ampere or newer, switch to bf16.

## Pipeline results

| Stage | Wall time | Result |
|---|---|---|
| Manifest + parquet extraction | 1 m 20 s | 2012 clips, 3.01 h, 37-symbol vocabulary |
| CTC forced alignment | 9 m 15 s | 22,497 word spans, 1.7% dropped, 0 non-monotonic |
| Span embeddings | ~3 m | 22,497 × 1024 |
| **Pronunciation-code discovery** | ~30 s | 6605 types examined, **9 declared ambiguous** |
| Teacher cache | 21 s | head fits the discovered codes at 100% train accuracy |
| Mimi codec | 1 m 19 s | 136,296 frames, 181.7 min of audio |
| Context encoder, 1500 steps | 1 m 06 s | 22.6 steps/s, 80 MB peak |
| Acoustic model, 300 steps | 44 m 45 s | loss 15.9 -> 10.6, coarse accuracy 0.014 -> 0.125 |

After training, `scripts/smoke_generate.py` produced real audio on CPU:

| | |
|---|---|
| Real-time factor | **0.36** (faster than real time, on CPU) |
| Adaptive depth | depth 3 = 1.21 s, depth 5 = 1.38 s, depth 8 = 1.50 s |
| Exit losses | 5.341 / 5.336 / 5.334, so self-distillation is working |
| Override | rendered both readings of a discovered word, no diacritics typed |

At 300 steps the audio is not intelligible, which is expected. What this
establishes is that every mechanism in the design functions: codes resolve,
overrides reach the model, shallower exits genuinely cost less, and a waveform
comes out.

Text normalization left **zero** digits or Latin characters across all 2012
transcripts.

## What exp0 does and does not prove

**Proves.** The machinery works end to end on real Egyptian speech: audio is
force-aligned without a lexicon, word spans are embedded, the four statistical
gates accept a small conservative set of splits, those become labels, and both
models train on them.

**Does not prove.** That the specific homographs from the brief are resolved.
This corpus is 3 hours, and in it `علم` occurs 10 times and `مصر` 4. Discovery
needs enough occurrences per word type to be confident, so the showcase
homographs need the 68-hour corpus in `exp1_egyptian.yaml`. Held-out accuracy
of 83% on 6 ambiguous words is not a meaningful number; it is a smoke signal.

## Bugs this run found

Every one of these was invisible to the synthetic test suite and would have
failed a rented-GPU run.

**1. Pretrained checkpoints blocked.** Transformers 5.x refuses `.bin`
checkpoints unless torch is at least 2.6 (CVE-2025-32434). Three of the four
configured models were `.bin` only. Replaced with safetensors equivalents and
each verified to load and produce the expected shapes.

**2. CTC blank token.** On the MMS aligner the blank is `<blank>` at index 0
while `pad_token_id` is 1. Using the pad id produces alignments that look
plausible and are wrong. The blank is now detected explicitly.

**3. Romanized aligner vocabulary.** The MMS aligner emits 31 Latin tokens, not
Arabic script. Added `text/romanize.py`, which transliterates while keeping a
character index map so Latin frame spans map back to Arabic word spans.

**4. Alignment thresholds discarded valid data.** Measured: the median aligned
Egyptian word lasts 0.24 s and the 5th percentile is 0.06 s, so the 0.08 s
floor cut real short words, and a romanized aligner scores lower because
transliteration is approximate. Relaxing to 0.04 s and -6.5 keeps 99.4% of
spans against 88.3%.

**5. Duration confound in discovery.** This is the important one. The first
three gates all passed for the wrong reason: the accepted splits separated the
*elongated* realization of a word from the *fast* one. For `عارفين` the two
clusters averaged 0.27 s and 1.15 s. The clusters were stable, clean and well
separated, and they were not two readings.

The cause is structural, since a mean-pooled self-supervised embedding over a
span correlates with span length. A fourth gate now measures how much of the
split duration alone explains and rejects it when that dominates. The threshold
was calibrated on the data rather than guessed: genuine splits sit at R² ≤ 0.21
and duration-driven ones at 0.31 and 0.46, so 0.25 falls in the gap. Accepted
words went from 12 to 9.

**6. Windows and fp16 specifics.** A lambda `collate_fn` cannot be pickled for
spawned dataloader workers, and `binary_cross_entropy` is rejected under fp16
autocast. Both fixed.

**7. The difficulty head collapsed to zero.** Found by running inference
against the trained context encoder: a word whose posterior entropy was 0.99,
meaning maximum uncertainty for a two-way choice, received a predicted
difficulty of 0.0018.

The loss averaged over every word in the batch, and on real text more than 95%
of words are unambiguous with a target of exactly 0, so predicting 0 everywhere
is near-optimal. The difficulty loss now trains on ambiguous words only.

This one mattered beyond the metric. Difficulty selects the acoustic model's
exit depth, so a collapsed head would have routed every sentence to the
shallowest exit regardless of how hard it was, silently disabling the
adaptive-compute mechanism while every loss curve still looked healthy.

**8. The duration predictor could not see sequence length.** Every sentence,
from 38 to 79 characters, predicted exactly 26.2 frames against true lengths of
63 to 77.

The head mean-pooled the text memory before predicting, and mean pooling
divides by sequence length, so length was removed from its input by
construction. No amount of training could fix that. It now predicts a speaking
*rate* in frames per character and multiplies by the real count: length is
arithmetic and needs no learning, while rate is the part that genuinely depends
on content.

**9. The central learning test was measuring noise.** It asserted that a model
trained on 10 sentences generalizes to 2 held-out ones. Measured across seeds it
passed 3 to 8 times out of 8 regardless of what the code did.

Investigating properly exposed a second issue: on a larger corpus the model was
keying on the function word في, which was accidentally correlated with one
reading. Balancing that away lowered accuracy further, and duplicating
sentences did not help.

The honest conclusion is that a character-level model cannot learn semantics
from a few dozen synthetic sentences, and no fixture can hold enough varied
Arabic to change that. Semantic generalization is what the 68-hour corpus and
the distilled teacher are for, and it is measured there by held-out code
accuracy during real training. The unit test now checks what it can establish
deterministically: that the supervision pathway fits the discovered codes, that
the same word in two contexts receives two different codes, and that an
unambiguous word is structurally incapable of receiving a second reading.

## A note on running two jobs at once

On a 6 GB card, do not. The acoustic run holds 4.9 GB, so starting a second
training process leaves too little headroom and both stall against each other
rather than queueing cleanly. Run one stage at a time.

## Measured performance notes

The dataloader is not a bottleneck. At batch 32 it delivers 543 batches/s with
`num_workers=0`, and workers make it *slower* on Windows because of process
spawning:

| workers | batches/s |
|---|---|
| 0 | 543 |
| 2 | 327 |
| 4 | 242 |

So the 4.2 s/step of the acoustic stage is GPU compute, not data starvation.
Batch size matters a great deal on this card:

| batch | frames/s | peak MB |
|---|---|---|
| 8 | 377 | 1245 |
| 32 | 1102 | 4361 |

Small batches leave the GPU launch-bound rather than compute-bound.

## Reproducing

```bash
conda create -n adaptts python=3.11 -y
conda activate adaptts
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m ipykernel install --user --name adaptts --display-name "AdapTTS (conda)"

python scripts/preprocess.py --config configs/exp0_small.yaml --stage all
python scripts/train_context.py  --config configs/exp0_small.yaml
python scripts/train_acoustic.py --config configs/exp0_small.yaml
```
