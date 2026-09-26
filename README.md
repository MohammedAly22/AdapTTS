# AdapTTS

**Adaptive, context-aware text-to-speech for Egyptian Arabic.**

Built around one idea: *pronunciation is a discrete decision the model makes
explicitly, from context, and you can see and change it.*

---

## The problem

Arabic is written without diacritics, so one written word maps to several
pronunciations. Context is the only thing that disambiguates it:

| Written | Reading 1 | Reading 2 | Reading 3 |
|---|---|---|---|
| `علم` | `عَلَم` flag | `عِلْم` science | `عَلَّم` taught |
| `مصر` | `مَصْر` Egypt | `مُصِرّ` insisting | |
| `دول` | `دُول` these | `دِوَل` countries | |

A single sentence can contain four of them:

```
انا كنت مصر على ان مصر عندها امكانيات تخليها تتفوق على دول من اللي شايفين نفسهم دول
        ^مُصِرّ         ^مَصْر                              ^الدِوَل                    ^دُول
     insisting        Egypt                              countries                these
```

Most Arabic TTS solves this by demanding diacritized transcripts or bolting on
a rule-based diacritizer. AdapTTS does neither.

---

## The core insight

The problem is not model capacity. It is **missing supervision**.

Undiacritized text carries no label saying which reading occurred, so an
end-to-end model has nothing to learn from and collapses to the most frequent
reading. The homograph is one word in thirty, and roughly 0.4 seconds of a
ten-second clip, so its contribution to reconstruction loss is a rounding
error.

But **the audio contains the answer**. When the narrator says `عَلَم` the
acoustics carry `/ʕalam/`; when they say `عِلْم` they carry `/ʕilm/`. So we
recover the label from the waveform:

```
            ┌─────────────────────────────────────────────────────┐
            │  every occurrence of the word علم in the corpus     │
            └─────────────────────────────────────────────────────┘
                                    │
                    force-align, embed each occurrence
                                    │
                                    ▼
        acoustic embedding space (wav2vec2 middle layers)

              ● ●                                  ▲ ▲ ▲
            ● ● ● ●                              ▲ ▲ ▲ ▲
              ● ●                                  ▲ ▲
           cluster 0                            cluster 1
          /ʕalam/ flag                       /ʕilm/ science
                                    │
              a split is accepted only if it is
              reproducible + separated + balanced
                                    │
                                    ▼
                  PRONUNCIATION CODE 0        CODE 1
                  (these become the training labels)
```

Because the decision is a **discrete code, about 3 bits wide**, the model
cannot smear it across a hidden vector. It has to commit, which means we can
read the commitment and overwrite it.

---

## Architecture

```
┌──────────────────── OFFLINE, once per dataset ──────────────────────┐
│                                                                      │
│  audio + undiacritized text                                          │
│         │                                                            │
│         ├──► A1  CTC forced alignment ────► word time spans          │
│         │        (no lexicon, no MFA, implemented in-repo)           │
│         │                                                            │
│         ├──► A2  wav2vec2 mid layers ─────► span embeddings          │
│         │                                                            │
│         ├──► A3  stability clustering ────► PRONUNCIATION CODES ★    │
│         │        per word type: how many readings, and which one     │
│         │        each occurrence is                                  │
│         │                                                            │
│         ├──► A4  Arabic BERT (frozen) ────► cached teacher states    │
│         │                                                            │
│         └──► A5  Mimi encode ─────────────► RVQ codes @ 12.5 Hz      │
│                                                                      │
│  everything lands in memmaps, so training never decodes audio        │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────── TRAINING, two stages ───────────────────────┐
│                                                                      │
│  STAGE B  Context Encoder  ~3.5M params              ~20 min        │
│  ─────────────────────────────────────────────────                  │
│     characters ─► tiny transformer ─► per-word code logits          │
│                                     └─► difficulty score            │
│     losses:  cross-entropy on discovered codes                      │
│            + KL distillation from the frozen Arabic BERT teacher    │
│            + representation distillation                            │
│                                                                      │
│  STAGE C  Acoustic Model  ~56M params                 7-9 h         │
│  ─────────────────────────────────────────────────                  │
│     characters + codes + speaker ─► Mimi RVQ codes                  │
│                                                                      │
│     text encoder ──► temporal backbone ──► depth transformer        │
│      (bidir)          (causal, 12 layers)   (8 RVQ levels)          │
│                              │                                       │
│                    early-exit heads at 4, 8, 12                     │
│                    each trained by distillation                     │
│                    from the deepest exit                            │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────── INFERENCE, on CPU ──────────────────────────┐
│                                                                      │
│   text ─► context encoder ─► codes + difficulty                     │
│                    │                                                 │
│                    ├──►  YOU SEE THE DECISION                       │
│                    │     and can override it (no diacritics)        │
│                    ▼                                                 │
│           acoustic model at a depth chosen by difficulty            │
│                    │                                                 │
│                    ▼                                                 │
│           Mimi decoder ─► 24 kHz waveform                           │
└──────────────────────────────────────────────────────────────────────┘
```

### Why the context encoder is separate

This is the decision the whole design turns on. If disambiguation were just a
branch of the acoustic model, its gradient would compete with reconstruction
loss and lose. As a separate model it gets:

- its own balanced cross-entropy, so the signal is not diluted;
- a 20-minute training loop, so it can be iterated cheaply;
- an inspectable output, which is what makes control possible;
- a swappable interface, so adding a language later means a new head, not a
  rewrite.

---

## Adaptive depth

The context encoder reports how uncertain it is. That uncertainty picks the
depth:

```
  EASY SENTENCE                    HARD SENTENCE
  "الجو النهارده حلو"               "كنت مصر على ان مصر ..."
  no ambiguous words               four ambiguous tokens
  difficulty 0.02                  difficulty 0.71

  fixed    ████████████████ 12L    fixed    ████████████████ 12L
  adaptive █████            4L     adaptive ████████████████ 12L
           ~35% of compute                  100% of compute
```

Every exit is trained by distillation from the deepest one, so a shallow exit
is a *smaller model*, not a truncated one. That is what keeps quality from
collapsing when the model takes the short path.

---

## Size and speed

| Component | Parameters | Role |
|---|---|---|
| Context encoder | 3.5M | resolves readings, once per sentence |
| Acoustic model | 56.0M | RVQ language model at 12.5 Hz |
| Mimi decoder (frozen) | ~25M | codes to 24 kHz audio |
| **Total deployed** | **~85M** | CPU target |

Ten seconds of speech is 125 frames at Mimi's 12.5 Hz. That short sequence is
why CPU generation is fast.

---

## Quick start

```bash
git clone https://github.com/MohammedAly22/AdapTTS.git
cd AdapTTS

conda create -n adaptts python=3.11 -y
conda activate adaptts
pip install -r requirements.txt
python -m ipykernel install --user --name adaptts

python tests/test_egyptian.py     # text normalization
python tests/test_models.py       # causality, KV cache, masking
python tests/test_integration.py  # the real pipeline, synthetic data
```

**Start here:** [PIPELINE.md](PIPELINE.md) is the current end-to-end guide. It
explains what the first 150k-step run got wrong, what changed, and the cheap
checkpoint that tests the whole hypothesis for about fifty cents.

**Renting a GPU?** [RUNPOD.md](RUNPOD.md) covers the workflow, what to run in a
terminal versus a notebook, expected training milestones, and the cheap
checkpoint that tests the central hypothesis for under a dollar. Run
`python scripts/preflight_check.py --config <your-config>` first; it verifies
every known failure mode before you spend anything.

Every stage was run end to end on a GTX 1660 Ti against the real dataset.
[VERIFICATION.md](VERIFICATION.md) records the measured timings, what the run
proved, what it did not, and the six bugs it exposed.

### Two experiments, two configs

| Config | Dataset | Size | Purpose |
|---|---|---|---|
| `configs/exp0_small.yaml` | `OmarAhmedSobhy/tts-egyption-dataset` | 2013 clips, 2 GB | prove the pipeline works, on a consumer GPU |
| `configs/exp1_egyptian.yaml` | `ehabnegm/100-hour-Egyptian-dataset-single-speaker` | 15.6k clips, 68 h | the real run |

Start with **exp0**. It is sized for a 6 GB card and answers the only question
that matters early: does discovery find real homographs, and does the model
produce audio?

### Notebooks

| Notebook | Does | exp0 time | exp1 time |
|---|---|---|---|
| `00_setup.ipynb` | install, check GPU, run tests, cache models | 5 min | 5 min |
| `01_prepare_data.ipynb` | align, embed, **discover codes**, encode | ~25 min | ~1.8 h |
| `02_train_context.ipynb` | train the disambiguator | ~5 min | ~20 min |
| `03_train_acoustic.ipynb` | train the acoustic model | ~1.5 h | 7-9 h |
| `04_inference.ipynb` | CPU benchmark, control, export | 10 min | 10 min |

---

## Using it

```python
from adaptts.infer.pipeline import AdapTTS

tts = AdapTTS.from_checkpoints("configs/exp0_small.yaml", device="cpu")

plan = tts.analyze("انا شوفت علم مصر بيرفرف")
print(plan)
```

```
text: انا شوفت علم مصر بيرفرف
sentence difficulty: 0.184   depth: 5

  #  word             readings  chosen  confidence  difficulty
----------------------------------------------------------------
  2  علم                     2       0       0.947       0.118
  3  مصر                     2       0       0.981       0.061
```

Correct a reading by setting the code. **No diacritics are ever typed:**

```python
plan.set_code("علم", 1)                  # every occurrence
plan.set_code("مصر", 0, occurrence=0)    # only the first one
plan.set_depth(12)                       # force full depth

wav, stats = tts.synthesize(plan)
```

---

## Text normalization

A separate, self-testing module turns anything a transcript can contain into
speakable Egyptian Arabic:

```bash
python src/adaptts/text/egyptian.py     # runs its own test suite
```

It handles numbers, dates, times, currency, percentages, phone numbers, URLs,
emails, abbreviations and Latin tokens. The number rule is Egyptian, not
Modern Standard: **no linking waw between magnitude groups.**

| Input | Egyptian (correct) | MSA (wrong here) |
|---|---|---|
| `2024` | الفين اربعة و عشرين | ~~الفين **و** اربعة و عشرين~~ |
| `876` | تمنمية ستة و سبعين | ~~تمنمية **و** ستة و سبعين~~ |
| `1986` | الف تسعمية ستة و تمانين | ~~الف **و** تسعمية و ستة و تمانين~~ |

The waw belongs to the tens group alone, never between groups. A test asserts
this across every number from 0 to 10000.

Phone numbers are grouped the way people actually say them, not spelled out
digit by digit. The 01X prefix is read as a number, and a fourth digit of zero
is absorbed into it:

| Input | Spoken |
|---|---|
| `01027756313` | زيرو عشرة، اتنين سبعة سبعة، خمسة ستة، تلاتة واحد تلاتة |
| `01116953882` | زيرو حداشر، واحد ستة تسعة، خمسة تلاتة، تمانية تمانية اتنين |
| `01002776313` | زيرو مية، اتنين سبعة سبعة، ستة تلاتة، واحد تلاتة |

Other Egyptian conventions the module follows: hours are plain cardinals, so
7:45 is تمانية الا ربع rather than the Modern Standard الساعة الثامنة الا ربع;
the separator in an address is the borrowed دوت, not نقطة; and a Latin name in
an email is transliterated, so `ahmed@gmail.com` reads احمد ات جيميل دوت كوم
rather than being spelled out letter by letter.

---

## Roadmap

```
  ✅ EXPERIMENT 0   small corpus, prove the pipeline
     2013 clips · consumer GPU · real audio out
                    │
                    ▼
  ▶  EXPERIMENT 1   Egyptian Arabic, single speaker
     68 h · homograph accuracy is the metric
                    │
                    ▼
     EXPERIMENT 2   multi-speaker, zero-shot cloning
     YouTube podcasts · the speaker vector is already a slot
     in the acoustic model, so this is a fine-tune
                    │
                    ▼
     EXPERIMENT 3   English + Arabic, code-switching
     the character vocabulary already covers Latin script
                    │
                    ▼
     EXPERIMENT 4   language adapters
     add languages without disturbing existing pronunciations
```

Each step reuses the previous model. Nothing in the interface changes between
them, which is the point of putting the pronunciation decision in a discrete
slot rather than in the weights.

---

## Repository layout

```
ARCHITECTURE.md              the design and the reasoning behind it
PIPELINE.md                  current end-to-end guide, and what the first run taught
VERIFICATION.md              measured results from a real end-to-end run
RUNPOD.md                    rented-GPU workflow, milestones, costs
configs/
  base.yaml                  every tunable, documented
  exp0_small.yaml            small dataset, consumer GPU
  exp1_egyptian.yaml         the 68-hour run
  exp1_smoke.yaml            tiny config for a fast shakedown
src/adaptts/
  text/egyptian.py           Egyptian normalization (self-testing)
  text/normalize.py          Unicode cleanup, word tokenization
  data/ctc_aligner.py        CTC forced alignment, no lexicon
  data/discovery.py          pronunciation-code discovery ★
  data/preprocess.py         offline stages, parquet + audiofolder
  data/dataset.py            memmap dataset, length bucketing
  models/context_encoder.py  the homograph disambiguator
  models/acoustic.py         adaptive-depth RVQ model
  modules/transformer.py     attention, RoPE, KV cache, SwiGLU
  infer/pipeline.py          analyze / override / synthesize
  utils/logging_utils.py     tqdm progress, aligned log tables
scripts/                     preprocess, train, build notebooks
tests/                       52 tests, all passing
```

---

## What is deliberately not here

- No handwritten homograph list, no pronunciation lexicon, no G2P rules, no
  diacritization rules. Every pronunciation distinction is discovered from
  audio.
- No per-token adaptive halting. It destroys batching and the KV-cache layout,
  and on CPU the control overhead exceeds the compute saved.
- No external aligner binary. Montreal Forced Aligner needs a lexicon, and a
  lexicon is the rule table this project rejects.

---

## Tests

```bash
python tests/test_egyptian.py      # 18 - normalization, the waw rule
python tests/test_models.py        # 20 - causality, KV cache, masking
python tests/test_data.py          #  7 - collation, bucketing, config
python tests/test_end_to_end.py    #  6 - learning check
python tests/test_integration.py   #  1 - full pipeline, synthetic data
```

`test_end_to_end.py` is the one that matters most: it builds a corpus where
context fully determines the reading, and asserts the model resolves held-out
sentences correctly. If that regresses, the central claim is broken.
