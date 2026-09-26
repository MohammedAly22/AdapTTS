# AdapTTS — Adaptive Context-Aware TTS for Egyptian Arabic

## 0. The core problem, stated precisely

Egyptian Arabic is written **undiacritized**. One grapheme string maps to many
pronunciations:

```
علم  ->  عَلَم (flag)  |  عِلْم (science)  |  عَلَّم (taught)  |  عُلِم (was known)
مصر  ->  مَصْر (Egypt)  |  مُصِرّ (insisting)
دول  ->  دُول (these, colloquial)  |  دِوَل (countries)
```

Standard TTS pipelines resolve this by requiring diacritized transcripts, or by
bolting on a rule-based diacritizer. Both are rejected by the project brief:
no handwritten rules, no curated homograph lists.

### 0.1 Why naive "just add BERT" does not work

The tempting design is: run MARBERTv2 over the sentence, concatenate the
contextual vector to the character embedding, train TTS end-to-end, hope the
model figures it out. This **fails silently**, and it is worth being explicit
about why, because the entire architecture is a response to it.

1. **The gradient is too weak.** The homograph is about 1 word in 30. Its
   acoustic realization is about 0.4 s in a 10 s clip. Codec reconstruction loss
   is dominated by the other 96% of the signal. The disambiguation signal is a
   rounding error in the loss, so the model learns a *prior*, always producing
   the frequent reading, and the rare reading is never recovered.
2. **No supervision exists.** The transcript is undiacritized. There is no label
   saying "in this clip, علم was عَلَم". So there is nothing to attach a
   discriminative loss to.
3. **Frozen BERT embeddings are not pronunciation-aligned.** MARBERTv2 separates
   *meaning*. Meaning and pronunciation correlate but are not the same axis.
   عَلَم (flag) and راية (banner) are near neighbours in meaning space and
   maximally distant in pronunciation space.

So the real problem is **absent supervision**, not model capacity. Everything
below exists to manufacture that supervision from the data itself.

---

## 1. The central idea: Pronunciation Codes

**Pronunciation Code (PC)**: for each orthographic word type `w`, a small
discrete set of classes `{0 … K_w-1}`, one per way the word is actually voiced
in the corpus. `K_w = 1` means unambiguous. For `علم` it is 2 or more.

This gives a **discrete, inspectable, controllable** latent: the user can see
which reading the model chose and change it without typing diacritics.

### 1.1 How the codes are labelled

Labels come from a **diacritizer run once over the transcripts**. For every
word occurrence the diacritizer states the vowels; word types that take more
than one distinct vowel pattern across the corpus are the homographs, and each
occurrence is labelled by its own pattern.

The shipped model never sees a diacritic. They label training data exactly as
the forced aligner finds spans, and neither runs at inference.

### 1.2 Why not discover the codes from audio

The first version of this system clustered self-supervised acoustic embeddings
of each word occurrence and treated the clusters as readings. It was gated
three ways: bootstrap stability, silhouette margin, and centroid separation.

**On 68 hours of real Egyptian narration it failed, and it failed in a way
that looked like success.** The top "discovered homographs" were:

    الجرس  التعليقات  لايك  الوصف  الرابط  البلاي  اكتبوه

Those are the channel's subscribe pitch. They occur in exactly two acoustic
contexts, the scripted promo read in a fixed fast cadence and ordinary
narration, and the clustering found that split. Meanwhile `علم`, `مصر` and
`دول` each received a single code.

The cause is structural, not a matter of tuning. A mean-pooled SSL embedding
over a word span encodes speaking rate, energy, pitch and recording session far
more strongly than vowel identity. The three gates filter *noise*; a confound
that is reproducible and well separated passes all of them. A fourth gate for
duration confound helped and was not enough, because register differs in more
than duration.

Diacritics observe the vowels directly instead of through a proxy that is
dominated by something else.

### 1.3 The diacritizer is not treated as ground truth

It is noisy, so a pattern becomes a reading only when it clears three bars:

* seen at least `min_pattern_count` times, since one odd diacritization is an
  error rather than a homograph;
* holding at least `min_pattern_frac` of that word's occurrences;
* surviving artifact suppression.

That last one matters most. Measured on 3000 real sentences, raw patterns gave
912 "ambiguous" word types, the large majority differing only by marks that
cannot encode a pronunciation:

| Word | Two forms | What differs |
|---|---|---|
| `في` | فِيْ / فْيْ | sukun on the initial letter |
| `قال` | قَالْ / قَاْلْ | sukun on a long vowel |
| `الناس` | النَّاسْ / اْلنَّاسْ | mark inside the article |

Each is a phonological impossibility: a word cannot begin vowelless, a long
vowel cannot also be vowelless, and `ال` is invariant. Suppressing them cut the
count from 912 to **145** while every genuine homograph survived. Case endings
are also ignored, since Egyptian speech drops them and keeping them would make
a homograph of every noun.

### 1.4 What this is verified to do

Measured with CATT-ECA on real corpus sentences:

| Word | Result |
|---|---|
| `علم` | عَلَم (flag) vs عِلْم (science) |
| `مصر` | مَصْر (Egypt) vs مُصِرّ (insisting) |
| `الدول` | إِلدِّوَل (countries) |
| `عالم` | عَالَم (world) vs عَالِم (scholar) |
| `الجرس`, `لايك`, `الوصف` | one reading each, correctly |

Token alignment between plain and diacritized text is 100% on that sample, and
the pass runs at 13 sentences/s, about 20 minutes for the whole corpus.

## 2. System architecture

```
                   ┌─────────── OFFLINE (once per dataset) ───────────┐
                   │                                                  │
audio + text  ────►│  CTC forced alignment  (wav2vec2-MMS-ar)          │
                   │            ↓                                     │
                   │  word-span SSL embeddings (wav2vec2 layer 6-9)    │
                   │            ↓                                     │
                   │  per-word-type stability clustering               │
                   │            ↓                                     │
                   │  pronunciation_codes.json  (w -> K_w, centroids)  │
                   │  + per-occurrence PC labels  (the SUPERVISION)    │
                   │                                                  │
                   │  MARBERTv2 frozen forward pass                    │
                   │            ↓                                     │
                   │  cached contextual embeddings (fp16 memmap)       │
                   │            ↓                                     │
                   │  Mimi encode -> audio codes (8 RVQ x 12.5 Hz)     │
                   └──────────────────────────────────────────────────┘
                                       │
                                       ▼
 ┌──────────────────────── TRAINING (2 stages) ───────────────────────────┐
 │                                                                        │
 │  Stage B: Context Encoder (student)                                    │
 │     chars -> tiny transformer -> per-word PC logits                    │
 │     losses: CE on discovered PC labels                                 │
 │           + KL distillation from MARBERTv2 teacher head                │
 │           + entropy-based difficulty head (the adaptive signal)        │
 │     about 3.5M params. Runs standalone on CPU in a few ms.             │
 │                                                                        │
 │  Stage C: Acoustic Model                                               │
 │     (chars + PC embeddings + speaker) -> Mimi RVQ codes                │
 │     backbone: depth-adaptive transformer with early exit               │
 │     depth transformer over the 8 RVQ levels (Pocket-TTS style)         │
 │     about 56M params.                                                  │
 └────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
 ┌──────────────────────── INFERENCE (CPU) ───────────────────────────────┐
 │  text -> context encoder -> PC per word + difficulty                   │
 │       -> [USER MAY INSPECT AND OVERRIDE HERE]                          │
 │       -> acoustic model (depth chosen by difficulty)                   │
 │       -> Mimi decoder -> 24 kHz waveform                               │
 └────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Why this decomposition is the right one

The **critical design decision** is that the context encoder is a separate,
independently supervised model, not an end-to-end branch. Consequences:

- The disambiguation gradient no longer competes with reconstruction loss. It
  has its own dedicated cross-entropy on a balanced task.
- It trains in minutes, so it can be iterated fast.
- It is inspectable. `encoder.explain(sentence)` returns per-word codes,
  probabilities and difficulty. This is the controllability requirement.
- It is overridable. The acoustic model consumes a *code*, so substituting a
  different code is a legal, in-distribution edit. No diacritics needed.
- It is replaceable per language. Adding English later means training a second
  context encoder head; the acoustic model interface does not change. This is
  the "add languages without affecting other pronunciation" requirement.

### 2.2 The teacher and student split

The teacher is a frozen Arabic BERT, about 135M params: too heavy for CPU-first
inference at our budget, but we only need it **at training time**:

- **Teacher**: frozen, cached once to disk as fp16, and never run again.
  The default is `aubmindlab/bert-base-arabertv02-twitter`, chosen because it
  is trained on dialectal Arabic and ships safetensors. MARBERTv2 is the
  stronger Egyptian encoder but is distributed as `.bin` only, which
  transformers 5.x refuses without torch 2.6 or newer. Both are 768-dim
  12-layer BERTs, so swapping back is a one-line config change.
- **Student**: a 3.5M-param char-level transformer, trained to match the
  teacher PC posterior via KL divergence plus hard CE on discovered labels.

Distillation transfers the semantic discrimination into a model far smaller.
Caching means the teacher costs one forward pass over the corpus, roughly 8
minutes on a 4090, not one pass per training step. This is the main reason
training is cheap.

---

## 3. Adaptive depth — spending compute where it is needed

The brief's bar chart is implemented literally.

### 3.1 Difficulty signal

The context encoder emits per-word difficulty

```
d_i = H(p_i) / log(K_{w_i})        normalized posterior entropy, in [0,1]
```

with `d_i = 0` when `K_w = 1`, so a word with one pronunciation is free. The
sentence difficulty is a soft-max pool over words, so one hard word in an easy
sentence still raises the budget.

This is not a heuristic bolted on. It is the calibrated uncertainty of a head
trained with cross-entropy, so it is a genuine probability statement.

### 3.2 Early-exit backbone

The acoustic backbone has `L` layers with exit heads at a configurable subset,
by default after layers 4, 8 and 12. Training uses **layer dropout with depth
consistency distillation**: the deep exit teaches the shallow exits, so every
exit is a valid model and the shallow path never degrades to garbage. This is
what guarantees no hallucination at low depth. A shallow exit is a distilled
model, not a truncated one.

At inference, depth is selected by difficulty:

```
d <= tau_lo            -> exit at layer 4   (easy: about 35% of full compute)
tau_lo < d <= tau_hi   -> exit at layer 8   (about 65%)
d > tau_hi             -> full depth        (100%)
```

Thresholds live in config and are tunable at inference with no retraining. There
is also a `force_depth` override for A/B comparison.

### 3.3 Halting is per-sentence, not per-token

We deliberately do **not** do per-token adaptive halting of the PonderNet or ACT
kind. Per-token halting destroys batching and KV-cache layout, and on CPU the
control overhead exceeds the compute saved. Per-sentence depth selection keeps
the inner loop a dense, static-shape matmul, which is what actually runs fast on
CPU. The measured compute saving is real rather than theoretical.

---

## 4. Acoustic model — why an RVQ language model

Requirements: CPU-fast, streaming, small, zero-shot cloning, no hallucination.

| Option | Verdict |
|---|---|
| Mel + diffusion/flow (F5, E2) | 30+ function evaluations, poor CPU latency, no streaming |
| Mel + GAN vocoder (VITS style) | Fast, but cloning is weak and mel is speaker-entangled |
| **RVQ code LM + Mimi decoder** | Streaming, 12.5 Hz, strong cloning, proven on CPU |

Chosen: **RVQ code language model over Mimi tokens**, the Pocket-TTS and Moshi
family. Mimi at 12.5 Hz means a 10 s utterance is 125 frames, an extremely short
sequence, which is precisely why CPU inference is fast.

### 4.1 Two-transformer structure

Following the Moshi and Pocket-TTS decomposition:

- **Temporal backbone** (`d=384`, `L=12`): one step per 12.5 Hz frame. Carries
  long-range context. This is where early exit applies. Twelve layers is not
  arbitrary: three exit depths need enough layers that each is a meaningfully
  different amount of compute.
- **Depth transformer** (`d=256`, `L=4`): runs 8 micro-steps inside each frame,
  one per RVQ level, predicting level `q` conditioned on levels below `q`. Tiny
  and cheap, but it is what makes the RVQ levels mutually consistent. Modelling
  them independently is the most common cause of artefacts.

Measured parameter counts for the shipped config:

| Component | Parameters | Runs on |
|---|---|---|
| Context encoder (deployed) | 3.5M | CPU, once per sentence |
| Acoustic model | 56.0M | CPU, autoregressive |
| Mimi decoder (frozen) | about 25M | CPU, once per utterance |
| **Total deployed** | **about 85M** | |

That is comparable to Pocket TTS at 100M. The backbone was deliberately sized
down from `d=512` (which reaches 84M on its own) because 68 hours of
single-speaker audio does not support that capacity; the larger model overfits
rather than improving.

The context encoder also carries a `teacher_proj` layer used only for
representation distillation during training. It is not needed at inference and
is excluded from the deployed count above.

### 4.2 Anti-hallucination measures

Autoregressive TTS fails by looping or terminating early. Four mechanisms:

1. **Monotonic alignment prior.** A learned bias added to text cross-attention
   that penalizes non-monotonic jumps. Cheap and effective.
2. **Duration-anchored length control.** A duration predictor gives an expected
   frame count; sampling is bounded to 0.6x to 1.6x that estimate.
3. **Classifier-free guidance on the text condition** with a trained
   unconditional branch, so the model can be pushed toward the transcript.
4. **Repetition guard** on the coarse RVQ level: an n-gram loop detector that
   forces resampling. Bounded and deterministic, with no silent failure.

### 4.3 Zero-shot voice cloning

Speaker conditioning is a 192-dim vector prepended as a prefix token. In stage 1
with a single speaker it is a learned constant. In stage 2 it is produced by a
small speaker encoder over the reference clip. **The acoustic model interface
does not change between stages**, only what fills the slot. This is why stage 2
is a fine-tune rather than a rewrite.

---

## 5. Text frontend — characters, no G2P

No phonemizer, no lexicon, no rules. The unit is the **Arabic character**, with
a vocabulary built from the corpus. Characters are the right unit here because:

- Arabic orthography is shallow *given* the vowel decision, and the vowel
  decision is exactly what the PC token supplies.
- Code-switching becomes trivial: Latin characters share the same vocabulary, so
  stage 3 needs no new tokenizer.
- It is robust to the out-of-vocabulary words a G2P would choke on.

Word boundaries are tracked so PC embeddings can be broadcast onto the
characters of their word. Normalization is Unicode NFC plus diacritic stripping,
so that if a diacritized transcript ever appears it is harmlessly folded. This is
character-class normalization, not a pronunciation rule.

---

## 6. Training economics

Target: **under 12 GPU-hours total on a single 4090**, roughly 6 to 10 dollars
on RunPod.

| Stage | What | Time (4090) |
|---|---|---|
| A1 | CTC forced alignment over 68 h | about 35 min |
| A2 | SSL span embeddings | about 25 min |
| A3 | PC clustering (CPU) | about 8 min |
| A4 | MARBERTv2 teacher cache | about 8 min |
| A5 | Mimi encode to RVQ codes | about 30 min |
| B | Context encoder | about 20 min |
| C | Acoustic model | about 7 to 9 h |
| **Total** | | **about 10 h** |

Efficiency measures, all implemented:

- **Everything cached to memmap.** No audio decoding, no BERT and no Mimi encode
  inside the training loop. The dataloader reads fp16 and int16 memmaps, so the
  GPU stays fed.
- **bf16 autocast and `torch.compile`** on the backbone.
- **SDPA / flash attention** via `scaled_dot_product_attention`.
- **Length-bucketed batch sampler** so padding waste stays under about 8%.
- **Fused AdamW**, gradient accumulation, cosine schedule with warmup.

---

## 7. Experiment sequence

- **Exp 1 (now)**: single-speaker Egyptian Arabic, homograph correctness.
  Metric: PC accuracy on held-out homograph occurrences, plus a listening set of
  8 minimal-pair sentences rendered to TensorBoard every N steps.
- **Exp 2**: multi-speaker from YouTube podcasts, speaker encoder enabled,
  zero-shot cloning.
- **Exp 3**: add English, shared character vocabulary, code-switching.
- **Exp 4**: language adapters so new languages do not disturb existing ones.

Each experiment is one YAML file in `configs/`. Changing the dataset path and
the config path is the only edit required.

---

## 8. What is deliberately NOT in this system

- No handwritten homograph list, no diacritization rules, no pronunciation
  lexicon, no G2P rules. Every pronunciation distinction is discovered.
- No per-token adaptive halting, since it harms CPU throughput (see 3.3).
- No external aligner binary. CTC alignment is implemented in-repo.
- No cloud embedding API in the training path. Gemini embeddings were considered
  and rejected: per-sentence API calls over 15k utterances add cost and latency
  for a signal MARBERTv2 already provides locally, and they cannot be distilled
  offline as cheaply.
