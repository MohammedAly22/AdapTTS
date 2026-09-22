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

## 1. The central idea: Pronunciation Codes discovered from audio

The audio *contains* the answer. When the narrator says عَلَم, the acoustics
carry /ʕalam/; when they say عِلْم, they carry /ʕilm/. The label is not in the
text, it is in the waveform. We recover it.

**Pronunciation Code (PC)**: for each orthographic word type `w`, a small
discrete set of latent classes `{0 … K_w-1}`, where `K_w` is *discovered*, not
declared. Each class corresponds to one way that word is actually pronounced in
the corpus. For an unambiguous word `K_w = 1`. For علم the procedure should
recover `K_w` of about 3 or 4.

This gives a **discrete, inspectable, controllable** latent: exactly what the
brief asks for when it says the user must be able to see what the model thinks a
hard word means and correct it without typing diacritics.

### 1.1 How PCs are discovered (Stage A, offline, one pass over data)

```
for each word type w with corpus frequency >= min_freq:
    1. Force-align the corpus  -> time span (t0, t1) of every occurrence of w
    2. Extract a pronunciation-bearing acoustic embedding over (t0, t1)
    3. Cluster those embeddings, selecting k by a stability criterion
    4. k == 1  -> w is unambiguous
       k >  1  -> w is a homograph; each cluster is one Pronunciation Code
```

Three sub-problems, each solved without rules or lists:

**(a) Alignment without an aligner binary.** We do not ship Montreal Forced
Aligner, which is heavy and needs a pronunciation lexicon, and a lexicon is the
handwritten rule list this project rejects. Instead we run **CTC forced
alignment**: a Viterbi search over the CTC posterior lattice, implemented from
scratch in `data/ctc_aligner.py`.

The acoustic model is `MahmoudAshraf/mms-300m-1130-forced-aligner`. Two
practical details drove that choice:

* It ships **safetensors**. Transformers 5.x refuses to load `.bin`
  checkpoints unless torch is at least 2.6 (CVE-2025-32434), and most Arabic
  wav2vec2 checkpoints are `.bin` only.
* Its CTC vocabulary is **romanized**, 31 Latin tokens with no Arabic script.
  So `text/romanize.py` transliterates the transcript while recording, for
  every Latin character, which Arabic character produced it. Latin frame spans
  are mapped back through that index to Arabic character spans, and from there
  to words.

The romanization is a script mapping used only to find frame boundaries. It
makes no pronunciation decisions; those remain with the discovered codes.

One trap worth naming: on this model the CTC blank is `<blank>` at index 0
while `pad_token_id` is 1. Using the tokenizer's pad id as the blank produces
alignments that look plausible and are wrong, so the blank is detected
explicitly.

**Threshold calibration.** Measured on Egyptian conversational speech, the
median aligned word lasts 0.24 s and the 5th percentile is 0.06 s. A 0.08 s
minimum therefore discards real short words, and a romanized aligner scores
lower than a native-script one because transliteration is approximate. The
shipped thresholds (0.04 s, mean log-prob -6.5) keep 99.4% of spans against
88.3% for the stricter defaults. Discovery needs occurrences far more than it
needs a pristine tail.

**(b) A pronunciation-bearing embedding.** Raw mel is speaker and prosody
dominated. The right representation is a **self-supervised speech layer known to
encode phonetic identity**: wav2vec2/HuBERT middle layers. We mean-pool the
aligned span, then apply per-word standardization to strip the global
speaking-rate and energy axes. We also whiten against the sentence context so
the embedding captures *this word's* realization rather than the utterance mood.

**(c) Choosing k without declaring it.** For each word we run k-means for
k = 1..K_max and select k by a **bootstrap stability score** (Fowlkes-Mallows on
resampled fits) combined with a silhouette margin, then apply a **minimum
between-cluster distance gate** so that clusters differing only in prosody
collapse back to k = 1. A word is declared a homograph only if its split is both
reproducible and acoustically substantive.

This is a discovery procedure, not a rule table. Run on a different dialect or
language, it rediscovers that language's homographs.

### 1.2 Why this is the novel contribution

Existing Arabic TTS: **text -> diacritizer -> phonemes -> acoustics**, where the
diacritizer is a separate supervised model needing diacritized corpora.

AdapTTS: **text -> context encoder -> discrete pronunciation code -> acoustics**,
where the codes are self-discovered from audio and the context encoder is
distilled from a large language model but deployed as a tiny one.

The pronunciation code is a *bottleneck*. It is about 3 bits wide, not 768
dimensions. The model cannot smear the decision, it must commit, and we can read
the commitment and override it.

---

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
