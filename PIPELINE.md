# The pipeline, end to end

What changed after the first 150k-step run, why, and how to run the corrected
version. Read [VERIFICATION.md](VERIFICATION.md) for the earlier findings and
[ARCHITECTURE.md](ARCHITECTURE.md) for the design.

---

## What the failed run taught

Three separate failures, each now fixed and each guarded by a test.

### 1. The labels were wrong

Unsupervised clustering of acoustic spans returned the channel's subscribe
pitch as "homographs":

```
الجرس  التعليقات  لايك  الوصف  الرابط  البلاي
```

Those words appear in two acoustic registers, the scripted promo and ordinary
narration, and the clustering found that. Meanwhile `علم`, `مصر` and `دول` each
got one code, so the model was trained on labels that encoded *reading speed*
rather than pronunciation.

**Fixed** by taking labels from a diacritizer instead. Verified on real corpus
text: `علم` separates into عَلَم and عِلْم, `مصر` into مَصْر and مُصِرّ, while
`الجرس` and `لايك` correctly get one reading each.

### 2. Inference was ~10x slower than it needed to be

Profiling showed the per-level depth transformer was 72-80% of CPU time, at
21 ms per frame against 6.3 ms for the entire 12-layer backbone. It made 32
module calls per frame.

The cost was **dispatch, not arithmetic**: one block on a single token needs
67 µs of matmul and took 453 µs through PyTorch, and a batch of 8 cost barely
more than a batch of 1.

**Fixed** by predicting all RVQ levels from the backbone state in one shot,
keeping coarse-to-fine conditioning through a per-level embedding. Measured
4.8x faster to generate, 7.6x faster in the training forward, half the
parameters. Real-time factor went from about 1.0 to **0.11**.

### 3. Three of five hours were spent overfitting

The run ended with `ce_exit0` = 4.14 beating `ce_exit2` = 4.41. Every exit is
distilled from the deepest, so a shallow exit winning means the model had more
capacity than the data supports. Eval loss had turned upward around step 100k
and training continued to 150k anyway.

**Fixed** three ways: the model is now 27M rather than 56M, training stops when
eval loss stalls, and an alarm fires when a shallow exit beats the deepest.
Replayed against the real loss curve, the stopper halts at step 14000.

---

## Running it

### 0. Setup and preflight

```bash
cd /workspace
git clone https://github.com/MohammedAly22/AdapTTS.git && cd AdapTTS
pip install -r requirements.txt
python scripts/preflight_check.py --config configs/exp1_egyptian.yaml
```

### 1. Manifest

```bash
python scripts/preprocess.py --config configs/exp1_egyptian.yaml --stage manifest
```

Expect **15650 clips, 71.8 hours**. If you see 14.4 hours, `audio.max_duration`
is below the corpus median again; the manifest now warns loudly when filters
discard more than 10%.

### 2. Diacritize  ← new stage

This needs the CATT environment, which has its own torch pin, so run it with
that interpreter:

```bash
python scripts/diacritize.py \
    --config configs/exp1_egyptian.yaml \
    --catt-root /path/to/Marwa-Code-Switching-TTS-V1.2
```

About 20 minutes at 13 sentences/s. Writes `cache/exp1/diacritized.jsonl`, and
nothing downstream needs CATT again.

### 3. Alignment and codec

```bash
python scripts/preprocess.py --config configs/exp1_egyptian.yaml --stage align
python scripts/preprocess.py --config configs/exp1_egyptian.yaml --stage codec
python scripts/preprocess.py --config configs/exp1_egyptian.yaml --stage teacher
```

Alignment gives word spans and the codec gives the RVQ targets. Span embeddings
are no longer needed for labels but the stage remains for analysis.

### 4. Readings  ← the decision point

```bash
python scripts/preprocess.py --config configs/exp1_egyptian.yaml --stage discover
head -40 cache/exp1/reading_report.txt
```

**Read this before training anything.** It costs minutes and tells you whether
the labels are sound. What to look for:

* Real homographs present: `علم`, `مصر`, `دول`, `عالم`, `كتب`.
* Promo words absent: no `الجرس`, `لايك`, `الوصف`, `التعليقات`.
* A few hundred ambiguous types, not thousands. Thousands means artifact
  suppression is not working.

If the report looks wrong, adjust `discovery.min_pattern_count` and rerun this
stage alone. `--force` does not cascade upstream.

### 5. Context encoder

```bash
python scripts/train_context.py --config configs/exp1_egyptian.yaml
```

About 20 minutes. Watch `eval/code_acc`, which is held-out homograph accuracy.
Above 0.90 means the disambiguator works, and this is the project's actual
deliverable.

Then verify on the probes:

```python
tts = AdapTTS.from_checkpoints(CONFIG, device="cpu", load_codec=False)
a = tts.analyze("انا شوفت علم مصر بيرفرف")            # flag
b = tts.analyze("علم الفيزيا من اهم العلوم البشرية")  # science
```

The two must give `علم` **different codes**. If they do not, stop here; the
acoustic model will inherit the failure.

### 6. Acoustic model

```bash
nohup python scripts/train_acoustic.py --config configs/exp1_egyptian.yaml \
  > acoustic.log 2>&1 &
tail -f acoustic.log
```

Now early-stopped, so it ends when it stops improving rather than at a fixed
step count. Expect roughly 4 to 8 hours.

**Abort if you see the overfit warning**, which means a shallow exit is beating
the deepest. Reduce `acoustic.d_model` or `n_layers` and restart.

### 7. Generate

```bash
python scripts/smoke_generate.py --config configs/exp1_egyptian.yaml --device cpu
```

Expect real-time factor near **0.1**, roughly 10x faster than real time.

---

## Cost

| Stage | Time | Cost |
|---|---|---|
| Manifest | 1 min | — |
| Diacritize | 20 min | ~$0.15 |
| Align + codec + teacher | ~45 min | ~$0.35 |
| **Readings** | **2 min** | **the decision point** |
| Context encoder | 20 min | ~$0.15 |
| Acoustic model | 4-8 h | $2-4 |

Everything up to and including the reading report costs about **$0.50** and
answers the question the project is about. Spend that first.
