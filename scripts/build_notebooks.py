"""Generate the RunPod notebooks from source.

Keeping notebooks generated from a plain-Python source file means they stay
diff-able and cannot drift into an unparseable state. Run this after changing
any notebook content:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": s.strip("\n").splitlines(keepends=True),
    }


def nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


BOOT = """
import os, sys
REPO = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ["PYTHONIOENCODING"] = "utf-8"

# ---------------------------------------------------------------------------
# PICK YOUR EXPERIMENT HERE. This is the only line to change.
#
#   configs/exp0_small.yaml     2013 clips, ~2 GB   -> proves the pipeline,
#                                                     runs on a 6 GB GPU
#   configs/exp1_egyptian.yaml  15.6k clips, 68 h   -> the real run
# ---------------------------------------------------------------------------
CONFIG = "configs/exp0_small.yaml"

from adaptts.utils.config import load_config
from adaptts.utils.logging_utils import setup_logging
setup_logging()
cfg = load_config(CONFIG)
print("repo   :", REPO)
print("config :", CONFIG, "->", cfg.name)
print("dataset:", cfg.paths.hf_dataset_id)

# The homographs named in the brief, read from the probe file so the notebooks
# never hardcode a word list of their own.
import json as _json
PROBE_WORDS = sorted({
    w for _s in _json.load(open("assets/probe_sentences.json", encoding="utf-8"))["sentences"]
    for w in _s["focus"].split(" / ") if w and w != "none"
})
print("probe  :", " ".join(PROBE_WORDS))
"""


# ---------------------------------------------------------------------------
# 00 - environment setup
# ---------------------------------------------------------------------------

NB00 = [
    md("""
# 00 - Environment setup

Run this once per machine or pod. It checks the GPU, verifies the environment,
runs the test suite, and pre-downloads the pretrained models so later notebooks
never stall mid-run.

## Creating the environment (shell, once)

```bash
conda create -n adaptts python=3.11 -y
conda activate adaptts

# CUDA build of PyTorch. cu121 works on Turing (GTX 16xx) through Ada (4090).
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
python -m ipykernel install --user --name adaptts --display-name "AdapTTS (conda)"
```

Then pick the **AdapTTS (conda)** kernel in Jupyter before running anything
below. The first cell prints which interpreter you are actually on, so a
wrong-kernel mistake shows up immediately rather than as a confusing import
error later.

Expected: about 5 minutes plus roughly 4 GB of model downloads.
"""),
    code("!nvidia-smi"),
    code("""
import sys, os

print("interpreter:", sys.executable)
print("python     :", sys.version.split()[0])
in_conda = "adaptts" in sys.executable.lower() or os.environ.get("CONDA_DEFAULT_ENV") == "adaptts"
print("env        :", os.environ.get("CONDA_DEFAULT_ENV", "(none)"))
if not in_conda:
    print()
    print("WARNING: this does not look like the adaptts environment.")
    print("Select the 'AdapTTS (conda)' kernel from the kernel picker.")
"""),
    code("""
import os, sys
REPO = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ["PYTHONIOENCODING"] = "utf-8"
print("repo:", REPO)
"""),
    md("""
## Verify dependencies

If anything is missing here, run the pip commands from the shell block above in
a terminal, then restart the kernel.
"""),
    code("""
import importlib

required = [
    ("torch", "torch"), ("torchaudio", "torchaudio"), ("transformers", "transformers"),
    ("numpy", "numpy"), ("scipy", "scipy"), ("soundfile", "soundfile"),
    ("pyarrow", "pyarrow"), ("yaml", "pyyaml"), ("tqdm", "tqdm"),
    ("tensorboard", "tensorboard"), ("huggingface_hub", "huggingface_hub"),
]
missing = []
for mod, pkg in required:
    try:
        m = importlib.import_module(mod)
        print(f"  ok      {pkg:<18} {getattr(m, '__version__', '')}")
    except ImportError:
        missing.append(pkg)
        print(f"  MISSING {pkg}")
if missing:
    raise SystemExit("install these first: pip install " + " ".join(missing))
"""),
    code("""
import torch

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print()
    print("No GPU visible. Training will run, but very slowly.")
else:
    p = torch.cuda.get_device_properties(0)
    cc = p.major + p.minor / 10
    print(f"gpu     : {p.name}")
    print(f"memory  : {p.total_memory / 1024 ** 3:.1f} GB")
    print(f"compute : {p.major}.{p.minor}")
    print()
    # torch reports is_bf16_supported() True on Turing, but that is emulation,
    # not hardware. Only Ampere and newer have bf16 tensor cores.
    if cc >= 8.0:
        print("Ampere or newer: use train.precision = bf16 (no gradient scaler needed).")
        print("  -> configs/exp1_egyptian.yaml is already set up this way.")
    else:
        print("Turing or older: NO hardware bf16, despite what torch reports.")
        print("Use train.precision = fp16 with a gradient scaler.")
        print("  -> configs/exp0_small.yaml is already set up this way.")
    if p.total_memory / 1024 ** 3 < 10:
        print()
        print("Under 10 GB: start with configs/exp0_small.yaml.")
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    print()
    print("GPU matmul check:", bool(torch.isfinite((x @ x).sum())))
"""),
    md("""
## Run the test suite

These check causality, KV-cache equivalence, collation alignment and, most
importantly, that the model actually learns homograph disambiguation on a
controlled corpus. All must pass before you spend GPU time.
"""),
    code("""
import subprocess, sys, os

tests = [
    "tests/test_egyptian.py",     # text normalization, the waw rule
    "tests/test_models.py",       # causality, KV cache, masking
    "tests/test_data.py",         # collation, bucketing, config
    "tests/test_end_to_end.py",   # does it actually learn to disambiguate
    "tests/test_integration.py",  # the real pipeline on synthetic data
]
env = dict(os.environ, PYTHONIOENCODING="utf-8")
for t in tests:
    print()
    print("=" * 60)
    print(t)
    print("=" * 60)
    r = subprocess.run([sys.executable, t], capture_output=True, text=True, env=env)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
        raise SystemExit(t + " FAILED - fix this before continuing")
print()
print("All tests passed. The code is ready to train.")
"""),
    md("""
## Check the Egyptian text normalizer

This runs its own suite. The headline rule: no linking waw between magnitude
groups, so 2024 is "الفين اربعة و عشرين", never "الفين و اربعة و عشرين".
"""),
    code("!python src/adaptts/text/egyptian.py"),
    md("""
## Pre-download the pretrained models
"""),
    code("""
import yaml
from transformers import (
    AutoFeatureExtractor, AutoModel, AutoModelForCTC, AutoProcessor, AutoTokenizer, MimiModel,
)

cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))

print("1/4 CTC aligner")
AutoProcessor.from_pretrained(cfg["align"]["model_id"])
AutoModelForCTC.from_pretrained(cfg["align"]["model_id"])

print("2/4 SSL span encoder")
AutoFeatureExtractor.from_pretrained(cfg["spanemb"]["model_id"])
AutoModel.from_pretrained(cfg["spanemb"]["model_id"])

print("3/4 MARBERTv2 teacher")
AutoTokenizer.from_pretrained(cfg["teacher"]["model_id"])
AutoModel.from_pretrained(cfg["teacher"]["model_id"])

print("4/4 Mimi codec")
MimiModel.from_pretrained(cfg["codec_model_id"])

print()
print("all models cached")
"""),
    md("Setup is done. Continue to **01_prepare_data.ipynb**."),
]


# ---------------------------------------------------------------------------
# 01 - data preparation and pronunciation-code discovery
# ---------------------------------------------------------------------------

NB01 = [
    md("""
# 01 - Data preparation and pronunciation-code discovery

All the offline work happens here. Afterwards training reads only memmaps, so
the GPU never waits on data.

| Stage | What it does | Time on a 4090 |
|---|---|---|
| A0 | manifest: scan, normalize, filter | 1 min |
| A1 | CTC forced alignment to word spans | about 35 min |
| A2 | self-supervised embeddings per word span | about 25 min |
| A3 | **discover pronunciation codes** | about 8 min |
| A4 | MARBERTv2 teacher cache and head | about 8 min |
| A5 | Mimi encode to RVQ codes | about 30 min |

Stage A3 is the novel part: it decides from audio alone which words have more
than one pronunciation. Nothing is hardcoded.
"""),
    code(BOOT),
    md("""
## Download the dataset

About 12 GB. To use a different corpus, change `paths.hf_dataset_id` and
`paths.dataset_dir` in the config.
"""),
    code("""
from huggingface_hub import snapshot_download

target = cfg.paths.dataset_dir
print("downloading to:", target)
snapshot_download(
    repo_id=cfg.paths.hf_dataset_id, repo_type="dataset",
    local_dir=target, max_workers=8,
)
print("done")
"""),
    code("""
import glob, os

wavs = glob.glob(os.path.join(target, "clips", "**", "*.wav"), recursive=True)
parquet = glob.glob(os.path.join(target, "**", "*.parquet"), recursive=True)
meta = os.path.join(target, "metadata")
print("wav clips     :", len(wavs))
print("parquet shards:", len(parquet))
print("metadata dir  :", os.listdir(meta) if os.path.isdir(meta) else "none")
if parquet and not wavs:
    print()
    print("This is a parquet dataset. The manifest stage below unpacks the")
    print("embedded audio to wav once, so later stages never decode it again.")
"""),
    md("""
## Stage A0 - manifest

Normalizes every transcript once, so alignment, the teacher and training all see
byte-identical strings.
"""),
    code("!python scripts/preprocess.py --config $CONFIG --stage manifest"),
    code("""
import json, collections

rows = [json.loads(l) for l in open(cfg.paths.manifest_path, encoding="utf-8")]
hours = sum(r["duration"] for r in rows) / 3600
print(f"{len(rows)} utterances, {hours:.1f} hours")
print("splits:", collections.Counter(r["split"] for r in rows))
for r in rows[:3]:
    print(f'  [{r["duration"]:.1f}s] {r["text"][:70]}')
"""),
    md("""
### Check the text normalization

Every transcript passed through the Egyptian normalizer. Numbers, dates and
Latin tokens should all be spoken words by now, with no digits left.
"""),
    code("""
import json, random

rows = [json.loads(l) for l in open(cfg.paths.manifest_path, encoding="utf-8")]
random.seed(0)
for r in random.sample(rows, min(8, len(rows))):
    print(f"[{r['duration']:5.1f}s] {r['text'][:100]}")

leftover = [r for r in rows if any(c.isdigit() for c in r["text"])]
print()
print(f"utterances still containing digits: {len(leftover)} of {len(rows)}")
for r in leftover[:3]:
    print("   ", r["text"][:100])
"""),
    md("""
## Stage A1 - CTC forced alignment

Finds the time span of every word with no pronunciation lexicon. The Viterbi
alignment is implemented in-repo, so there is no Montreal Forced Aligner
dependency.
"""),
    code("!python scripts/preprocess.py --config $CONFIG --stage align"),
    md("""
## Stages A2 and A3 - span embeddings and code discovery

The heart of the system. Each occurrence of each word type is embedded with a
self-supervised speech model, then we ask whether those embeddings form one
cluster or several. A split is accepted only when it is reproducible under
bootstrap resampling, geometrically clean, and acoustically well separated.
"""),
    code("!python scripts/preprocess.py --config $CONFIG --stage discover"),
    md("""
### What did it discover?

This is the moment of truth. Words like the ones named in the brief should
appear with more than one reading.
"""),
    code("""
from adaptts.data.discovery import PronunciationLexicon

lex = PronunciationLexicon.load(cfg.paths.lexicon_path)
amb = lex.ambiguous_words
print(f"{len(amb)} ambiguous word types discovered")
print()

entries = sorted((lex.entries[w] for w in amb), key=lambda e: -e.occurrence_count)
header = f"{'word':<18}{'codes':>6}{'occ':>7}{'stab':>8}{'sil':>8}{'sep':>8}  counts"
print(header)
print("-" * 74)
for e in entries[:40]:
    print(f"{e.word:<18}{e.n_codes:>6}{e.occurrence_count:>7}{e.stability:>8.2f}"
          f"{e.silhouette:>8.2f}{e.separation:>8.2f}  {e.counts}")
"""),
    code("""
# Check the specific homographs named in the project brief.
for w in PROBE_WORDS:
    k = lex.n_codes(w)
    print(f"{w:<10} -> {k} code(s)   " + ("AMBIGUOUS" if k > 1 else "single reading"))
"""),
    code("""
# Read the sentences behind each code. This is how you confirm the clusters
# track meaning rather than recording conditions.
import json, collections, os

labels = json.load(open(os.path.join(cfg.paths.cache_dir, "code_labels.json"), encoding="utf-8"))
manifest = [json.loads(l) for l in open(cfg.paths.manifest_path, encoding="utf-8")]
by_uid = {r["uid"]: r["text"] for r in manifest}

WORD = PROBE_WORDS[0]      # change to inspect any discovered homograph
groups = collections.defaultdict(list)
for key, code in labels.items():
    uid, widx = key.split(chr(9))
    text = by_uid.get(uid, "")
    words = text.split()
    if int(widx) < len(words) and words[int(widx)] == WORD:
        groups[code].append(text)

for code in sorted(groups):
    print()
    print(f"=== {WORD}  code {code}  ({len(groups[code])} occurrences) ===")
    for t in groups[code][:6]:
        print("   ", t[:95])
"""),
    md("""
If the sentences under each code share a meaning, discovery worked. If they look
mixed, raise `discovery.min_separation` or `discovery.stability_threshold` in the
config and rerun this stage with `--force`.
"""),
    md("""
## Stage A4 - teacher cache

One frozen MARBERTv2 pass over the corpus, cached as fp16. The teacher never
runs again, which is the main reason training is cheap.
"""),
    code("!python scripts/preprocess.py --config $CONFIG --stage teacher"),
    md("""
## Stage A5 - Mimi codec encoding

Every clip becomes 8 RVQ streams at 12.5 Hz. A 10 second clip is 125 frames,
which is why generation is fast on a CPU.
"""),
    code("!python scripts/preprocess.py --config $CONFIG --stage codec"),
    md("## Verify the cache is complete"),
    code("""
from adaptts.data.dataset import AdapTTSDataset, collate
from adaptts.text.vocab import CharVocab

vocab = CharVocab.load(cfg.paths.charvocab_path)
ds = AdapTTSDataset(cfg, "train", vocab, lex, need_codes=True, need_teacher=True)
b = collate(
    [ds[i] for i in range(4)], vocab.pad_id, cfg.discovery.max_codes_per_word,
    cfg.audio.n_quantizers, cfg.teacher.hidden_size,
)
for k, v in b.items():
    print(f"  {k:<20} {tuple(v.shape)}  {v.dtype}")
print()
print("ambiguous words in this batch:", int((b["n_codes"] > 1).sum()))
ds.close()
print()
print("Data is ready. Continue to 02_train_context.ipynb")
"""),
]


# ---------------------------------------------------------------------------
# 02 - context encoder
# ---------------------------------------------------------------------------

NB02 = [
    md("""
# 02 - Train the context encoder (homograph disambiguation)

This is the cheap, high-value stage: about 20 minutes on a 4090. It is also
where homograph accuracy actually comes from, so iterate here before spending
money on the acoustic model.

The model is about 6M parameters. For every word it predicts which of that
word's discovered readings applies in this context.
"""),
    code(BOOT),
    md("""
## Start TensorBoard

Watch `eval/code_acc`. That is held-out homograph accuracy, the number that
matters. Above roughly 0.90 means the approach is working.
"""),
    code("""
%load_ext tensorboard
%tensorboard --logdir $cfg.paths.tb_dir --port 6006 --bind_all
"""),
    md("## Train"),
    code("!python scripts/train_context.py --config $CONFIG"),
    md("""
## Evaluate what it learned

The two `علم` sentences must get *different* codes. If they do not,
disambiguation is not working and the acoustic model will inherit the failure.
"""),
    code("""
import json
from adaptts.infer.pipeline import AdapTTS

tts = AdapTTS.from_checkpoints(CONFIG, device="cpu", load_codec=False)
probes = json.load(open("assets/probe_sentences.json", encoding="utf-8"))["sentences"]

for p in probes:
    plan = tts.analyze(p["text"])
    hard = [f"{w.word}=code{w.code}({w.confidence:.2f})" for w in plan.hard_words]
    print()
    print(p["tag"])
    print("  text      :", p["text"])
    print("  expected  :", p["expected"])
    print(f"  difficulty: {plan.sentence_difficulty:.3f} -> depth {plan.depth}")
    print("  decisions :", ", ".join(hard) if hard else "no ambiguous words")
"""),
    code("""
# The critical comparison: one word, two contexts, two readings.
a = tts.analyze("انا شوفت علم مصر بيرفرف")             # flag
b = tts.analyze("علم الفيزيا من اهم العلوم البشرية")   # science
ca = [w.code for w in a.hard_words if w.word == "علم"]
cb = [w.code for w in b.hard_words if w.word == "علم"]
print("flag context    -> code", ca)
print("science context -> code", cb)
print()
print("DISAMBIGUATION WORKS" if ca and cb and ca != cb
      else "NOT disambiguating: investigate before training the acoustic model")
"""),
    code("""
# The multi-homograph stress sentence from the brief.
plan = tts.analyze(
    "انا كنت مصر على ان مصر عندها امكانيات و موارد تخليها تتفوق على دول من اللي شايفين نفسهم دول"
)
print(plan)
"""),
    md("""
## Inference speed on CPU

The context encoder must be negligible next to the acoustic model.
"""),
    code("""
import time

txt = "انا كنت مصر على ان مصر عندها امكانيات تخليها تتفوق على دول"
for _ in range(3):
    tts.analyze(txt)                      # warm up
t0 = time.perf_counter()
for _ in range(50):
    tts.analyze(txt)
print(f"context encoder: {(time.perf_counter() - t0) / 50 * 1000:.2f} ms per sentence on CPU")
"""),
    md("If accuracy looks good, continue to **03_train_acoustic.ipynb**."),
]


# ---------------------------------------------------------------------------
# 03 - acoustic model
# ---------------------------------------------------------------------------

NB03 = [
    md("""
# 03 - Train the acoustic model

About 7 to 9 hours on a 4090, roughly 5 to 7 dollars on RunPod.

Every `sample_every` steps the probe sentences are rendered to TensorBoard, so
you can *listen* to whether homographs are pronounced correctly instead of
guessing from a loss curve.
"""),
    code(BOOT),
    md("## Check the model size before committing GPU hours"),
    code("""
import torch
from adaptts.models.acoustic import AcousticModel
from adaptts.text.vocab import CharVocab

vocab = CharVocab.load(cfg.paths.charvocab_path)
m = AcousticModel(
    len(vocab), n_quantizers=cfg.audio.n_quantizers, codebook_size=cfg.audio.codebook_size,
    d_model=cfg.acoustic.d_model, n_layers=cfg.acoustic.n_layers, n_heads=cfg.acoustic.n_heads,
    d_ff=cfg.acoustic.d_ff, text_d_model=cfg.acoustic.text_d_model,
    text_n_layers=cfg.acoustic.text_n_layers, text_n_heads=cfg.acoustic.text_n_heads,
    depth_d_model=cfg.acoustic.depth_d_model, depth_n_layers=cfg.acoustic.depth_n_layers,
    depth_n_heads=cfg.acoustic.depth_n_heads, speaker_dim=cfg.acoustic.speaker_dim,
    max_codes=cfg.discovery.max_codes_per_word, pc_embed_dim=cfg.acoustic.pc_embed_dim,
    exit_layers=cfg.acoustic.exit_layers, pad_id=vocab.pad_id,
)
n = sum(p.numel() for p in m.parameters())
print(f"acoustic model: {n / 1e6:.1f}M parameters")
print(f"fp32 weights:   {n * 4 / 1024 ** 2:.0f} MB")
print(f"total deployed with the frozen Mimi decoder: about {(n + 25e6) / 1e6:.0f}M")
del m
"""),
    md("""
## TensorBoard

Watch `train/acc_q0` for the coarse RVQ level, and the `probe/` audio tab.
"""),
    code("""
%load_ext tensorboard
%tensorboard --logdir $cfg.paths.tb_dir --port 6006 --bind_all
"""),
    md("""
## Train

If the pod restarts, resume with
`--resume runs/exp1/checkpoints/acoustic/last.pt`.
"""),
    code("!python scripts/train_acoustic.py --config $CONFIG"),
    md("""
## Quick machinery check

Before listening critically, confirm the parts are working: real samples, a
measurable difference between exit depths, and an override that actually
changes the output. Audio quality at this point depends entirely on how long
the model trained.
"""),
    code("!python scripts/smoke_generate.py --config $CONFIG --device cpu"),
    md("## Listen to the probe set"),
    code("""
import json
from IPython.display import Audio, display
from adaptts.infer.pipeline import AdapTTS

tts = AdapTTS.from_checkpoints(CONFIG, device="cpu")
probes = json.load(open("assets/probe_sentences.json", encoding="utf-8"))["sentences"]

for p in probes:
    plan = tts.analyze(p["text"])
    wav, st = tts.synthesize(plan)
    print()
    print(f"=== {p['tag']} | expected {p['expected']} ===")
    print("   ", p["text"])
    print(f"    depth {st['depth']}  difficulty {st['sentence_difficulty']:.2f}  "
          f"RTF {st['real_time_factor']:.2f}")
    display(Audio(wav, rate=st["sample_rate"]))
"""),
    md("""
## The controllability demo

The same sentence rendered with each reading, by overriding the code. No
diacritics are typed anywhere.
"""),
    code("""
text = "انا شوفت علم مصر بيرفرف"
plan = tts.analyze(text)
print(plan)

w = [x for x in plan.hard_words if x.word == "علم"][0]
for c in range(w.n_codes):
    plan.set_code("علم", c)
    wav, st = tts.synthesize(plan)
    print()
    print(f"--- علم forced to code {c} ---")
    display(Audio(wav, rate=st["sample_rate"]))
"""),
    md("""
## Adaptive depth: measure the saving

An easy sentence should route to a shallow exit and be measurably faster.
"""),
    code("""
import time

easy = "الجو النهارده حلو جدا و الشمس طالعة"
hard = "انا كنت مصر على ان مصر عندها امكانيات تخليها تتفوق على دول"

for name, txt in [("easy", easy), ("hard", hard)]:
    plan = tts.analyze(txt)
    t0 = time.perf_counter()
    wav, st = tts.synthesize(plan)
    dt = time.perf_counter() - t0
    print(f"{name:5s} difficulty {plan.sentence_difficulty:.3f} -> depth {st['depth']:2d}  "
          f"{dt:.2f}s for {st['audio_seconds']:.1f}s audio  RTF {st['real_time_factor']:.2f}")
"""),
    code("""
# Force each depth on the same sentence to isolate the compute saving.
plan = tts.analyze(hard)
for d in cfg.acoustic.exit_layers:
    plan.set_depth(d)
    t0 = time.perf_counter()
    wav, st = tts.synthesize(plan)
    dt = time.perf_counter() - t0
    print(f"depth {d:2d}: {dt:.2f}s  RTF {st['real_time_factor']:.2f}")
    display(Audio(wav, rate=st["sample_rate"]))
"""),
]


# ---------------------------------------------------------------------------
# 04 - inference and CPU benchmark
# ---------------------------------------------------------------------------

NB04 = [
    md("""
# 04 - Inference, control and CPU benchmarking

Everything here runs on CPU, which is the deployment target. Use this notebook
to measure real-time factor, to inspect and correct pronunciations, and to
export a deployable bundle.
"""),
    code(BOOT),
    code("""
import torch
torch.set_num_threads(os.cpu_count() or 4)
from adaptts.infer.pipeline import AdapTTS, save_wav

tts = AdapTTS.from_checkpoints(CONFIG, device="cpu")
print("loaded on CPU with", torch.get_num_threads(), "threads")
"""),
    md("""
## 1. Inspect: what does the model think each word means?

`analyze` returns a plan. It lists every ambiguous word, the reading chosen, the
confidence, and the difficulty that drives adaptive depth.
"""),
    code("""
plan = tts.analyze(
    "انا كنت مصر على ان مصر عندها امكانيات و موارد تخليها تتفوق على دول من اللي شايفين نفسهم دول"
)
print(plan)
"""),
    md("""
## 2. Correct: override a reading without diacritics

The pronunciation decision is a discrete code, so changing it is a legal edit
that the acoustic model was trained to consume.
"""),
    code("""
from IPython.display import Audio, display

plan = tts.analyze("انا شوفت علم مصر بيرفرف")
wav, st = tts.synthesize(plan)
print("model's own choice, code", plan.hard_words[0].code)
display(Audio(wav, rate=st["sample_rate"]))

plan.set_code("علم", 1)      # force the other reading
wav, st = tts.synthesize(plan)
print("forced to code 1")
display(Audio(wav, rate=st["sample_rate"]))
"""),
    code("""
# When a word appears twice with different readings, target one occurrence.
plan = tts.analyze("انا كنت مصر على ان مصر عندها امكانيات")
print(plan)
plan.set_code("مصر", 0, occurrence=0)   # first مصر only
plan.set_code("مصر", 1, occurrence=1)   # second مصر only
wav, st = tts.synthesize(plan)
display(Audio(wav, rate=st["sample_rate"]))
"""),
    md("""
## 3. Benchmark on CPU

Real-time factor below 1.0 means faster than real time. Pocket TTS reports about
6x real time on an M4; expect a similar order here at the shallow exits.
"""),
    code("""
import time
import numpy as np

sentences = [
    "الجو النهارده حلو جدا",
    "احنا رايحين السوق بكرة الصبح ان شاء الله",
    "انا كنت مصر على ان مصر عندها امكانيات تخليها تتفوق على دول",
]

rows = []
for txt in sentences:
    plan = tts.analyze(txt)
    tts.synthesize(plan)                       # warm up
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        wav, st = tts.synthesize(plan)
        times.append(time.perf_counter() - t0)
    rows.append((len(txt), plan.sentence_difficulty, st["depth"],
                 st["audio_seconds"], float(np.median(times))))

print(f"{'chars':>6}{'difficulty':>12}{'depth':>7}{'audio_s':>9}{'wall_s':>8}{'RTF':>7}")
print("-" * 49)
for n, d, dep, a, t in rows:
    print(f"{n:>6}{d:>12.3f}{dep:>7}{a:>9.2f}{t:>8.2f}{t / a:>7.2f}")
"""),
    code("""
# Adaptive versus fixed depth, on the same sentences.
import numpy as np

def bench(txt, depth=None):
    plan = tts.analyze(txt)
    if depth:
        plan.set_depth(depth)
    tts.synthesize(plan)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        _, st = tts.synthesize(plan)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), st["depth"]

full = cfg.acoustic.exit_layers[-1]
print(f"{'sentence':<34}{'adaptive':>12}{'fixed-full':>12}{'saving':>9}")
print("-" * 67)
for txt in sentences:
    ta, da = bench(txt)
    tf, _ = bench(txt, full)
    print(f"{txt[:32]:<34}{ta:>10.2f}s{tf:>10.2f}s{(1 - ta / tf) * 100:>8.0f}%")
"""),
    md("""
## 4. Save audio to disk
"""),
    code("""
wav, st = tts.tts("اهلا بيكم في التجربة الاولى من النظام الجديد")
save_wav("runs/exp1/samples/demo.wav", wav, st["sample_rate"])
print("wrote runs/exp1/samples/demo.wav", st)
display(Audio(wav, rate=st["sample_rate"]))
"""),
    md("""
## 5. Export a deployment bundle

Collects the two checkpoints, the vocabulary, the discovered codes and the
config into one directory that can be copied to a device.
"""),
    code("""
import shutil, json
from pathlib import Path

out = Path("runs/exp1/deploy")
out.mkdir(parents=True, exist_ok=True)
for src, dst in [
    (Path(cfg.paths.ckpt_dir) / "context_encoder" / "best.pt", "context_encoder.pt"),
    (Path(cfg.paths.ckpt_dir) / "acoustic" / "best.pt", "acoustic.pt"),
    (Path(cfg.paths.charvocab_path), "char_vocab.json"),
    (Path(cfg.paths.lexicon_path), "pronunciation_codes.json"),
    (Path("configs/exp1_egyptian.yaml"), "config.yaml"),
]:
    if Path(src).exists():
        shutil.copy(src, out / dst)
        print("copied", dst, f"{Path(src).stat().st_size / 1e6:.1f} MB")
    else:
        print("MISSING", src)
print()
print("bundle at", out.resolve())
"""),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cells in [
        ("00_setup.ipynb", NB00),
        ("01_prepare_data.ipynb", NB01),
        ("02_train_context.ipynb", NB02),
        ("03_train_acoustic.ipynb", NB03),
        ("04_inference.ipynb", NB04),
    ]:
        path = OUT / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb(cells), f, ensure_ascii=False, indent=1)
        print("wrote", path)


if __name__ == "__main__":
    main()
