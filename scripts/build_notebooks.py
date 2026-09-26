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

## The second environment: CATT

The diacritizer pins an older torch and needs `pytorch_lightning`, which does
not coexist cleanly with this project's pins. It therefore gets its own
environment, used by exactly one step in notebook 01 and never again.

```bash
conda create -n CATT python=3.11 -y
conda activate CATT
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning num2words tqdm pyyaml
conda deactivate
```

Upload the `catt_tashkeel` folder so it sits at `/workspace/catt_parent/catt_tashkeel`,
and set `paths.catt_root: /workspace/catt_parent` in your config. Only the ECA
checkpoint is needed; the MSA weights are never loaded.

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
    "tests/test_diacritics.py",   # reading labels, artifact suppression
    "tests/test_phonology.py",    # variants, taa marbuta, partial overrides
    "tests/test_control.py",      # controllability and the complexity view
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

print("2/4 SSL span encoder (analysis only; labels no longer use it)")
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
    md("""
## Check the CATT environment

The diacritizer runs in its own environment, so it cannot be imported here.
This checks it the way notebook 01 will actually invoke it: a subprocess under
the `CATT` interpreter that loads the ECA checkpoint and diacritizes two probe
sentences.

Catching a broken CATT setup now costs a minute. Catching it in notebook 01
costs the 40 minutes of alignment you already paid for.
"""),
    code("""
import os, subprocess, sys, yaml

_cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
CATT_ROOT = os.environ.get("CATT_ROOT") or _cfg["paths"].get("catt_root", "")
CATT_PY = os.environ.get(
    "CATT_PY", os.path.expanduser("~/miniconda3/envs/CATT/bin/python")
)

print("catt_root:", CATT_ROOT or "(not set)")
print("catt python:", CATT_PY)

ok = True
if not CATT_ROOT or not os.path.isdir(os.path.join(CATT_ROOT, "catt_tashkeel")):
    ok = False
    print()
    print("No catt_tashkeel/ under catt_root.")
    print("Upload the folder and set paths.catt_root in your config.")
elif not os.path.isfile(CATT_PY):
    ok = False
    print()
    print("CATT interpreter not found. Create the env (see the shell block above),")
    print("or set CATT_PY to its python.")
else:
    ckpt = os.path.join(CATT_ROOT, "catt_tashkeel", "checkpoints", "eca_model_weights.pt")
    # Size, not just existence. A part-transferred checkpoint is present and
    # then fails inside torch.load with "pickle data was truncated", which reads
    # like a code fault instead of an upload that stopped early. Browser uploads
    # of a 74 MB file truncate often enough to be worth checking here.
    EXPECTED = 78_059_891
    if not os.path.isfile(ckpt):
        ok = False
        print("eca checkpoint: MISSING")
    else:
        size = os.path.getsize(ckpt)
        if size != EXPECTED:
            ok = False
            print(f"eca checkpoint: WRONG SIZE {size:,} bytes, expected {EXPECTED:,}")
            print(f"  short by {EXPECTED - size:,} bytes - the upload did not finish.")
            print("  Re-transfer with scp or huggingface-cli, then check:")
            print("    md5sum should be 4fc95acd1d70d7b372f94cb40b0b4339")
        else:
            print(f"eca checkpoint: {size:,} bytes (size verified)")

if ok:
    # Only worth probing once the file is known to be complete: otherwise this
    # fails inside torch.load and buries the real cause in a traceback.
    probe = (
        "import sys; sys.path.insert(0, %r); sys.path.insert(0, 'scripts')\\n"
        "from diacritize import load_eca_model\\n"
        "m, pre, post, tok = load_eca_model(%r)\\n"
        "t = ['انا شوفت علم مصر بيرفرف', 'علم الفيزيا من اهم العلوم']\\n"
        "p = [pre.process_text(tok.remove_tashkeel(x), verbose=False) for x in t]\\n"
        "o = [post.process(x) for x in m.do_tashkeel_batch(p, batch_size=2, verbose=False)]\\n"
        "print(o[0]); print(o[1])\\n"
    ) % (CATT_ROOT, CATT_ROOT)
    r = subprocess.run([CATT_PY, "-c", probe], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    print()
    print(r.stdout.strip() or "(no output)")
    if r.returncode != 0:
        ok = False
        print("FAILED:", r.stderr.strip()[-1200:])

print()
print("CATT ready" if ok else "CATT NOT ready - fix before notebook 01 stage A0b")
"""),
    md("Setup is done. Continue to **01_prepare_data.ipynb**."),
]


# ---------------------------------------------------------------------------
# 01 - data preparation and pronunciation-code discovery
# ---------------------------------------------------------------------------

NB01 = [
    md("""
# 01 - Data preparation and pronunciation labels

All the offline work happens here. Afterwards training reads only memmaps, so
the GPU never waits on data.

| Stage | What it does | Time on a 4090 |
|---|---|---|
| A0 | manifest: scan, normalize, filter | 1 min |
| A0b | **diacritize with CATT-ECA** (own conda env) | about 20 min |
| A1 | CTC forced alignment to word spans | about 35 min |
| A3 | **derive readings** -> the decision point | about 2 min |
| A4 | MARBERTv2 teacher cache and head | about 8 min |
| A5 | Mimi encode to RVQ codes | about 30 min |

Run the cells in order, top to bottom.

## What changed, and why it matters

The first full run cost $15 and learned nothing, because labels came from
unsupervised clustering of acoustic word spans. That returned the channel's
subscribe pitch as "homographs" (الجرس, لايك, الوصف) while علم, مصر and دول each
got a single code. A mean-pooled speech embedding encodes speaking rate and
recording session far more strongly than vowel identity, so no threshold on that
signal can separate the two.

Labels now come from a **diacritizer**, which observes the vowels directly.
Stage A2 (span embeddings) is no longer on the critical path and is skipped.

Stage A3 is the decision point: it costs two minutes and tells you whether the
labels are sound *before* you spend money on training.
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
## Stage A0b - diacritize with CATT-ECA

This is the label source. It runs **in the CATT environment**, not this one, so
it goes through that interpreter explicitly rather than through `!python`.

The diacritics are a labelling device only. They tell us which reading each word
occurrence takes; the shipped model never sees a diacritic and you never type
one at inference. This is the same role the forced aligner plays for word spans.

About 20 minutes for 15.6k sentences. Runs once, then caches.
"""),
    code("""
import os, subprocess, sys, yaml

_raw = yaml.safe_load(open(CONFIG, encoding="utf-8"))
_base = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
CATT_ROOT = (os.environ.get("CATT_ROOT")
             or _raw.get("paths", {}).get("catt_root")
             or _base["paths"].get("catt_root", ""))
CATT_PY = os.environ.get(
    "CATT_PY", os.path.expanduser("~/miniconda3/envs/CATT/bin/python")
)
print("catt_root  :", CATT_ROOT or "(not set)")
print("catt python:", CATT_PY)

assert CATT_ROOT and os.path.isdir(os.path.join(CATT_ROOT, "catt_tashkeel")), (
    "catt_tashkeel/ not found under catt_root. Upload the folder and set "
    "paths.catt_root in your config."
)
assert os.path.isfile(CATT_PY), (
    "CATT interpreter not found. Create the CATT env (notebook 00) or set CATT_PY."
)
"""),
    code("""
# Streams output so you can watch progress rather than waiting on a block.
import subprocess, sys, os

cmd = [CATT_PY, "scripts/diacritize.py", "--config", CONFIG,
       "--catt-root", CATT_ROOT, "--batch-size", "32"]
print(" ".join(cmd))
print()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1,
                     env=dict(os.environ, PYTHONIOENCODING="utf-8"))
for line in p.stdout:
    print(line.rstrip(), flush=True)
p.wait()
assert p.returncode == 0, f"diacritize failed with code {p.returncode}"
"""),
    code("""
# Inspect the diacritized output. The homographs should differ in their marks.
import json

rows = [json.loads(l) for l in open(cfg.paths.diacritized_path, encoding="utf-8")]
print(f"{len(rows)} diacritized utterances")
print()
for r in rows[:3]:
    print("plain:", r["text"][:80])
    print("diac :", r["diacritized"][:80])
    print()

# Token counts must match, or an occurrence would be mislabelled. The reading
# stage skips any mismatch rather than guessing, but a high rate means trouble.
bad = sum(1 for r in rows if len(r["text"].split()) != len(r["diacritized"].split()))
print(f"token-count mismatches: {bad} of {len(rows)} ({100*bad/max(len(rows),1):.1f}%)")
"""),
    md("""
## Stage A1 - CTC forced alignment

Finds the time span of every word with no pronunciation lexicon. The Viterbi
alignment is implemented in-repo, so there is no Montreal Forced Aligner
dependency.
"""),
    code("!python scripts/preprocess.py --config $CONFIG --stage align"),
    md("""
## Stage A3 - derive the readings

### This is the decision point of the whole project

Two minutes here decides whether training is worth paying for. The stage groups
every occurrence of every word by its vowel pattern and keeps a split only when
it survives the artifact filters and the context-agreement gate.

Span embeddings (the old stage A2) are not needed for labels and are skipped.
"""),
    code("!python scripts/preprocess.py --config $CONFIG --stage discover"),
    code("""
from adaptts.text.diacritics import ReadingLexicon

lex = ReadingLexicon.load(cfg.paths.reading_lexicon_path)
amb = lex.ambiguous_words
print(f"{len(lex)} word types, {len(amb)} with more than one reading")
print()

entries = sorted((lex.entries[w] for w in amb), key=lambda e: -e.total)
print(f"{'word':<16}{'readings':>9}{'uses':>7}  counts  examples")
print("-" * 74)
for e in entries[:40]:
    print(f"{e.word:<16}{e.n_codes:>9}{e.total:>7}  {e.counts}  {' '.join(e.examples)}")
"""),
    code("""
# The homographs named in the brief. These are the reason the project exists.
for w in PROBE_WORDS:
    e = lex.entries.get(w)
    k = lex.n_codes(w)
    ex = " ".join(e.examples) if e else ""
    print(f"{w:<10} -> {k} reading(s)  {'AMBIGUOUS' if k > 1 else 'single':<10} {ex}")
"""),
    code("""
# The failure mode from the first run: promo words must NOT be ambiguous.
# Clustering split these on speaking register. If any shows up here, the
# labels have regressed and training would waste money again.
PROMO = ["الجرس", "التعليقات", "لايك", "الوصف", "الرابط", "البلاي", "اكتبوه", "عندكم"]
bad = [w for w in PROMO if lex.n_codes(w) > 1]
for w in PROMO:
    k = lex.n_codes(w)
    print(f"{w:<14} {k} reading(s)" + ("   <-- REGRESSION" if k > 1 else ""))
print()
print("promo words clean" if not bad else f"PROBLEM: {bad} split again")
"""),
    code("""
# Read the sentences behind each reading. This is how you confirm the labels
# track meaning rather than an artifact of the diacritizer.
import json, collections

rows = [json.loads(l) for l in open(cfg.paths.diacritized_path, encoding="utf-8")]

WORD = PROBE_WORDS[0]      # change to inspect any ambiguous word
e = lex.entries.get(WORD)
if e is None or e.n_codes < 2:
    print(f"{WORD} has a single reading here; pick another from the table above.")
else:
    groups = collections.defaultdict(list)
    for r in rows:
        plain, diac = r["text"].split(), r["diacritized"].split()
        if len(plain) != len(diac):
            continue
        for w, d in zip(plain, diac):
            if w == WORD:
                c = e.code_of(d)
                if c >= 0:
                    groups[c].append((d, r["text"]))
    for c in sorted(groups):
        print()
        print(f"=== {WORD}  reading {c}  ({e.examples[c]})  "
              f"{len(groups[c])} occurrences ===")
        for d, t in groups[c][:6]:
            print(f"   {d:<14} {t[:80]}")
"""),
    md("""
**How to read this.** Sentences under one reading should share a meaning: علم as
flag in one group, as science in the other. If the groups look mixed, raise
`discovery.min_pattern_count` in the config and rerun this stage with `--force`.

If the promo-word check above shows a regression, stop. Do not train.
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
"""),
    md("""
## The majority baseline

The number notebook 02 has to beat. If every ambiguous word were always given
its most common reading, this is the accuracy you would get for free. A context
encoder that scores at or below this has learned nothing, whatever the loss
curve looks like.

Write it down before training.
"""),
    code("""
best = sum(max(lex.entries[w].counts) for w in amb)
total = sum(lex.entries[w].total for w in amb)
baseline = best / max(total, 1)
print(f"ambiguous word types : {len(amb)}")
print(f"labelled occurrences : {total}")
print(f"MAJORITY BASELINE    : {baseline:.3f}")
print()
print("Notebook 02 must beat this by a clear margin, not by 1%.")
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
matters, and it has to be read **against the majority baseline** from notebook
01, not against zero. A model that always guesses the commonest reading already
scores the baseline while having learned nothing.
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
    md("""
## The gate: did it beat the majority baseline?

The probe check above is necessary but not sufficient. Two different codes on
two sentences is consistent with a model that has genuinely learned context, and
also with one that memorised a single split. This cell measures held-out
accuracy against the baseline over every ambiguous occurrence.

**Do not start notebook 03 until this passes.** The acoustic model inherits
these labels; if the disambiguator is at the prior, the expensive run produces a
system that reads homographs by frequency, which is what the first $15 bought.
"""),
    code("""
import json, collections
from adaptts.text.diacritics import ReadingLexicon

lex = ReadingLexicon.load(cfg.paths.reading_lexicon_path)
rows = [json.loads(l) for l in open(cfg.paths.diacritized_path, encoding="utf-8")]
manifest = {json.loads(l)["uid"]: json.loads(l) for l in
            open(cfg.paths.manifest_path, encoding="utf-8")}

# Held-out splits only: training accuracy proves nothing about generalisation.
# The manifest names them "dev" and "test".
eval_uids = {u for u, r in manifest.items() if r.get("split") in ("dev", "test")}
print(f"held-out utterances: {len(eval_uids)}")

# Group the gold labels by sentence so each sentence is analyzed once rather
# than once per ambiguous word in it.
gold_by_text = collections.defaultdict(list)   # text -> [(word_index, word, code)]
for r in rows:
    if r["uid"] not in eval_uids:
        continue
    plain, diac = r["text"].split(), r["diacritized"].split()
    if len(plain) != len(diac):
        continue
    for i, (w, d) in enumerate(zip(plain, diac)):
        e = lex.entries.get(w)
        if e is None or e.n_codes < 2:
            continue
        c = e.code_of(d)
        if c >= 0:
            gold_by_text[r["text"]].append((i, w, c))

gold = [(t, i, w, c) for t, items in gold_by_text.items() for i, w, c in items]
print(f"held-out ambiguous occurrences: {len(gold)} "
      f"across {len(gold_by_text)} sentences")

# Majority baseline on exactly this set.
majority = {w: max(range(e.n_codes), key=lambda k: e.counts[k])
            for w, e in lex.entries.items() if e.n_codes > 1}
base_hits = sum(1 for _, _, w, c in gold if majority.get(w) == c)

# Model predictions, one analyze() per sentence.
from tqdm.auto import tqdm

hits = 0
by_word = collections.defaultdict(lambda: [0, 0])
for text, items in tqdm(gold_by_text.items(), desc="scoring", unit="sent"):
    plan = tts.analyze(text)
    # Match on word index, not on the word string: a sentence can repeat an
    # ambiguous word with two different readings (دول ... دول), and matching by
    # string alone would score the wrong occurrence.
    pred_by_idx = {hw.index: hw.code for hw in plan.hard_words}
    for idx, w, c in items:
        ok = pred_by_idx.get(idx) == c
        hits += int(ok)
        by_word[w][0] += int(ok)
        by_word[w][1] += 1

n = max(len(gold), 1)
acc, base = hits / n, base_hits / n
print()
print(f"majority baseline : {base:.3f}")
print(f"model accuracy    : {acc:.3f}")
print(f"margin            : {acc - base:+.3f}")
print()
if acc > base + 0.05:
    print("PASS - the model is using context. Continue to notebook 03.")
elif acc > base:
    print("MARGINAL - above the prior but within noise. More data or more")
    print("homograph contrast is needed before paying for the acoustic run.")
else:
    print("FAIL - at or below the prior. The model has learned nothing beyond")
    print("word frequency. Do NOT train the acoustic model yet.")

print()
print(f"{'word':<14}{'acc':>6}{'n':>6}")
for w, (h, t) in sorted(by_word.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"{w:<14}{h/max(t,1):>6.2f}{t:>6}")
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

# The model decides on its own from context.
plan = tts.analyze("انا شوفت علم مصر بيرفرف")
print(plan)
wav, st = tts.synthesize(plan)
display(Audio(wav, rate=st["sample_rate"]))
"""),
    md("""
### Correcting a reading: write the vowels, not a code number

If a reading is wrong, add diacritics to that one word. Nothing else changes, and
the rest of the sentence stays undiacritized.

```
انا كنت مصر على ان مصر عندها امكانيات          # model decides both
انا كنت مُصِرّ على ان مصر عندها امكانيات        # first pinned, second free
```

Why not a code number? Codes are assigned by corpus frequency, so there is no way
to know that code 1 means مُصِرّ, and the answer changes whenever the lexicon is
rebuilt. Writing the vowel is the notation people already use.

**The diacritics never reach the model.** They are read as instructions, resolved
to a code, and stripped, so the model's input is identical either way. A partial
marking is enough: only the letters that distinguish the readings matter.
"""),
    code("""
free   = "انا كنت مصر على ان مصر عندها امكانيات"
pinned = "انا كنت مُصِرّ على ان مصر عندها امكانيات"

for label, txt in [("model decides", free), ("first word pinned", pinned)]:
    plan = tts.analyze(txt)
    print(f"--- {label} ---")
    print("  text the model sees:", plan.text)
    for w in plan.hard_words:
        src = "user" if w.from_user else "model"
        print(f"    #{w.index} {w.word}: code {w.code} ({src}), "
              f"confidence {w.confidence:.3f}")
    if plan.unresolved_marks:
        print("  could not apply:", plan.unresolved_marks)
    print()
"""),
    code("""
# Listen to the difference the correction makes.
for label, txt in [("model's choice", free), ("corrected", pinned)]:
    plan = tts.analyze(txt)
    wav, st = tts.synthesize(plan)
    print(label)
    display(Audio(wav, rate=st["sample_rate"]))
"""),
    code("""
# A correction that cannot be resolved is reported, never guessed at.
for attempt in ["مُصِرّ", "مصِر", "مُصر", "مصُر"]:
    code_, reason = tts.lexicon.code_from_partial_marks(attempt)
    verdict = f"code {code_}" if code_ >= 0 else "unresolved"
    print(f"  {attempt:<8} -> {verdict:<12} {reason}")
"""),
    md("""
### Phoneme variants

Some letters have a second realisation that the diacritizer marks and the model
learns. Type `~` after the letter to force it:

| Written | Effect |
|---|---|
| `ق~` | qaf pronounced as a glottal stop, as in دلوق~تي |
| `ج~` | geem as /zh/ rather than /g/, as in تكنولوج~يا |
| `ف~` | faa as /v/, as in ف~يديو |
| `...ةْ` | sukun on a final ة sounds the /t/: مَدِينَةْ is /madiinat/, مَدِينَة is /madiina/ |

These are independent of the reading: a word can take a variant without being
ambiguous at all.
"""),
    code("""
for txt in ["دلوقتي احنا مشغولين", "دلوق~تي احنا مشغولين",
            "المدينة كبيره", "المدينةْ كبيره"]:
    plan = tts.analyze(txt)
    v = plan.inputs["variants"][0].tolist()
    marked = [(i, c) for i, c in enumerate(v) if c]
    print(f"{txt:<26} variants at {marked}")
"""),
    md("""
## 2b. The complexity view

Where the compute goes, per word. This is the adaptive claim made checkable: a
word with one reading should be cheap, a homograph should not.

If single-reading words cost as much as homographs, the difficulty head has
learned nothing and the adaptive-depth story is empty.
"""),
    code("""
print(tts.analyze(
    "انا كنت مصر على ان مصر عندها امكانيات و موارد تخليها تتفوق على دول"
).complexity_report())
"""),
    code("""
# Easy sentence against hard sentence: the depth should differ.
for txt in ["الجو النهارده حلو", free]:
    plan = tts.analyze(txt)
    n_amb = len(plan.hard_words)
    print(f"difficulty {plan.sentence_difficulty:.3f} -> depth {plan.depth}  "
          f"({n_amb} ambiguous)  {txt[:44]}")
"""),
    md("""
## 2c. The other controls

| Control | What it does |
|---|---|
| `set_tempo(x)` | speaking rate; 1.0 is the model's own pace |
| `set_cfg_scale(x)` | how strongly to follow the conditioning / reference voice |
| `set_budget(f)` | compute ceiling as a fraction of the deepest exit |
| `set_depth(n)` | fix the depth, overriding adaptivity |
| `set_temperature(x)` | sampling randomness |

A budget is a ceiling, not a target: an easy sentence still runs shallow, so it
trades quality for speed only where the model wanted the compute.
"""),
    code("""
plan = tts.analyze(free)
print("adaptive depth:", plan.depth, "of", list(cfg.acoustic.exit_layers))

for t in [0.8, 1.0, 1.3]:
    p = tts.analyze(free).set_tempo(t)
    wav, st = tts.synthesize(p)
    print(f"tempo {t}: {st['audio_seconds']:.2f}s audio, "
          f"RTF {st['total_seconds']/st['audio_seconds']:.3f}")
    display(Audio(wav, rate=st["sample_rate"]))
"""),
    code("""
# A budget lowers the depth; it can never raise it.
for b in [0.4, 0.7, 1.0]:
    p = tts.analyze(free).set_budget(b)
    print(f"budget {b} -> depth {p.depth}")
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
    # The reading lexicon, not the retired clustering one. Inference needs it to
    # know how many readings a word has and what each code means.
    (Path(cfg.paths.reading_lexicon_path), "reading_lexicon.json"),
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
