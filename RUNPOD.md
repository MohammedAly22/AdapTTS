# Running on RunPod

Short answer to "do I open the notebooks and run cells one by one?": **yes for
setup and data, no for the long training runs.** Notebooks lose their kernel
when a browser tab closes or the connection drops, and an acoustic run is 15 to
25 hours. Launch that from a terminal and watch it from the notebook.

---

## 1. Pod setup

Pick a **4090 or A100**. Any Ampere or newer card supports bf16 in hardware,
which the `exp1` config assumes. A 1660 Ti or other Turing card works but is
several times slower and must use fp16.

**There is no conda on a RunPod image, and you do not need one.** The pod is
already an isolated container, so the system Python is the environment. Conda
is only useful on a shared machine such as your own laptop.

First, see what the image already provides:

```bash
cd /workspace
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

A RunPod PyTorch template normally reports a working CUDA torch, which saves
you the slowest install step.

```bash
git clone https://github.com/MohammedAly22/AdapTTS.git
cd AdapTTS

pip install -r requirements.txt
python -m ipykernel install --user --name adaptts --display-name "AdapTTS"
```

**Only if the check above printed `False`, an error, or a torch below 2.4:**

```bash
pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Do not run that otherwise. Replacing a working CUDA build costs several minutes
and occasionally installs a wheel that does not match the pod's driver.

## 2. Preflight, before spending anything

```bash
python scripts/preflight_check.py --config configs/exp1_egyptian.yaml
```

This downloads and loads all four pretrained models and verifies every known
failure mode: the CTC blank index, autocast safety, the duration head, the
difficulty head, the discovery gates, picklable dataloaders and GPU precision
support. It takes a few minutes and tells you specifically what it checked.

**If anything fails here, do not start training.** That is the whole point of
the script.

## 3. Preprocessing

Run this from a terminal. It is about two hours on a 4090 and each stage is
resumable, so a dropped connection costs nothing.

```bash
python scripts/preprocess.py --config configs/exp1_egyptian.yaml --stage all 2>&1 | tee preprocess.log
```

### The decision point

When it finishes, read the discovery report **before** training anything:

```bash
head -40 cache/exp1/discovery_report.txt
grep -E "علم|مصر|دول" cache/exp1/discovery_report.txt
```

This is the cheapest possible test of the central hypothesis. If words like
`علم` and `مصر` appear with two or more codes, and the sentences behind each
code group by meaning, the approach is working and the rest is just compute. If
they do not, no amount of acoustic training will fix it, and you have spent two
hours instead of twenty.

Notebook `01_prepare_data.ipynb` has cells that print the sentences behind each
discovered code, which is the readable way to judge this.

## 4. Context encoder

About 20 minutes. Safe to run in a notebook, but a terminal is still simpler:

```bash
python scripts/train_context.py --config configs/exp1_egyptian.yaml 2>&1 | tee context.log
```

Watch `eval/code_acc`, which is held-out homograph accuracy. Above roughly 0.90
means the disambiguator works. This number is the actual deliverable of the
project, and it is available after 20 minutes rather than 20 hours.

## 5. Acoustic model

**Run this detached.** 15 to 25 hours on a 4090.

```bash
nohup python scripts/train_acoustic.py --config configs/exp1_egyptian.yaml \
  > acoustic.log 2>&1 &

tail -f acoustic.log          # watch it
```

Resume after any interruption:

```bash
python scripts/train_acoustic.py --config configs/exp1_egyptian.yaml \
  --resume runs/exp1/checkpoints/acoustic/last.pt
```

### TensorBoard

**RunPod fixes a pod's exposed ports when the pod is created.** There is no way
to add one to a running pod, and the Connect tab only lists what the template
declared. So do not plan on exposing 6006; use the port you already have.

**Easiest: run it inside JupyterLab.** Your pod already exposes 8888, and
JupyterLab proxies TensorBoard through it. In a notebook cell:

```python
%load_ext tensorboard
%tensorboard --logdir runs/exp1/tensorboard --port 6006
```

It renders inline, no port changes needed. Notebooks 02 and 03 already contain
this cell. For a full browser tab instead, `pip install jupyter-server-proxy`,
restart JupyterLab, and open
`https://<pod-id>-8888.proxy.runpod.net/proxy/6006/`.

**Most responsive: an SSH tunnel.** The "SSH over exposed TCP" entry on the
Connect tab forwards any port to your own machine:

```bash
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/id_ed25519 -L 6006:localhost:6006
```

Leave that open, start TensorBoard on the pod, then browse to
`http://localhost:6006` locally.

**Only if you want a permanent public URL:** recreate the pod and add 6006 to
*HTTP Ports* under Edit Template. Not worth losing a running job for.

Whichever route you take, bind to all interfaces or the proxy cannot reach it:

```bash
tensorboard --logdir runs/exp1/tensorboard --port 6006 --host 0.0.0.0
```

Without `--host 0.0.0.0` TensorBoard listens on localhost only. The in-notebook
magic works either way; the proxy and tunnel routes need it.

The `probe/` tab renders the eight homograph sentences as audio every
`sample_every` steps, so you can listen to progress rather than infer it from a
curve.

### What to expect, and when

The number to watch is `train/acc_q0`, the coarse RVQ level accuracy. These
bands come from the measured level-0 code entropy of 9.27 bits on this data:

| acc_q0 | What it sounds like |
|---|---|
| below 0.15 | noise with the right speaker timbre |
| 0.15 to 0.30 | speech-like rhythm and energy, no words |
| 0.30 to 0.45 | syllables audible, mostly unintelligible |
| 0.45 to 0.60 | words emerge, intelligible in places |
| above 0.60 | consistently intelligible |

Roughly: recognizable rhythm in the low thousands of steps, first words in the
low tens of thousands, clean speech near the end.

**Stop and investigate if `acc_q0` is still below 0.15 at 5000 steps.** That is
not slow progress, that is something wrong.

## 6. Generate and check

```bash
python scripts/smoke_generate.py --config configs/exp1_egyptian.yaml --device cpu
```

Writes wav files plus the numbers: real-time factor, per-depth timing, and an
override demonstration. Notebook `04_inference.ipynb` does the same
interactively with playback.

---

## Which parts belong in a notebook

| Task | Where | Why |
|---|---|---|
| Setup, preflight | notebook or terminal | short, interactive |
| Preprocessing | terminal, `tee` to a log | two hours, resumable |
| **Reading the discovery report** | **notebook** | this is the decision, and it wants Arabic rendering |
| Context encoder | either | only 20 minutes |
| Acoustic training | **terminal, detached** | survives a dropped connection |
| TensorBoard | **notebook cell** | 6006 is not exposed; the magic proxies through 8888 |
| Listening, overrides | notebook | needs inline audio playback |

## Cost

| Stage | 4090 hours | Approximate cost |
|---|---|---|
| Preprocessing | 2 | under $1 |
| Context encoder | 0.3 | negligible |
| Acoustic model | 15 to 25 | $8 to $15 |

Preprocessing plus the context encoder is under $1 and answers the question the
project is actually about. Spend that first and read the discovery report before
committing to the acoustic run.

## If something goes wrong

**Out of memory.** Lower `train.batch_size` to 16, then 8. Peak memory is
roughly linear in batch size.

**Slower than about 1.5 s/step on a 4090.** Check `train.precision` is `bf16`
and `train.compile` is `true`. The dataloader is not the bottleneck; it was
measured at 543 batches/s.

**Discovery finds no ambiguous words.** Lower `discovery.min_word_freq`, or
relax `stability_threshold` and `min_separation`. Rerun only that stage with
`--stage discover --force`.

`--force` applies to the stage you name and does not cascade upstream, so the
alignment and span embeddings are reused. That matters: span embeddings alone
take 11 minutes on the 68-hour corpus. Pass `--force-upstream` only when an
upstream cache is genuinely stale, such as after changing the aligner model or
the duration window.

**Discovery finds implausible words.** Tighten `max_duration_confound` below
0.25. Clusters that differ mainly in how long the word was spoken are prosody,
not separate readings.
