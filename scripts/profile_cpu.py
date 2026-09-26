"""Does CPU thread count explain the pod's 6x slowdown?

Local: RTF 0.18 at depth 8. Pod: RTF 1.01, same code and config.

A container that reports many cores makes PyTorch spawn a thread per core, and
for a batch-1 autoregressive loop with tiny per-step matmuls, synchronisation
between those threads costs more than the work. Run this on the pod.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import torch  # noqa: E402

from adaptts.models.acoustic import AcousticModel  # noqa: E402
from adaptts.text.diacritics import N_VARIANTS  # noqa: E402
from adaptts.utils.config import load_config  # noqa: E402

cfg = load_config(os.environ.get("CONFIG", "configs/exp1_egyptian.yaml"))
print(f"os.cpu_count()          : {os.cpu_count()}")
try:
    print(f"sched_getaffinity       : {len(os.sched_getaffinity(0))}")
except AttributeError:
    pass
print(f"torch default threads   : {torch.get_num_threads()}")
print(f"config d_model={cfg.acoustic.d_model} n_layers={cfg.acoustic.n_layers} "
      f"exits={tuple(cfg.acoustic.exit_layers)}")
print()

m = AcousticModel(
    vocab_size=41, n_quantizers=cfg.audio.n_quantizers,
    codebook_size=cfg.audio.codebook_size, d_model=cfg.acoustic.d_model,
    n_layers=cfg.acoustic.n_layers, n_heads=cfg.acoustic.n_heads,
    d_ff=cfg.acoustic.d_ff, text_d_model=cfg.acoustic.text_d_model,
    text_n_layers=cfg.acoustic.text_n_layers,
    text_n_heads=cfg.acoustic.text_n_heads,
    depth_d_model=cfg.acoustic.depth_d_model,
    depth_n_layers=cfg.acoustic.depth_n_layers,
    depth_n_heads=cfg.acoustic.depth_n_heads,
    speaker_dim=cfg.acoustic.speaker_dim,
    max_codes=cfg.discovery.max_codes_per_word,
    pc_embed_dim=cfg.acoustic.pc_embed_dim,
    exit_layers=cfg.acoustic.exit_layers, pad_id=0,
    n_variants=N_VARIANTS,
).eval()

S, FRAMES = 23, 100
char_ids = torch.randint(1, 41, (1, S))
pc = torch.full((1, S), 4)
spk = torch.zeros(1, cfg.acoustic.speaker_dim)
variants = torch.zeros(1, S, dtype=torch.long)
DEPTH = max(cfg.acoustic.exit_layers)

print(f"generating {FRAMES} frames ({FRAMES/12.5:.0f}s audio) at depth {DEPTH}")
print()
print(f"{'threads':>8}{'total ms':>11}{'ms/frame':>11}{'RTF':>8}")
print("-" * 40)

best_cfg = None
for n in [1, 2, 4, 8, 16, os.cpu_count() or 8]:
    if n > (os.cpu_count() or 8):
        continue
    torch.set_num_threads(n)
    with torch.no_grad():
        m.generate(char_ids, pc, spk, depth=DEPTH, max_frames=8, variants=variants)
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        with torch.no_grad():
            codes, _ = m.generate(
                char_ids, pc, spk, depth=DEPTH,
                max_frames=FRAMES, min_frames=FRAMES, variants=variants,
            )
        best = min(best, time.perf_counter() - t0)
    rtf = best / (codes.shape[1] / 12.5)
    print(f"{n:>8}{best*1000:>11.0f}{best/codes.shape[1]*1000:>11.2f}{rtf:>8.2f}")
    if best_cfg is None or rtf < best_cfg[1]:
        best_cfg = (n, rtf)

print()
print(f"fastest: {best_cfg[0]} thread(s) at RTF {best_cfg[1]:.2f}")
print()
print("Local reference on the same code/config: RTF 0.18 at depth 8, 4 threads.")
if best_cfg[1] > 0.5:
    print("Still slow with the best thread count -> not a threading problem alone.")
else:
    print(f"Threading explains it: set torch.set_num_threads({best_cfg[0]}).")
