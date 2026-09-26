"""Stage C: train the adaptive-depth acoustic model.

    python scripts/train_acoustic.py --config configs/exp1_egyptian.yaml

Reads cached Mimi codes and pronunciation codes, so the loop is pure GPU work.
Every ``sample_every`` steps it renders the homograph probe sentences to
TensorBoard so you can hear whether disambiguation is working, rather than
guessing from a loss curve.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.data.dataset import (  # noqa: E402
    AdapTTSDataset,
    Collator,
    LengthBucketSampler,
)
from adaptts.data.discovery import PronunciationLexicon  # noqa: E402
from adaptts.models.acoustic import AcousticModel, acoustic_loss  # noqa: E402
from adaptts.models.context_encoder import ContextEncoder  # noqa: E402
from adaptts.text.vocab import CharVocab  # noqa: E402
from adaptts.train.common import (  # noqa: E402
    CheckpointManager,
    MetricTracker,
    Throughput,
    amp_context,
    build_optimizer,
    count_parameters,
    gpu_memory_mb,
    lr_at_step,
    maybe_compile,
    move_to_device,
    set_lr,
    set_seed,
    setup_backend,
)
from adaptts.utils.console import enable_utf8_console  # noqa: E402
from adaptts.utils.logging_utils import (  # noqa: E402
    format_duration,
    log_config_summary,
    log_table,
    progress,
    setup_logging,
)
from adaptts.utils.preflight import check_preprocessing  # noqa: E402
from adaptts.utils.config import load_config, save_config  # noqa: E402

enable_utf8_console()
setup_logging()
logger = logging.getLogger("train_acoustic")


def build_loader(cfg, ds, shuffle: bool, batch_size: int) -> DataLoader:
    sampler = LengthBucketSampler(
        ds.lengths, batch_size, cfg.train.bucket_boundaries, shuffle=shuffle, seed=cfg.train.seed
    )
    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers and cfg.train.num_workers > 0,
        prefetch_factor=cfg.train.prefetch_factor if cfg.train.num_workers > 0 else None,
        collate_fn=Collator(
            ds.vocab.pad_id, cfg.discovery.max_codes_per_word,
            cfg.audio.n_quantizers, cfg.teacher.hidden_size,
        ),
    )


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches: int) -> Dict[str, float]:
    model.eval()
    tracker = MetricTracker()
    n = min(max_batches, len(loader))
    for i, batch in enumerate(progress(loader, desc="  evaluating", unit="batch",
                                       total=n, leave=False)):
        if i >= max_batches:
            break
        batch = move_to_device(batch, device)
        with amp_context(cfg.train.precision, device):
            out = model(
                batch["char_ids"], batch["pc_per_char"], batch["codes"],
                make_speaker(batch, cfg, device), batch["char_padding_mask"],
                batch["frame_mask"], cfg_dropout=0.0,
                monotonic_strength=cfg.acoustic.monotonic_prior_weight,
            )
            _, stats = acoustic_loss(
                out, batch["codes"], batch["frame_mask"], cfg.acoustic.rvq_loss_weights,
                cfg.acoustic.exit_loss_weights, batch["n_frames"].float(),
                cfg.acoustic.duration_weight, cfg.acoustic.self_distill_weight,
            )
        tracker.update(stats)
    model.train()
    return {f"eval/{k}": v for k, v in tracker.mean().items()}


def make_speaker(batch: Dict[str, torch.Tensor], cfg, device) -> torch.Tensor:
    """Speaker vector slot.

    Experiment 1 is single speaker, so this is a constant. Experiment 2 replaces
    it with a speaker encoder over a reference clip; the model interface does
    not change, which is what makes that a fine-tune rather than a rewrite.
    """
    B = batch["char_ids"].shape[0]
    return torch.zeros(B, cfg.acoustic.speaker_dim, device=device)


class HomographProbe:
    """Renders fixed test sentences to TensorBoard so progress is audible."""

    def __init__(self, cfg, vocab: CharVocab, lexicon, device) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.lexicon = lexicon
        self.device = device
        path = Path(cfg.paths.root).parent / "probes.json"
        fallback = Path(__file__).resolve().parent.parent / "assets" / "probe_sentences.json"
        src = path if path.exists() else fallback
        if not src.exists():
            logger.warning("probe sentences not found at %s; audio logging disabled", src)
            self.sentences = []
        else:
            self.sentences = json.load(open(src, encoding="utf-8"))["sentences"]
        self.codec = None

    def _load_codec(self):
        if self.codec is None:
            from transformers import MimiModel

            self.codec = MimiModel.from_pretrained(self.cfg.codec_model_id).to(self.device).eval()
        return self.codec

    @torch.no_grad()
    def run(self, model, ctx_encoder, writer: SummaryWriter, step: int) -> None:
        if not self.sentences:
            return
        from adaptts.infer.pipeline import prepare_inputs

        codec = self._load_codec()
        model.eval()
        for i, item in enumerate(
            progress(self.sentences, desc="  rendering probes", unit="clip", leave=False)
        ):
            text = item["text"]
            try:
                inp = prepare_inputs(text, self.vocab, self.lexicon, self.device)
                pc = resolve_codes(ctx_encoder, inp, self.cfg, self.device)
                spk = torch.zeros(1, self.cfg.acoustic.speaker_dim, device=self.device)
                with amp_context(self.cfg.train.precision, self.device):
                    codes, stats = model.generate(
                        inp["char_ids"], pc, spk, inp["char_padding_mask"],
                        depth=self.cfg.acoustic.n_layers,
                        temperature=self.cfg.inference.temperature,
                        top_k=self.cfg.inference.top_k, top_p=self.cfg.inference.top_p,
                        cfg_scale=self.cfg.inference.cfg_scale,
                        monotonic_strength=self.cfg.acoustic.monotonic_prior_weight,
                    )
                wav = codec.decode(codes.transpose(1, 2)).audio_values[0, 0].float().cpu()
                writer.add_audio(
                    f"probe/{i:02d}_{item.get('tag', 'x')}", wav.unsqueeze(0),
                    step, sample_rate=codec.config.sampling_rate,
                )
                writer.add_text(f"probe/{i:02d}_text", text, step)
            except Exception as exc:  # noqa: BLE001
                logger.warning("probe %d failed: %s", i, exc)
        model.train()


@torch.no_grad()
def resolve_codes(ctx_encoder, inp, cfg, device) -> torch.Tensor:
    """Predict pronunciation codes and broadcast them onto characters."""
    from adaptts.models.context_encoder import broadcast_words_to_chars

    if ctx_encoder is None:
        return torch.full_like(inp["char_ids"], cfg.discovery.max_codes_per_word)
    out = ctx_encoder(
        inp["char_ids"], inp["word_index"], inp["n_codes"], inp["char_padding_mask"]
    )
    codes = out.code_logits.argmax(-1)  # [1, W]
    pc = torch.full_like(inp["char_ids"], cfg.discovery.max_codes_per_word)
    W = inp["n_codes"].shape[1]
    for w in range(W):
        if int(inp["n_codes"][0, w]) > 1:
            pc[inp["word_index"] == w] = int(codes[0, w])
    return pc


class EarlyStopper:
    """Halt training once evaluation loss stops improving.

    The previous run spent its last three hours past the point where eval loss
    began rising. Patience is counted in evaluations rather than steps so the
    behaviour does not change when eval_every does.
    """

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_step = 0
        self.bad = 0

    def update(self, loss: float, step: int) -> bool:
        """Record an evaluation. Returns True when training should stop."""
        if self.patience <= 0:
            return False
        if loss < self.best - self.min_delta:
            self.best, self.best_step, self.bad = loss, step, 0
            return False
        self.bad += 1
        return self.bad >= self.patience

    def status(self) -> str:
        return (
            f"best {self.best:.4f} at step {self.best_step}, "
            f"{self.bad}/{self.patience} evals without improvement"
        )


def check_exit_ordering(metrics: dict, logger) -> None:
    """Warn when a shallow exit beats the deepest one.

    Each exit is trained by distillation from the deepest, so the deepest should
    always be the best. When it is not, the model is memorizing and the extra
    depth is doing harm. The failed run ended at ce_exit0 = 4.14 against
    ce_exit2 = 4.41 and reported it as a normal metric.
    """
    ces = sorted(
        (k, v) for k, v in metrics.items() if k.startswith("eval/ce_exit")
    )
    if len(ces) < 2:
        return
    shallow, deep = ces[0][1], ces[-1][1]
    if deep > shallow + 1e-6:
        logger.warning("")
        logger.warning("=" * 68)
        logger.warning(
            "OVERFIT WARNING: the deepest exit (%.4f) is WORSE than the "
            "shallowest (%.4f).", deep, shallow,
        )
        logger.warning(
            "Every exit is distilled from the deepest, so this ordering means "
            "the model has more capacity than the data supports."
        )
        logger.warning(
            "Consider a smaller acoustic.d_model / n_layers, or stop here and "
            "use the best checkpoint."
        )
        logger.warning("=" * 68)
        logger.warning("")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the AdapTTS acoustic model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--context-ckpt", default="", help="context encoder for probes")
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_steps:
        cfg.optim.max_steps = args.max_steps
    set_seed(cfg.train.seed)
    setup_backend()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg.paths.root) / "acoustic"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    writer = SummaryWriter(str(Path(cfg.paths.tb_dir) / "acoustic"))
    logger.info("AdapTTS stage C: adaptive-depth acoustic model")
    logger.info("device: %s", device)
    log_config_summary(logger, cfg)

    # Fail early, with instructions, if a preprocessing stage was skipped.
    check_preprocessing(cfg, need_codes=True, need_teacher=False)

    vocab = CharVocab.load(Path(cfg.paths.charvocab_path))
    lexicon = (
        PronunciationLexicon.load(Path(cfg.paths.lexicon_path))
        if Path(cfg.paths.lexicon_path).exists() else None
    )

    train_ds = AdapTTSDataset(cfg, "train", vocab, lexicon, need_codes=True)
    try:
        dev_ds = AdapTTSDataset(cfg, "dev", vocab, lexicon, need_codes=True)
    except ValueError:
        dev_ds = AdapTTSDataset(cfg, "test", vocab, lexicon, need_codes=True)
    train_loader = build_loader(cfg, train_ds, True, cfg.train.batch_size)
    dev_loader = build_loader(cfg, dev_ds, False, cfg.train.eval_batch_size)

    model = AcousticModel(
        vocab_size=len(vocab),
        n_quantizers=cfg.audio.n_quantizers,
        codebook_size=cfg.audio.codebook_size,
        d_model=cfg.acoustic.d_model,
        n_layers=cfg.acoustic.n_layers,
        n_heads=cfg.acoustic.n_heads,
        d_ff=cfg.acoustic.d_ff,
        dropout=cfg.acoustic.dropout,
        text_d_model=cfg.acoustic.text_d_model,
        text_n_layers=cfg.acoustic.text_n_layers,
        text_n_heads=cfg.acoustic.text_n_heads,
        depth_d_model=cfg.acoustic.depth_d_model,
        depth_n_layers=cfg.acoustic.depth_n_layers,
        depth_n_heads=cfg.acoustic.depth_n_heads,
        speaker_dim=cfg.acoustic.speaker_dim,
        max_codes=cfg.discovery.max_codes_per_word,
        pc_embed_dim=cfg.acoustic.pc_embed_dim,
        exit_layers=cfg.acoustic.exit_layers,
        pad_id=vocab.pad_id,
    ).to(device)
    total, _ = count_parameters(model)
    log_table(
        logger, ["model", "value"],
        [("parameters", f"{total / 1e6:.2f}M"),
         ("backbone", f"d={cfg.acoustic.d_model} L={cfg.acoustic.n_layers}"),
         ("exit layers", str(cfg.acoustic.exit_layers)),
         ("quantizers", cfg.audio.n_quantizers),
         ("train utterances", len(train_ds)),
         ("dev utterances", len(dev_ds))],
    )

    ctx_encoder = None
    cpath = args.context_ckpt or str(Path(cfg.paths.ckpt_dir) / "context_encoder" / "best.pt")
    if Path(cpath).exists():
        ctx_encoder = ContextEncoder(
            vocab_size=len(vocab), d_model=cfg.context_encoder.d_model,
            n_layers=cfg.context_encoder.n_layers, n_heads=cfg.context_encoder.n_heads,
            d_ff=cfg.context_encoder.d_ff, dropout=0.0,
            max_codes=cfg.context_encoder.max_codes,
            teacher_dim=cfg.teacher.hidden_size, pad_id=vocab.pad_id,
        ).to(device)
        CheckpointManager.load(Path(cpath), ctx_encoder)
        ctx_encoder.eval()
        logger.info("probe context encoder loaded from %s", cpath)
    else:
        logger.warning("no context encoder at %s; probes will use null codes", cpath)

    optimizer = build_optimizer(model, cfg)
    ckpt = CheckpointManager(Path(cfg.paths.ckpt_dir) / "acoustic", cfg.train.keep_last_n)
    step = 0
    resume = args.resume or cfg.train.resume
    if resume:
        step = CheckpointManager.load(Path(resume), model, optimizer)

    stopper = EarlyStopper(
        cfg.train.early_stop_patience, cfg.train.early_stop_min_delta
    )
    probe = HomographProbe(cfg, vocab, lexicon, device)
    model = maybe_compile(model, cfg)
    model.train()

    tracker, thr = MetricTracker(), Throughput()
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.precision == "fp16" and device.type == "cuda"))
    epoch, t_start = 0, time.time()
    accum = max(1, cfg.optim.accum_steps)

    bar = progress(total=cfg.optim.max_steps, desc="training acoustic model",
                   unit="step")
    bar.update(step)
    should_stop = False
    while step < cfg.optim.max_steps and not should_stop:
        train_loader.batch_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.optim.max_steps:
                break
            lr = lr_at_step(step, cfg)
            set_lr(optimizer, lr)
            batch = move_to_device(batch, device)

            with amp_context(cfg.train.precision, device):
                out = model(
                    batch["char_ids"], batch["pc_per_char"], batch["codes"],
                    make_speaker(batch, cfg, device), batch["char_padding_mask"],
                    batch["frame_mask"], cfg_dropout=cfg.acoustic.cfg_dropout,
                    monotonic_strength=cfg.acoustic.monotonic_prior_weight,
                )
                loss, stats = acoustic_loss(
                    out, batch["codes"], batch["frame_mask"], cfg.acoustic.rvq_loss_weights,
                    cfg.acoustic.exit_loss_weights, batch["n_frames"].float(),
                    cfg.acoustic.duration_weight, cfg.acoustic.self_distill_weight,
                )
                loss = loss / accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                tracker.update({"grad_norm": gnorm})

            tracker.update({**stats, "lr": lr})
            thr.update(int(batch["frame_mask"].sum()))
            step += 1
            bar.update(1)

            if step % cfg.train.log_every == 0:
                m = tracker.mean()
                sps, fps = thr.rate()
                bar.set_postfix(
                    loss=f"{m.get('loss', 0):.3f}",
                    acc_q0=f"{m.get('acc_q0', 0):.3f}",
                    lr=f"{lr:.1e}",
                    mem=f"{gpu_memory_mb():.0f}MB",
                )
                for k, v in m.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("perf/steps_per_sec", sps, step)
                writer.add_scalar("perf/frames_per_sec", fps, step)
                tracker.reset()
                thr.reset()

            if step % cfg.train.eval_every == 0:
                em = evaluate(model, dev_loader, cfg, device, cfg.train.max_eval_batches)
                eval_loss = em.get("eval/loss", float("nan"))
                logger.info(
                    "step %d: eval loss %.4f  coarse-level accuracy %.4f  (%s)",
                    step, eval_loss, em.get("eval/acc_q0", float("nan")),
                    stopper.status(),
                )
                for k, v in em.items():
                    writer.add_scalar(k, v, step)
                check_exit_ordering(em, logger)
                ckpt.save(model, optimizer, step, cfg,
                          {**em, "monitor": eval_loss})

                if stopper.update(float(eval_loss), step):
                    logger.info("")
                    logger.info("=" * 68)
                    logger.info(
                        "EARLY STOP at step %d: eval loss has not improved for "
                        "%d evaluations.", step, stopper.patience,
                    )
                    logger.info(
                        "Best was %.4f at step %d; that checkpoint is saved as "
                        "best.pt.", stopper.best, stopper.best_step,
                    )
                    logger.info("=" * 68)
                    should_stop = True
                    break

            if step % cfg.train.sample_every == 0:
                probe.run(getattr(model, "_orig_mod", model), ctx_encoder, writer, step)

            if step % cfg.train.save_every == 0:
                ckpt.save(model, optimizer, step, cfg, {"monitor": None})
        epoch += 1

    bar.close()
    em = evaluate(model, dev_loader, cfg, device, 10_000)
    ckpt.save(model, optimizer, step, cfg, {**em, "monitor": em.get("eval/loss")})
    probe.run(getattr(model, "_orig_mod", model), ctx_encoder, writer, step)
    logger.info("")
    logger.info("=" * 68)
    logger.info("ACOUSTIC TRAINING COMPLETE in %s", format_duration(time.time() - t_start))
    log_table(
        logger, ["metric", "value"],
        [(k.replace("eval/", ""), f"{v:.4f}") for k, v in sorted(em.items())],
    )
    logger.info("checkpoints: %s", ckpt.dir)
    logger.info("samples logged to TensorBoard under probe/")
    logger.info("=" * 68)
    writer.close()


if __name__ == "__main__":
    main()
