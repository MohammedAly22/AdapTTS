"""Stage B: train the context encoder (the homograph disambiguator).

    python scripts/train_context.py --config configs/exp1_egyptian.yaml

This is the cheap, high-value stage. It trains in about twenty minutes and it is
where homograph accuracy actually comes from, so iterate here before spending
money on the acoustic model.
"""

from __future__ import annotations

import argparse
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
from adaptts.models.context_encoder import ContextEncoder, context_encoder_loss  # noqa: E402
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
logger = logging.getLogger("train_context")


def build_loader(cfg, ds, shuffle: bool, batch_size: int) -> DataLoader:
    sampler = LengthBucketSampler(
        ds.lengths, batch_size, cfg.train.bucket_boundaries,
        shuffle=shuffle, seed=cfg.train.seed,
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


def teacher_logits_for(
    batch: Dict[str, torch.Tensor], head: Optional[Dict[str, torch.Tensor]]
) -> Optional[torch.Tensor]:
    """Apply the cached teacher head to the cached teacher word states."""
    if head is None or "teacher_hidden" not in batch:
        return None
    th = batch["teacher_hidden"]
    z = (th - head["mu"]) / head["sd"]
    return z @ head["W"] + head["b"]


@torch.no_grad()
def evaluate(model, loader, cfg, device, head, max_batches: int) -> Dict[str, float]:
    """Accuracy on held-out ambiguous words, which is the metric that matters."""
    model.eval()
    correct = total = 0
    per_word_correct: Dict[int, List[int]] = {}
    loss_sum, nb = 0.0, 0

    n = min(max_batches, len(loader))
    for i, batch in enumerate(progress(loader, desc="  evaluating", unit="batch",
                                       total=n, leave=False)):
        if i >= max_batches:
            break
        batch = move_to_device(batch, device)
        with amp_context(cfg.train.precision, device):
            out = model(
                batch["char_ids"], batch["word_index"], batch["n_codes"],
                batch["char_padding_mask"],
            )
            loss, _ = context_encoder_loss(
                out, batch["code_target"], batch["word_mask"], batch["n_codes"],
                teacher_logits_for(batch, head), batch.get("teacher_hidden"),
                getattr(model, "_orig_mod", model),
                cfg.context_encoder.ce_weight, cfg.context_encoder.distill_weight,
                cfg.context_encoder.distill_temperature,
                cfg.context_encoder.difficulty_weight,
                cfg.context_encoder.label_smoothing,
            )
        loss_sum += float(loss)
        nb += 1

        sel = (batch["n_codes"] > 1) & batch["word_mask"] & (batch["code_target"] >= 0)
        if bool(sel.any()):
            pred = out.code_logits.argmax(-1)[sel]
            tgt = batch["code_target"][sel]
            nk = batch["n_codes"][sel]
            correct += int((pred == tgt).sum())
            total += int(sel.sum())
            for k in nk.unique():
                m = nk == k
                per_word_correct.setdefault(int(k), [0, 0])
                per_word_correct[int(k)][0] += int((pred[m] == tgt[m]).sum())
                per_word_correct[int(k)][1] += int(m.sum())

    model.train()
    out_metrics = {
        "eval/loss": loss_sum / max(nb, 1),
        "eval/code_acc": correct / max(total, 1),
        "eval/n_ambiguous": float(total),
    }
    for k, (c, n) in sorted(per_word_correct.items()):
        out_metrics[f"eval/acc_k{k}"] = c / max(n, 1)
    return out_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the AdapTTS context encoder")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="override config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_steps:
        cfg.optim.max_steps = args.max_steps
    set_seed(cfg.train.seed)
    setup_backend()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg.paths.root) / "context_encoder"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    writer = SummaryWriter(str(Path(cfg.paths.tb_dir) / "context_encoder"))
    logger.info("AdapTTS stage B: context encoder (homograph disambiguation)")
    logger.info("device: %s", device)
    log_config_summary(logger, cfg)

    # Fail early, with instructions, if a preprocessing stage was skipped.
    check_preprocessing(cfg, need_codes=False, need_teacher=True)

    vocab = CharVocab.load(Path(cfg.paths.charvocab_path))
    lex_path = Path(cfg.paths.lexicon_path)
    if not lex_path.exists():
        raise SystemExit(
            f"pronunciation codes not found at {lex_path}; run scripts/preprocess.py"
        )
    lexicon = PronunciationLexicon.load(lex_path)
    n_amb = len(lexicon.ambiguous_words)
    logger.info("lexicon: %d ambiguous word types discovered", n_amb)
    if n_amb == 0:
        raise SystemExit(
            "discovery found no ambiguous words. Check alignment quality, or relax "
            "discovery.stability_threshold / min_separation in the config."
        )

    train_ds = AdapTTSDataset(cfg, "train", vocab, lexicon, need_codes=False, need_teacher=True)
    dev_split = "dev"
    try:
        dev_ds = AdapTTSDataset(cfg, dev_split, vocab, lexicon, need_codes=False, need_teacher=True)
    except ValueError:
        dev_split = "test"
        dev_ds = AdapTTSDataset(cfg, dev_split, vocab, lexicon, need_codes=False, need_teacher=True)
    logger.info("splits: train=%d  %s=%d", len(train_ds), dev_split, len(dev_ds))

    train_loader = build_loader(cfg, train_ds, True, cfg.train.batch_size)
    dev_loader = build_loader(cfg, dev_ds, False, cfg.train.eval_batch_size)

    # Cached teacher head, used for soft targets.
    head = None
    hp = Path(cfg.paths.teacher_dir) / "teacher_head.npz"
    if hp.exists():
        z = np.load(hp)
        if "W" in z:
            head = {
                "W": torch.from_numpy(z["W"]).float().to(device),
                "b": torch.from_numpy(z["b"]).float().to(device),
                "mu": torch.from_numpy(z["mu"]).float().to(device),
                "sd": torch.from_numpy(z["sd"]).float().to(device),
            }
            logger.info("teacher head loaded (train acc %.3f)", float(z["acc"][0]))
    if head is None:
        logger.warning("no teacher head; training on discovered hard labels only")

    model = ContextEncoder(
        vocab_size=len(vocab),
        d_model=cfg.context_encoder.d_model,
        n_layers=cfg.context_encoder.n_layers,
        n_heads=cfg.context_encoder.n_heads,
        d_ff=cfg.context_encoder.d_ff,
        dropout=cfg.context_encoder.dropout,
        max_codes=cfg.context_encoder.max_codes,
        teacher_dim=cfg.teacher.hidden_size,
        pad_id=vocab.pad_id,
    ).to(device)
    total, trainable = count_parameters(model)
    log_table(
        logger, ["model", "value"],
        [("parameters", f"{total / 1e6:.2f}M"),
         ("trainable", f"{trainable / 1e6:.2f}M"),
         ("vocab size", len(vocab)),
         ("ambiguous word types", n_amb),
         ("train utterances", len(train_ds)),
         ("dev utterances", len(dev_ds))],
    )

    optimizer = build_optimizer(model, cfg)
    ckpt = CheckpointManager(Path(cfg.paths.ckpt_dir) / "context_encoder",
                             cfg.train.keep_last_n, higher_is_better=True)
    start_step = 0
    resume = args.resume or cfg.train.resume
    if resume:
        start_step = CheckpointManager.load(Path(resume), model, optimizer)

    model = maybe_compile(model, cfg)
    model.train()

    tracker, thr = MetricTracker(), Throughput()
    step = start_step
    epoch = 0
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.precision == "fp16" and device.type == "cuda"))
    t_start = time.time()

    bar = progress(total=cfg.optim.max_steps, desc="training context encoder",
                   unit="step")
    bar.update(step)
    while step < cfg.optim.max_steps:
        train_loader.batch_sampler.set_epoch(epoch)
        for batch in train_loader:
            if step >= cfg.optim.max_steps:
                break
            lr = lr_at_step(step, cfg)
            set_lr(optimizer, lr)
            batch = move_to_device(batch, device)

            with amp_context(cfg.train.precision, device):
                out = model(
                    batch["char_ids"], batch["word_index"], batch["n_codes"],
                    batch["char_padding_mask"],
                )
                loss, stats = context_encoder_loss(
                    out, batch["code_target"], batch["word_mask"], batch["n_codes"],
                    teacher_logits_for(batch, head), batch.get("teacher_hidden"),
                    getattr(model, "_orig_mod", model),
                    cfg.context_encoder.ce_weight, cfg.context_encoder.distill_weight,
                    cfg.context_encoder.distill_temperature,
                    cfg.context_encoder.difficulty_weight,
                    cfg.context_encoder.label_smoothing,
                )

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                optimizer.step()

            tracker.update({**stats, "grad_norm": gnorm, "lr": lr})
            thr.update(batch["char_ids"].shape[0])
            step += 1
            bar.update(1)

            if step % cfg.train.log_every == 0:
                m = tracker.mean()
                sps, ips = thr.rate()
                bar.set_postfix(
                    loss=f"{m.get('loss', 0):.3f}",
                    acc=f"{m.get('code_acc', 0):.3f}",
                    lr=f"{lr:.1e}",
                    mem=f"{gpu_memory_mb():.0f}MB",
                )
                for k, v in m.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("perf/steps_per_sec", sps, step)
                tracker.reset()
                thr.reset()

            if step % cfg.train.eval_every == 0 or step == cfg.optim.max_steps:
                em = evaluate(model, dev_loader, cfg, device, head, cfg.train.max_eval_batches)
                logger.info(
                    "step %d: held-out homograph accuracy %.4f on %d ambiguous words "
                    "(loss %.4f)",
                    step, em["eval/code_acc"], int(em["eval/n_ambiguous"]),
                    em["eval/loss"],
                )
                for k, v in em.items():
                    writer.add_scalar(k, v, step)
                ckpt.save(model, optimizer, step, cfg,
                          {**em, "monitor": em["eval/code_acc"]})

            if step % cfg.train.save_every == 0:
                ckpt.save(model, optimizer, step, cfg, {"monitor": None})
        epoch += 1

    bar.close()
    em = evaluate(model, dev_loader, cfg, device, head, 10_000)
    ckpt.save(model, optimizer, step, cfg, {**em, "monitor": em["eval/code_acc"]})
    logger.info("")
    logger.info("=" * 68)
    logger.info("CONTEXT ENCODER TRAINING COMPLETE in %s",
                format_duration(time.time() - t_start))
    log_table(
        logger, ["metric", "value"],
        [(k.replace("eval/", ""), f"{v:.4f}") for k, v in sorted(em.items())],
    )
    logger.info("checkpoints: %s", ckpt.dir)
    logger.info("=" * 68)
    writer.close()


if __name__ == "__main__":
    main()
