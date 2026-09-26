"""Datasets and collation.

Everything here reads from memmaps produced by preprocessing. No audio decoding,
no tokenizer, no codec runs inside the training loop, which is what keeps the
GPU busy and the epoch time low.

The batch sampler buckets by length so padding waste stays small; with the
default boundaries, padded frames are under ten percent of the batch on the
Masri corpus.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..text.normalize import tokenize_words
from ..text.vocab import CharVocab
from ..utils.config import Config
from ..text.diacritics import ReadingLexicon
from .discovery import PronunciationLexicon

logger = logging.getLogger(__name__)


class RaggedArray:
    """Read-only view over a ragged store (buffer + offsets).

    The buffer is memory mapped, so a 12 GB cache costs no resident memory and
    the OS page cache does the work. ``close`` releases the mapping, which
    Windows requires before the file can be deleted.
    """

    def __init__(self, path: Path) -> None:
        path = Path(path)
        self.path = path
        self.buf = np.load(path, mmap_mode="r")
        self.offsets = np.load(path.with_name(path.stem + "_offsets.npy"))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.buf[self.offsets[i] : self.offsets[i + 1]])

    def close(self) -> None:
        mm = getattr(self.buf, "_mmap", None)
        if mm is not None:
            mm.close()
        self.buf = None

    def __enter__(self) -> "RaggedArray":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class Sample:
    uid: str
    char_ids: np.ndarray  # [L]
    word_index: np.ndarray  # [L] word id per char, -1 outside words
    n_codes: np.ndarray  # [W]
    code_target: np.ndarray  # [W], -1 where unsupervised
    codes: Optional[np.ndarray]  # [T, Q] Mimi codes
    teacher_hidden: Optional[np.ndarray]  # [W, H]
    n_words: int


class AdapTTSDataset(Dataset):
    """Serves preprocessed utterances for either training stage."""

    def __init__(
        self,
        cfg: Config,
        split: str,
        vocab: CharVocab,
        lexicon: Optional["ReadingLexicon"],
        need_codes: bool = True,
        need_teacher: bool = False,
    ) -> None:
        self.cfg = cfg
        self.split = split
        self.vocab = vocab
        self.lexicon = lexicon
        self.need_codes = need_codes
        self.need_teacher = need_teacher

        manifest = Path(cfg.paths.manifest_path)
        if not manifest.exists():
            raise FileNotFoundError(
                f"manifest not found at {manifest}; run scripts/preprocess.py first"
            )
        rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no utterances for split {split!r} in {manifest}")

        # Code labels from discovery.
        self.labels: Dict[Tuple[str, int], int] = {}
        lp = Path(cfg.paths.cache_dir) / "code_labels.json"
        if lp.exists():
            raw = json.load(open(lp, encoding="utf-8"))
            for k, v in raw.items():
                uid, widx = k.split("\t")
                self.labels[(uid, int(widx))] = int(v)

        self.codes: Optional[RaggedArray] = None
        self.code_index: Dict[str, int] = {}
        if need_codes:
            cpath = Path(cfg.paths.codes_dir) / "codes.npy"
            if not cpath.exists():
                raise FileNotFoundError(
                    f"codec cache not found at {cpath}; run preprocessing stage 'codec'"
                )
            self.codes = RaggedArray(cpath)
            order = json.load(open(Path(cfg.paths.codes_dir) / "uid_order.json", encoding="utf-8"))
            self.code_index = {uid: i for i, uid in enumerate(order)}
            self.rows = [r for r in self.rows if r["uid"] in self.code_index]

        self.teacher: Optional[RaggedArray] = None
        self.teacher_index: Dict[str, int] = {}
        if need_teacher:
            tpath = Path(cfg.paths.teacher_dir) / "word_hidden.npy"
            if tpath.exists():
                self.teacher = RaggedArray(tpath)
                order = json.load(
                    open(Path(cfg.paths.teacher_dir) / "uid_order.json", encoding="utf-8")
                )
                self.teacher_index = {uid: i for i, uid in enumerate(order)}
            else:
                logger.warning("teacher cache missing at %s; distillation disabled", tpath)

        # Precompute lengths for the bucketing sampler.
        self.lengths = np.array(
            [
                self.codes[self.code_index[r["uid"]]].shape[0]
                if self.codes is not None
                else len(r["text"])
                for r in self.rows
            ],
            dtype=np.int64,
        )
        logger.info(
            "dataset[%s]: %d utterances, length range %d..%d",
            split, len(self.rows), int(self.lengths.min()), int(self.lengths.max()),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def close(self) -> None:
        """Release the memory maps. Needed before deleting the cache on Windows."""
        for name in ("codes", "teacher"):
            ra = getattr(self, name, None)
            if ra is not None:
                ra.close()
                setattr(self, name, None)

    def __enter__(self) -> "AdapTTSDataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __getitem__(self, i: int) -> Sample:
        r = self.rows[i]
        text = r["text"]
        uid = r["uid"]
        words, spans = tokenize_words(text)
        ids, widx = self.vocab.encode_with_word_index(text, spans)

        n_words = len(words)
        n_codes = np.ones(n_words, dtype=np.int64)
        targets = np.full(n_words, -1, dtype=np.int64)
        if self.lexicon is not None:
            for wi, w in enumerate(words):
                k = self.lexicon.n_codes(w)
                n_codes[wi] = k
                if k > 1:
                    targets[wi] = self.labels.get((uid, wi), -1)

        codes = None
        if self.codes is not None:
            codes = self.codes[self.code_index[uid]].astype(np.int64)

        th = None
        if self.teacher is not None and uid in self.teacher_index:
            th = self.teacher[self.teacher_index[uid]].astype(np.float32)
            if th.shape[0] < n_words:
                pad = np.zeros((n_words - th.shape[0], th.shape[1]), dtype=np.float32)
                th = np.concatenate([th, pad], axis=0)
            th = th[:n_words]

        return Sample(
            uid=uid,
            char_ids=np.asarray(ids, dtype=np.int64),
            word_index=np.asarray(widx, dtype=np.int64),
            n_codes=n_codes,
            code_target=targets,
            codes=codes,
            teacher_hidden=th,
            n_words=n_words,
        )


def collate(
    batch: Sequence[Sample],
    pad_id: int,
    max_codes: int,
    n_quantizers: int,
    teacher_dim: int = 768,
) -> Dict[str, torch.Tensor]:
    """Pad a list of samples into dense batch tensors."""
    B = len(batch)
    L = max(s.char_ids.shape[0] for s in batch)
    W = max(s.n_words for s in batch)
    has_codes = batch[0].codes is not None
    T = max(s.codes.shape[0] for s in batch) if has_codes else 0
    has_teacher = any(s.teacher_hidden is not None for s in batch)

    char_ids = np.full((B, L), pad_id, dtype=np.int64)
    word_index = np.full((B, L), -1, dtype=np.int64)
    char_pad = np.ones((B, L), dtype=bool)
    n_codes = np.zeros((B, W), dtype=np.int64)
    code_target = np.full((B, W), -1, dtype=np.int64)
    word_mask = np.zeros((B, W), dtype=bool)
    # Per-character pronunciation code; `max_codes` is the "no code" slot.
    pc_per_char = np.full((B, L), max_codes, dtype=np.int64)

    codes = np.zeros((B, T, n_quantizers), dtype=np.int64) if has_codes else None
    frame_mask = np.zeros((B, T), dtype=bool) if has_codes else None
    n_frames = np.zeros((B,), dtype=np.int64)
    teacher = np.zeros((B, W, teacher_dim), dtype=np.float32) if has_teacher else None

    for i, s in enumerate(batch):
        l = s.char_ids.shape[0]
        char_ids[i, :l] = s.char_ids
        word_index[i, :l] = s.word_index
        char_pad[i, :l] = False
        w = s.n_words
        n_codes[i, :w] = s.n_codes
        code_target[i, :w] = s.code_target
        word_mask[i, :w] = True

        # Broadcast the ground-truth code onto the characters of its word. An
        # unambiguous word, or one with no label, gets the null slot.
        for wi in range(w):
            if s.n_codes[wi] > 1 and s.code_target[wi] >= 0:
                sel = (s.word_index == wi)
                pc_per_char[i, :l][sel] = int(s.code_target[wi])

        if has_codes:
            t = s.codes.shape[0]
            codes[i, :t] = s.codes
            frame_mask[i, :t] = True
            n_frames[i] = t
        if has_teacher and s.teacher_hidden is not None:
            th = s.teacher_hidden
            teacher[i, : th.shape[0]] = th[:W]

    out = {
        "char_ids": torch.from_numpy(char_ids),
        "word_index": torch.from_numpy(word_index),
        "char_padding_mask": torch.from_numpy(char_pad),
        "n_codes": torch.from_numpy(n_codes),
        "code_target": torch.from_numpy(code_target),
        "word_mask": torch.from_numpy(word_mask),
        "pc_per_char": torch.from_numpy(pc_per_char),
        "n_frames": torch.from_numpy(n_frames),
    }
    if has_codes:
        out["codes"] = torch.from_numpy(codes)
        out["frame_mask"] = torch.from_numpy(frame_mask)
    if has_teacher:
        out["teacher_hidden"] = torch.from_numpy(teacher)
    return out


class Collator:
    """Picklable ``collate_fn`` for :class:`torch.utils.data.DataLoader`.

    Windows spawns dataloader workers instead of forking them, so the collate
    function has to survive pickling. A lambda does not, which makes
    ``num_workers > 0`` fail outright. This class carries the same arguments and
    is picklable, so worker counts behave the same on Windows and Linux.
    """

    def __init__(
        self,
        pad_id: int,
        max_codes: int,
        n_quantizers: int,
        teacher_dim: int = 768,
    ) -> None:
        self.pad_id = pad_id
        self.max_codes = max_codes
        self.n_quantizers = n_quantizers
        self.teacher_dim = teacher_dim

    def __call__(self, batch: Sequence[Sample]) -> Dict[str, torch.Tensor]:
        return collate(
            batch, self.pad_id, self.max_codes, self.n_quantizers, self.teacher_dim
        )


class LengthBucketSampler(Sampler[List[int]]):
    """Groups similar-length utterances so padding waste stays low.

    Buckets are shuffled every epoch and each bucket is shuffled internally, so
    the model still sees varied batches while padding stays near the minimum.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        boundaries: Sequence[int],
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.boundaries = list(boundaries)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.buckets: List[List[int]] = [[] for _ in range(len(self.boundaries) + 1)]
        for i, l in enumerate(self.lengths):
            b = int(np.searchsorted(self.boundaries, l, side="right"))
            self.buckets[b].append(i)
        sizes = [len(b) for b in self.buckets]
        logger.info("length buckets: %s (boundaries %s)", sizes, self.boundaries)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: List[List[int]] = []
        for bucket in self.buckets:
            if not bucket:
                continue
            idx = np.array(bucket)
            if self.shuffle:
                idx = idx[rng.permutation(len(idx))]
            for s in range(0, len(idx), self.batch_size):
                chunk = idx[s : s + self.batch_size].tolist()
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                batches.append(chunk)
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return iter(batches)

    def __len__(self) -> int:
        n = 0
        for bucket in self.buckets:
            if not bucket:
                continue
            if self.drop_last:
                n += len(bucket) // self.batch_size
            else:
                n += (len(bucket) + self.batch_size - 1) // self.batch_size
        return n
