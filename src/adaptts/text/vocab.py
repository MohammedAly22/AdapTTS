"""Character vocabulary, built from the corpus rather than declared.

The vocabulary is derived by counting characters in the training transcripts.
Characters below a frequency floor map to <unk>, which keeps the embedding table
small and makes the model robust to stray symbols at inference time.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD, BOS, EOS, UNK, SPACE = "<pad>", "<bos>", "<eos>", "<unk>", " "
SPECIALS: Tuple[str, ...] = (PAD, BOS, EOS, UNK)


class CharVocab:
    """Bidirectional character/id mapping with word-boundary aware encoding."""

    def __init__(self, chars: Sequence[str]) -> None:
        itos: List[str] = list(SPECIALS)
        seen = set(itos)
        if SPACE not in seen:
            itos.append(SPACE)
            seen.add(SPACE)
        for ch in chars:
            if ch not in seen:
                itos.append(ch)
                seen.add(ch)
        self.itos: List[str] = itos
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]
        self.space_id = self.stoi[SPACE]

    def __len__(self) -> int:
        return len(self.itos)

    # -- building ---------------------------------------------------------

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2) -> "CharVocab":
        counts: Counter = Counter()
        for t in texts:
            counts.update(t)
        for sp in SPECIALS:
            counts.pop(sp, None)
        chars = sorted(ch for ch, n in counts.items() if n >= min_freq and ch != SPACE)
        return cls(chars)

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.itos = list(data["itos"])
        obj.stoi = {ch: i for i, ch in enumerate(obj.itos)}
        for name, tok in (
            ("pad_id", PAD),
            ("bos_id", BOS),
            ("eos_id", EOS),
            ("unk_id", UNK),
            ("space_id", SPACE),
        ):
            if tok not in obj.stoi:
                raise ValueError(f"vocab file {path} is missing the {tok!r} token")
            setattr(obj, name, obj.stoi[tok])
        return obj

    # -- encoding ---------------------------------------------------------

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_bos_eos:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def encode_with_word_index(
        self, text: str, word_spans: Sequence[Tuple[int, int]], add_bos_eos: bool = True
    ) -> Tuple[List[int], List[int]]:
        """Encode text and return, per character, the index of its word.

        Characters outside any word (spaces, punctuation) get ``-1``. This is
        what lets a per-word pronunciation code be broadcast onto the characters
        it governs.
        """
        word_of_char = [-1] * len(text)
        for wi, (s, e) in enumerate(word_spans):
            if s < 0 or e > len(text) or s >= e:
                raise ValueError(f"invalid word span ({s}, {e}) for text of length {len(text)}")
            for ci in range(s, e):
                word_of_char[ci] = wi

        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]
            word_of_char = [-1] + word_of_char + [-1]
        return ids, word_of_char

    def decode(self, ids: Iterable[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        return "".join(self.itos[i] for i in ids if i not in specials)
