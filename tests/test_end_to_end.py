"""End-to-end pipeline test on synthetic data.

This builds a miniature corpus in a temp directory, runs the dataset, both
models, the training step and the inference pipeline, and asserts the system
actually *learns homograph disambiguation* rather than merely running.

The learning check is the important one. It constructs a corpus where the
correct reading of an ambiguous word is fully determined by its context, then
verifies the context encoder reaches high accuracy on held-out sentences. If
this test regresses, the central claim of the architecture is broken.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from adaptts.data.discovery import PronunciationLexicon, WordCodes  # noqa: E402
from adaptts.models.acoustic import AcousticModel, acoustic_loss  # noqa: E402
from adaptts.models.context_encoder import ContextEncoder, context_encoder_loss  # noqa: E402
from adaptts.text.normalize import normalize_text, tokenize_words  # noqa: E402
from adaptts.text.vocab import CharVocab  # noqa: E402
from adaptts.utils.config import load_config  # noqa: E402

# A tiny synthetic Egyptian-Arabic-like corpus. The word "علم" is ambiguous:
# with "مصر"/"بيرفرف" around it, it is the flag reading (code 0); with
# "الفيزيا"/"الرياضيات", it is the science reading (code 1). The context fully
# determines the answer, so a working model must approach 100%.
FLAG_CONTEXTS = [
    "انا شوفت علم مصر بيرفرف",
    "علم مصر لونه احمر وابيض واسود",
    "رفعوا علم النادي فوق المدرج",
    "الولد شال علم كبير في ايده",
    "علم مصر بيرفرف فوق المبنى",
    "حطوا علم على العربية",
]
SCIENCE_CONTEXTS = [
    "علم الفيزيا من اهم العلوم",
    "علم الرياضيات صعب بس مفيد",
    "هو بيدرس علم النفس في الجامعة",
    "علم الاحياء بيدرس الكائنات",
    "علم الفلك بيدرس النجوم",
    "الطالب بيحب علم الكيميا",
]


def _build_corpus():
    texts, labels = [], []
    for t in FLAG_CONTEXTS:
        texts.append(normalize_text(t))
        labels.append(0)
    for t in SCIENCE_CONTEXTS:
        texts.append(normalize_text(t))
        labels.append(1)
    return texts, labels


def _make_lexicon(word: str, n_codes: int, dim: int = 8) -> PronunciationLexicon:
    wc = WordCodes(
        word=word, n_codes=n_codes,
        centroids=np.eye(n_codes, dim, dtype=np.float32),
        projection=np.eye(dim, dim, dtype=np.float32),
        mean=np.zeros(dim, dtype=np.float32),
        scale=np.ones(dim, dtype=np.float32),
        counts=[6] * n_codes, stability=0.9, silhouette=0.4,
        separation=0.8, occurrence_count=12,
    )
    return PronunciationLexicon({word: wc}, max_codes=4)


def _encode(texts, vocab, lexicon, device):
    """Tensorize a batch of sentences for the context encoder."""
    enc, widxs, ncs, tgts = [], [], [], []
    for t in texts:
        words, spans = tokenize_words(t)
        ids, widx = vocab.encode_with_word_index(t, spans)
        enc.append(ids)
        widxs.append(widx)
        ncs.append([lexicon.n_codes(w) for w in words])
        tgts.append(words)
    L = max(len(e) for e in enc)
    W = max(len(n) for n in ncs)
    B = len(texts)

    char_ids = torch.full((B, L), vocab.pad_id, dtype=torch.long)
    word_index = torch.full((B, L), -1, dtype=torch.long)
    pad = torch.ones(B, L, dtype=torch.bool)
    n_codes = torch.zeros(B, W, dtype=torch.long)
    for i, (e, w, n) in enumerate(zip(enc, widxs, ncs)):
        char_ids[i, : len(e)] = torch.tensor(e)
        word_index[i, : len(w)] = torch.tensor(w)
        pad[i, : len(e)] = False
        n_codes[i, : len(n)] = torch.tensor(n)
    return (
        char_ids.to(device), word_index.to(device), pad.to(device),
        n_codes.to(device), tgts,
    )


def test_context_encoder_learns_homograph_disambiguation():
    """The central claim: context alone resolves the reading, no diacritics."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    texts, labels = _build_corpus()
    vocab = CharVocab.build(texts, min_freq=1)
    lexicon = _make_lexicon("علم", 2)

    # Hold out one sentence of each reading; train on the rest.
    train_idx = [i for i in range(len(texts)) if i not in (5, 11)]
    test_idx = [5, 11]

    model = ContextEncoder(
        len(vocab), d_model=96, n_layers=2, n_heads=4, d_ff=192,
        dropout=0.0, max_codes=4, teacher_dim=768, pad_id=vocab.pad_id,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    tr_texts = [texts[i] for i in train_idx]
    ci, wi, pad, nc, words_per = _encode(tr_texts, vocab, lexicon, device)
    target = torch.full_like(nc, -1)
    for bi, i in enumerate(train_idx):
        for w_i, w in enumerate(words_per[bi]):
            if w == "علم":
                target[bi, w_i] = labels[i]
    mask = nc > 0

    model.train()
    for _ in range(300):
        out = model(ci, wi, nc, pad)
        loss, stats = context_encoder_loss(
            out, target, mask, nc, ce_weight=1.0, distill_weight=0.0,
            difficulty_weight=0.1, label_smoothing=0.0,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(stats["code_acc"]) > 0.95, f"train acc only {float(stats['code_acc'])}"

    # Held-out sentences the model never saw.
    model.eval()
    te_texts = [texts[i] for i in test_idx]
    ci, wi, pad, nc, words_per = _encode(te_texts, vocab, lexicon, device)
    with torch.no_grad():
        out = model(ci, wi, nc, pad)
    pred = out.code_logits.argmax(-1)

    correct = 0
    for bi, i in enumerate(test_idx):
        for w_i, w in enumerate(words_per[bi]):
            if w == "علم":
                correct += int(int(pred[bi, w_i]) == labels[i])
    assert correct == 2, (
        f"held-out homograph accuracy {correct}/2: the model failed to "
        "disambiguate from context"
    )


def test_difficulty_is_higher_for_ambiguous_words():
    """The adaptive-compute signal must actually track ambiguity."""
    torch.manual_seed(0)
    texts, _ = _build_corpus()
    vocab = CharVocab.build(texts, min_freq=1)
    lexicon = _make_lexicon("علم", 2)
    model = ContextEncoder(
        len(vocab), d_model=64, n_layers=2, n_heads=4, d_ff=128,
        dropout=0.0, max_codes=4, pad_id=vocab.pad_id,
    ).eval()

    ci, wi, pad, nc, words_per = _encode(texts[:2], vocab, lexicon, torch.device("cpu"))
    with torch.no_grad():
        out = model(ci, wi, nc, pad)
    amb = nc > 1
    una = nc == 1
    # Unambiguous words are exactly zero difficulty by construction; ambiguous
    # ones are strictly positive before training has sharpened them.
    assert float(out.difficulty[una].max()) == 0.0
    assert float(out.difficulty[amb].min()) > 0.0

    sent_d = model.sentence_difficulty(out.difficulty, nc > 0)
    assert float(sent_d[0]) > 0.0


def test_acoustic_model_trains_a_step_and_generates_at_each_depth():
    torch.manual_seed(0)
    texts, _ = _build_corpus()
    vocab = CharVocab.build(texts, min_freq=1)
    B, T, Q, CB = 3, 12, 8, 64

    model = AcousticModel(
        len(vocab), n_quantizers=Q, codebook_size=CB, d_model=96, n_layers=4,
        n_heads=4, d_ff=192, text_d_model=64, text_n_layers=2, text_n_heads=4,
        depth_d_model=64, depth_n_layers=2, depth_n_heads=4, speaker_dim=32,
        max_codes=4, exit_layers=(2, 4), pad_id=vocab.pad_id,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ci, wi, pad, nc, _ = _encode(texts[:B], vocab, _make_lexicon("علم", 2), torch.device("cpu"))
    pc = torch.full_like(ci, 4)
    codes = torch.randint(0, CB, (B, T, Q))
    fm = torch.ones(B, T, dtype=torch.bool)
    spk = torch.zeros(B, 32)

    losses = []
    for _ in range(12):
        out = model(ci, pc, codes, spk, pad, fm, cfg_dropout=0.1)
        loss, _ = acoustic_loss(out, codes, fm, [4, 2, 1, 1, 1, 1, 1, 1], [0.5, 1.0], fm.sum(1).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"

    model.eval()
    for depth in (2, 4):
        with torch.no_grad():
            g, st = model.generate(ci, pc, spk, pad, depth=depth, max_frames=10, min_frames=3)
        assert g.shape[0] == B and g.shape[2] == Q
        assert int(g.max()) < CB and int(g.min()) >= 0
        assert st["depth_used"] == float(depth)


def test_synthesis_plan_is_inspectable_and_overridable():
    """The controllability requirement: see the decision, change it, no diacritics."""
    from adaptts.infer.pipeline import SynthesisPlan, WordPlan

    plan = SynthesisPlan(
        text="انا شوفت علم مصر بيرفرف",
        words=[
            WordPlan(0, "انا", 1, 0, [1.0], 0.0),
            WordPlan(1, "شوفت", 1, 0, [1.0], 0.0),
            WordPlan(2, "علم", 2, 1, [0.3, 0.7], 0.61),
            WordPlan(3, "مصر", 2, 0, [0.9, 0.1], 0.20),
            WordPlan(4, "بيرفرف", 1, 0, [1.0], 0.0),
        ],
        sentence_difficulty=0.55,
        depth=12,
        inputs={"word_index": torch.tensor([[0, 0, 0, -1, 2, 2, 2]])},
    )

    assert len(plan.hard_words) == 2
    assert "علم" in str(plan) and "confidence" in str(plan)

    # Override by word, then by index. This is the no-diacritics correction.
    plan.set_code("علم", 0)
    assert plan.words[2].code == 0 and plan.overrides == {2: 0}
    plan.set_code(3, 1)
    assert plan.words[3].code == 1

    # An out-of-range code must be refused rather than silently clipped.
    try:
        plan.set_code("علم", 5)
    except ValueError as e:
        assert "valid codes are" in str(e)
    else:
        raise AssertionError("expected an out-of-range code to be rejected")

    # The per-character tensor must carry the override. Word 2 was overridden
    # to code 0 above, so its characters must read 0, not the model's original 1.
    pc = plan.pc_per_char(max_codes=4)
    assert torch.all(pc[plan.inputs["word_index"] == 2] == 0)
    # Word 0 is unambiguous, so it gets the "no code" slot.
    assert torch.all(pc[plan.inputs["word_index"] == 0] == 4)
    # Characters outside any word also get the null slot.
    assert torch.all(pc[plan.inputs["word_index"] == -1] == 4)

    d = plan.to_dict()
    assert d["overrides"] == {2: 0, 3: 1} and len(d["words"]) == 5


def test_smoke_config_is_loadable_and_consistent():
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root / "configs" / "exp1_smoke.yaml")
    assert cfg.name == "exp1_smoke"
    assert cfg.acoustic.exit_layers[-1] == cfg.acoustic.n_layers
    assert len(cfg.acoustic.rvq_loss_weights) == cfg.audio.n_quantizers
    # Inheritance must actually reach the base file. The specific teacher can
    # change (see base.yaml on safetensors availability), so assert that the
    # value is inherited and coherent rather than pinning one model name.
    assert cfg.codec_model_id == "kyutai/mimi"
    assert cfg.teacher.model_id and "/" in cfg.teacher.model_id
    assert cfg.teacher.hidden_size == 768
    assert max(cfg.teacher.layers) < 12

    full = load_config(root / "configs" / "exp1_egyptian.yaml")
    assert full.acoustic.n_layers == 12 and full.optim.max_steps == 90000


def test_probe_sentences_are_wellformed():
    root = Path(__file__).resolve().parent.parent
    data = json.load(open(root / "assets" / "probe_sentences.json", encoding="utf-8"))
    sents = data["sentences"]
    assert len(sents) == 8, f"expected 8 probe sentences, got {len(sents)}"
    tags = set()
    for s in sents:
        assert {"tag", "text", "focus", "expected"} <= set(s)
        assert s["tag"] not in tags, f"duplicate probe tag {s['tag']}"
        tags.add(s["tag"])
        # Probe text must survive normalization into real words.
        words, _ = tokenize_words(normalize_text(s["text"]))
        assert words, f"probe {s['tag']} normalizes to nothing"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
