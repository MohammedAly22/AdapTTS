"""Pronunciation-code discovery.

This is the novel core of AdapTTS. Given, for each occurrence of each word type,
an acoustic embedding of how that word was *actually pronounced*, we ask:

    does this word type have one pronunciation, or several?

and if several, we assign each occurrence a discrete code. Those codes become
the supervision signal for the context encoder. No homograph list, no
diacritization rules, no lexicon is involved: the number of readings per word
is discovered, and it is discovered from audio.

The statistical problem is that k-means will happily split any cloud of points.
We must only declare a split when it is *real*. Four independent gates:

1. **Bootstrap stability.** Refit on resampled subsets; if the partition is not
   reproducible (Fowlkes-Mallows below threshold), it is noise.
2. **Silhouette margin.** The clustering must be geometrically better than the
   k=1 null hypothesis by a margin.
3. **Separation gate.** Centroids must be far apart relative to within-cluster
   spread, which rejects splits whose clusters merely touch.
4. **Duration-confound gate.** A split that span duration alone explains is
   prosody, not pronunciation.

Gate 4 was added after observing the first three pass on real speech for the
wrong reason. On a 3-hour Egyptian corpus the accepted splits separated the
elongated realization of a word from the fast one: for عارفين the two clusters
averaged 0.27 s and 1.15 s. The clusters were stable, clean and well separated,
and they were not two readings. The cause is structural, since a mean-pooled
self-supervised embedding over a span correlates with how long that span is, so
the gate measures that correlation directly and rejects the split when it
dominates.

Only a split passing all four is accepted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small, dependency-free numerical helpers
# ---------------------------------------------------------------------------


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def pca_reduce(x: np.ndarray, dim: int) -> np.ndarray:
    """Project to ``dim`` principal components via SVD on the centred matrix."""
    if x.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {x.shape}")
    n, d = x.shape
    k = int(min(dim, d, max(1, n - 1)))
    xc = x - x.mean(axis=0, keepdims=True)
    # economy SVD; for our sizes (<=600 x 1024) this is microseconds
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:k]
    out = xc @ comps.T
    # Whiten so no single high-variance direction (usually loudness) dominates.
    scale = s[:k] / np.sqrt(max(n - 1, 1))
    out = out / (scale + 1e-6)
    return out


def kmeans(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 6,
    max_iter: int = 120,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """k-means with k-means++ init. Returns ``(labels, centroids, inertia)``."""
    n, d = x.shape
    if k <= 1:
        centroid = x.mean(axis=0, keepdims=True)
        inertia = float(((x - centroid) ** 2).sum())
        return np.zeros(n, dtype=np.int64), centroid, inertia
    if k > n:
        raise ValueError(f"k={k} exceeds n={n}")

    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    for _ in range(n_init):
        # k-means++ seeding
        centroids = np.empty((k, d), dtype=x.dtype)
        centroids[0] = x[rng.integers(n)]
        closest = ((x - centroids[0]) ** 2).sum(axis=1)
        for ci in range(1, k):
            total = closest.sum()
            if total <= 0:
                centroids[ci] = x[rng.integers(n)]
            else:
                probs = closest / total
                centroids[ci] = x[rng.choice(n, p=probs)]
            closest = np.minimum(closest, ((x - centroids[ci]) ** 2).sum(axis=1))
        # (seeding stays point-wise: k is small and this runs once per init)

        labels = np.zeros(n, dtype=np.int64)
        prev_inertia = np.inf
        inertia = np.inf
        x_sq = (x * x).sum(axis=1)[:, None]
        for _ in range(max_iter):
            # Gram-matrix distances: one matmul instead of an n x k x d broadcast.
            d2 = x_sq + (centroids * centroids).sum(axis=1)[None, :] - 2.0 * (x @ centroids.T)
            np.maximum(d2, 0.0, out=d2)
            labels = d2.argmin(axis=1)
            inertia = float(d2[np.arange(n), labels].sum())
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    centroids[ci] = x[mask].mean(axis=0)
                else:
                    # Re-seed an empty cluster at the worst-fit point.
                    centroids[ci] = x[int(d2[np.arange(n), labels].argmax())]
            if abs(prev_inertia - inertia) <= tol * max(abs(prev_inertia), 1.0):
                break
            prev_inertia = inertia

        if best is None or inertia < best[2]:
            best = (labels.copy(), centroids.copy(), inertia)

    assert best is not None
    return best


def _pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances via the Gram matrix.

    The expanded form ||a||^2 - 2a.b + ||b||^2 is one BLAS matmul, where the
    broadcast form materializes an n x n x d array. At n=600, d=48 that is the
    difference between a few milliseconds and tens of them, before counting the
    memory traffic.
    """
    sq = (x * x).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient. Returns 0.0 for degenerate partitions.

    Vectorized: the per-point Python loop that this replaces cost 91 ms at
    n=600, and it runs once per candidate k. Same quantity, same thresholds.
    """
    uniq = np.unique(labels)
    n = x.shape[0]
    k = uniq.size
    if k < 2 or n <= k:
        return 0.0

    dist = np.sqrt(_pairwise_sq_dists(np.ascontiguousarray(x, dtype=np.float64)))

    # One-hot membership lets every per-cluster sum become a single matmul.
    onehot = np.zeros((n, k), dtype=np.float64)
    remap = {c: i for i, c in enumerate(uniq)}
    idx = np.array([remap[c] for c in labels])
    onehot[np.arange(n), idx] = 1.0

    sums = dist @ onehot          # [n, k] distance from each point to each cluster
    counts = onehot.sum(axis=0)   # [k]

    own_counts = counts[idx] - 1.0
    own_sums = sums[np.arange(n), idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(own_counts > 0, own_sums / np.maximum(own_counts, 1.0), 0.0)

        means = sums / np.maximum(counts[None, :], 1.0)
        means[np.arange(n), idx] = np.inf   # exclude a point's own cluster
        means[:, counts == 0] = np.inf
        b = means.min(axis=1)

    valid = (own_counts > 0) & np.isfinite(b)
    sil = np.zeros(n, dtype=np.float64)
    denom = np.maximum(np.maximum(a, b), 1e-12)
    sil[valid] = ((b - a) / denom)[valid]
    return float(sil.mean())


def fowlkes_mallows(a: np.ndarray, b: np.ndarray) -> float:
    """Fowlkes-Mallows index between two labellings of the same points.

    Computed from the contingency table rather than from two n x n boolean
    matrices. The pair counts follow directly from the table:

        TP + FP = sum over clusters of b of C(|cluster|, 2)
        TP + FN = the same over clusters of a
        TP      = the same over each cell of the table

    That is O(n + k^2) instead of O(n^2), and exact.
    """
    if a.shape != b.shape:
        raise ValueError("labellings must have the same length")
    n = a.shape[0]
    if n < 2:
        return 1.0

    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ua.size, ub.size), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)

    def pairs(counts: np.ndarray) -> float:
        c = counts.astype(np.float64)
        return float((c * (c - 1.0) / 2.0).sum())

    tp = pairs(table)
    tp_fn = pairs(table.sum(axis=1))
    tp_fp = pairs(table.sum(axis=0))

    denom = np.sqrt(tp_fp * tp_fn)
    return float(tp / denom) if denom > 0 else 0.0


def duration_confound(labels: np.ndarray, durations: np.ndarray) -> float:
    """How much of the split is explained by span duration alone?

    Returns the between-group share of total duration variance, an R-squared in
    [0, 1]. A value near 1 means the clusters are a fast/slow split rather than
    two pronunciations.

    This gate exists because mean-pooled SSL embeddings correlate with span
    length, so a clustering can look stable, well separated and clean while
    only having discovered that a word is sometimes drawn out. Measured on
    Egyptian conversational speech, unfiltered discovery returned exactly that:
    clusters whose mean durations differed four-fold.
    """
    if durations.size != labels.size or durations.size < 4:
        return 0.0
    total = float(((durations - durations.mean()) ** 2).sum())
    if total <= 1e-12:
        return 0.0
    within = 0.0
    for c in np.unique(labels):
        grp = durations[labels == c]
        if grp.size:
            within += float(((grp - grp.mean()) ** 2).sum())
    return max(0.0, min(1.0, 1.0 - within / total))


def cluster_separation(x: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Minimum centroid gap divided by pooled within-cluster spread.

    Values near 0 mean the clusters overlap (the split is cosmetic); values
    above ~0.5 mean genuinely distinct acoustic realizations.
    """
    k = centroids.shape[0]
    if k < 2:
        return 0.0
    spreads = []
    for ci in range(k):
        pts = x[labels == ci]
        if pts.shape[0] == 0:
            return 0.0
        spreads.append(np.sqrt(((pts - centroids[ci]) ** 2).sum(axis=1)).mean())
    pooled = float(np.mean(spreads))
    gaps = [
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(k)
        for j in range(i + 1, k)
    ]
    return float(min(gaps) / (pooled + 1e-8)) / 2.0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@dataclass
class WordCodes:
    """The discovered pronunciation inventory for one orthographic word type."""

    word: str
    n_codes: int
    centroids: np.ndarray  # [n_codes, pca_dim] in the reduced space
    projection: np.ndarray  # [pca_dim, feat_dim] PCA basis (for new occurrences)
    mean: np.ndarray  # [feat_dim] centring vector
    scale: np.ndarray  # [pca_dim] whitening scale
    counts: List[int]
    stability: float
    silhouette: float
    separation: float
    occurrence_count: int

    def assign(self, feats: np.ndarray) -> np.ndarray:
        """Assign new occurrences (``[n, feat_dim]``) to codes."""
        if self.n_codes == 1:
            return np.zeros(feats.shape[0], dtype=np.int64)
        feats = _l2_normalize(np.asarray(feats, dtype=np.float64))
        z = (feats - self.mean[None, :]) @ self.projection.T
        z = z / (self.scale[None, :] + 1e-8)
        d2 = ((z[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return d2.argmin(axis=1).astype(np.int64)

    def to_json(self) -> dict:
        return {
            "word": self.word,
            "n_codes": self.n_codes,
            "centroids": self.centroids.astype(np.float32).tolist(),
            "projection": self.projection.astype(np.float32).tolist(),
            "mean": self.mean.astype(np.float32).tolist(),
            "scale": self.scale.astype(np.float32).tolist(),
            "counts": list(self.counts),
            "stability": float(self.stability),
            "silhouette": float(self.silhouette),
            "separation": float(self.separation),
            "occurrence_count": int(self.occurrence_count),
        }

    @classmethod
    def from_json(cls, d: dict) -> "WordCodes":
        return cls(
            word=d["word"],
            n_codes=int(d["n_codes"]),
            centroids=np.asarray(d["centroids"], dtype=np.float32),
            projection=np.asarray(d["projection"], dtype=np.float32),
            mean=np.asarray(d["mean"], dtype=np.float32),
            scale=np.asarray(d["scale"], dtype=np.float32),
            counts=list(d["counts"]),
            stability=float(d["stability"]),
            silhouette=float(d["silhouette"]),
            separation=float(d["separation"]),
            occurrence_count=int(d["occurrence_count"]),
        )


def _fit_projection(
    feats: np.ndarray, dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(z, projection, mean, scale)`` for reproducible re-projection.

    Note on scaling: we deliberately do **not** whiten per component. Per
    component whitening divides every principal direction by its own standard
    deviation, which rescales a genuine between-reading axis down to the same
    magnitude as a pure noise axis and destroys exactly the separation we are
    trying to measure. Instead we apply a single global scale, which preserves
    the relative geometry while keeping distances in a numerically sane range.
    """
    n, d = feats.shape
    k = int(min(dim, d, max(1, n - 1)))
    mean = feats.mean(axis=0)
    xc = feats - mean[None, :]
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    projection = vt[:k]
    proj = xc @ projection.T
    # One shared scale for all components: preserves geometry, normalizes units.
    global_scale = float(np.sqrt((proj**2).sum(axis=1).mean())) + 1e-8
    scale = np.full(k, global_scale, dtype=np.float64)
    z = proj / global_scale
    return (
        z.astype(np.float32),
        projection.astype(np.float32),
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


def discover_word_codes(
    word: str,
    feats: np.ndarray,
    *,
    max_codes: int = 4,
    n_bootstrap: int = 24,
    bootstrap_frac: float = 0.8,
    stability_threshold: float = 0.72,
    silhouette_threshold: float = 0.10,
    min_separation: float = 0.55,
    min_cluster_frac: float = 0.12,
    pca_dim: int = 48,
    seed: int = 0,
    durations: Optional[np.ndarray] = None,
    max_duration_confound: float = 0.5,
) -> WordCodes:
    """Decide how many pronunciations ``word`` has, and cluster its occurrences.

    ``feats`` is ``[n_occurrences, feat_dim]`` of acoustic span embeddings.
    Returns a :class:`WordCodes`; ``n_codes == 1`` means unambiguous.

    ``durations`` is the span length of each occurrence in seconds. When given,
    a split whose duration R-squared exceeds ``max_duration_confound`` is
    rejected as prosody rather than pronunciation. Strongly recommended: without
    it, discovery reliably finds fast/slow splits instead of homographs.
    """
    if feats.ndim != 2:
        raise ValueError(f"feats must be 2-D, got {feats.shape}")
    n = feats.shape[0]
    rng = np.random.default_rng(seed)

    feats = _l2_normalize(feats.astype(np.float64))
    z, projection, mean, scale = _fit_projection(feats, pca_dim)
    z = z.astype(np.float64)

    def _unambiguous(reason: str) -> WordCodes:
        logger.debug("word %s -> 1 code (%s)", word, reason)
        return WordCodes(
            word=word,
            n_codes=1,
            centroids=z.mean(axis=0, keepdims=True).astype(np.float32),
            projection=projection,
            mean=mean,
            scale=scale,
            counts=[n],
            stability=1.0,
            silhouette=0.0,
            separation=0.0,
            occurrence_count=n,
        )

    # Need enough occurrences that a minority reading could even be seen.
    min_needed = int(np.ceil(2.0 / max(min_cluster_frac, 1e-6)))
    if n < max(8, min_needed):
        return _unambiguous(f"too few occurrences ({n})")

    best: Optional[Tuple[int, np.ndarray, np.ndarray, float, float, float]] = None

    for k in range(2, min(max_codes, n // 2) + 1):
        labels, centroids, _ = kmeans(z, k, np.random.default_rng(seed + k))

        counts = np.bincount(labels, minlength=k)
        if (counts < max(2, int(np.ceil(min_cluster_frac * n)))).any():
            continue  # a code nobody uses is not a reading

        sil = silhouette_score(z, labels)
        if sil < silhouette_threshold:
            continue

        sep = cluster_separation(z, labels, centroids)
        if sep < min_separation:
            continue

        # Reject a split that duration alone explains. A mean-pooled embedding
        # correlates with span length, so without this the procedure finds
        # elongated-versus-fast realizations and calls them separate readings.
        if durations is not None:
            conf = duration_confound(labels, np.asarray(durations, dtype=np.float64))
            if conf > max_duration_confound:
                logger.debug(
                    "word %s k=%d rejected: duration explains %.0f%% of the split",
                    word, k, 100 * conf,
                )
                continue

        # Bootstrap stability: refit on subsamples and compare partitions
        # restricted to the shared points.
        agreements: List[float] = []
        m = max(4, int(round(bootstrap_frac * n)))
        for b in range(n_bootstrap):
            brng = np.random.default_rng(seed * 1000 + k * 97 + b)
            idx = brng.choice(n, size=m, replace=False)
            if np.unique(labels[idx]).size < k:
                continue
            sub_labels, _, _ = kmeans(z[idx], k, brng, n_init=3, max_iter=60)
            agreements.append(fowlkes_mallows(labels[idx], sub_labels))
        stability = float(np.mean(agreements)) if agreements else 0.0
        if stability < stability_threshold:
            continue

        score = stability * sil * sep
        if best is None or score > best[3] * best[4] * best[5]:
            best = (k, labels, centroids, stability, sil, sep)

    if best is None:
        return _unambiguous("no split passed the gates")

    k, labels, centroids, stability, sil, sep = best
    # Order codes by frequency so code 0 is always the dominant reading. This
    # makes the codes stable and human-readable across runs.
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    remap = np.zeros(k, dtype=np.int64)
    remap[order] = np.arange(k)
    labels = remap[labels]
    centroids = centroids[order]
    counts = counts[order]

    logger.info(
        "word %s -> %d codes (counts=%s stability=%.3f sil=%.3f sep=%.3f)",
        word, k, counts.tolist(), stability, sil, sep,
    )
    final_confound = (
        duration_confound(labels, np.asarray(durations, dtype=np.float64))
        if durations is not None else 0.0
    )
    logger.info("   duration confound for %s: %.2f", word, final_confound)
    return WordCodes(
        word=word,
        n_codes=k,
        centroids=centroids.astype(np.float32),
        projection=projection,
        mean=mean,
        scale=scale,
        counts=counts.tolist(),
        stability=stability,
        silhouette=sil,
        separation=sep,
        occurrence_count=n,
    )


class PronunciationLexicon:
    """The discovered inventory: word type -> pronunciation codes.

    Despite the name this is *not* a handwritten pronunciation dictionary. It is
    the serialized output of the discovery procedure and is regenerated from
    audio whenever the dataset changes.
    """

    def __init__(self, entries: Dict[str, WordCodes], max_codes: int) -> None:
        self.entries = entries
        self.max_codes = max_codes

    def __len__(self) -> int:
        return len(self.entries)

    def n_codes(self, word: str) -> int:
        e = self.entries.get(word)
        return e.n_codes if e is not None else 1

    def is_ambiguous(self, word: str) -> bool:
        return self.n_codes(word) > 1

    @property
    def ambiguous_words(self) -> List[str]:
        return sorted(w for w, e in self.entries.items() if e.n_codes > 1)

    def assign(self, word: str, feats: np.ndarray) -> np.ndarray:
        e = self.entries.get(word)
        if e is None:
            return np.zeros(feats.shape[0], dtype=np.int64)
        return e.assign(feats)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_codes": self.max_codes,
            "entries": [e.to_json() for e in self.entries.values()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "PronunciationLexicon":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        entries = {}
        for d in payload["entries"]:
            wc = WordCodes.from_json(d)
            entries[wc.word] = wc
        return cls(entries, int(payload["max_codes"]))
