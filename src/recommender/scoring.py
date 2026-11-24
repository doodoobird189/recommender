"""Scoring models shared by offline evaluation and the web app.

Keeping a single implementation means the metrics in the README are measured on exactly the
code path that serves recommendations.

* ``ALSScorer``      collaborative filtering. Users are *folded in* from their interactions against
                     fixed item factors, so brand-new users (the web demo) are handled identically
                     to training users and need no retraining.
* ``ContentScorer``  cosine similarity between a user's tag-genome taste profile and each movie.
* ``blend``          per-user normalisation of both signals (see _scale_to_top), then a
                     weighted sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .genome import Genome

# Ratings above this pull the content profile towards a movie, ratings below push away from it.
NEUTRAL_RATING = 2.5


@dataclass
class FoldIn:
    x: np.ndarray  # user factor vector
    cho: tuple  # Cholesky factor of A = YᵀCᵤY + λI, reused for explanations
    items: np.ndarray  # item indices the user interacted with
    conf: np.ndarray  # their confidences


class ALSScorer:
    def __init__(self, item_factors: np.ndarray, regularization: float):
        self.Y = np.ascontiguousarray(item_factors, dtype=np.float32)
        y64 = self.Y.astype(np.float64)
        self._A0 = y64.T @ y64 + regularization * np.eye(self.Y.shape[1])

    @property
    def n_items(self) -> int:
        return self.Y.shape[0]

    def fold_in(self, items: np.ndarray, conf: np.ndarray) -> FoldIn:
        """Solve the ALS user step  x = (YᵀY + Yᵀ(Cᵤ−I)Y + λI)⁻¹ YᵀCᵤp(u)  for one user."""
        items = np.asarray(items, dtype=np.int64)
        conf = np.asarray(conf, dtype=np.float64)
        yu = self.Y[items].astype(np.float64)
        a = self._A0 + (yu.T * (conf - 1.0)) @ yu
        b = yu.T @ conf
        cho = cho_factor(a)
        return FoldIn(x=cho_solve(cho, b), cho=cho, items=items, conf=conf)

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Scores for one user vector (f,) or a batch (B, f)."""
        return np.asarray(x, dtype=np.float32) @ self.Y.T

    def contributions(self, fold: FoldIn, targets: np.ndarray) -> np.ndarray:
        """Decompose each target item's score into per-interaction contributions.

        score(t) = y_tᵀ A⁻¹ b = Σ_i c_i · (y_tᵀ A⁻¹ y_i), so entry [t, i] is how much the user's
        interaction with item i pushed item t up. Rows sum exactly to the item's score.
        (Hu, Koren & Volinsky 2008, section 5.)
        """
        w = cho_solve(fold.cho, self.Y[targets].astype(np.float64).T)  # (f, T)
        return (self.Y[fold.items].astype(np.float64) @ w).T * fold.conf  # (T, n_interactions)


class ContentScorer:
    def __init__(self, genome: Genome):
        self.genome = genome

    def profile(self, items: np.ndarray, ratings: np.ndarray) -> np.ndarray | None:
        """Unit-length taste vector in genome space, or None if no rated movie has genome data."""
        items = np.asarray(items)
        w = np.asarray(ratings, dtype=np.float32) - NEUTRAL_RATING
        mask = self.genome.has[items] & (w != 0)
        if not mask.any():
            return None
        p = w[mask] @ self.genome.unit[items[mask]]
        norm = np.linalg.norm(p)
        return p / norm if norm > 0 else None

    def taste(self, items: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """Rating-weighted mean tag z-scores: interpretable ("+2.1σ dark")."""
        items = np.asarray(items)
        w = np.asarray(ratings, dtype=np.float32) - NEUTRAL_RATING
        mask = self.genome.has[items]
        if not mask.any() or not np.abs(w[mask]).sum():
            return np.zeros(self.genome.z.shape[1], dtype=np.float32)
        return (w[mask] @ self.genome.z[items[mask]]) / np.abs(w[mask]).sum()

    def scores(self, profile: np.ndarray) -> np.ndarray:
        """Cosine similarity of every movie to a profile (or a (B, tags) batch). 0 if no genome."""
        return np.asarray(profile, dtype=np.float32) @ self.genome.unit.T


def _scale_to_top(scores: np.ndarray, valid: np.ndarray, head: int = 10) -> np.ndarray:
    """Row-wise affine rescale: catalogue mean -> 0, mean of the top-`head` scores -> 1.

    A plain z-score is not comparable across signals here: ALS scores are heavy-tailed (top items
    sit at z≈12) while genome cosine tops out at z≈3, so under z-scoring ALS dominates any blend
    and cold-start items can never surface. Anchoring both signals at the head of the ranking —
    the only region a top-K list ever sees — makes the blend weight behave as intended.
    Invalid entries -> 0 (neutral).
    """
    s = np.where(valid, scores, np.nan)
    mean = np.nanmean(s, axis=-1, keepdims=True)
    filled = np.where(valid, scores, -np.inf)
    k = min(head, filled.shape[-1])
    top = -np.partition(-filled, k - 1, axis=-1)[..., :k]
    top_mean = np.mean(np.where(np.isfinite(top), top, np.nan), axis=-1, keepdims=True)
    span = np.nan_to_num(top_mean - mean, nan=0.0)
    span = np.where(span > 0, span, 1.0)
    return np.where(valid, (scores - np.nan_to_num(mean)) / span, 0.0).astype(np.float32)


def blend_parts(
    als: np.ndarray,
    content: np.ndarray,
    cf_weight: float,
    has_genome: np.ndarray,
    exclude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The two weighted terms of the hybrid score: (w·norm(ALS), (1−w)·norm(content)).

    Both signals are normalised per user over the candidate set (unseen items) so the weight is
    meaningful regardless of their raw scales. Movies without genome data get a neutral (0)
    content term, so they are ranked purely by ALS rather than penalised.
    """
    candidates = ~exclude
    n_als = _scale_to_top(als, candidates)
    n_content = _scale_to_top(content, candidates & has_genome)
    return cf_weight * n_als, (1.0 - cf_weight) * n_content


def blend(
    als: np.ndarray,
    content: np.ndarray,
    cf_weight: float,
    has_genome: np.ndarray,
    exclude: np.ndarray,
) -> np.ndarray:
    """Hybrid score = w·norm(ALS) + (1−w)·norm(content)."""
    cf_part, content_part = blend_parts(als, content, cf_weight, has_genome, exclude)
    return cf_part + content_part


def top_k(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k highest scores per row (descending), skipping excluded items."""
    s = np.array(scores, dtype=np.float32, copy=True, ndmin=2)
    if exclude is not None:
        s[np.broadcast_to(exclude, s.shape)] = -np.inf
    k = min(k, s.shape[1])
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(part, order, axis=1)
    return top if np.ndim(scores) > 1 else top[0]
