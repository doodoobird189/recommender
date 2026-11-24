"""Offline evaluation: baselines vs ALS vs content vs hybrid, under random and temporal splits.

Accuracy metrics (Precision / Recall / NDCG / MAP @K) treat every held-out interaction as relevant
("will the user watch this?"). Beyond-accuracy metrics capture what accuracy hides:

* Coverage@10    share of the catalogue that is ever recommended to anyone
* Novelty@10     mean self-information −log2(popularity) of recommended items (higher = rarer)
* Diversity@10   mean pairwise genome distance within a list (intra-list diversity)
* ColdRecall@20  recall restricted to held-out items that had < N training interactions
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import ALSConfig, EvalConfig
from .data import confidence, to_csr
from .genome import Genome
from .scoring import ALSScorer, ContentScorer, blend, top_k

log = logging.getLogger(__name__)

TrainFn = Callable[[csr_matrix, ALSConfig], np.ndarray]


# ── per-list metrics ─────────────────────────────────────────────────────────


def ranking_metrics(recs: np.ndarray, relevant: set[int], k: int) -> dict[str, float]:
    """Precision, Recall, NDCG and AP at k for one ranked list."""
    hits = np.array([1.0 if i in relevant else 0.0 for i in recs[:k]])
    n_rel = len(relevant)
    ideal = min(n_rel, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(hits @ discounts[: len(hits)])
    idcg = float(discounts[:ideal].sum())
    precision_at_hits = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return {
        "precision": hits.sum() / k,
        "recall": hits.sum() / n_rel,
        "ndcg": dcg / idcg if idcg else 0.0,
        "map": float((precision_at_hits * hits).sum() / ideal) if ideal else 0.0,
    }


def intra_list_diversity(recs: np.ndarray, genome: Genome) -> float | None:
    covered = recs[genome.has[recs]]
    if len(covered) < 2:
        return None
    v = genome.unit[covered]
    sim = v @ v.T
    n = len(covered)
    return float(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))


@dataclass
class _Accumulator:
    k_values: tuple[int, ...]
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    recommended: set[int] = field(default_factory=set)

    def add(self, name: str, value: float | None) -> None:
        if value is None:
            return
        self.sums[name] = self.sums.get(name, 0.0) + value
        self.counts[name] = self.counts.get(name, 0) + 1

    def result(self, n_items: int) -> dict[str, float]:
        out = {k: round(self.sums[k] / self.counts[k], 4) for k in self.sums}
        out["coverage@10"] = round(len(self.recommended) / n_items, 4)
        return out


# ── evaluation of one split ──────────────────────────────────────────────────


def evaluate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_users: int,
    n_items: int,
    genome: Genome,
    als_cfg: ALSConfig,
    eval_cfg: EvalConfig,
    train_fn: TrainFn,
    recent_since: int | None = None,
) -> dict:
    rng = np.random.default_rng(eval_cfg.seed)
    k_max = max(max(eval_cfg.k_values), 20)

    train_ratings = to_csr(train, n_users, n_items)
    train_conf = train_ratings.copy()
    train_conf.data = confidence(train_conf.data, als_cfg.alpha)
    test_csr = to_csr(test, n_users, n_items)

    log.info("Training ALS on the training split...")
    als = ALSScorer(train_fn(train_conf, als_cfg), als_cfg.regularization)
    content = ContentScorer(genome)

    # popularity = number of distinct training users per item
    item_pop = np.bincount(train_ratings.indices, minlength=n_items).astype(np.float64)
    n_train_users = int((np.diff(train_ratings.indptr) > 0).sum())
    self_info = -np.log2((item_pop + 1) / (n_train_users + 1))
    cold_items = item_pop < eval_cfg.cold_item_threshold

    fixed_scores: dict[str, np.ndarray] = {"Popularity (all-time)": np.log1p(item_pop)}
    if recent_since is not None:
        recent = train[train["timestamp"] >= recent_since]
        fixed_scores["Popularity (last 12 months)"] = np.log1p(
            np.bincount(recent["item_idx"], minlength=n_items).astype(np.float64)
        )

    eligible = np.flatnonzero((np.diff(train_ratings.indptr) > 0) & (np.diff(test_csr.indptr) > 0))
    users = np.sort(
        rng.choice(eligible, size=min(eval_cfg.n_eval_users, len(eligible)), replace=False)
    )
    log.info("Evaluating %s of %s eligible users", f"{len(users):,}", f"{len(eligible):,}")

    model_names = [*fixed_scores, "Content (genome)", "ALS"] + [
        f"Hybrid (cf={w})" for w in eval_cfg.cf_weights
    ]
    acc = {m: _Accumulator(eval_cfg.k_values) for m in model_names}
    n_cold_users = 0

    for chunk in np.array_split(users, max(1, len(users) // 500)):
        rows = train_ratings[chunk]
        exclude = rows.toarray() > 0
        xs, profiles = [], []
        for r in range(len(chunk)):
            items = rows.indices[rows.indptr[r] : rows.indptr[r + 1]]
            ratings = rows.data[rows.indptr[r] : rows.indptr[r + 1]]
            xs.append(als.fold_in(items, confidence(ratings, als_cfg.alpha)).x)
            p = content.profile(items, ratings)
            profiles.append(p if p is not None else np.zeros(genome.unit.shape[1]))
        als_s = als.scores(np.vstack(xs))
        content_s = content.scores(np.vstack(profiles))

        scores = {name: np.broadcast_to(s, exclude.shape) for name, s in fixed_scores.items()}
        scores["Content (genome)"] = content_s
        scores["ALS"] = als_s
        for w in eval_cfg.cf_weights:
            scores[f"Hybrid (cf={w})"] = blend(als_s, content_s, w, genome.has, exclude)

        relevant = [set(test_csr[u].indices.tolist()) for u in chunk]
        cold_relevant = [{i for i in rel if cold_items[i]} for rel in relevant]
        n_cold_users += sum(1 for c in cold_relevant if c)

        for name, s in scores.items():
            tops = top_k(s, k_max, exclude)
            a = acc[name]
            for top, rel, cold in zip(tops, relevant, cold_relevant, strict=True):
                for k in eval_cfg.k_values:
                    for metric, v in ranking_metrics(top, rel, k).items():
                        a.add(f"{metric}@{k}", v)
                if cold:
                    a.add("cold_recall@20", len(cold.intersection(top[:20].tolist())) / len(cold))
                a.add("novelty@10", float(self_info[top[:10]].mean()))
                a.add("diversity@10", intra_list_diversity(top[:10], genome))
                a.recommended.update(top[:10].tolist())

    return {
        "n_train_interactions": int(len(train)),
        "n_test_interactions": int(len(test)),
        "n_eval_users": int(len(users)),
        "n_users_with_cold_test_items": int(n_cold_users),
        "n_cold_items": int(cold_items.sum()),
        "models": {m: a.result(n_items) for m, a in acc.items()},
    }


def select_cf_weight(models: dict[str, dict[str, float]], cf_weights, metric="ndcg@10") -> float:
    """Pick the blend weight (including the pure endpoints) with the best validation metric."""
    candidates = {0.0: models["Content (genome)"], 1.0: models["ALS"]}
    candidates.update({w: models[f"Hybrid (cf={w})"] for w in cf_weights})
    return max(candidates, key=lambda w: candidates[w][metric])


# ── reporting ────────────────────────────────────────────────────────────────

_COLUMNS = [
    ("precision@10", "P@10"),
    ("recall@10", "R@10"),
    ("ndcg@10", "NDCG@10"),
    ("map@10", "MAP@10"),
    ("cold_recall@20", "ColdR@20"),
    ("coverage@10", "Coverage"),
    ("novelty@10", "Novelty"),
    ("diversity@10", "Diversity"),
]


def markdown_table(models: dict[str, dict[str, float]], include_cold: bool = True) -> str:
    cols = [c for c in _COLUMNS if include_cold or c[0] != "cold_recall@20"]
    best = {key: max(m.get(key, float("-inf")) for m in models.values()) for key, _ in cols}
    lines = [
        "| Model | " + " | ".join(label for _, label in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for name, m in models.items():
        cells = []
        for key, _ in cols:
            v = m.get(key)
            cell = "–" if v is None else (f"{v:.1%}" if key == "coverage@10" else f"{v:.4f}")
            cells.append(f"**{cell}**" if v is not None and v == best[key] else cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
