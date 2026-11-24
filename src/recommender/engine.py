"""Serving-time recommender: loads artifacts once, answers requests from any set of ratings."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CF_WEIGHT, Paths
from .data import confidence
from .genome import Genome
from .scoring import ALSScorer, ContentScorer, blend_parts, top_k

log = logging.getLogger(__name__)

LIKED_RATING = 5.0  # rating assumed for movies a visitor clicks "like" on


@dataclass
class Movie:
    item_idx: int
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    tmdb_id: int | None
    n_ratings: int
    mean_rating: float

    def to_dict(self) -> dict:
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "tmdb_id": self.tmdb_id,
            "n_ratings": self.n_ratings,
            "mean_rating": self.mean_rating,
        }


class Engine:
    def __init__(
        self,
        items: pd.DataFrame,
        item_factors: np.ndarray,
        genome: Genome,
        model_card: dict,
        sample_users: pd.DataFrame | None = None,
    ):
        self.items = items.sort_values("item_idx").reset_index(drop=True)
        self.n_items = len(self.items)
        if item_factors.shape[0] != self.n_items:
            raise ValueError("item factors and items table are misaligned")
        self.als = ALSScorer(item_factors, model_card["als"]["regularization"])
        self.content = ContentScorer(genome)
        self.genome = genome
        self.alpha = model_card["als"]["alpha"]
        self.model_card = model_card
        self.sample_users = sample_users

        self._movie_to_idx = dict(zip(self.items["movieId"], self.items["item_idx"], strict=True))
        self._popular = np.argsort(-self.items["n_ratings"].to_numpy(), kind="stable")
        self._search_titles = self.items["title"].str.casefold().to_numpy()
        self._movies = [self._row_to_movie(r) for r in self.items.itertuples(index=False)]

    # ── loading ──────────────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(cls, paths: Paths) -> Engine:
        missing = [
            p
            for p in (paths.items, paths.item_factors, paths.genome, paths.model_card)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts {[str(m) for m in missing]}. "
                "Run `uv run recommender prepare && uv run recommender train` first."
            )
        items = pd.read_parquet(paths.items)
        sample = pd.read_parquet(paths.sample_users) if paths.sample_users.exists() else None
        engine = cls(
            items=items,
            item_factors=np.load(paths.item_factors),
            genome=Genome.load(paths.genome, len(items)),
            model_card=json.loads(paths.model_card.read_text()),
            sample_users=sample,
        )
        log.info("Engine ready: %s movies", f"{engine.n_items:,}")
        return engine

    @staticmethod
    def _row_to_movie(r) -> Movie:
        return Movie(
            item_idx=int(r.item_idx),
            movie_id=int(r.movieId),
            title=str(r.title),
            year=None if pd.isna(r.year) else int(r.year),
            genres=[g for g in str(r.genres or "").split("|") if g],
            tmdb_id=None if pd.isna(r.tmdbId) else int(r.tmdbId),
            n_ratings=int(r.n_ratings),
            mean_rating=float(r.mean_rating),
        )

    # ── catalogue ────────────────────────────────────────────────────────────

    def movie(self, movie_id: int) -> Movie | None:
        idx = self._movie_to_idx.get(movie_id)
        return None if idx is None else self._movies[idx]

    def popular(self, n: int, pool: int = 300, seed: int | None = None) -> list[Movie]:
        """A random draw from the `pool` most-rated movies (recognisable picks for onboarding)."""
        candidates = self._popular[:pool]
        if seed is not None:
            candidates = np.random.default_rng(seed).permutation(candidates)
        return [self._movies[i] for i in candidates[:n]]

    def search(self, query: str, limit: int = 10) -> list[Movie]:
        q = query.strip().casefold()
        if len(q) < 2:
            return []
        hits = [i for i in self._popular if q in self._search_titles[i]]
        # exact-prefix matches first, then by popularity
        hits.sort(key=lambda i: not self._search_titles[i].startswith(q))
        return [self._movies[i] for i in hits[:limit]]

    def random_sample_user(self, seed: int | None = None) -> dict | None:
        if self.sample_users is None or self.sample_users.empty:
            return None
        rng = np.random.default_rng(seed)
        uid = rng.choice(self.sample_users["user_idx"].unique())
        rows = self.sample_users[self.sample_users["user_idx"] == uid]
        ratings = {
            int(self.items.at[int(i), "movieId"]): float(r)
            for i, r in zip(rows["item_idx"], rows["rating"], strict=True)
        }
        return {"user_idx": int(uid), "ratings": ratings}

    # ── recommendation ───────────────────────────────────────────────────────

    def recommend(
        self, ratings: dict[int, float], n: int = 12, cf_weight: float = DEFAULT_CF_WEIGHT
    ) -> dict:
        """Recommend for an arbitrary user given {movieId: rating}. Unknown movieIds are ignored."""
        known = [(self._movie_to_idx[m], r) for m, r in ratings.items() if m in self._movie_to_idx]
        if not known:
            raise ValueError("None of the provided movies are in the catalogue")
        items = np.array([i for i, _ in known], dtype=np.int64)
        rvals = np.array([r for _, r in known], dtype=np.float32)

        fold = self.als.fold_in(items, confidence(rvals, self.alpha))
        als_scores = self.als.scores(fold.x)
        profile = self.content.profile(items, rvals)
        content_scores = (
            self.content.scores(profile) if profile is not None else np.zeros(self.n_items)
        )

        exclude = np.zeros(self.n_items, dtype=bool)
        exclude[items] = True
        cf_part, content_part = blend_parts(
            als_scores, content_scores, cf_weight, self.genome.has, exclude
        )
        scores = cf_part + content_part
        top = top_k(scores, n, exclude)

        taste = self.content.taste(items, rvals)
        explanations = self._explain(fold, top, items, rvals, taste, cf_part >= content_part)
        recs = []
        for rank, (i, why) in enumerate(zip(top, explanations, strict=True), start=1):
            m = self._movies[i].to_dict()
            m.update(
                rank=rank,
                score=round(float(scores[i]), 3),
                driven_by="people like you" if cf_part[i] >= content_part[i] else "similar content",
                cf_score=round(float(als_scores[i]), 4),
                content_score=round(float(content_scores[i]), 4) if self.genome.has[i] else None,
                **why,
            )
            recs.append(m)

        return {
            "cf_weight": cf_weight,
            "n_known_ratings": len(known),
            "taste_profile": self.genome.top_tags(taste, 8),
            "recommendations": recs,
        }

    def _explain(self, fold, top, items, rvals, taste, cf_dominant) -> list[dict]:
        """'Because you liked …' from whichever signal drove each recommendation.

        * CF-driven: the user's movies with the largest exact ALS score contributions.
        * Content-driven: the user's movies closest to it in tag-genome space.
        Shared tags are those strong in both the movie and the user's taste profile.
        """
        liked = rvals > 3.0
        contrib = self.als.contributions(fold, top)  # (n, n_interactions)
        genome_sim = self.genome.unit[top] @ self.genome.unit[items].T  # (n, n_interactions)
        out = []
        for row, sim, item in zip(contrib, genome_sim, top, strict=True):
            use_cf = cf_dominant[item] or not self.genome.has[item]
            strength = row if use_cf else sim
            because = [
                {"movie_id": self._movies[items[j]].movie_id, "title": self._movies[items[j]].title}
                for j in np.argsort(-strength)
                if strength[j] > 0 and liked[j]
            ][:2]
            shared = (
                self.genome.top_tags(np.minimum(self.genome.z[item], taste), 3, min_value=0.3)
                if self.genome.has[item]
                else []
            )
            out.append({"because_you_liked": because, "shared_tags": shared})
        return out

    @staticmethod
    def liked_to_ratings(movie_ids: list[int]) -> dict[int, float]:
        return dict.fromkeys(movie_ids, LIKED_RATING)


def load_metrics(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
