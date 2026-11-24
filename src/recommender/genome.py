"""MovieLens Tag Genome: a dense 1,128-dimensional content descriptor per movie.

Each (movie, tag) pair has a relevance score in [0, 1] (e.g. how "mind-bending" a movie is).
Raw relevances are dominated by tags that are mildly relevant to everything, so we z-score each
tag across the catalog: a value of +3 then means "far more <tag> than a typical movie".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Tags that judge quality, record awards or catalogue metadata rather than describe content.
# They are kept in the similarity model (they carry signal) but never shown as explanations.
_UNINFORMATIVE_PREFIXES = (
    "oscar",
    "afi 100",
    "saturn award",
    "golden palm",
    "best of",
    "imdb",
    "movielens",
    "nudity",
    "criterion",
    "free to download",
    "1001 movies",
)
UNINFORMATIVE_TAGS = frozenset(
    {
        "good",
        "great",
        "great movie",
        "excellent",
        "very good",
        "original",
        "classic",
        "masterpiece",
        "awesome",
        "brilliant",
        "genius",
        "perfect",
        "stunning",
        "beautiful",
        "interesting",
        "very interesting",
        "entertaining",
        "fun",
        "fun movie",
        "cool",
        "bad",
        "boring",
        "boring!",
        "dumb",
        "stupid",
        "stupidity",
        "lame",
        "pointless",
        "overrated",
        "underrated",
        "disappointing",
        "predictable",
        "pretentious",
        "catastrophe",
        "excellent script",
        "good acting",
        "great acting",
        "bad acting",
        "exceptional acting",
        "good dialogue",
        "great dialogue",
        "good story",
        "story",
        "storytelling",
        "plot",
        "script",
        "original plot",
        "bad plot",
        "bad script",
        "no plot",
        "good soundtrack",
        "great soundtrack",
        "awesome soundtrack",
        "good music",
        "great music",
        "notable soundtrack",
        "great ending",
        "bad ending",
        "great cinematography",
        "amazing cinematography",
        "cinematography",
        "amazing photography",
        "good action",
        "good sequel",
        "bad sequel",
        "not as good as the first",
        "highly quotable",
        "quotable",
        "potential oscar nom",
        "oscar winner",
        "waste of time",
        "too long",
        "long",
        "slow",
        "time",
        "mentor",
        "sexy",
        "male nudity",
        "notable nudity",
        "mindfuck",
        "funny as hell",
        "stupid as hell",
        "bullshit history",
        "watch the credits",
        "dvd-video",
        "tumey's dvds",
        "clv",
        "netflix finland",
        "reviewed",
        "seen more than once",
        "seen at the cinema",
    }
)


def is_informative(tag: str) -> bool:
    return tag not in UNINFORMATIVE_TAGS and not tag.startswith(_UNINFORMATIVE_PREFIXES)


def build_genome(raw_dir: Path, item_movie_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Return arrays aligned to our item index, ready to be saved with np.savez_compressed."""
    tags = pd.read_csv(raw_dir / "genome-tags.csv").sort_values("tagId")
    scores = pd.read_csv(
        raw_dir / "genome-scores.csv",
        engine="pyarrow",
        dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"},
    ).sort_values(["movieId", "tagId"])

    n_tags = len(tags)
    movie_ids = scores["movieId"].unique()
    if len(scores) != len(movie_ids) * n_tags:
        raise ValueError("genome-scores.csv is not a complete movie x tag grid")
    relevance = scores["relevance"].to_numpy(np.float32).reshape(len(movie_ids), n_tags)

    # keep only movies that are in our (filtered) catalog
    item_pos = pd.Series(np.arange(len(item_movie_ids)), index=item_movie_ids)
    in_catalog = np.isin(movie_ids, item_movie_ids)
    relevance = relevance[in_catalog]
    item_idx = item_pos.loc[movie_ids[in_catalog]].to_numpy(np.int32)

    mean, std = relevance.mean(axis=0), relevance.std(axis=0) + 1e-6
    z = (relevance - mean) / std
    return {
        "item_idx": item_idx,
        "z": z.astype(np.float16),
        "tags": tags["tag"].to_numpy(dtype=str),
    }


@dataclass
class Genome:
    """Genome vectors aligned to the full item index (rows of zeros for uncovered items)."""

    z: np.ndarray  # (n_items, n_tags) per-tag z-scores
    unit: np.ndarray  # (n_items, n_tags) L2-normalised z, for cosine similarity
    has: np.ndarray  # (n_items,) bool coverage mask
    tags: np.ndarray  # (n_tags,) tag names
    informative: np.ndarray  # (n_tags,) bool, False for UNINFORMATIVE_TAGS

    @classmethod
    def from_arrays(cls, item_idx: np.ndarray, z: np.ndarray, tags: np.ndarray, n_items: int):
        full = np.zeros((n_items, z.shape[1]), dtype=np.float32)
        full[item_idx] = z.astype(np.float32)
        norms = np.linalg.norm(full, axis=1, keepdims=True)
        unit = np.divide(full, norms, out=np.zeros_like(full), where=norms > 0)
        has = np.zeros(n_items, dtype=bool)
        has[item_idx] = True
        informative = np.array([is_informative(str(t)) for t in tags])
        return cls(z=full, unit=unit, has=has, tags=np.asarray(tags), informative=informative)

    @classmethod
    def load(cls, path: Path, n_items: int) -> Genome:
        with np.load(path, allow_pickle=False) as f:
            return cls.from_arrays(f["item_idx"], f["z"], f["tags"], n_items)

    def top_tags(self, vector: np.ndarray, n: int, min_value: float = 0.5) -> list[str]:
        v = np.where(self.informative, vector, -np.inf)
        order = np.argsort(-v)[:n]
        return [str(self.tags[i]) for i in order if v[i] >= min_value]
