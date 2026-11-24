"""Loading, cleaning, encoding and splitting MovieLens interactions."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

_TITLE_YEAR = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})(?:[-–]\d{0,4})?\)\s*$")
_TRAILING_ARTICLE = re.compile(r"^(?P<rest>.*), (?P<article>The|A|An)$")


# ── loading ──────────────────────────────────────────────────────────────────


def load_ratings(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "ratings.csv", engine="pyarrow")
    return df.astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    )


def load_movies(raw_dir: Path) -> pd.DataFrame:
    movies = pd.read_csv(raw_dir / "movies.csv")
    links = pd.read_csv(raw_dir / "links.csv", dtype={"movieId": "int32", "tmdbId": "Int64"})
    parsed = movies["title"].map(parse_title)
    movies["title"] = parsed.map(lambda p: p[0])
    movies["year"] = parsed.map(lambda p: p[1]).astype("Int16")
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    return movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")


def parse_title(raw: str) -> tuple[str, int | None]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999)."""
    raw = raw.strip()
    year = None
    m = _TITLE_YEAR.match(raw)
    if m:
        raw, year = m.group("title"), int(m.group("year"))
    a = _TRAILING_ARTICLE.match(raw)
    if a:
        raw = f"{a.group('article')} {a.group('rest')}"
    return raw, year


# ── cleaning ─────────────────────────────────────────────────────────────────


def filter_k_core(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """Iteratively drop users/items below the interaction thresholds until stable.

    Removing sparse items can push users below the threshold (and vice versa), hence the loop.
    """
    while True:
        before = len(df)
        df = df[df.groupby("userId")["userId"].transform("size") >= min_user]
        df = df[df.groupby("movieId")["movieId"].transform("size") >= min_item]
        if len(df) == before:
            return df.reset_index(drop=True)


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """Map raw user/movie ids to contiguous 0..n-1 indices (items ordered by movieId)."""
    df = df.copy()
    df["user_idx"], users = pd.factorize(df["userId"], sort=True)
    df["item_idx"], items = pd.factorize(df["movieId"], sort=True)
    df["user_idx"] = df["user_idx"].astype(np.int32)
    df["item_idx"] = df["item_idx"].astype(np.int32)
    return df, users, items


def build_items_table(item_index: pd.Index, interactions: pd.DataFrame, movies: pd.DataFrame):
    stats = interactions.groupby("item_idx").agg(
        n_ratings=("rating", "size"), mean_rating=("rating", "mean")
    )
    items = pd.DataFrame({"item_idx": np.arange(len(item_index), dtype=np.int32)})
    items["movieId"] = item_index.to_numpy().astype(np.int32)
    items = items.merge(movies, on="movieId", how="left").join(stats, on="item_idx")
    items["mean_rating"] = items["mean_rating"].astype("float32").round(3)
    return items[
        ["item_idx", "movieId", "title", "year", "genres", "tmdbId", "n_ratings", "mean_rating"]
    ]


# ── matrices ─────────────────────────────────────────────────────────────────


def confidence(ratings: np.ndarray, alpha: float) -> np.ndarray:
    """Hu et al. confidence. Always >= 1, so every observed item counts as a positive signal."""
    return (1.0 + alpha * np.asarray(ratings, dtype=np.float32)).astype(np.float32)


def to_csr(
    df: pd.DataFrame, n_users: int, n_items: int, values: np.ndarray | None = None
) -> csr_matrix:
    vals = df["rating"].to_numpy(np.float32) if values is None else values
    m = csr_matrix(
        (vals, (df["user_idx"].to_numpy(), df["item_idx"].to_numpy())),
        shape=(n_users, n_items),
        dtype=np.float32,
    )
    m.sum_duplicates()
    m.sort_indices()
    return m


# ── splits ───────────────────────────────────────────────────────────────────


def random_split(df: pd.DataFrame, test_fraction: float, seed: int):
    """Hold out a uniformly random fraction of interactions (the classic, optimistic setup)."""
    mask = np.random.default_rng(seed).random(len(df)) < test_fraction
    return df[~mask], df[mask]


def temporal_split(df: pd.DataFrame, test_fraction: float):
    """Global time cutoff: train on the past, test on the future.

    This mirrors deployment — the model never sees interactions that happen after it was trained —
    and exposes cold-start items that only start receiving ratings after the cutoff.
    """
    cutoff = int(df["timestamp"].quantile(1 - test_fraction))
    return df[df["timestamp"] < cutoff], df[df["timestamp"] >= cutoff], cutoff
