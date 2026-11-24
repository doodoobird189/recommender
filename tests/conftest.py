"""Tiny synthetic catalogue so tests run in milliseconds without MovieLens on disk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.engine import Engine
from recommender.genome import Genome

N_ITEMS, N_FACTORS, N_TAGS = 40, 8, 6
TAGS = np.array(["dark", "funny", "space", "romance", "great movie", "heist"])


@pytest.fixture
def genome() -> Genome:
    rng = np.random.default_rng(1)
    covered = np.arange(30)  # items 30..39 have no genome data
    z = rng.normal(size=(len(covered), N_TAGS)).astype(np.float16)
    return Genome.from_arrays(covered, z, TAGS, N_ITEMS)


@pytest.fixture
def items() -> pd.DataFrame:
    idx = np.arange(N_ITEMS, dtype=np.int32)
    return pd.DataFrame(
        {
            "item_idx": idx,
            "movieId": idx * 10 + 1,
            "title": [f"Movie {i}" for i in idx],
            "year": pd.array([1990 + i for i in idx], dtype="Int16"),
            "genres": ["Drama|Comedy"] * N_ITEMS,
            "tmdbId": pd.array([None] + list(range(1, N_ITEMS)), dtype="Int64"),
            "n_ratings": (N_ITEMS - idx) * 100,
            "mean_rating": np.full(N_ITEMS, 3.5, dtype=np.float32),
        }
    )


@pytest.fixture
def item_factors() -> np.ndarray:
    return np.random.default_rng(2).normal(scale=0.3, size=(N_ITEMS, N_FACTORS)).astype(np.float32)


@pytest.fixture
def engine(items, item_factors, genome) -> Engine:
    sample = pd.DataFrame(
        {
            "user_idx": [7, 7, 7, 9, 9],
            "item_idx": [1, 2, 3, 4, 5],
            "rating": [5.0, 4.0, 2.0, 3.5, 4.5],
        }
    )
    card = {
        "n_users": 100,
        "n_items": N_ITEMS,
        "n_interactions": 1000,
        "als": {"regularization": 0.1, "alpha": 1.0, "factors": N_FACTORS},
    }
    return Engine(items, item_factors, genome, card, sample_users=sample)
