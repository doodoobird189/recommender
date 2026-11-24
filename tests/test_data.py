import numpy as np
import pandas as pd
import pytest

from recommender.data import (
    confidence,
    encode_ids,
    filter_k_core,
    parse_title,
    random_split,
    temporal_split,
    to_csr,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        ("City of God (Cidade de Deus) (2002)", ("City of God (Cidade de Deus)", 2002)),
        ("Babylon 5", ("Babylon 5", None)),
        ("Fawlty Towers (1975-1979)", ("Fawlty Towers", 1975)),
    ],
)
def test_parse_title(raw, expected):
    assert parse_title(raw) == expected


def _ratings(n_users=30, n_items=20, density=0.6, seed=0):
    rng = np.random.default_rng(seed)
    u, i = np.nonzero(rng.random((n_users, n_items)) < density)
    return pd.DataFrame(
        {
            "userId": u + 100,
            "movieId": i + 1000,
            "rating": rng.choice([1.0, 3.5, 5.0], len(u)),
            "timestamp": rng.integers(0, 10_000, len(u)),
        }
    )


def test_k_core_is_stable_and_respects_thresholds():
    df = filter_k_core(_ratings(density=0.3), min_user=5, min_item=5)
    assert (df.groupby("userId").size() >= 5).all()
    assert (df.groupby("movieId").size() >= 5).all()
    assert len(filter_k_core(df, 5, 5)) == len(df)


def test_encode_ids_contiguous():
    df, users, items = encode_ids(_ratings())
    assert df["user_idx"].max() + 1 == len(users) == df["userId"].nunique()
    assert set(df["item_idx"]) == set(range(len(items)))
    # mapping round-trips
    assert (items[df["item_idx"]] == df["movieId"].to_numpy()).all()


def test_confidence_at_least_one():
    c = confidence(np.array([0.5, 5.0]), alpha=2.0)
    np.testing.assert_allclose(c, [2.0, 11.0])
    assert (c >= 1).all()


def test_random_split_partitions():
    df = _ratings()
    train, test = random_split(df, 0.2, seed=0)
    assert len(train) + len(test) == len(df)
    assert set(train.index).isdisjoint(test.index)
    assert 0.1 < len(test) / len(df) < 0.3


def test_temporal_split_has_no_future_leakage():
    df = _ratings()
    train, test, cutoff = temporal_split(df, 0.1)
    assert train["timestamp"].max() < cutoff <= test["timestamp"].min()
    assert len(train) + len(test) == len(df)


def test_to_csr_shape_and_values():
    df = pd.DataFrame({"user_idx": [0, 0, 2], "item_idx": [1, 3, 0], "rating": [4.0, 2.0, 5.0]})
    m = to_csr(df, 3, 4)
    assert m.shape == (3, 4)
    assert m[0, 1] == 4.0 and m[2, 0] == 5.0 and m.nnz == 3
